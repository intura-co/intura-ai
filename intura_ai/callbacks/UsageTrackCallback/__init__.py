import time
from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.agents import AgentAction, AgentFinish

from intura_ai.shared.external.intura_api import InturaFetch
from intura_ai.shared.utils.logging import get_component_logger

# Get component-specific logger
logger = get_component_logger("usage_track")

@dataclass
class CallbackConfig:
    """Configuration for the UsageTrackCallback."""
    intura_api_key: str
    experiment_id: str
    treatment_id: str
    treatment_name: str
    session_id: str
    model_name: str


class UsageTrackCallback(BaseCallbackHandler):
    """
    Callback handler for tracking LLM usage and sending logs to Intura API.
    
    This callback captures usage metrics such as tokens, latency, and
    sends the data to the Intura API for analysis and logging.
    """

    def __init__(
        self, 
        intura_api_key: str,
        experiment_id: str,
        treatment_id: str,
        treatment_name: str,
        session_id: str,
        model_name: str,
        verbose: bool = False
    ):
        """
        Initialize the UsageTrackCallback.
        
        Args:
            intura_api_key: API key for Intura services
            experiment_id: ID of the experiment
            treatment_id: ID of the treatment
            treatment_name: Name of the treatment
            session_id: Session ID for tracking
            verbose: Enable verbose logging for this component
        """
        super().__init__()
        
        # Store configuration
        self._config = CallbackConfig(
            intura_api_key=intura_api_key,
            experiment_id=experiment_id,
            treatment_id=treatment_id,
            treatment_name=treatment_name,
            session_id=session_id,
            model_name=model_name
        )
        
        # Initialize API client
        self._intura_api = InturaFetch(intura_api_key)
        
        # Tracking variables
        self._start_time: float = 0.0
        self._input_chat: List[Dict[str, str]] = []
        
        # Configure component-specific logging if verbose is specified
        if verbose:
            from intura_ai.shared.utils.logging import set_component_level
            set_component_level("usage_track", "debug")
            
        logger.debug(f"Initialized UsageTrackCallback for experiment: {experiment_id}")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self._start_time = time.perf_counter()
        logger.debug(f"LLM started with {len(prompts)} prompts")

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""
        self._start_time = time.perf_counter()
        
        # Extract messages in a format suitable for logging
        self._input_chat = []
        for message_list in messages:
            for message in message_list:
                self._input_chat.append({
                    "role": message.type,
                    "content": message.content
                })
        
        logger.debug(f"Chat model started with {len(self._input_chat)} messages")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """
        Run when LLM ends generating. Logs usage data to Intura API.
        
        This method calculates metrics like latency and token usage,
        then sends the data to the Intura API for tracking.
        
        Args:
            response: LLM generation result
            **kwargs: Additional arguments
        """
        end_time = time.perf_counter()
        latency = (end_time - self._start_time) * 1000
        
        logger.debug(f"LLM ended. Latency: {latency:.2f}ms")
        
        for resp_idx, resp in enumerate(response.generations):
            for inner_resp in resp:
                try:
                    # Extract usage metrics
                    usage_metadata = getattr(inner_resp.message, 'usage_metadata', {})
                    input_tokens = usage_metadata.get('input_tokens', 0)
                    output_tokens = usage_metadata.get('output_tokens', 0)
                    total_tokens = input_tokens + output_tokens
                    
                    # Get message content and ID
                    message_content = inner_resp.message.content
                    message_id = getattr(inner_resp.message, 'id', 'unknown')
                    
                    logger.debug(
                        f"Response {resp_idx+1}: {total_tokens} tokens "
                        f"({input_tokens} input, {output_tokens} output)"
                    )
                    
                    # Prepare payload for API
                    payload = self._create_log_payload(
                        prediction_id=message_id,
                        result=message_content,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency=latency
                    )
                    
                    # Send to API
                    self._log_to_api(payload)
                    
                except Exception as e:
                    logger.error(f"Error processing LLM end event: {str(e)}")
    
    def _create_log_payload(
        self,
        prediction_id: str,
        result: str,
        input_tokens: int,
        output_tokens: int,
        latency: float,
        cached_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create the payload for the Intura API log.
        
        Args:
            prediction_id: ID of the prediction
            result: Text result from the LLM
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency: Latency in milliseconds
            cached_tokens: Number of cached tokens, if any
            
        Returns:
            Payload dictionary for the API
        """
        return {
            "session_id": self._config.session_id,
            "experiment_id": self._config.experiment_id,
            "content": self._input_chat,
            "latency": latency,
            "result": [
                {
                    "treatment_id": self._config.treatment_id,
                    "treatment_name": self._config.treatment_name,
                    "prediction_id": prediction_id,
                    "model_name": self._config.model_name,
                    "predictions": {
                        "result": result,
                        "cost": {
                            "total_tokens": input_tokens + output_tokens,
                            "output_tokens": output_tokens,
                            "input_tokens": input_tokens,
                            "cached_tokens": cached_tokens
                        },
                        "latency": latency
                    },
                    "prediction_attribute": {
                        "source": "SDK",
                        "package_name": "intura-ai"
                    }
                }
            ]
        }
    
    def _log_to_api(self, payload: Dict[str, Any]) -> None:
        """
        Send log data to the Intura API.
        
        Args:
            payload: Data payload to send
        """
        try:
            self._intura_api.insert_log_inference(payload=payload)
            logger.debug("Successfully sent log to Intura API")
        except Exception as e:
            logger.error(f"Failed to send log to Intura API: {str(e)}")

    # Implementing the remaining callback methods with minimal stubs
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when LLM errors."""
        logger.error(f"LLM error occurred: {str(error)}")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain starts running."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        pass

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when chain errors."""
        pass

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """Run when tool starts running."""
        pass

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        pass

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when tool errors."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        pass