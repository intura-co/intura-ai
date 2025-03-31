import time
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List, Union
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.agents import AgentAction, AgentFinish

from intura_ai.shared.external.intura_api import InturaFetch

class UsageTrackCallback(BaseCallbackHandler):
    """Base callback handler that can be used to handle callbacks from langchain."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._intura_api = InturaFetch(kwargs["intura_api_key"])
        self._experiment_id = kwargs["experiment_id"]
        self._treatment_id = kwargs["treatment_id"]
        self._treatment_name = kwargs["treatment_name"]
        self._session_id = kwargs["session_id"]
        self._start_time = time.perf_counter() 
  
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        pass

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""
        # print("OK", self._experiment_id, messages, kwargs)
        self._start_time = time.perf_counter() 
        result = []
        for message in messages:
            for row in message:
                result.append({
                    "role": row.type,
                    "content": row.content
                })
        self._input_chat = result

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        for resp in response.generations:
            for inner_resp in resp:
                end_time = time.perf_counter()
                input_token = inner_resp.message.usage_metadata["input_tokens"]
                output_token = inner_resp.message.usage_metadata["output_tokens"]
                latency = (end_time - self._start_time) * 1000
                payload = {
                    "session_id": self._session_id,
                    "experiment_id": self._experiment_id,
                    "content": self._input_chat,
                    "latency": latency,
                    "result": [
                            {
                        "treatment_id": self._treatment_id,
                        "treatment_name": self._treatment_name,
                        "prediction_id": inner_resp.message.id,
                        "predictions": {
                            "result": inner_resp.message.content,
                            "cost": {
                                "total_tokens": input_token + output_token,
                                "output_tokens": output_token,
                                "input_tokens": input_token,
                                "cached_tokens": None
                            },
                            "latency": latency
                        },
                        "prediction_attribute": {
                            "source": "SDK"
                        }
                    }]
                }
                self._intura_api.insert_log_inference(payload=payload)
                
    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""