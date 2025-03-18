from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List, Union
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.agents import AgentAction, AgentFinish

from intura_ai.shared.middlewares import validate_class, validate_api_key
from intura_ai.client import get_intura_client

@validate_class(validate_api_key)
class UsageTrackCallback(BaseCallbackHandler):
    """Base callback handler that can be used to handle callbacks from langchain."""

    def __init__(self, experiment_id=None):
        super().__init__()
        self._initialized = False
        self._intura_api = get_intura_client()
        if self._intura_api:
            if self._intura_api.check_experiment_id(experiment_id=experiment_id):
                self._experiment_id = experiment_id
                self._initialized = True

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
        if self._initialized:
            result = []
            for message in messages:
                for row in message:
                    result.append({
                        "role": row.type,
                        "content": row.content
                    })
            self._intura_api.insert_chat_input(result)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        for resp in response.generations:
            for inner_resp in resp:
                self._intura_api.insert_chat_usage(inner_resp.message.usage_metadata)
                self._intura_api.insert_chat_output({
                    "content": inner_resp.message.content
                })
                
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