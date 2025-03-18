import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from intura_ai.client import get_intura_client
from intura_ai.callbacks import UsageTrackCallback
class ChatModelExperiment:
    def __init__(self, experiment_id=None):
        self._initialized = False
        self._intura_api = get_intura_client()
        if self._intura_api:
            if self._intura_api.check_experiment_id(experiment_id=experiment_id):
                self._experiment_id = experiment_id
                resp = self._intura_api.get_experiment_detail(self._experiment_id)
                self._data = resp["data"]
                self._initialized = True
                self._usage_callback = UsageTrackCallback(experiment_id=experiment_id)
                
    @property
    def treatment(self):
        return self._data["treatment_id"]
    
    def build(self, api_key=None):
        if self._initialized:
            if self._data["model_provider"] == "Google":
                if not api_key:
                    api_key = os.getenv("GOOGLE_API_KEY")
                model = ChatGoogleGenerativeAI(
                    api_key=api_key,
                    max_retries=0,
                    callbacks=[self._usage_callback],
                    **self._data["model_configuration"]
                )
            elif self._data["model_provider"] == "Anthropic":
                if not api_key:
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                model = ChatAnthropic(
                    api_key=api_key,
                    max_retries=0,
                    callbacks=[self._usage_callback],
                    **self._data["model_configuration"]
                )
            elif self._data["model_provider"] == "Deepseek":
                if not api_key:
                    api_key = os.getenv("DEEPSEEK_API_KEY")
                model = ChatDeepSeek(
                    api_key=api_key,
                    max_retries=0,
                    callbacks=[self._usage_callback],
                    **self._data["model_configuration"]
                )
            elif self._data["model_provider"] == "OpenAI":
                if not api_key:
                    api_key = os.getenv("OPENAI_API_KEY")
                model = ChatOpenAI(
                    api_key=api_key,
                    max_retries=0,
                    callbacks=[self._usage_callback],
                    **self._data["model_configuration"]
                )
            elif self._data["model_provider"] == "OnPremise":
                model = ChatOllama(
                    max_retries=0,
                    callbacks=[self._usage_callback],
                    **self._data["model_configuration"]
                )
            else:
                raise NotImplementedError("Model not implemented")
            messages = [("system", f"{prompt['name']} {prompt['value']}") for prompt in self._data["prompts"]]
            return model, messages
        else:
            raise ValueError("Need to initialized first")
