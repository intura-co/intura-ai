from uuid import uuid4
from intura_ai.libs.wrappers.langchain_chat_model import (
    InturaChatOpenAI,
    InturaChatAnthropic,
    InturaChatDeepSeek,
    InturaChatGoogleGenerativeAI
)
from intura_ai.shared.external.intura_api import InturaFetch
from intura_ai.callbacks import UsageTrackCallback

class ChatModelExperiment:
    def __init__(self, intura_api_key=None):
        self._choiced_model = None
        self._intura_api_key = intura_api_key
        self._intura_api = InturaFetch(intura_api_key)
        
    @property
    def choiced_model(self):
        return self._choiced_model

    def build(self, experiment_id, session_id=None, features={}, max_pred=1):
        resp = self._intura_api.get_experiment_detail(experiment_id, features=features)
        if not resp:
            return None, {}, []
        self._data = resp["data"]
        results = []
        result = {}
        for row in self._data:
            if row["model_provider"] == "Google":
                model = InturaChatGoogleGenerativeAI
            elif row["model_provider"] == "Anthropic":
                model = InturaChatAnthropic
            elif row["model_provider"] == "Deepseek":
                model = InturaChatDeepSeek
            elif row["model_provider"] == "OpenAI":
                model = InturaChatOpenAI
            else:
                raise NotImplementedError("Model not implemented")
            
            chat_templates = [("system", f"{prompt['name']} {prompt['value']}") for prompt in row["prompts"]]
            session_id = session_id or str(uuid4())
            model_configuration = {}
            for key in row["model_configuration"]:
                if row["model_configuration"][key]:
                    model_configuration[key] = row["model_configuration"][key]
            result = model, {
                **model_configuration,
                "callbacks": [
                    UsageTrackCallback(
                        intura_api_key= self._intura_api_key,
                        experiment_id= experiment_id,
                        treatment_id= row["treatment_id"],
                        treatment_name= row["treatment_name"],
                        session_id= session_id,
                    )  
                ],
                "metadata": {
                    "experiment_id": experiment_id,
                    "treatment_id": row["treatment_id"],
                    "treatment_name": row["treatment_name"],
                    "session_id": session_id
                }
            }, chat_templates
            results.append(result)
            if len(results) == 1 and max_pred == 1:
                self._choiced_model = row["model_configuration"]["model"]
                return results[0]
            elif len(result) == max_pred:
                break
        return results
