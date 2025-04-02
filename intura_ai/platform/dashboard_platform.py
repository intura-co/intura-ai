import os
from .domain import ExperimentModel
from intura_ai.shared.external.intura_api import InturaFetch


class DashboardPlatform:
    def __init__(self, intura_api_key=None):
        if not intura_api_key:
            intura_api_key = os.environ.get("INTURA_API_KEY")
        if not intura_api_key:
            raise ValueError("API Key Intura not found")
        self._intura_api_key = intura_api_key
        self._intura_api = InturaFetch(intura_api_key)
    
    def create_experiment(self, payload: ExperimentModel):
        return self._intura_api.insert_experiment(payload.model_dump_json())    