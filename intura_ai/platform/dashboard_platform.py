import os
from .domain import ExperimentModel
from intura_ai.shared.external.intura_api import InturaFetch


class DashboardPlatform:
    def __init__(self, intura_api_key=None):
        if not intura_api_key:
            intura_api_key = os.environ.get("INTURA_API_KEY")

        self._intura_api_key = intura_api_key
        self._intura_api = InturaFetch(intura_api_key)
    
    def create_experiment(self, payload: ExperimentModel):
        return self._intura_api.insert_experiment(payload.model_dump_json())    
    
    def list_models(self):
        return self._intura_api.get_list_models()   

    def list_experiments(self):
        return self._intura_api.get_list_experiment()
    
    def get_experiment_detail(self, experiment_id: str):
        return self._intura_api.get_experiment_detail(experiment_id=experiment_id)