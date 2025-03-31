import os
from intura_ai.shared.external.intura_api import InturaFetch
from .domain import Experiment, Treatment
class InturaPlatform:
    def __init__(self, intura_api_key=None):
        if not intura_api_key:
            intura_api_key = os.environ.get("INTURA_API_KEY")
        self._intura_api_key = intura_api_key
        self._intura_api = InturaFetch(intura_api_key)
    
    def create_experiment(self, payload: Experiment):
        print(payload)
        pass