import requests, os
from datetime import datetime
from uuid import uuid4
from intura_ai.shared.variables.api_host import INTURA_API_HOST

class InturaFetch:
    _endpoint_check_api_key         = "external/validate-api-key"
    _endpoint_check_experiment_id   = "external/validate-experiment"
    _endpoint_get_detail_experiment = "external/experiment/detail"
    _endpoint_insert_inference      = "external/insert/inference"
    _endpoint_track_reward          = "ai/track"
    
    def __init__(self, intura_api_key=None):
        self._api_host      = INTURA_API_HOST
        self._api_version   = "v1"
        
        if not intura_api_key:
            intura_api_key =  os.environ.get("INTURA_API_KEY")
            
        if not intura_api_key:
            raise ValueError("Intura API Key Not Found")
        
        self._headers       = {
            'x-request-id': str(uuid4()),
            'x-timestamp': str(datetime.now().timestamp() * 1000),
            'x-api-key': intura_api_key,
            'Content-Type': 'application/json',
        }
        
        if not self._check_api_key():
            raise ValueError("Incorrect Intura API Key")
        
    def _check_api_key(self):
        endpoint = "/".join([self._api_host, self._api_version, self._endpoint_check_api_key])
        resp = requests.get(endpoint, headers=self._headers)
        if resp.status_code == 200:
            return True
        else:
            return False
        
    def check_experiment_id(self, experiment_id):
        endpoint = "/".join([self._api_host, self._api_version, self._endpoint_check_experiment_id])
        endpoint = endpoint + f"?experiment_id={experiment_id}"
        resp = requests.get(endpoint, headers=self._headers)
        if resp.status_code == 200:
            return True
        else:
            return False
    
    def get_experiment_detail(self, experiment_id, features={}):
        endpoint = "/".join([self._api_host, self._api_version, self._endpoint_get_detail_experiment])
        endpoint = endpoint + f"?experiment_id={experiment_id}"
        resp = requests.post(endpoint, json={
            "features": features
        }, headers=self._headers)
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
        
    def insert_log_inference(self,payload):
        endpoint = "/".join([self._api_host, self._api_version, self._endpoint_insert_inference])
        resp = requests.post(endpoint, json=payload, headers=self._headers)
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
        
        
    def insert_chat_usage(self, values):
        return True
        request_body = {
            "body": {
                "event_name": "CHAT_MODEL_USAGE",
                "event_value": values,
                "attributes": {},
                "prediction_id": str(uuid4())
            },
            "reward_type": "RESERVED_REWARD",
            "reward_category": "CHAT_USAGE"
        }
        endpoint = "/".join([self._api_host, self._api_version, self._endpoint_track_reward])
        resp = requests.post(endpoint, json=request_body, headers=self._headers)
        if resp.status_code == 200:
            return True
        else:
            return False
        
    def insert_chat_output(self, values):
        request_body = {
            "body": {
                "event_name": "CHAT_MODEL_OUTPUT",
                "event_value": values,
                "attributes": {},
                "prediction_id": str(uuid4())
            },
            "reward_type": "RESERVED_REWARD",
            "reward_category": "CHAT_LOG"
        }
        endpoint = "/".join([self._api_host, self._api_version, self._endpoint_track_reward])
        resp = requests.post(endpoint, json=request_body, headers=self._headers)
        if resp.status_code == 200:
            return True
        else:
            return False
        
    def insert_chat_input(self, values):
        request_body = {
            "body": {
                "event_name": "CHAT_MODEL_INPUT",
                "event_value": values,
                "attributes": {},
                "prediction_id": str(uuid4())
            },
            "reward_type": "RESERVED_REWARD",
            "reward_category": "CHAT_LOG"
        }
        endpoint = "/".join([self._api_host, self._api_version, self._endpoint_track_reward])
        resp = requests.post(endpoint, json=request_body, headers=self._headers)
        if resp.status_code == 200:
            return True
        else:
            return False