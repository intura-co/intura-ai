import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from uuid import uuid4

from intura_ai.shared.variables.api_host import INTURA_API_HOST
from intura_ai.shared.utils.logging import get_component_logger

# Get component-specific logger
logger = get_component_logger("intura_api")

class InturaAPIError(Exception):
    """Exception raised for Intura API errors."""
    pass

class InturaFetch:
    """
    Client for interacting with the Intura API.
    
    This class provides methods to communicate with various Intura API endpoints
    for experiment management, model building, and logging.
    """
    
    # API endpoints
    ENDPOINTS = {
        # Authentication and validation
        "validate_api_key": "external/validate-api-key",
        "validate_experiment": "external/validate-experiment",
        
        # Inference and logging
        "insert_inference": "external/insert/inference",
        
        # Experiment management
        "experiment": "experiment",
        "list_models": "experiment/models",
        "experiment_detail": "experiment/detail",
        "build_chat_model": "experiment/build/chat",
        
        # Tracking and rewards
        "track_reward": "ai/track"
    }
    
    def __init__(self, intura_api_key: Optional[str] = None, verbose: bool = False):
        """
        Initialize the Intura API client.
        
        Args:
            intura_api_key: API key for authentication (falls back to INTURA_API_KEY env var)
            verbose: Enable verbose logging for this component
        """
        self._api_host = INTURA_API_HOST
        self._api_version = "v1"
        
        # Get API key from parameter or environment
        api_key = intura_api_key or os.environ.get("INTURA_API_KEY")
        if not api_key:
            logger.error("Intura API Key not found")
            raise ValueError("Intura API Key not found")
        
        # Configure component-specific logging if verbose is specified
        if verbose:
            from intura_ai.shared.utils.logging import set_component_level
            set_component_level("intura_api", "debug")
        
        # Initialize headers
        self._headers = self._create_headers(api_key)
        
        # Validate API key
        if not self._check_api_key():
            logger.error("Invalid Intura API Key")
            raise ValueError("Invalid Intura API Key")
            
        logger.debug("InturaFetch initialized successfully")
    
    def _create_headers(self, api_key: str) -> Dict[str, str]:
        """
        Create HTTP headers for API requests.
        
        Args:
            api_key: API key for authentication
            
        Returns:
            Dictionary of HTTP headers
        """
        return {
            'x-request-id': str(uuid4()),
            'x-timestamp': str(int(datetime.now().timestamp() * 1000)),
            'x-api-key': api_key,
            'Content-Type': 'application/json',
        }
    
    def _get_endpoint_url(self, endpoint_key: str) -> str:
        """
        Build the full URL for an API endpoint.
        
        Args:
            endpoint_key: Key of the endpoint in the ENDPOINTS dictionary
            
        Returns:
            Full URL for the endpoint
        """
        if endpoint_key not in self.ENDPOINTS:
            raise ValueError(f"Unknown endpoint: {endpoint_key}")
            
        return "/".join([self._api_host, self._api_version, self.ENDPOINTS[endpoint_key]])
    
    def _make_request(
        self, 
        method: str, 
        endpoint_key: str, 
        params: Optional[Dict[str, Any]] = None, 
        data: Optional[Union[Dict[str, Any], str]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make an HTTP request to the Intura API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint_key: Key of the endpoint in the ENDPOINTS dictionary
            params: URL parameters
            data: Request data (for POST/PUT)
            json_data: JSON data (for POST/PUT)
            
        Returns:
            Response data on success, None on failure
            
        Raises:
            InturaAPIError: If the API request fails
        """
        url = self._get_endpoint_url(endpoint_key)
        
        # Update request ID and timestamp for each request
        self._headers.update({
            'x-request-id': str(uuid4()),
            'x-timestamp': str(int(datetime.now().timestamp() * 1000)),
        })
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=self._headers)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, data=data, json=json_data, headers=self._headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Log response code
            logger.debug(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                # For endpoints that return JSON
                if response.headers.get('Content-Type', '').startswith('application/json'):
                    return response.json()
                # For endpoints that return plain text or other formats
                return {"status": "success", "code": response.status_code}
            else:
                logger.warning(
                    f"API request failed: {response.status_code} - {response.text[:100]}"
                )
                return None
                
        except requests.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise InturaAPIError(f"API request failed: {str(e)}")
    
    def _check_api_key(self) -> bool:
        """
        Validate the API key.
        
        Returns:
            True if valid, False otherwise
        """
        logger.debug("Validating API key")
        result = self._make_request("GET", "validate_api_key")
        return result is not None
    
    def get_list_experiment(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get a list of experiments.
        
        Returns:
            List of experiments or None if the request fails
        """
        logger.debug("Fetching list of experiments")
        response = self._make_request("GET", "experiment")
        return response.get("data") if response else None
    
    def insert_experiment(self, payload: Union[Dict[str, Any], str]) -> Optional[str]:
        """
        Create a new experiment.
        
        Args:
            payload: Experiment data
            
        Returns:
            Experiment ID on success, None on failure
        """
        logger.debug("Creating new experiment")
        response = self._make_request("POST", "experiment", data=payload)
        if response and "data" in response and "experiment_id" in response["data"]:
            experiment_id = response["data"]["experiment_id"]
            logger.info(f"Created experiment with ID: {experiment_id}")
            return experiment_id
        return None
    
    def get_list_models(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get a list of available models.
        
        Returns:
            List of models or None if the request fails
        """
        logger.debug("Fetching list of models")
        response = self._make_request("GET", "list_models")
        return response.get("data") if response else None
    
    def check_experiment_id(self, experiment_id: str) -> bool:
        """
        Check if an experiment ID is valid.
        
        Args:
            experiment_id: ID of the experiment to check
            
        Returns:
            True if valid, False otherwise
        """
        logger.debug(f"Validating experiment ID: {experiment_id}")
        params = {"experiment_id": experiment_id}
        response = self._make_request("GET", "validate_experiment", params=params)
        return response is not None
    
    def get_experiment_detail(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details of an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Experiment details or None if the request fails
        """
        logger.debug(f"Fetching details for experiment: {experiment_id}")
        params = {"experiment_id": experiment_id}
        return self._make_request("GET", "experiment_detail", params=params)
    
    def build_chat_model(
        self, experiment_id: str, features: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Build a chat model based on an experiment configuration.
        
        Args:
            experiment_id: ID of the experiment
            features: Features to include in the model
            
        Returns:
            Model configuration or None if the request fails
        """
        logger.debug(f"Building chat model for experiment: {experiment_id}")
        features = features or {}
        params = {"experiment_id": experiment_id}
        json_data = {"features": features}
        
        return self._make_request(
            "POST", "build_chat_model", params=params, json_data=json_data
        )
    
    def insert_log_inference(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Log inference data.
        
        Args:
            payload: Inference data
            
        Returns:
            Response data or None if the request fails
        """
        try:
            session_id = payload.get("session_id", "unknown")
            logger.debug(f"Logging inference for session: {session_id}")
            
            return self._make_request("POST", "insert_inference", json_data=payload)
        except Exception as e:
            logger.error(f"Error logging inference: {str(e)}")
            return None
    
    def _track_event(
        self, 
        event_name: str, 
        event_value: Any, 
        reward_category: str,
        prediction_id: Optional[str] = None
    ) -> bool:
        """
        Track an event in the Intura system.
        
        Args:
            event_name: Name of the event
            event_value: Value/data for the event
            reward_category: Category of the reward
            prediction_id: Optional prediction ID
            
        Returns:
            True on success, False on failure
        """
        prediction_id = prediction_id or str(uuid4())
        
        request_body = {
            "body": {
                "event_name": event_name,
                "event_value": event_value,
                "attributes": {},
                "prediction_id": prediction_id
            },
            "reward_type": "RESERVED_REWARD",
            "reward_category": reward_category
        }
        
        logger.debug(f"Tracking event: {event_name} in category: {reward_category}")
        response = self._make_request("POST", "track_reward", json_data=request_body)
        
        return response is not None
    
    def insert_chat_usage(self, values: Any) -> bool:
        """
        Log chat model usage.
        
        Args:
            values: Usage data
            
        Returns:
            True on success, False on failure
        """
        # NOTE: The original code returns True directly, preserved here
        logger.debug("Logging chat usage (disabled)")
        return True
        
        # This is the implementation if we want to enable it:
        # return self._track_event(
        #     event_name="CHAT_MODEL_USAGE",
        #     event_value=values,
        #     reward_category="CHAT_USAGE"
        # )
    
    def insert_chat_output(self, values: Any) -> bool:
        """
        Log chat model output.
        
        Args:
            values: Output data
            
        Returns:
            True on success, False on failure
        """
        return self._track_event(
            event_name="CHAT_MODEL_OUTPUT",
            event_value=values,
            reward_category="CHAT_LOG"
        )
    
    def insert_chat_input(self, values: Any) -> bool:
        """
        Log chat model input.
        
        Args:
            values: Input data
            
        Returns:
            True on success, False on failure
        """
        return self._track_event(
            event_name="CHAT_MODEL_INPUT",
            event_value=values,
            reward_category="CHAT_LOG"
        )