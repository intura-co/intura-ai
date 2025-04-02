import os
from typing import Optional, List, Dict, Any, Union

from .domain import ExperimentModel
from intura_ai.shared.external.intura_api import InturaFetch
from intura_ai.shared.utils.logging import get_component_logger

# Get component-specific logger
logger = get_component_logger("dashboard")

class DashboardError(Exception):
    """Exception raised for dashboard-related errors."""
    pass

class DashboardPlatform:
    """
    Platform for interacting with Intura's experiment dashboard.
    
    This class provides a simplified interface for creating and managing
    experiments through the Intura API.
    """
    
    def __init__(
        self, 
        intura_api_key: Optional[str] = None, 
        verbose: bool = False
    ):
        """
        Initialize the dashboard platform.
        
        Args:
            intura_api_key: API key for Intura services (falls back to INTURA_API_KEY env var)
            verbose: Enable verbose logging for this component
        """
        # Try to get API key from parameter or environment
        api_key = intura_api_key or os.environ.get("INTURA_API_KEY")
        if not api_key:
            logger.error("Intura API Key not found")
            raise ValueError("Intura API Key not found")
            
        self._intura_api_key = api_key
        
        # Configure component-specific logging if verbose is specified
        if verbose:
            from intura_ai.shared.utils.logging import set_component_level
            set_component_level("dashboard", "debug")
            
        # Initialize API client with same verbosity setting
        self._intura_api = InturaFetch(api_key, verbose=verbose)
        logger.debug("DashboardPlatform initialized successfully")
    
    def create_experiment(self, experiment: ExperimentModel) -> Optional[str]:
        """
        Create a new experiment.
        
        Args:
            experiment: Experiment model containing configuration
            
        Returns:
            Experiment ID on success, None on failure
            
        Raises:
            DashboardError: If experiment creation fails
        """
        try:
            logger.info("Creating new experiment")
            logger.debug(f"Experiment name: {getattr(experiment, 'name', 'unknown')}")
            
            # Convert model to JSON and send to API
            experiment_json = experiment.model_dump_json()
            experiment_id = self._intura_api.insert_experiment(experiment_json)
            
            if not experiment_id:
                logger.error("Failed to create experiment")
                raise DashboardError("Failed to create experiment")
                
            logger.info(f"Created experiment with ID: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment: {str(e)}")
            raise DashboardError(f"Error creating experiment: {str(e)}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models.
        
        Returns:
            List of model configurations
            
        Raises:
            DashboardError: If retrieving models fails
        """
        try:
            logger.debug("Fetching list of available models")
            models = self._intura_api.get_list_models()
            
            if models is None:
                logger.error("Failed to retrieve models")
                raise DashboardError("Failed to retrieve models")
                
            logger.debug(f"Retrieved {len(models)} models")
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            raise DashboardError(f"Error listing models: {str(e)}")

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        Get a list of existing experiments.
        
        Returns:
            List of experiment summaries
            
        Raises:
            DashboardError: If retrieving experiments fails
        """
        try:
            logger.debug("Fetching list of experiments")
            experiments = self._intura_api.get_list_experiment()
            
            if experiments is None:
                logger.error("Failed to retrieve experiments")
                raise DashboardError("Failed to retrieve experiments")
                
            logger.debug(f"Retrieved {len(experiments)} experiments")
            return experiments
            
        except Exception as e:
            logger.error(f"Error listing experiments: {str(e)}")
            raise DashboardError(f"Error listing experiments: {str(e)}")
    
    def get_experiment_detail(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get detailed information about an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Experiment details
            
        Raises:
            DashboardError: If retrieving experiment details fails
        """
        try:
            if not experiment_id:
                raise ValueError("Experiment ID is required")
                
            logger.debug(f"Fetching details for experiment: {experiment_id}")
            details = self._intura_api.get_experiment_detail(experiment_id=experiment_id)
            
            if details is None:
                logger.error(f"Failed to retrieve details for experiment: {experiment_id}")
                raise DashboardError(f"Failed to retrieve details for experiment: {experiment_id}")
                
            logger.debug(f"Retrieved details for experiment: {experiment_id}")
            return details
            
        except Exception as e:
            logger.error(f"Error getting experiment details: {str(e)}")
            raise DashboardError(f"Error getting experiment details: {str(e)}")
    
    def validate_experiment(self, experiment_id: str) -> bool:
        """
        Validate if an experiment ID exists.
        
        Args:
            experiment_id: ID of the experiment to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not experiment_id:
                return False
                
            logger.debug(f"Validating experiment ID: {experiment_id}")
            is_valid = self._intura_api.check_experiment_id(experiment_id)
            
            if is_valid:
                logger.debug(f"Experiment ID is valid: {experiment_id}")
            else:
                logger.debug(f"Experiment ID is invalid: {experiment_id}")
                
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating experiment: {str(e)}")
            return False