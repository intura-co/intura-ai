from uuid import uuid4
from typing import Dict, List, Tuple, Optional, Any, Type, Union

from intura_ai.libs.wrappers.langchain_chat_model import (
    InturaChatOpenAI,
    InturaChatAnthropic,
    InturaChatDeepSeek,
    InturaChatGoogleGenerativeAI,
    InturaChatOllama
)
from intura_ai.shared.external.intura_api import InturaFetch
from intura_ai.callbacks import UsageTrackCallback
from intura_ai.shared.utils.logging import get_component_logger

# Get component-specific logger
logger = get_component_logger("chat_model_experiment")

# Type definitions for better type hinting
ModelClass = Type[Union[
    InturaChatOpenAI,
    InturaChatAnthropic,
    InturaChatDeepSeek, 
    InturaChatGoogleGenerativeAI,
    InturaChatOllama
]]
ChatTemplate = List[Tuple[str, str]]
ModelResult = Tuple[ModelClass, Dict[str, Any], ChatTemplate]

class ChatModelExperiment:
    """
    Manages experiments with different chat models.
    
    This class provides functionality to build and configure chat models
    based on experiment configurations retrieved from the Intura API.
    """
    
    # Map provider names to model classes for easier lookup
    PROVIDER_TO_MODEL = {
        "Google": InturaChatGoogleGenerativeAI,
        "Anthropic": InturaChatAnthropic,
        "Deepseek": InturaChatDeepSeek,
        "OpenAI": InturaChatOpenAI,
        "Ollama": InturaChatOllama
    }
    
    def __init__(self, intura_api_key: Optional[str] = None, verbose: bool = False):
        """
        Initialize a new chat model experiment.
        
        Args:
            intura_api_key: API key for Intura services
            verbose: Enable verbose logging for this component
        """
        self._choiced_model = None
        self._intura_api_key = intura_api_key
        self._intura_api = InturaFetch(intura_api_key)
        self._data = []
        
        # Configure component-specific logging if verbose is specified
        if verbose:
            from intura_ai.shared.utils.logging import set_component_level
            set_component_level("chat_model_experiment", "debug")
            
        logger.debug("Initialized ChatModelExperiment")
    
    @property
    def choiced_model(self) -> Optional[str]:
        """Get the selected model for the experiment."""
        return self._choiced_model
    
    @property
    def data(self) -> List[Dict[str, Any]]:
        """Get the experiment data retrieved from the API."""
        return self._data
    
    def _create_model_result(
        self, 
        model_data: Dict[str, Any], 
        experiment_id: str, 
        session_id: str
    ) -> ModelResult:
        """
        Create a model result tuple from model data.
        
        Args:
            model_data: Model configuration data from the API
            experiment_id: ID of the experiment
            session_id: Session ID for the experiment
            
        Returns:
            A tuple of (model_class, configuration, chat_templates)
            
        Raises:
            ValueError: If the model provider is not supported
        """
        provider = model_data["model_provider"]
        
        # Get the appropriate model class
        if provider not in self.PROVIDER_TO_MODEL:
            logger.error(f"Unsupported model provider: {provider}")
            raise ValueError(f"Unsupported model provider: {provider}")
            
        model_class = self.PROVIDER_TO_MODEL[provider]
        logger.debug(f"Using model class: {model_class.__name__} for provider {provider}")
        
        # Create chat templates
        chat_templates = [("system", model_data["prompt"])]
        
        # Filter out None values from model configuration
        model_configuration = {
            k: v for k, v in model_data["model_configuration"].items() if v is not None
        }
        
        # Create the callback and metadata
        callback = UsageTrackCallback(
            intura_api_key=self._intura_api_key,
            experiment_id=experiment_id,
            treatment_id=model_data["treatment_id"],
            treatment_name=model_data["treatment_name"],
            session_id=session_id,
            model_name=model_configuration["model"]
        )
        
        metadata = {
            "experiment_id": experiment_id,
            "treatment_id": model_data["treatment_id"],
            "treatment_name": model_data["treatment_name"],
            "session_id": session_id
        }
        
        # Combine everything into configuration
        configuration = {
            **model_configuration,
            "callbacks": [callback],
            "metadata": metadata
        }
        
        return model_class, configuration, chat_templates
    
    def build(
        self, 
        experiment_id: str, 
        session_id: Optional[str] = None, 
        features: Dict[str, Any] = None, 
        max_models: int = 1,
        verbose: bool = False
    ) -> Union[ModelResult, List[ModelResult], Tuple[None, Dict, List]]:
        """
        Build chat models based on experiment configuration.
        
        Args:
            experiment_id: ID of the experiment to build
            session_id: Optional session ID (will generate one if not provided)
            features: Features to include in the experiment
            max_models: Maximum number of models to return
            verbose: Enable verbose logging for this specific build
            
        Returns:
            If max_models=1: A single model result tuple
            If max_models>1: A list of model result tuples
            If error: (None, {}, [])
        """
        # Temporarily increase logging level if requested for this operation
        original_level = None
        if verbose:
            from intura_ai.shared.utils.logging import set_component_level
            original_level = logger.level
            set_component_level("chat_model_experiment", "debug")
        
        try:
            features = features or {}
            session_id = session_id or str(uuid4())
            
            logger.info(f"Building chat model for experiment: {experiment_id}")
            logger.debug(f"Features: {features}, Session ID: {session_id}")
            
            # Fetch model data from API
            resp = self._intura_api.build_chat_model(experiment_id, features=features)
            if not resp:
                logger.warning(f"Failed to build chat model for experiment: {experiment_id}")
                return None, {}, []
            
            self._data = resp["data"]
            logger.debug(f"Retrieved {len(self._data)} model configurations")
            
            results = []
            for model_data in self._data:
                try:
                    result = self._create_model_result(model_data, experiment_id, session_id)
                    results.append(result)
                    
                    logger.debug(f"Added model: {model_data.get('model_configuration', {}).get('model')}")
                    
                    # If we've reached the desired number of models, break
                    if len(results) >= max_models:
                        break
                        
                except Exception as e:
                    logger.error(f"Error creating model result: {str(e)}")
            
            # Set the chosen model if we have results
            if results and max_models == 1:
                model_name = self._data[0]["model_configuration"].get("model")
                self._choiced_model = model_name
                logger.info(f"Selected model: {model_name}")
                return results[0]
            
            return results
            
        except Exception as e:
            logger.error(f"Error in build: {str(e)}")
            return None, {}, []
            
        finally:
            # Restore original logging level if we changed it
            if verbose and original_level is not None:
                from intura_ai.shared.utils.logging import set_component_level
                set_component_level("chat_model_experiment", original_level)