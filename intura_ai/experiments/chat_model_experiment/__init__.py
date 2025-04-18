import os
from uuid import uuid4
from typing import Dict, List, Tuple, Optional, Any, Type, Union
import importlib
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate

from intura_ai.shared.external.intura_api import InturaFetch
from intura_ai.callbacks import UsageTrackCallback
from intura_ai.shared.utils.logging import get_component_logger, set_component_level

# Type definitions
ModelClass = Any  # Type relaxed for lazy loading
ModelResult = Any  # The result of the model chain creation

# Get component-specific logger
logger = get_component_logger("chat_model_experiment")

class ChatModelExperiment:
    """
    Manages experiments with different chat models.
    
    This class provides functionality to build and configure chat models
    based on experiment configurations retrieved from the Intura API.
    Uses lazy loading to avoid unnecessary dependency installations.
    """
    
    def __init__(self, intura_api_key: Optional[str] = None, verbose: bool = False):
        """
        Initialize a new chat model experiment.
        
        Args:
            intura_api_key: API key for Intura services
            verbose: Enable verbose logging for this component
        """
        self._chosen_model = None  # Fixed spelling
        api_key = intura_api_key or os.environ.get("INTURA_API_KEY")
        self._intura_api_key = api_key
        self._intura_api = InturaFetch(api_key)
        self._data = []
        self._model_cache = {}  # Cache for imported model classes
        
        # Configure component-specific logging if verbose is specified
        if verbose:
            set_component_level("chat_model_experiment", "debug")
            
        logger.debug("Initialized ChatModelExperiment")
    
    @property
    def chosen_model(self) -> Optional[str]:
        """Get the selected model for the experiment."""
        return self._chosen_model  # Fixed spelling
    
    @property
    def data(self) -> List[Dict[str, Any]]:
        """Get the experiment data retrieved from the API."""
        return self._data
        
    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Validate that the messages list contains at least one human message
        and that each message follows the required schema.
        
        Args:
            messages: List of message dictionaries to validate
            
        Raises:
            ValueError: If there is no human message in the list or if messages don't follow the schema
        """
        if  len(messages) == 0:
            raise ValueError("At least one message with role='human' is required in the messages list")
            
        # Check for at least one human message
        has_human_message = any(
            message.get("role") == "human" for message in messages
        )
        
        if not has_human_message:
            logger.warning("Messages provided without a human message")
            raise ValueError("At least one message with role='human' is required in the messages list")
        
        # Validate each message follows the schema (only role and content fields)
        for i, message in enumerate(messages):
            keys = set(message.keys())
            if keys != {"role", "content"}:
                extra_keys = keys - {"role", "content"}
                missing_keys = {"role", "content"} - keys
                
                error_parts = []
                if extra_keys:
                    error_parts.append(f"extra keys: {', '.join(extra_keys)}")
                if missing_keys:
                    error_parts.append(f"missing keys: {', '.join(missing_keys)}")
                
                error_msg = f"Message at index {i} has invalid schema ({', '.join(error_parts)})"
                logger.warning(error_msg)
                raise ValueError(error_msg)

    
    def _lazy_import_model_class(self, provider: str, module_path: str, class_name: str) -> Type:
        """
        Lazily import the model class for the given provider.
        
        Args:
            provider: The model provider name
            module_path: The import path for the module
            class_name: The name of the class to import
            
        Returns:
            The imported model class
            
        Raises:
            ImportError: If the module cannot be imported or the class is not found
        """
        # Check if we've already imported this class
        cache_key = f"{provider}:{module_path}.{class_name}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # Import the module and get the class
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            
            # Cache the imported class
            self._model_cache[cache_key] = model_class
            logger.debug(f"Lazily imported {class_name} for provider {provider}")
            
            return model_class
        except ImportError as e:
            logger.error(f"Failed to import model class for provider {provider}: {str(e)}")
            raise ImportError(
                f"The {provider} provider requires additional dependencies. "
                f"Please install them with 'pip install intura_ai[{provider.lower()}]'"
            ) from e
        except AttributeError as e:
            logger.error(f"Model class for provider {provider} not found: {str(e)}")
            raise ImportError(f"Model class {class_name} for provider {provider} not found") from e
    
    def _create_model_result(
        self, 
        model_data: Dict[str, Any], 
        experiment_id: str, 
        session_id: str,
        messages: List[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        additional_model_configs: Optional[Dict[str, Any]] = None, 
        api_key_mapping: Optional[Dict[str, str]] = None
    ) -> ModelResult:
        """
        Create a model result from model data.
        
        Args:
            model_data: Model configuration data from the API
            experiment_id: ID of the experiment
            session_id: Session ID for the experiment
            messages: Optional list of messages to pre-populate the chat
            api_key: Optional API key to override configuration
            additional_model_configs: Additional model configuration parameters
            
        Returns:
            A langchain chain combining the chat template and model
        """
        messages = messages or []
        additional_model_configs = additional_model_configs or {}
        
        # Validate that there's at least one human message if messages are provided
        self._validate_messages(messages)
        
        provider = model_data["model_provider"]
        module_path = model_data["sdk_config"]["module_path"]
        class_name = model_data["sdk_config"]["class_name"]
        
        # Get the appropriate model class (lazy import)
        model_class = self._lazy_import_model_class(provider, module_path, class_name)
        logger.debug(f"Using model class: {model_class.__name__} for provider {provider}")
        
        # Create chat templates
        chat_prompts = [SystemMessagePromptTemplate.from_template(template=model_data["prompt"])]
        
        # Add any existing conversation messages
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "human":
                chat_prompts.append(HumanMessagePromptTemplate.from_template(template=content))
            elif role == "ai":
                chat_prompts.append(AIMessagePromptTemplate.from_template(template=content))
        
        chat_template = ChatPromptTemplate.from_messages(chat_prompts)
        
        # Filter out None values from model configuration
        model_configuration = {
            k: v for k, v in model_data["model_configuration"].items() if v is not None
        }
        
        # Add API key if provided
        if api_key:
            model_configuration["api_key"] = api_key
        elif api_key_mapping:
            model_configuration["api_key"] = api_key_mapping[model_configuration.get("model", "unknown")]
        
        # Create the callback and metadata
        callback = UsageTrackCallback(
            intura_api_key=self._intura_api_key,
            experiment_id=experiment_id,
            treatment_id=model_data["treatment_id"],
            treatment_name=model_data["treatment_name"],
            session_id=session_id,
            model_name=model_configuration.get("model", "unknown")
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
            "metadata": metadata,
            **additional_model_configs
        }
        
        # Initialize the model and combine with the chat template
        model = model_class(**configuration)
        return model, chat_template
    
    def build(
        self, 
        experiment_id: str, 
        session_id: Optional[str] = None, 
        features: Optional[Dict[str, Any]] = None, 
        max_models: int = 1,
        verbose: bool = False,
        messages: List[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        api_key_mapping: Optional[Dict[str, str]] = None,
        additional_model_configs: Optional[Dict[str, Any]] = None
    ) -> Union[ModelResult, List[ModelResult]]:
        """
        Build chat models based on experiment configuration.
        
        Args:
            experiment_id: ID of the experiment to build
            session_id: Optional session ID (will generate one if not provided)
            features: Features to include in the experiment
            max_models: Maximum number of models to return
            verbose: Enable verbose logging for this specific build
            messages: Optional list of messages to pre-populate the chat
            api_key: Optional API key to override configuration
            additional_model_configs: Additional model configuration parameters
            
        Returns:
            If max_models=1: A single model result
            If max_models>1: A list of model results
            If error: None
        """
        messages = messages or []
        features = features or {}
        additional_model_configs = additional_model_configs or {}
        session_id = session_id or str(uuid4())
        
        # Validate that there's at least one human message if messages are provided
        try:
            self._validate_messages(messages)
        except ValueError as e:
            logger.error(f"Invalid messages format: {str(e)}")
            return None
        
        # Temporarily increase logging level if requested for this operation
        original_level = None
        if verbose:
            set_component_level("chat_model_experiment", "debug")
        
        try:
            logger.info(f"Building chat model for experiment: {experiment_id}")
            logger.debug(f"Features: {features}, Session ID: {session_id}")
            
            # Fetch model data from API
            resp = self._intura_api.build_chat_model(experiment_id, features=features)
            if not resp:
                logger.warning(f"Failed to build chat model for experiment: {experiment_id}")
                return None
            
            self._data = resp["data"]
            logger.debug(f"Retrieved {len(self._data)} model configurations")
            
            if not self._data:
                logger.warning(f"No model configurations found for experiment: {experiment_id}")
                return None
                
            results = []
            for model_data in self._data:
                try:
                    result = self._create_model_result(
                        model_data, 
                        experiment_id, 
                        session_id,
                        messages,
                        api_key,
                        additional_model_configs,
                        api_key_mapping
                    )
                    results.append(result)
                    
                    model_name = model_data.get("model_configuration", {}).get("model", "unknown")
                    logger.debug(f"Added model: {model_name}")
                    
                    # If we've reached the desired number of models, break
                    if len(results) >= max_models:
                        break
                        
                except ImportError as e:
                    # Log the import error but continue with other models
                    logger.warning(f"Skipping model due to missing dependencies: {str(e)}")
                except Exception as e:
                    logger.error(f"Error creating model result: {str(e)}", exc_info=True)
            
            # Set the chosen model if we have results
            if results:
                if max_models == 1:
                    model_name = self._data[0].get("model_configuration", {}).get("model", "unknown")
                    self._chosen_model = model_name  # Fixed spelling
                    logger.info(f"Selected model: {model_name}")
                    return results[0]
                return results
            
            logger.warning("No models were successfully created")
            return None
            
        except Exception as e:
            logger.error(f"Error in build: {str(e)}", exc_info=True)
            return None
            
        finally:
            # Restore original logging level if we changed it
            if verbose and original_level is not None:
                set_component_level("chat_model_experiment", original_level)