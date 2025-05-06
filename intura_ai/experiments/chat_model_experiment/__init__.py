import os
from uuid import uuid4
from typing import Dict, List, Tuple, Optional, Any, Type, Union, TypeVar, Callable, Protocol
import importlib
from functools import lru_cache
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from intura_ai.shared.external.intura_api import InturaFetch
from intura_ai.callbacks import UsageTrackCallback
from intura_ai.shared.utils.logging import get_component_logger, set_component_level

# Type definitions
ModelClass = TypeVar('ModelClass')
ModelResult = Tuple[Any, ChatPromptTemplate]

@dataclass
class Message:
    """Schema for chat messages."""
    role: str
    content: str

@dataclass
class ModelConfig:
    """Configuration for a model."""
    provider: str
    module_path: str
    class_name: str
    prompt: str
    treatment_id: str
    treatment_name: str
    model_configuration: Dict[str, Any]

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_id: str
    treatment_id: Optional[str]
    session_id: str
    request_id: Optional[str]
    features: Dict[str, Any]
    messages: List[Dict[str, str]]
    max_models: int
    verbose: bool
    api_key: Optional[str]
    api_key_mapping: Optional[Dict[str, str]]
    additional_model_configs: Optional[Dict[str, Any]]

class ModelFactory(Protocol):
    """Protocol for model creation."""
    def create_model(self, config: ModelConfig, experiment_config: ExperimentConfig) -> ModelResult:
        """Create a model instance."""
        ...

class ChatModelFactory:
    """Factory for creating chat models."""
    
    def __init__(self, intura_api_key: Optional[str] = None):
        self._intura_api_key = intura_api_key
        self._model_cache = {}
    
    @lru_cache(maxsize=32)
    def _lazy_import_model_class(self, provider: str, module_path: str, class_name: str) -> Type[ModelClass]:
        """Lazily import a model class."""
        cache_key = f"{provider}:{module_path}.{class_name}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            self._model_cache[cache_key] = model_class
            return model_class
        except ImportError as e:
            raise ImportError(
                f"The {provider} provider requires additional dependencies. "
                f"Please install them with 'pip install intura_ai[{provider.lower()}]'"
            ) from e
        except AttributeError as e:
            raise ImportError(f"Model class {class_name} for provider {provider} not found") from e
    
    def create_model(self, config: ModelConfig, experiment_config: ExperimentConfig) -> ModelResult:
        """Create a model instance."""
        model_class = self._lazy_import_model_class(
            config.provider, 
            config.module_path, 
            config.class_name
        )
        
        chat_template = self._create_chat_template(config.prompt, experiment_config.messages)
        model_config = self._prepare_model_config(config, experiment_config)
        
        model = model_class(**model_config)
        return model, chat_template
    
    def _create_chat_template(self, system_prompt: str, messages: List[Dict[str, str]]) -> ChatPromptTemplate:
        """Create a chat template."""
        chat_prompts = [SystemMessage(content=system_prompt)]
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "human":
                chat_prompts.append(HumanMessage(content=content))
            elif role == "ai":
                chat_prompts.append(AIMessage(content=content))
            elif role == "system":
                chat_prompts.append(SystemMessage(content=content))
            elif role == "tool":
                chat_prompts.append(ToolMessage(content=content))
        return ChatPromptTemplate.from_messages(chat_prompts)
    
    def _prepare_model_config(
        self, 
        model_config: ModelConfig, 
        experiment_config: ExperimentConfig
    ) -> Dict[str, Any]:
        """Prepare the model configuration."""
        config = {
            k: v for k, v in model_config.model_configuration.items() 
            if v is not None
        }
        
        model_name = config.get("model", "unknown")
        if experiment_config.api_key:
            config["api_key"] = experiment_config.api_key
        elif experiment_config.api_key_mapping and model_name in experiment_config.api_key_mapping:
            config["api_key"] = experiment_config.api_key_mapping[model_name]
        
        callback = UsageTrackCallback(
            intura_api_key=self._intura_api_key,
            experiment_id=experiment_config.experiment_id,
            treatment_id=model_config.treatment_id,
            treatment_name=model_config.treatment_name,
            session_id=experiment_config.session_id,
            model_name=model_name,
            model_provider=model_config.provider,
            request_id=experiment_config.request_id
        )
        
        metadata = {
            "experiment_id": experiment_config.experiment_id,
            "treatment_id": model_config.treatment_id,
            "treatment_name": model_config.treatment_name,
            "session_id": experiment_config.session_id,
            "model_name": model_name
        }
        
        return {
            **config,
            "callbacks": [callback],
            "metadata": metadata,
            **(experiment_config.additional_model_configs or {})
        }

class MessageValidator:
    """Validator for chat messages."""
    
    @staticmethod
    def validate(messages: List[Dict[str, str]]) -> None:
        """Validate chat messages."""
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        if not any(message.get("role") == "human" for message in messages):
            raise ValueError("At least one message with role='human' is required")
        
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
                
                raise ValueError(f"Message at index {i} has invalid schema ({', '.join(error_parts)})")

class ChatModelExperiment:
    """Manages experiments with different chat models."""
    
    COMPONENT_NAME = "chat_model_experiment"
    
    def __init__(self, intura_api_key: Optional[str] = None, verbose: bool = False):
        """Initialize a new chat model experiment."""
        self._chosen_model = None
        self._intura_api_key = intura_api_key or os.environ.get("INTURA_API_KEY")
        self._intura_api = InturaFetch(self._intura_api_key)
        self._data = []
        self._model_factory = ChatModelFactory(self._intura_api_key)
        self._logger = get_component_logger(self.COMPONENT_NAME)
        
        if verbose:
            self._set_verbose_logging(True)
        
        self._logger.debug("Initialized ChatModelExperiment")
    
    def _set_verbose_logging(self, verbose: bool) -> Optional[str]:
        """Set or restore logging level."""
        if verbose:
            return set_component_level(self.COMPONENT_NAME, "debug")
        return None
    
    @property
    def chosen_model(self) -> Optional[str]:
        """Get the selected model."""
        return self._chosen_model
    
    @property
    def data(self) -> List[Dict[str, Any]]:
        """Get the experiment data."""
        return self._data
    
    def invoke(
        self,
        experiment_id: str,
        treatment_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
        max_inferences: int = 1,
        latency_threshold: int = 30,
        max_timeout: int = 120,
        verbose: bool = False,
        messages: List[Dict[str, str]] = None,
    ) -> Union[Dict[str, Any], None]:
        """Run inference based on experiment configuration."""
        config = self._prepare_invoke_config(
            experiment_id, treatment_id, session_id, request_id, features,
            messages, verbose
        )
        
        if config.messages:
            try:
                MessageValidator.validate(config.messages)
            except ValueError as e:
                self._logger.error(f"Invalid messages format: {str(e)}")
                return None
        
        original_level = None
        if config.verbose:
            original_level = self._set_verbose_logging(True)
        
        try:
            self._logger.info(f"Running invoke for experiment: {config.experiment_id}")
            self._logger.debug(f"Features: {config.features}, Session ID: {config.session_id}")
            
            resp = self._intura_api.inference_chat_model(
                config.experiment_id,
                treatment_id=config.treatment_id,
                request_id=config.request_id,
                features=config.features,
                messages=config.messages,
                max_inferences=max_inferences,
                session_id=config.session_id,
                latency_threshold=latency_threshold,
                max_timeout=max_timeout
            )
            
            if not resp:
                self._logger.warning(f"Failed to run invoke for experiment: {config.experiment_id}")
                return None
            
            return self._process_inference_results(resp["data"], max_inferences)
            
        except Exception as e:
            self._logger.error(f"Error in invoke: {str(e)}", exc_info=True)
            return None
            
        finally:
            if config.verbose and original_level is not None:
                set_component_level(self.COMPONENT_NAME, original_level)
    
    def _process_inference_results(
        self, 
        data: List[Dict[str, Any]], 
        max_inferences: int
    ) -> Union[AIMessage, List[AIMessage]]:
        """Process inference results."""
        results = []
        for item in data:
            results.append(AIMessage(
                content=item["predictions"]["content"],
                id=item["prediction_id"],
                name=item["treatment_name"],
                response_metadata={
                    "model_name": item["model_name"],
                    "model_provider": item["model_provider"],
                    "treatment_id": item["treatment_id"],
                    "treatment_name": item["treatment_name"],
                    "latency": item["predictions"]["latency"]
                },
                usage_metadata=item["predictions"]["usage_metadata"]
            ))
        
        return results[0] if max_inferences == 1 else results
    
    def build(
        self,
        experiment_id: str,
        treatment_id: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
        max_models: int = 1,
        verbose: bool = False,
        messages: List[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        api_key_mapping: Optional[Dict[str, str]] = None,
        additional_model_configs: Optional[Dict[str, Any]] = None
    ) -> Union[ModelResult, List[ModelResult], None]:
        """Build chat models based on experiment configuration."""
        config = self._prepare_build_config(
            experiment_id, treatment_id, session_id, request_id, features, 
            max_models, verbose, messages, api_key,
            api_key_mapping, additional_model_configs,
        )
        
        if config.messages:
            try:
                MessageValidator.validate(config.messages)
            except ValueError as e:
                self._logger.error(f"Invalid messages format: {str(e)}")
                return None
        
        original_level = None
        if config.verbose:
            original_level = self._set_verbose_logging(True)
        
        try:
            self._logger.info(f"Building chat model for experiment: {config.experiment_id}")
            self._logger.debug(f"Features: {config.features}, Session ID: {config.session_id}")
            
            resp = self._intura_api.build_chat_model(
                config.experiment_id,
                treatment_id=config.treatment_id,
                features=config.features,
                messages=config.messages,
                request_id=config.request_id,
            )
            
            if not resp:
                self._logger.warning(f"Failed to build chat model for experiment: {config.experiment_id}")
                return None
            
            self._data = resp["data"]
            self._logger.debug(f"Retrieved {len(self._data)} model configurations")
            
            if not self._data:
                self._logger.warning(f"No model configurations found for experiment: {config.experiment_id}")
                return None
            
            results = self._build_models(config)
            return self._process_build_results(results, config.max_models)
            
        except Exception as e:
            self._logger.error(f"Error in build: {str(e)}", exc_info=True)
            return None
            
        finally:
            if config.verbose and original_level is not None:
                set_component_level(self.COMPONENT_NAME, original_level)
    
    def _build_models(self, config: ExperimentConfig) -> List[ModelResult]:
        """Build models from configurations."""
        results = []
        for model_data in self._data[:config.max_models]:
            try:
                model_config = ModelConfig(
                    provider=model_data["model_provider"],
                    module_path=model_data["sdk_config"]["module_path"],
                    class_name=model_data["sdk_config"]["class_name"],
                    prompt=model_data["prompt"],
                    treatment_id=model_data["treatment_id"],
                    treatment_name=model_data["treatment_name"],
                    model_configuration=model_data["model_configuration"]
                )
                
                result = self._model_factory.create_model(model_config, config)
                results.append(result)
                
                model_name = model_data.get("model_configuration", {}).get("model", "unknown")
                self._logger.debug(f"Added model: {model_name}")
                
            except ImportError as e:
                self._logger.warning(f"Skipping model due to missing dependencies: {str(e)}")
            except Exception as e:
                self._logger.error(f"Error creating model result: {str(e)}", exc_info=True)
        
        return results
    
    def _process_build_results(
        self, 
        results: List[ModelResult], 
        max_models: int
    ) -> Union[ModelResult, List[ModelResult], None]:
        """Process build results."""
        if not results:
            self._logger.warning("No models were successfully created")
            return None
        
        if max_models == 1:
            model_name = self._data[0].get("model_configuration", {}).get("model", "unknown")
            self._chosen_model = model_name
            self._logger.info(f"Selected model: {model_name}")
            return results[0]
        
        return results
    
    def _prepare_invoke_config(
        self,
        experiment_id: str,
        treatment_id: Optional[str],
        request_id: Optional[str],
        session_id: Optional[str],
        features: Optional[Dict[str, Any]],
        messages: Optional[List[Dict[str, str]]],
        verbose: bool
    ) -> ExperimentConfig:
        """Prepare configuration for invoke."""
        return ExperimentConfig(
            experiment_id=experiment_id,
            treatment_id=treatment_id,
            request_id=request_id or str(uuid4()),
            session_id=session_id or str(uuid4()),
            features=features or {},
            messages=messages or [],
            max_models=1,
            verbose=verbose,
            api_key=None,
            api_key_mapping=None,
            additional_model_configs=None
        )
    
    def _prepare_build_config(
        self,
        experiment_id: str,
        treatment_id: Optional[str],
        request_id: Optional[str],
        session_id: Optional[str],
        features: Optional[Dict[str, Any]],
        max_models: int,
        verbose: bool,
        messages: Optional[List[Dict[str, str]]],
        api_key: Optional[str],
        api_key_mapping: Optional[Dict[str, str]],
        additional_model_configs: Optional[Dict[str, Any]]
    ) -> ExperimentConfig:
        """Prepare configuration for build."""
        return ExperimentConfig(
            experiment_id=experiment_id,
            treatment_id=treatment_id,
            session_id=session_id or str(uuid4()),
            request_id=request_id or str(uuid4()),
            features=features or {},
            messages=messages or [],
            max_models=max_models,
            verbose=verbose,
            api_key=api_key,
            api_key_mapping=api_key_mapping,
            additional_model_configs=additional_model_configs
        )