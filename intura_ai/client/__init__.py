"""
Intura AI client for interacting with LLM services.
"""

from typing import Optional

from intura_ai.shared.utils.logging import get_component_logger
from .config import get_config

# Get component-specific logger
logger = get_component_logger("client")

class InturaClient:
    """Client for interacting with LLM services."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
    ):
        """
        Initialize the Intura client.
        
        Args:
            api_key: API key for authentication (overrides global config)
            base_url: Base URL for API endpoints (overrides global config)
            timeout: Request timeout in seconds (overrides global config)
            verbose: Enable verbose logging for this client instance
        """
        config = get_config()
        
        self.api_key = api_key or config.api_key
        self.base_url = base_url or config.api_base_url
        self.timeout = timeout or config.timeout
        
        # Configure component-specific logging if verbose is specified
        if verbose is not None:
            from intura_ai.shared.utils.logging import set_component_level
            set_component_level("client", "debug" if verbose else "info")
        
        logger.debug(f"Initialized client with base_url={self.base_url}, timeout={self.timeout}")
        
    # def generate(self, prompt: str, **kwargs: Any) -> str:
    #     """
    #     Generate text from a prompt.
        
    #     Args:
    #         prompt: Input prompt
    #         **kwargs: Additional parameters
            
    #     Returns:
    #         Generated text
    #     """
    #     logger.info(f"Generating text for prompt: {prompt[:50]}...")
    #     logger.debug(f"Full prompt: {prompt}")
    #     logger.debug(f"Additional parameters: {kwargs}")
        
    #     # Simulate API call
    #     time.sleep(0.5)
        
    #     response = f"Response to: {prompt[:10]}..."
    #     logger.info(f"Received response of length {len(response)}")
    #     logger.debug(f"Full response: {response}")
        
    #     return response
        
    # def experiment(
    #     self, 
    #     prompts: List[str], 
    #     models: List[str],
    #     verbose: bool = False
    # ) -> Dict[str, Any]:
    #     """
    #     Run experiments with multiple prompts and models.
        
    #     Args:
    #         prompts: List of prompts to test
    #         models: List of models to test
    #         verbose: Whether to log detailed information for this experiment
            
    #     Returns:
    #         Experiment results
    #     """
    #     # Temporarily increase logging verbosity if requested for this experiment
    #     original_level = logger.level
    #     if verbose:
    #         from intura_ai.shared.utils.logging import set_component_level

    #         set_component_level("client", "debug")
        
    #     try:
    #         logger.info(f"Starting experiment with {len(prompts)} prompts and {len(models)} models")
            
    #         # Simulation of experiment
    #         results = {
    #             "num_prompts": len(prompts),
    #             "num_models": len(models),
    #             "samples": []
    #         }
            
    #         for prompt_idx, prompt in enumerate(prompts):
    #             logger.debug(f"Processing prompt {prompt_idx+1}/{len(prompts)}")
    #             for model in models:
    #                 logger.debug(f"Testing model: {model}")
    #                 # Simulation of API call
    #                 time.sleep(0.1)
                    
    #         logger.info("Experiment completed successfully")
    #         return results
            
    #     finally:
    #         # Restore original logging level if we changed it
    #         if verbose:
    #             from shared.utils.logging import set_component_level
    #             set_component_level("client", original_level)