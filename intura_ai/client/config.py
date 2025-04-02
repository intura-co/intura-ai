"""
Configuration management for intura-ai.

This module provides a global configuration system for the package.
"""

import os
from typing import Any, Optional
from pydantic import BaseModel, Field

class InturaConfig(BaseModel):
    """Global configuration for intura-ai."""
    
    # Logging settings
    verbose: bool = Field(
        default=False, 
        description="Enable verbose output (debug logging)"
    )
    log_level: str = Field(
        default="info",
        description="Logging level (debug, info, warning, error, critical, none)"
    )
    
    # API settings
    api_key: Optional[str] = Field(
        default=None,
        description="API key for external services"
    )
    api_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for API endpoints"
    )
    
    # Other settings
    timeout: float = Field(
        default=60.0,
        description="Default timeout for API calls in seconds"
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory to use for caching"
    )
    
    # Add more configuration options as needed

# Global configuration instance
_config = InturaConfig()

def get_config() -> InturaConfig:
    """Get the current configuration."""
    return _config

def configure(
    verbose: Optional[bool] = None,
    log_level: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any
) -> InturaConfig:
    """
    Configure intura-ai globally.
    
    Args:
        verbose: Enable verbose output (debug logging)
        log_level: Logging level (debug, info, warning, error, critical, none)
        api_key: API key for external services
        api_base_url: Base URL for API endpoints
        timeout: Default timeout for API calls in seconds
        cache_dir: Directory to use for caching
        **kwargs: Additional configuration options
        
    Returns:
        Updated configuration
    """
    # Update configuration
    update_dict = {k: v for k, v in locals().items() 
                   if k != 'kwargs' and v is not None}
    update_dict.update(kwargs)
    
    for key, value in update_dict.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
    
    # Apply logging configuration if verbose or log_level changed
    if verbose is not None or log_level is not None:
        from intura_ai.shared.utils.logging import configure_logging
        
        # verbose takes precedence over log_level
        if verbose is not None:
            configure_logging(verbose=verbose)
        elif log_level is not None:
            configure_logging(level=log_level)
    
    return _config

def load_from_env() -> InturaConfig:
    """
    Load configuration from environment variables.
    
    Environment variables should be prefixed with INTURA_,
    e.g., INTURA_VERBOSE=true, INTURA_API_KEY=abc123
    
    Returns:
        Updated configuration
    """
    config_updates = {}
    
    # Check for environment variables
    for key in _config.model_fields:
        env_key = f"INTURA_{key.upper()}"
        if env_key in os.environ:
            value = os.environ[env_key]
            
            # Convert string to appropriate type
            field_info = _config.model_fields[key]
            if field_info.annotation == bool:
                value = value.lower() in ('true', 'yes', '1', 't', 'y')
            elif field_info.annotation == int:
                value = int(value)
            elif field_info.annotation == float:
                value = float(value)
            
            config_updates[key] = value
    
    return configure(**config_updates)

# Initialize from environment variables on module load
load_from_env()