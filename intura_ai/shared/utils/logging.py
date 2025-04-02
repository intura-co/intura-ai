"""
Logging utilities for intura-ai.

This module provides a flexible logging system that can be configured 
globally or per-component.
"""

import logging
import os
import sys
from typing import Optional, Union, Literal, Dict, Any

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create the logger for the package
logger = logging.getLogger("intura_ai")

# Dictionary to store component-specific loggers
_component_loggers: Dict[str, logging.Logger] = {}

# Environment variable to control logging level
ENV_LOG_LEVEL = "INTURA_LOG_LEVEL"

# Log level mapping
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "none": logging.CRITICAL + 10,  # Higher than any standard level
}

# Initialize with default handler if not already configured
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    logger.addHandler(handler)
    
    # Set default level based on environment variable or default to INFO
    default_level = os.environ.get(ENV_LOG_LEVEL, "info").lower()
    logger.setLevel(LOG_LEVELS.get(default_level, logging.INFO))

def configure_logging(
    level: Optional[Union[str, int]] = None,
    format: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
    verbose: Optional[bool] = None,
) -> None:
    """
    Configure the global logger for intura-ai.
    
    Args:
        level: Log level (debug, info, warning, error, critical, or integer level)
        format: Log message format
        handler: Custom log handler
        verbose: If True, sets level to DEBUG. If False, sets level to INFO.
                 (Convenience parameter)
    """
    # Clear existing handlers
    logger.handlers.clear()

    # Create new handler if not provided
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter
    if format is not None:
        handler.setFormatter(logging.Formatter(format))
    else:
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    
    logger.addHandler(handler)
    
    # Set level based on parameters
    if verbose is not None:
        level = "debug" if verbose else "info"
    
    if level is not None:
        if isinstance(level, str):
            level = LOG_LEVELS.get(level.lower(), logging.INFO)
        logger.setLevel(level)

def get_component_logger(component_name: str) -> logging.Logger:
    """
    Get a logger for a specific component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        Logger instance for the component
    """
    if component_name not in _component_loggers:
        component_logger = logger.getChild(component_name)
        _component_loggers[component_name] = component_logger
    return _component_loggers[component_name]

def set_component_level(
    component_name: str, 
    level: Union[str, int, Literal["debug", "info", "warning", "error", "critical", "none"]]
) -> None:
    """
    Set the logging level for a specific component.
    
    Args:
        component_name: Name of the component
        level: Log level (debug, info, warning, error, critical, none, or integer level)
    """
    component_logger = get_component_logger(component_name)
    
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.lower(), logging.INFO)
    
    component_logger.setLevel(level)

# Convenience functions
def enable_debug() -> None:
    """Enable debug logging for all components."""
    configure_logging(level=logging.DEBUG)

def disable_logging() -> None:
    """Disable all logging."""
    configure_logging(level=LOG_LEVELS["none"])