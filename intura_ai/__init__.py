"""Intura AI package for LLM experimentation and production."""

from .__version__ import __version__
from intura_ai.client.config import configure

# Make key functions available at package level
def set_verbose(verbose: bool = True) -> None:
    """Enable or disable verbose mode (debug logging)."""
    configure(verbose=verbose)

# Your other imports and code here
__author__ = "Intura Developer"
