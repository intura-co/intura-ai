import os
from .config import InturaConfig
from intura_ai.shared.external.intura_api import InturaFetch

_config = None
_intura_api: InturaFetch = None
_initialized = False

def intura_initialization(api_key=None):
    """Initializes the intura_ai package."""
    global _config, _initialized, _intura_api
    if _initialized:
        print("intura_ai already initialized. Reinitializing.")

    _config = InturaConfig() #load config file.
    
    if api_key:
        _config.data['api_key'] = api_key

    if _config.get("api_key") is None:
         _config.data['api_key'] = os.environ.get("INTURA_API_KEY") #get api key from environment variable if not in config file.

    if _config.get("api_key") is None:
        raise ValueError("API Key not found")
    
    _intura_api = InturaFetch(api_key)
    if _intura_api.check_api_key():
        _initialized = True
        print("intura_ai initialized successfully.")
    else:
        raise ValueError("Invalid API Key")

def get_config():
    """Returns the config object if initialized, otherwise raises an exception."""
    if not _initialized:
        raise RuntimeError("intura_ai must be initialized before use.")
    return _config

def is_initialized():
    """Returns true if the api has been initialized"""
    return _initialized

def validate_client_key():
    if _intura_api:
        return _intura_api.check_api_key()
    else:
        return False
    
def get_intura_client():
    return _intura_api