from intura_ai.client import validate_client_key, is_initialized

def validate_api_key(_):
    if not is_initialized():
        raise RuntimeError("intura_ai must be initialized before using this.")
    if not validate_client_key():
        raise RuntimeError("intura_ai api key incorrect.")