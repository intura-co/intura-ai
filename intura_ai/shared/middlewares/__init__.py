from .validate_api_key import validate_api_key

def validate_class(validator):
    """Decorator to validate a class based on a given validator function."""
    def decorator(cls):
        def wrapper(*args, **kwargs):
            instance = cls(*args, **kwargs)
            validator(instance)  # Call the validation function
            return instance
        return wrapper
    return decorator