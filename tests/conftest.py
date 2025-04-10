import os
import pytest
from unittest.mock import patch

@pytest.fixture(autouse=True)
def load_env():
    """
    Fixture to automatically load environment variables for tests.
    This ensures that any environment variables needed by the tested modules
    are properly set up before each test.
    """
    with patch.dict(os.environ, {
        "INTURA_API_KEY": os.environ.get("INTURA_API_KEY"),
        "INTURA_API_BASE_URL": "https://intura-be-external-server-566556985624.asia-southeast2.run.app"
    }):
        yield