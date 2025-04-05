import warnings
from .__version__ import __version__

__author__ = "Intura Developer"

# Display beta warning
warnings.warn(
    "⚠️ Intura AI is currently in BETA and under active development. "
    "APIs may change without warning.",
    UserWarning,
    stacklevel=2
)