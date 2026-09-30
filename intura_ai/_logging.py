"""The package logger.

A library must not configure logging for the application that imports it, so
this attaches a `NullHandler` and nothing else. Anything the package wants to
say — a call being retried, a deployment answering on a degraded layer — goes
through here and is silent until the application calls `logging.basicConfig()`
or otherwise attaches a handler.

`intura_ai.set_verbose()` is the one-liner for a notebook or a script that does
want to see it, so the common case does not require anyone to know the logger's
name.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "intura_ai"

logger = logging.getLogger(LOGGER_NAME)
logger.addHandler(logging.NullHandler())


def set_verbose(enabled: bool = True, *, level: int = logging.INFO) -> None:
    """Print this package's log lines to stderr.

    Idempotent: calling it twice does not double every line.
    """
    if not enabled:
        logger.setLevel(logging.WARNING)
        return

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)


__all__ = ["LOGGER_NAME", "logger", "set_verbose"]
