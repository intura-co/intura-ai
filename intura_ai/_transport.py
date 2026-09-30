"""Auth, the envelope, and the retry policy — shared by both clients.

Everything here is a pure function over a request or a response, so the sync and
the async client differ only in how they await the network. There is one place
that decides what an error means and one place that decides whether to try
again, which is the only way the two stay honest with each other.

The envelope
------------
Every route answers in one shape, success or failure:

    {"status": "success", "code": 200, "data": {...}}
    {"status": "error",   "code": 402, "error": "..."}

So there is one parser, and `data` is the only thing a caller ever sees. The
service's own `error` text is written for whoever is wiring the call and is
raised verbatim rather than replaced with something of ours.

What gets retried
-----------------
A connection that never landed, a timeout, a 429 and a 5xx — nothing else. A
422 is a wrong body and will be wrong again; a 402 is an exhausted allowance and
retrying it is just a second refusal. Every inference endpoint is a pure
function of its request, so replaying one is safe: no state is created, and a
call the service never answered was never charged.

`Retry-After` wins when the service sends one, because it knows when the window
resets and a client guessing is how a limited caller stays limited. Past
`MAX_RETRY_WAIT` the wait is handed back to the caller as a `RateLimited` they
can schedule around instead of blocking a worker on it.
"""

from __future__ import annotations

import platform
import random
import sys
from collections.abc import Mapping
from typing import Any

import httpx

from .__version__ import __version__
from ._calls import Call
from ._errors import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    MissingAPIKey,
    RateLimited,
    error_for_status,
)

DEFAULT_BASE_URL = "https://ai.intura.co/api"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 2

# Half a second, doubling, with jitter — and never more than this between tries.
BASE_BACKOFF = 0.5
MAX_BACKOFF = 8.0

# Past this, a `Retry-After` is reported rather than slept through. A worker
# blocked for two minutes inside a library call is a worse outage than the rate
# limit it is waiting on.
MAX_RETRY_WAIT = 30.0

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})

USER_AGENT = f"intura-ai-python/{__version__} (python {platform.python_version()}; {sys.platform})"


def resolve_api_key(api_key: str | None, env: Mapping[str, str]) -> str:
    key = (api_key or env.get("INTURA_API_KEY") or "").strip()
    if not key:
        raise MissingAPIKey()
    return key


def resolve_base_url(base_url: str | None, env: Mapping[str, str]) -> str:
    """The service root, without a trailing slash.

    `INTURA_BASE_URL` is how a self-hosted deployment is pointed at: the same
    models, on-premise, reached by the same code. The default is Intura's own.
    """
    return (base_url or env.get("INTURA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def build_headers(api_key: str, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    headers = {
        # `X-API-Key` is the header the service reads by default; it also accepts
        # `Authorization: Bearer`. One of them is enough and sending both only
        # doubles the number of places a key can leak into a log.
        "X-API-Key": api_key,
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if extra:
        headers.update({str(k): str(v) for k, v in extra.items()})
    return headers


def request_kwargs(call: Call) -> dict[str, Any]:
    """`Call` -> the arguments httpx wants."""
    kwargs: dict[str, Any] = {"method": call.method, "url": call.path}
    if call.json is not None:
        kwargs["json"] = call.json
    if call.files is not None:
        kwargs["files"] = call.files
        if call.data:
            kwargs["data"] = call.data
    return kwargs


def request_id_of(response: httpx.Response) -> str | None:
    return response.headers.get("x-request-id")


def retry_after_of(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        # An HTTP-date is legal here and this service never sends one; treating
        # it as absent falls back to the backoff rather than raising inside the
        # error path.
        return None


def backoff(attempt: int, response: httpx.Response | None) -> float:
    """How long to wait before try number `attempt` (0-based)."""
    if response is not None:
        after = retry_after_of(response)
        if after is not None:
            return after
    # Full jitter: a fleet that retries in lockstep re-creates the spike it is
    # backing off from.
    return random.uniform(0, min(MAX_BACKOFF, BASE_BACKOFF * (2**attempt)))


def should_retry(response: httpx.Response, attempts_left: int) -> bool:
    if attempts_left <= 0 or response.status_code not in RETRY_STATUS:
        return False
    after = retry_after_of(response)
    return after is None or after <= MAX_RETRY_WAIT


def connection_error(exc: Exception) -> APIConnectionError:
    if isinstance(exc, httpx.TimeoutException):
        return APITimeoutError(
            f"the request to Intura timed out: {exc}. The call may still have "
            "been served — raise `timeout` rather than retrying in a loop.",
            cause=exc,
        )
    return APIConnectionError(
        f"could not reach the Intura API: {exc}. Check `base_url` and network "
        "egress; nothing was charged.",
        cause=exc,
    )


def interpret(response: httpx.Response, call: Call) -> Any:
    """The success payload, or the right exception.

    The service puts its own sentence in `error`; the only text this function
    writes is for the case where the body is not the envelope at all — a proxy
    page, an HTML error, a deployment that is not this API.
    """
    request_id = request_id_of(response)
    try:
        body = response.json()
    except ValueError:
        body = None

    if not isinstance(body, dict) or "status" not in body:
        snippet = (response.text or "").strip()[:200]
        raise error_for_status(response.status_code)(
            f"the response was not the Intura envelope. Check `base_url` points "
            f"at the API root and not at a proxy. Body: {snippet or '<empty>'}",
            status_code=response.status_code,
            request_id=request_id,
            body=body,
        )

    if response.status_code >= 400 or body.get("status") == "error":
        status = int(body.get("code") or response.status_code)
        message = str(body.get("error") or f"the API answered {status}")
        cls = error_for_status(status)
        if cls is RateLimited:
            raise RateLimited(
                message,
                status_code=status,
                request_id=request_id,
                body=body,
                retry_after=retry_after_of(response),
            )
        raise cls(message, status_code=status, request_id=request_id, body=body)

    return call.parse(body.get("data"), request_id)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT",
    "MAX_RETRY_WAIT",
    "USER_AGENT",
    "APIError",
    "backoff",
    "build_headers",
    "connection_error",
    "interpret",
    "request_kwargs",
    "resolve_api_key",
    "resolve_base_url",
    "should_retry",
]
