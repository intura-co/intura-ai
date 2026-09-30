"""Every exception this package raises, in one place.

They all descend from `InturaError`, so a caller who only wants to know that
*something* to do with Intura went wrong catches one class. Everything the API
answered with a status code descends from `APIError` and carries that code, the
message the service wrote, and the request id — which is the string support asks
for first.

The split that matters at a call site is not the status code, it is what you can
do about it:

    QuotaExceeded      top up or wait for the window to reset — no retry helps
    RateLimited        the same call will work shortly; `retry_after` says when
    AuthenticationError   the key is wrong, missing or revoked
    PermissionDenied   the key is real and is not scoped to this endpoint
    InvalidRequest     the body is wrong; the message names the field
    APIConnectionError never reached the service, so nothing was charged

Each message is written to be actionable on its own terms. A traceback from a
library is usually the only documentation anyone reads, so the message says what
to do rather than only what went wrong.

Detecting something bad is not an error. A blocked prompt, a rejected KTP and a
scam verdict are all a 200 with a `label` — the exceptions here are about the
call, never about the answer.
"""

from __future__ import annotations

from typing import Any

CONSOLE_URL = "https://ai.intura.co/console"


class InturaError(Exception):
    """Base class for everything this package raises deliberately."""


class MissingAPIKey(InturaError, ValueError):
    """No API key was passed and none is in the environment."""

    def __init__(self) -> None:
        super().__init__(
            "No API key. Pass one to the client, or set INTURA_API_KEY:\n"
            "    from intura_ai import Intura\n"
            '    client = Intura(api_key="sk_...")\n'
            "    # or: export INTURA_API_KEY=sk_...\n"
            f"Keys are created in the console: {CONSOLE_URL}"
        )


class APIConnectionError(InturaError):
    """The service could not be reached, so it never saw the request.

    Nothing was charged and nothing was decided — a retry is always safe.
    """

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class APITimeoutError(APIConnectionError):
    """The request was sent and no response arrived inside the timeout.

    Unlike its parent this one is genuinely ambiguous: the service may have
    answered the call and charged for it after the client stopped listening.
    """


class TransferError(InturaError):
    """A file transfer straight to or from Cloud Storage failed.

    The API signed the link and was not part of the transfer, so this is never
    an `APIError`: the status is the bucket's, not Intura's. Retrying is safe —
    a download resumes by re-fetching the file, and an upload that failed left
    nothing an inference call can read.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIError(InturaError):
    """The service answered, and the answer was a failure.

    `message` is the text the API put in `error` — written for whoever is wiring
    the call, so it is worth showing rather than replacing.
    """

    status_code: int = 0

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        request_id: str | None = None,
        body: Any = None,
    ) -> None:
        detail = f"{status_code} {message}"
        if request_id:
            detail = f"{detail} (request id: {request_id})"
        super().__init__(detail)
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        self.body = body


class InvalidRequest(APIError):
    """400 — the body could not be read. The message names what was wrong."""


class AuthenticationError(APIError):
    """401 — no key, an unknown key, or one that has been revoked."""


class QuotaExceeded(APIError):
    """402 — this account has no request allowance left in the current window.

    Retrying does not help; the window resets or the plan changes. Both are in
    the console.
    """


class PermissionDenied(APIError):
    """403 — a valid key that is not scoped to the endpoint it called.

    Key scopes are set in the console, per key.
    """


class NotFound(APIError):
    """404 — no such route. Usually a `base_url` with a path already on it."""


class PayloadTooLarge(APIError):
    """413 — the image is over the deployment's upload ceiling."""


class UnsupportedMediaType(APIError):
    """415 — the bytes are not an image this deployment decodes."""


class InvalidParameters(APIError):
    """422 — the shape is right and a value is not.

    A threshold outside its range or under an unknown key, a missing required
    field, text past the endpoint's ceiling. The message names the field and
    what it accepts.
    """


class RateLimited(APIError):
    """429 — too many calls too quickly.

    `retry_after` is the service's own `Retry-After`, in seconds, when it sent
    one. The client retries these for you up to `max_retries`; seeing this
    exception means the budget ran out.
    """

    def __init__(self, *args: Any, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.retry_after = retry_after


class ServerError(APIError):
    """5xx — the service failed. Retried automatically before you see it."""


_BY_STATUS: dict[int, type[APIError]] = {
    400: InvalidRequest,
    401: AuthenticationError,
    402: QuotaExceeded,
    403: PermissionDenied,
    404: NotFound,
    413: PayloadTooLarge,
    415: UnsupportedMediaType,
    422: InvalidParameters,
    429: RateLimited,
}


def error_for_status(status_code: int) -> type[APIError]:
    """Which exception class a status code maps to."""
    if status_code in _BY_STATUS:
        return _BY_STATUS[status_code]
    if status_code >= 500:
        return ServerError
    return APIError


__all__ = [
    "APIConnectionError",
    "APIError",
    "APITimeoutError",
    "AuthenticationError",
    "InturaError",
    "InvalidParameters",
    "InvalidRequest",
    "MissingAPIKey",
    "NotFound",
    "PayloadTooLarge",
    "PermissionDenied",
    "QuotaExceeded",
    "RateLimited",
    "ServerError",
    "TransferError",
    "UnsupportedMediaType",
    "error_for_status",
]
