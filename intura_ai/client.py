"""The two clients. Start here.

    from intura_ai import Intura

    client = Intura(api_key="sk_...")          # or set INTURA_API_KEY
    verdict = client.guardrails.prompt_injection.detect(text=user_turn)

    if verdict.action == "block":
        return refusal()

One key, one client, the whole tree. Create it once per process and keep it: it
holds an HTTP connection pool, and a guardrail that reconnects on every call
spends more time in TLS than in inference.

Both clients take the same arguments and answer with the same objects. Use
`AsyncIntura` inside an async framework — these endpoints usually run on the
request path, in front of a model call, and a blocking 12 ms in an event loop is
12 ms nothing else runs.

Configuration, in the order it is read:

    api_key     the argument, then INTURA_API_KEY
    base_url    the argument, then INTURA_BASE_URL, then https://ai.intura.co/api

`base_url` is how a self-hosted deployment is used — the same models on your own
box, reached by the same code. `client.models.download(slug)` fetches those
models: the key signs the links, and the files come from Cloud Storage.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from typing import Any

import httpx

from . import _calls
from ._calls import ENDPOINTS, Call
from ._transport import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    backoff,
    build_headers,
    connection_error,
    interpret,
    request_kwargs,
    resolve_api_key,
    resolve_base_url,
    should_retry,
)
from .resources import (
    AsyncChat,
    AsyncDocuments,
    AsyncFraud,
    AsyncGuardrails,
    AsyncModels,
    AsyncStorage,
    Chat,
    Documents,
    Fraud,
    Guardrails,
    Models,
    Storage,
)

__all__ = ["AsyncIntura", "Intura"]


class _BaseClient:
    """What the two clients share: configuration, and how a request is shaped."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        timeout: float | None,
        max_retries: int,
        default_headers: Mapping[str, str] | None,
        env: Mapping[str, str] | None,
    ) -> None:
        environ = os.environ if env is None else env
        self.api_key = resolve_api_key(api_key, environ)
        self.base_url = resolve_base_url(base_url, environ)
        self.timeout = DEFAULT_TIMEOUT if timeout is None else timeout
        # Whether the caller named it, which is what decides if it is sent per
        # request: a borrowed http client arrives with a timeout its owner chose,
        # and overwriting that with this class's default would be a silent
        # change to somebody else's configuration.
        self._explicit_timeout = timeout is not None
        self.max_retries = max(0, int(max_retries))
        self._headers = build_headers(self.api_key, default_headers)

    def _kwargs(self, call: Call) -> dict[str, Any]:
        kwargs = request_kwargs(call)
        kwargs["url"] = f"{self.base_url}{call.path}"
        kwargs["headers"] = self._headers
        if self._explicit_timeout:
            kwargs["timeout"] = self.timeout
        return kwargs

    def __repr__(self) -> str:
        # The key is never in here. A repr ends up in logs, exception context
        # and notebook output, and a credential that reaches any of those is
        # rotated rather than explained.
        return f"{type(self).__name__}(base_url={self.base_url!r})"

    @staticmethod
    def endpoints() -> tuple[str, ...]:
        """The eight endpoints, in the order every Intura surface lists them."""
        return tuple(ENDPOINTS)


class Intura(_BaseClient):
    """A synchronous client for the Intura AI API.

    Args:
        api_key: Your key, from the console. Falls back to `INTURA_API_KEY`.
        base_url: The service root. Falls back to `INTURA_BASE_URL`, then
            Intura's own. Point it at your deployment to run the same calls
            on-premise.
        timeout: Seconds to wait for one response, 30 by default. The text
            endpoints answer in milliseconds; a document photograph is the slow
            one. Left unset with an `http_client` of your own, that client's
            timeout is used rather than overwritten.
        max_retries: How many times to retry a call the service never answered,
            rate limited or failed on. Connection errors, timeouts, 429 and 5xx
            only — a wrong body is never retried.
        default_headers: Sent on every request. For a proxy or a trace header,
            not for auth.
        http_client: Bring your own `httpx.Client` — a proxy, a custom
            transport, a shared pool. Its own base URL and headers are ignored;
            this client sets both per request, and never closes a pool it did
            not open.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.Client | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(api_key, base_url, timeout, max_retries, default_headers, env)
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client(timeout=self.timeout)

        self.documents = Documents(self)
        self.guardrails = Guardrails(self)
        self.fraud = Fraud(self)
        self.chat = Chat(self)
        self.storage = Storage(self)
        self.models = Models(self)

    # -- the one place a request happens ------------------------------------ #

    def _call(self, call: Call) -> Any:
        kwargs = self._kwargs(call)
        attempts_left = self.max_retries
        attempt = 0

        while True:
            try:
                response = self._http.request(**kwargs)
            except httpx.HTTPError as exc:
                if attempts_left <= 0:
                    raise connection_error(exc) from exc
                attempts_left -= 1
                time.sleep(backoff(attempt, None))
                attempt += 1
                continue

            if should_retry(response, attempts_left):
                attempts_left -= 1
                time.sleep(backoff(attempt, response))
                attempt += 1
                continue

            return interpret(response, call)

    # -- metadata ----------------------------------------------------------- #

    def info(self, endpoint: str) -> dict[str, Any]:
        """What one endpoint loaded, by name: `"guardrails/pii-detection"`.

        The cuts you may move with their ranges and defaults, the
        languages that have a trained build here, and whether one is loaded at
        all. Free, and worth reading before tuning anything.
        """
        return self._call(_calls.info(_route(endpoint)))

    def health(self) -> dict[str, Any]:
        """Liveness of the deployment. Dependency-free and never rate limited."""
        return self._call(_calls.health())

    def verify_key(self) -> bool:
        """Check the key works, without spending a request on inference.

        Raises `AuthenticationError` if the key is unknown or revoked,
        `PermissionDenied` if it is scoped away from the guardrails tree, and
        `APIConnectionError` if the service could not be reached. Returns True
        otherwise. Worth calling once at start-up: a key that is wrong fails on
        the first real call otherwise, which is usually the worst moment.
        """
        self.guardrails.prompt_injection.info()
        return True

    # -- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        """Close the connection pool, if this client opened one."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Intura:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncIntura(_BaseClient):
    """The same tree, awaited. Arguments are `Intura`'s.

        async with AsyncIntura() as client:
            verdict = await client.guardrails.prompt_injection.detect(text=turn)

    `asyncio.gather` over two calls is the shape most integrations want — screen
    the input and classify the intent at once, then decide.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(api_key, base_url, timeout, max_retries, default_headers, env)
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=self.timeout)

        self.documents = AsyncDocuments(self)
        self.guardrails = AsyncGuardrails(self)
        self.fraud = AsyncFraud(self)
        self.chat = AsyncChat(self)
        self.storage = AsyncStorage(self)
        self.models = AsyncModels(self)

    async def _call(self, call: Call) -> Any:
        kwargs = self._kwargs(call)
        attempts_left = self.max_retries
        attempt = 0

        while True:
            try:
                response = await self._http.request(**kwargs)
            except httpx.HTTPError as exc:
                if attempts_left <= 0:
                    raise connection_error(exc) from exc
                attempts_left -= 1
                await asyncio.sleep(backoff(attempt, None))
                attempt += 1
                continue

            if should_retry(response, attempts_left):
                attempts_left -= 1
                await asyncio.sleep(backoff(attempt, response))
                attempt += 1
                continue

            return interpret(response, call)

    async def info(self, endpoint: str) -> dict[str, Any]:
        """What one endpoint loaded, by name: `"guardrails/pii-detection"`."""
        return await self._call(_calls.info(_route(endpoint)))

    async def health(self) -> dict[str, Any]:
        """Liveness of the deployment."""
        return await self._call(_calls.health())

    async def verify_key(self) -> bool:
        """Check the key works, without spending a request on inference."""
        await self.guardrails.prompt_injection.info()
        return True

    async def aclose(self) -> None:
        """Close the connection pool, if this client opened one."""
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncIntura:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


def _route(endpoint: str) -> str:
    """`"guardrails/pii-detection"` -> its path. A typo names the eight."""
    name = endpoint.strip().strip("/")
    if name in ENDPOINTS:
        return ENDPOINTS[name]
    raise KeyError(f"{endpoint!r} is not an Intura endpoint. Available: {', '.join(ENDPOINTS)}")
