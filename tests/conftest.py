"""Shared fixtures. Nothing in this suite touches the network.

Every test runs the real client against `httpx.MockTransport`, so the request
this package builds — path, headers, body, encoding — is asserted exactly as the
service would receive it, and a fresh clone runs green offline in under a
second. The parity this cannot check is whether the API still accepts that
request; `tests/test_calls.py` pins the paths and field names against
`intura-ai-api`, and the day one moves, that file is the one to update.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from intura_ai import AsyncIntura, Intura

API_KEY = "sk_test_0123456789abcdef"
BASE_URL = "https://api.test.intura"


class Recorder:
    """A fake service: hands back canned responses, keeps every request."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return response

    @property
    def request(self) -> httpx.Request:
        """The only request, when a test made one call."""
        assert len(self.requests) == 1, f"expected one request, got {len(self.requests)}"
        return self.requests[0]

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.request.content)


def envelope(data: Any, *, code: int = 200, request_id: str = "req_abc123") -> httpx.Response:
    return httpx.Response(
        code,
        json={"status": "success", "code": code, "data": data},
        headers={"X-Request-Id": request_id},
    )


def failure(code: int, error: str, **headers: str) -> httpx.Response:
    return httpx.Response(
        code,
        json={"status": "error", "code": code, "error": error},
        headers={"X-Request-Id": "req_err", **headers},
    )


def inference(**overrides: Any) -> dict[str, Any]:
    """A five-field payload with the shape every `/v1/ai` endpoint returns."""
    data = {
        "label": "injection",
        "score": 0.97,
        "thresholds": {"block": 0.7, "review": 0.4},
        "result": {"injection": True, "action": "block", "languages": ["id"]},
    }
    data.update(overrides)
    return data


def make_client(*responses: httpx.Response, **kwargs: Any) -> tuple[Intura, Recorder]:
    recorder = Recorder(list(responses) or [envelope(inference())])
    client = Intura(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=httpx.Client(transport=httpx.MockTransport(recorder)),
        env={},
        **kwargs,
    )
    return client, recorder


def make_async_client(*responses: httpx.Response, **kwargs: Any) -> tuple[AsyncIntura, Recorder]:
    recorder = Recorder(list(responses) or [envelope(inference())])
    client = AsyncIntura(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
        env={},
        **kwargs,
    )
    return client, recorder


@pytest.fixture
def client() -> Intura:
    built, _ = make_client()
    return built


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own key in the environment must not change a test's meaning."""
    monkeypatch.delenv("INTURA_API_KEY", raising=False)
    monkeypatch.delenv("INTURA_BASE_URL", raising=False)


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry tests assert the retry, not the wait."""
    monkeypatch.setattr("intura_ai.client.time.sleep", lambda _: None)

    async def _instant(_: float) -> None:
        return None

    monkeypatch.setattr("intura_ai.client.asyncio.sleep", _instant)
