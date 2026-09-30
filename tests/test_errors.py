"""What each failure becomes, and what gets tried again."""

from __future__ import annotations

import httpx
import pytest

from intura_ai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InturaError,
    InvalidParameters,
    PermissionDenied,
    QuotaExceeded,
    RateLimited,
    ServerError,
)

from .conftest import envelope, failure, inference, make_async_client, make_client


class TestStatusCodes:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (401, AuthenticationError),
            (402, QuotaExceeded),
            (403, PermissionDenied),
            (422, InvalidParameters),
            (429, RateLimited),
            (500, ServerError),
        ],
    )
    def test_each_code_has_its_own_class(self, code, expected):
        client, _ = make_client(failure(code, "nope"))
        with pytest.raises(expected):
            client.guardrails.prompt_injection.detect(text="halo")

    def test_everything_descends_from_one_class(self):
        client, _ = make_client(failure(402, "Credit quota exhausted"))
        with pytest.raises(InturaError):
            client.guardrails.prompt_injection.detect(text="halo")

    def test_the_services_own_message_survives(self):
        """The API writes its errors for whoever is wiring the call. Replacing
        one with ours would throw away the only useful sentence."""
        client, _ = make_client(failure(422, "thresholds.block must be between 0.1 and 0.99"))
        with pytest.raises(InvalidParameters) as caught:
            client.guardrails.prompt_injection.detect(text="halo", thresholds={"block": 5})
        assert "between 0.1 and 0.99" in caught.value.message
        assert caught.value.status_code == 422

    def test_the_request_id_is_kept_for_support(self):
        client, _ = make_client(failure(500, "InternalServerError"), max_retries=0)
        with pytest.raises(ServerError) as caught:
            client.guardrails.prompt_injection.detect(text="halo")
        assert caught.value.request_id == "req_err"
        assert "req_err" in str(caught.value)

    def test_rate_limit_carries_retry_after(self):
        client, _ = make_client(failure(429, "Too many requests", **{"Retry-After": "12"}))
        with pytest.raises(RateLimited) as caught:
            client.guardrails.prompt_injection.detect(text="halo")
        assert caught.value.retry_after == 12.0

    def test_a_body_that_is_not_the_envelope_says_so(self):
        """A proxy page or an HTML error means `base_url` is pointing at the
        wrong thing, and that is the useful thing to say."""
        client, _ = make_client(httpx.Response(502, text="<html>bad gateway</html>"), max_retries=0)
        with pytest.raises(ServerError) as caught:
            client.guardrails.prompt_injection.detect(text="halo")
        assert "base_url" in caught.value.message


class TestRetries:
    def test_a_server_error_is_retried_then_succeeds(self):
        client, recorder = make_client(failure(503, "unavailable"), envelope(inference()))
        verdict = client.guardrails.prompt_injection.detect(text="halo")
        assert len(recorder.requests) == 2
        assert verdict.label == "injection"

    def test_retries_are_bounded(self):
        client, recorder = make_client(failure(503, "unavailable"), max_retries=2)
        with pytest.raises(ServerError):
            client.guardrails.prompt_injection.detect(text="halo")
        assert len(recorder.requests) == 3

    def test_max_retries_zero_means_one_attempt(self):
        client, recorder = make_client(failure(503, "unavailable"), max_retries=0)
        with pytest.raises(ServerError):
            client.guardrails.prompt_injection.detect(text="halo")
        assert len(recorder.requests) == 1

    @pytest.mark.parametrize("code", [400, 401, 402, 403, 422])
    def test_a_call_that_will_fail_again_is_not_retried(self, code):
        """A wrong body stays wrong and an exhausted allowance stays exhausted.
        Retrying either is a second refusal at the same price."""
        client, recorder = make_client(failure(code, "no"))
        with pytest.raises(InturaError):
            client.guardrails.prompt_injection.detect(text="halo")
        assert len(recorder.requests) == 1

    def test_a_long_retry_after_is_handed_back_rather_than_slept_through(self):
        """Blocking a worker for ten minutes inside a library call is worse than
        the rate limit it is waiting on."""
        client, recorder = make_client(failure(429, "slow down", **{"Retry-After": "600"}))
        with pytest.raises(RateLimited) as caught:
            client.guardrails.prompt_injection.detect(text="halo")
        assert len(recorder.requests) == 1
        assert caught.value.retry_after == 600.0

    def test_a_connection_that_never_landed_is_retried(self):
        attempts = {"n": 0}

        def flaky(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.ConnectError("connection refused", request=request)
            return envelope(inference())

        from intura_ai import Intura

        client = Intura(
            api_key="sk_test",
            base_url="https://api.test.intura",
            http_client=httpx.Client(transport=httpx.MockTransport(flaky)),
            env={},
        )
        assert client.guardrails.prompt_injection.detect(text="halo").label == "injection"
        assert attempts["n"] == 2

    def test_an_unreachable_service_names_base_url(self):
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        from intura_ai import Intura

        client = Intura(
            api_key="sk_test",
            base_url="https://api.test.intura",
            http_client=httpx.Client(transport=httpx.MockTransport(refuse)),
            env={},
            max_retries=0,
        )
        with pytest.raises(APIConnectionError) as caught:
            client.guardrails.prompt_injection.detect(text="halo")
        assert "base_url" in str(caught.value)
        assert "nothing was charged" in str(caught.value)

    def test_a_timeout_is_its_own_class(self):
        def stall(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        from intura_ai import Intura

        client = Intura(
            api_key="sk_test",
            base_url="https://api.test.intura",
            http_client=httpx.Client(transport=httpx.MockTransport(stall)),
            env={},
            max_retries=0,
        )
        with pytest.raises(APITimeoutError):
            client.guardrails.prompt_injection.detect(text="halo")


class TestAsyncErrors:
    async def test_the_same_class_comes_out_of_the_async_client(self):
        client, _ = make_async_client(failure(402, "Credit quota exhausted"))
        with pytest.raises(QuotaExceeded):
            await client.guardrails.prompt_injection.detect(text="halo")
        await client.aclose()

    async def test_retries_happen_there_too(self):
        client, recorder = make_async_client(failure(503, "unavailable"), envelope(inference()))
        await client.guardrails.prompt_injection.detect(text="halo")
        assert len(recorder.requests) == 2
        await client.aclose()
