"""Constructing a client, and what one request looks like on the wire."""

from __future__ import annotations

import httpx
import pytest

from intura_ai import Intura, MissingAPIKey, _calls
from intura_ai._transport import DEFAULT_BASE_URL, USER_AGENT

from .conftest import API_KEY, BASE_URL, envelope, inference, make_async_client, make_client


class TestApiKey:
    def test_the_argument_wins(self):
        client = Intura(api_key="sk_explicit", env={"INTURA_API_KEY": "sk_environment"})
        assert client.api_key == "sk_explicit"

    def test_falls_back_to_the_environment(self):
        client = Intura(env={"INTURA_API_KEY": "sk_environment"})
        assert client.api_key == "sk_environment"

    def test_no_key_anywhere_says_what_to_do(self):
        with pytest.raises(MissingAPIKey) as caught:
            Intura(env={})
        message = str(caught.value)
        assert "INTURA_API_KEY" in message
        assert "ai.intura.co/console" in message

    def test_a_blank_key_is_no_key(self):
        with pytest.raises(MissingAPIKey):
            Intura(api_key="   ", env={})

    def test_the_key_is_not_in_the_repr(self):
        """A repr reaches logs and notebook output; a credential must not."""
        client = Intura(api_key=API_KEY, env={})
        assert API_KEY not in repr(client)
        assert "base_url" in repr(client)


class TestBaseUrl:
    def test_defaults_to_intura(self):
        assert Intura(api_key=API_KEY, env={}).base_url == DEFAULT_BASE_URL

    def test_environment_points_at_a_deployment(self):
        client = Intura(api_key=API_KEY, env={"INTURA_BASE_URL": "http://10.0.0.5:8000/"})
        assert client.base_url == "http://10.0.0.5:8000"

    def test_trailing_slash_does_not_double_up(self):
        """Paths are joined by concatenation, so the root is normalised once."""
        client = Intura(api_key=API_KEY, base_url="https://api.test.intura/", env={})
        assert client.base_url == "https://api.test.intura"


class TestTheRequest:
    def test_key_travels_in_the_documented_header(self):
        client, recorder = make_client()
        client.guardrails.prompt_injection.detect(text="halo")
        assert recorder.request.headers["x-api-key"] == API_KEY

    def test_user_agent_names_the_package_and_version(self):
        client, recorder = make_client()
        client.guardrails.prompt_injection.detect(text="halo")
        assert recorder.request.headers["user-agent"] == USER_AGENT
        assert "intura-ai-python/" in USER_AGENT

    def test_default_headers_are_sent(self):
        client, recorder = make_client(default_headers={"X-Trace": "abc"})
        client.guardrails.prompt_injection.detect(text="halo")
        assert recorder.request.headers["x-trace"] == "abc"

    def test_path_is_the_documented_one(self):
        client, recorder = make_client()
        client.guardrails.prompt_injection.detect(text="halo")
        assert recorder.request.url.path == "/v1/ai/guardrails/prompt-injection"

    def test_unset_arguments_are_absent_not_null(self):
        """A missing `thresholds` keeps the account's saved cuts; a null would
        not mean the same thing."""
        client, recorder = make_client()
        client.guardrails.prompt_injection.detect(text="halo")
        assert recorder.body == {"text": "halo"}

    def test_thresholds_are_sent_when_given(self):
        client, recorder = make_client()
        client.guardrails.prompt_injection.detect(text="halo", thresholds={"block": 0.85})
        assert recorder.body["thresholds"] == {"block": 0.85}


class TestMetadata:
    def test_info_is_addressed_by_endpoint_name(self):
        client, recorder = make_client(envelope({"module": "guardrails/pii-detection"}))
        data = client.info("guardrails/pii-detection")
        assert recorder.request.url.path == "/v1/ai/guardrails/pii-detection/info"
        assert data["module"] == "guardrails/pii-detection"

    def test_an_unknown_endpoint_names_the_eight(self):
        client, _ = make_client()
        with pytest.raises(KeyError) as caught:
            client.info("guard/pii")
        assert "guardrails/pii-detection" in str(caught.value)

    def test_health_is_unversioned(self):
        client, recorder = make_client(envelope({"status": "ok"}))
        client.health()
        assert recorder.request.url.path == "/health"

    def test_verify_key_spends_nothing_on_inference(self):
        """`/info` is authenticated and free — it declares no endpoint id, and a
        price row joins on one."""
        client, recorder = make_client(envelope({"module": "guardrails/prompt-injection"}))
        assert client.verify_key() is True
        assert recorder.request.method == "GET"
        assert recorder.request.url.path.endswith("/info")

    def test_endpoints_are_listed_in_taxonomy_order(self):
        assert Intura.endpoints() == (
            "documents/ktp-detection",
            "documents/npwp-detection",
            "guardrails/prompt-injection",
            "guardrails/pii-detection",
            "fraud/email-phishing",
            "fraud/chat-scam",
            "chat/commerce-intent",
            "chat/software-support-intent",
        )


class TestLifecycle:
    def test_context_manager_closes_its_own_pool(self):
        with Intura(api_key=API_KEY, base_url=BASE_URL, env={}) as client:
            pass
        assert client._http.is_closed

    def test_a_borrowed_http_client_is_left_open(self):
        """Closing a pool the caller owns would break the rest of their program."""
        borrowed = httpx.Client(transport=httpx.MockTransport(lambda r: envelope(inference())))
        with Intura(api_key=API_KEY, base_url=BASE_URL, http_client=borrowed, env={}):
            pass
        assert not borrowed.is_closed


class TestAsync:
    async def test_the_same_call_awaited(self):
        client, recorder = make_async_client()
        verdict = await client.guardrails.prompt_injection.detect(text="halo")
        assert recorder.request.url.path == "/v1/ai/guardrails/prompt-injection"
        assert verdict.label == "injection"
        await client.aclose()

    async def test_context_manager(self):
        client, _ = make_async_client()
        async with client:
            assert await client.verify_key() is True

    async def test_both_clients_answer_with_the_same_type(self):
        sync_client, _ = make_client()
        async_client, _ = make_async_client()
        one = sync_client.guardrails.prompt_injection.detect(text="halo")
        two = await async_client.guardrails.prompt_injection.detect(text="halo")
        assert type(one) is type(two)
        assert one.score == two.score
        await async_client.aclose()


class TestTimeout:
    def test_defaults_to_thirty_seconds(self):
        assert Intura(api_key=API_KEY, env={}).timeout == 30.0

    def test_a_named_timeout_is_sent_per_request(self):
        client, recorder = make_client(timeout=2.5)
        client.guardrails.prompt_injection.detect(text="halo")
        assert recorder.request.extensions["timeout"]["read"] == 2.5

    def test_an_unnamed_timeout_is_not_forced_onto_a_borrowed_client(self):
        """Overwriting the timeout its owner chose would silently change somebody
        else's configuration."""
        client, _ = make_client()
        assert client._explicit_timeout is False
        assert "timeout" not in client._kwargs(_calls.health())

    def test_a_named_one_is(self):
        client, _ = make_client(timeout=2.5)
        assert client._kwargs(_calls.health())["timeout"] == 2.5
