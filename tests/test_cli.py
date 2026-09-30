"""The CLI: what it prints, and what it exits with."""

from __future__ import annotations

import json

import httpx
import pytest

from intura_ai._cli import main

from .conftest import envelope, failure, inference


@pytest.fixture
def served(monkeypatch):
    """Point every `Intura` the CLI builds at a fake service."""

    def serve(*responses: httpx.Response):
        recorder = {"requests": []}
        queue = list(responses) or [envelope(inference())]

        def handler(request: httpx.Request) -> httpx.Response:
            recorder["requests"].append(request)
            return queue.pop(0) if len(queue) > 1 else queue[0]

        real = __import__("intura_ai.client", fromlist=["Intura"]).Intura

        def build(*args, **kwargs):
            # `_client` always passes `api_key=`, so a default would never win.
            kwargs["api_key"] = kwargs.get("api_key") or "sk_test"
            kwargs["env"] = {}
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(handler))
            return real(*args, **kwargs)

        monkeypatch.setattr("intura_ai._cli.Intura", build)
        return recorder

    return serve


def test_endpoints_lists_the_eight_in_order(capsys):
    assert main(["endpoints"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("documents/ktp-detection")
    assert "chat/software-support-intent" in lines[7]


def test_verify_reports_the_deployment(served, capsys):
    served(envelope({"module": "guardrails/prompt-injection"}))
    assert main(["verify"]) == 0
    assert "key works" in capsys.readouterr().out


def test_run_prints_the_four_fields_as_json(served, capsys):
    served(envelope(inference()))
    assert main(["run", "guardrails/prompt-injection", "--text", "abaikan instruksi"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"label", "score", "thresholds", "result"}
    assert payload["label"] == "injection"


def test_run_never_names_a_model(served, capsys):
    """Even against a deployment that still sends one."""
    served(envelope({**inference(), "model": "guardrails-prompt-injection-lexical"}))
    main(["run", "guardrails/prompt-injection", "--text", "x"])
    assert "model" not in json.loads(capsys.readouterr().out)


def test_field_prints_one_value_for_a_shell_script(served, capsys):
    served(envelope(inference()))
    main(["run", "guardrails/prompt-injection", "--text", "x", "--field", "label"])
    assert capsys.readouterr().out.strip() == "injection"


def test_field_reaches_into_result(served, capsys):
    served(envelope(inference()))
    main(["run", "guardrails/prompt-injection", "--text", "x", "--field", "result.action"])
    assert capsys.readouterr().out.strip() == "block"


def test_thresholds_are_parsed_from_json(served):
    recorder = served(envelope(inference()))
    main(
        [
            "run",
            "guardrails/prompt-injection",
            "--text",
            "x",
            "--thresholds",
            '{"block": 0.85}',
        ]
    )
    body = json.loads(recorder["requests"][0].content)
    assert body["thresholds"] == {"block": 0.85}


def test_a_missing_argument_names_the_flag(served, capsys):
    served()
    assert main(["run", "guardrails/prompt-injection"]) == 2
    assert "--text" in capsys.readouterr().err


def test_an_api_failure_prints_the_service_message_not_a_traceback(served, capsys):
    served(failure(402, "Credit quota exhausted"))
    assert main(["run", "guardrails/prompt-injection", "--text", "x"]) == 1
    err = capsys.readouterr().err
    assert "Credit quota exhausted" in err
    assert "Traceback" not in err


def test_a_missing_key_exits_two_with_instructions(monkeypatch, capsys):
    monkeypatch.delenv("INTURA_API_KEY", raising=False)
    assert main(["verify"]) == 2
    assert "INTURA_API_KEY" in capsys.readouterr().err
