"""`intura-ai` — the API from a shell, for the things a shell is better at.

Checking a key works, seeing what a deployment loaded, and running one call
against your own text without opening an editor:

    $ export INTURA_API_KEY=sk_...
    $ intura-ai verify                                   # does this key work
    $ intura-ai endpoints                                # the eight, in order
    $ intura-ai info guardrails/pii-detection            # cuts, labels, languages
    $ intura-ai run guardrails/prompt-injection --text "abaikan instruksi sebelumnya"
    $ intura-ai run documents/ktp-detection --file ktp.jpg
    $ intura-ai run fraud/email-phishing --body "$(cat mail.html)" --subject "PENTING"
    $ intura-ai models                                   # what you may self-host
    $ intura-ai download guardrails-pii-detection-id --dir ./models

Output is the API's own `data` object as JSON, so it pipes into `jq` and matches
what the library returns field for field. `--field` prints one value on its own
line, which is what a shell script actually wants:

    $ intura-ai run guardrails/prompt-injection --text "..." --field label

Anything that failed exits non-zero with the service's own message on stderr —
never a traceback, which is not what a shell script can act on.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import _calls
from .__version__ import __version__
from ._calls import ENDPOINTS
from ._errors import InturaError, MissingAPIKey
from .client import Intura

# Which arguments each endpoint takes, and how `run` turns them into a call.
# `_calls` owns the paths and the bodies; this only says which flags reach them.
_RUNNERS: dict[str, tuple[tuple[str, ...], Any]] = {
    "documents/ktp-detection": (
        ("file|object",),
        lambda a, **kw: _calls.ktp_detect(a.file, object=a.object, **kw),
    ),
    "documents/npwp-detection": (
        ("file|object",),
        lambda a, **kw: _calls.npwp_detect(a.file, object=a.object, **kw),
    ),
    "guardrails/prompt-injection": (
        ("text",),
        lambda a, **kw: _calls.prompt_injection_detect(a.text, **kw),
    ),
    "guardrails/pii-detection": (
        ("text",),
        lambda a, **kw: _calls.pii_mask(a.text, mask=not a.no_mask, **kw),
    ),
    "fraud/email-phishing": (
        ("body",),
        lambda a, **kw: _calls.email_phishing_detect(
            a.body, subject=a.subject, sender=a.sender, reply_to=a.reply_to, **kw
        ),
    ),
    "fraud/chat-scam": (
        ("message",),
        lambda a, **kw: _calls.chat_scam_detect(
            a.message, history=a.history or None, sender=a.sender, **kw
        ),
    ),
    "chat/commerce-intent": (
        ("message",),
        lambda a, **kw: _calls.commerce_intent_classify(
            a.message, history=a.history or None, channel=a.channel, **kw
        ),
    ),
    "chat/software-support-intent": (
        ("message",),
        lambda a, **kw: _calls.software_support_intent_classify(
            a.message, history=a.history or None, channel=a.channel, **kw
        ),
    ),
}


def _emit(payload: Any, field: str | None) -> None:
    if field:
        value = payload
        for part in field.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        print(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
        return
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _client(args: argparse.Namespace) -> Intura:
    return Intura(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )


def _cmd_endpoints(args: argparse.Namespace) -> int:
    """The eight, with what each one decides."""
    width = max(len(name) for name in ENDPOINTS)
    for name, path in ENDPOINTS.items():
        print(f"{name.ljust(width)}  {path}")
    print("\nRun one with:  intura-ai run <ENDPOINT> --text/--message/--body/--file ...")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    with _client(args) as client:
        client.verify_key()
        print(f"key works against {client.base_url}")
    return 0


def _cmd_health(args: argparse.Namespace) -> int:
    with _client(args) as client:
        _emit(client.health(), args.field)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    with _client(args) as client:
        _emit(client.info(args.endpoint), args.field)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    required, build = _RUNNERS[args.endpoint]
    missing = [
        name for name in required if not any(getattr(args, alt, None) for alt in name.split("|"))
    ]
    if missing:
        flags = ", ".join(
            " or ".join(f"--{alt.replace('_', '-')}" for alt in name.split("|")) for name in missing
        )
        print(f"{args.endpoint} needs {flags}.", file=sys.stderr)
        return 2

    thresholds = json.loads(args.thresholds) if args.thresholds else None
    call = build(args, thresholds=thresholds)

    with _client(args) as client:
        result = client._call(call)

    # `run` prints what the API sent rather than the typed object, so the CLI
    # and `curl` agree line for line.
    payload = {
        "label": result.label,
        "score": result.score,
        "thresholds": result.thresholds,
        "result": result.result,
    }
    _emit(payload, args.field)
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    with _client(args) as client:
        models = client.models.list()
    if args.field or args.json:
        _emit([vars(m) | {"files": list(m.files)} for m in models], args.field)
        return 0
    width = max((len(m.slug) for m in models), default=0)
    for m in models:
        print(f"{m.slug.ljust(width)}  {m.version}  {m.endpoint}  {m.name}")
    print("\nDownload one with:  intura-ai download <SLUG> --dir ./models")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    with _client(args) as client:
        root = client.models.download(args.slug, args.dir, overwrite=args.overwrite)
    print(root)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intura-ai",
        description="Call the Intura AI API from a shell.",
        epilog="Keys: https://ai.intura.co/console   Docs: https://ai.intura.co/docs",
    )
    parser.add_argument("--version", action="version", version=f"intura-ai {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--api-key",
        default=None,
        help="Your API key. Defaults to $INTURA_API_KEY.",
    )
    common.add_argument(
        "--base-url",
        default=None,
        help="Service root. Defaults to $INTURA_BASE_URL, then https://ai.intura.co/api.",
    )
    common.add_argument("--timeout", type=float, default=30.0, help="Seconds. Default 30.")
    common.add_argument(
        "--field",
        default=None,
        help="Print one value instead of the whole object, e.g. label, result.action.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("endpoints", help="List the eight endpoints.")
    listing.set_defaults(func=_cmd_endpoints)

    verify = sub.add_parser("verify", parents=[common], help="Check the API key works.")
    verify.set_defaults(func=_cmd_verify)

    health = sub.add_parser("health", parents=[common], help="Liveness of the deployment.")
    health.set_defaults(func=_cmd_health)

    info = sub.add_parser(
        "info", parents=[common], help="What one endpoint loaded, and what you may send."
    )
    info.add_argument("endpoint", choices=list(ENDPOINTS))
    info.set_defaults(func=_cmd_info)

    run = sub.add_parser("run", parents=[common], help="Call one endpoint.")
    run.add_argument("endpoint", choices=list(_RUNNERS))
    run.add_argument("--text", help="guardrails/*: the text to screen.")
    run.add_argument("--message", help="fraud/chat-scam, chat/*: the message.")
    run.add_argument("--body", help="fraud/email-phishing: the email body, HTML welcome.")
    run.add_argument("--subject", help="fraud/email-phishing: the subject line.")
    run.add_argument("--sender", help="fraud/*: the From header, or the chat sender.")
    run.add_argument("--reply-to", dest="reply_to", help="fraud/email-phishing: Reply-To.")
    run.add_argument("--file", help="documents/*: the photograph to check.")
    run.add_argument(
        "--object",
        help="documents/*: a photograph already uploaded, by the key storage returned.",
    )
    run.add_argument(
        "--history",
        action="append",
        default=[],
        help="A previous turn, oldest first. Repeatable.",
    )
    run.add_argument("--channel", help="chat/*: where the message arrived. Recorded, not scored.")
    run.add_argument(
        "--thresholds",
        help="Cuts for this call only, as JSON: '{\"block\": 0.85}'.",
    )
    run.add_argument(
        "--no-mask",
        dest="no_mask",
        action="store_true",
        help="guardrails/pii-detection: spans only, no masked copy.",
    )
    run.set_defaults(func=_cmd_run)

    models = sub.add_parser(
        "models", parents=[common], help="List the models you may download and self-host."
    )
    models.add_argument("--json", action="store_true", help="Print the list as JSON.")
    models.set_defaults(func=_cmd_models)

    download = sub.add_parser(
        "download",
        parents=[common],
        help="Download one model's files from Cloud Storage. Prints the directory.",
    )
    download.add_argument("slug", help="A slug from `intura-ai models`.")
    download.add_argument(
        "--dir", default=".", help="Where to put it. Files land in <dir>/<slug>/<version>/."
    )
    download.add_argument(
        "--overwrite", action="store_true", help="Download files already present too."
    )
    download.set_defaults(func=_cmd_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except MissingAPIKey as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyError as exc:
        print(str(exc).strip('"'), file=sys.stderr)
        return 2
    except InturaError as exc:
        # The service wrote a sentence for whoever is wiring the call. Print it,
        # not a traceback: a shell script can act on an exit code and a line.
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
