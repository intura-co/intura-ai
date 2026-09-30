"""Every endpoint, in one file: its path, its body, and how its answer is read.

This module is the parity point with `intura-ai-api`. Path strings, parameter
names and defaults are copied from that repo's `modules/v1/ai/**/schemas.py` and
`api.py` — never invented here. A client that is subtly wrong is worse than no
client, because the caller trusts it enough to stop reading the error message.

Nothing here performs I/O. Each builder returns a `Call` the transport executes,
which is what lets the sync and the async client share one definition of what an
endpoint *is*: two thin wrappers, one spec.

The eight endpoints, in the order every Intura surface lists them:

    documents    ktp-detection, npwp-detection
    guardrails   prompt-injection, pii-detection
    fraud        email-phishing, chat-scam
    chat         commerce-intent, software-support-intent

And two platform routes that run no model, both copied from
`modules/v1/platform/**`:

    POST /v1/platform/storage/uploads                   a signed PUT for one file
    GET  /v1/platform/downloads/models                  the models you may self-host
    POST /v1/platform/downloads/models/{slug}/download  signed GETs for one of them
"""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Union

from .types import (
    DocumentVerdict,
    Inference,
    IntentResult,
    ModelDownload,
    ModelInfo,
    PhishingVerdict,
    PIIResult,
    PromptInjectionVerdict,
    ScamVerdict,
)

# What a document endpoint accepts: a path, raw bytes, an open file, or a
# base64 `data:` URL.
ImageInput = Union[str, "Path", bytes, bytearray, memoryview, BinaryIO]

Thresholds = Mapping[str, float]

# Copied from `modules/v1/ai/documents/reading/constant.py`.
ACCEPTED_MEDIA_TYPES: tuple[str, ...] = ("image/jpeg", "image/jpg", "image/png", "image/webp")


@dataclass(frozen=True)
class Call:
    """One HTTP request, and the function that reads its `data` object."""

    method: str
    path: str
    json: dict[str, Any] | None = None
    files: dict[str, Any] | None = None
    data: dict[str, Any] = field(default_factory=dict)
    parse: Callable[[Any, str | None], Any] = lambda data, request_id: data


def _raw(data: Any, request_id: str | None) -> Any:
    """`/info` and `/health` answer with metadata, not a verdict.

    Returned as the dict the service sent. A typed model here would be a second
    place to update every time a deployment reports something new, and would
    hide the new field until this package caught up.
    """
    return data


def _parser(model: type[Inference]) -> Callable[[Any, str | None], Any]:
    def parse(data: Any, request_id: str | None) -> Any:
        if not isinstance(data, dict):  # pragma: no cover - the API always sends an object
            raise TypeError(f"expected an inference object, got {type(data).__name__}")
        return model.from_data(data, request_id=request_id)

    return parse


def _body(**fields: Any) -> dict[str, Any]:
    """Drop every unset argument.

    Absent and null are different to the API: a missing `thresholds` keeps what
    the account saved in the console, and an explicit one overrides it for this
    call only.
    """
    return {key: value for key, value in fields.items() if value is not None}


def _history(value: Iterable[str] | None) -> list[str] | None:
    return None if value is None else [str(turn) for turn in value]


def _cuts(value: Thresholds | None) -> dict[str, float] | None:
    return None if value is None else {str(k): float(v) for k, v in value.items()}


# --------------------------------------------------------------------------- #
# Images.


def _sniff(data: bytes) -> str | None:
    """Media type from the bytes themselves.

    So `detect(image=open("ktp.jpg", "rb").read())` works without the caller
    restating what they already handed over. The API requires a media type it
    accepts and refuses an unknown one with a 415, and a header no caller set is
    the most common way to earn that for a perfectly good photograph.
    """
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _decode_data_url(value: str) -> tuple[str, str]:
    """`data:image/jpeg;base64,…` -> (payload, media type)."""
    header, _, payload = value.partition(",")
    media_type = header[5:].split(";")[0].strip().lower()
    return payload, media_type


def read_file(
    source: ImageInput,
    media_type: str | None = None,
) -> tuple[str, bytes, str]:
    """(filename, bytes, media type) for a path, bytes or an open file.

    The media type is the one named, else guessed from the filename, else
    sniffed from the bytes; `application/octet-stream` when none of those knows.
    """
    filename = "upload"
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"no such file: {path}. Pass a path, the bytes themselves, an "
                "open file, or a base64 `data:` URL."
            ) from exc
        filename = path.name
        media_type = media_type or mimetypes.guess_type(path.name)[0]
    elif isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
    elif hasattr(source, "read"):
        data = source.read()
        filename = Path(str(getattr(source, "name", "upload"))).name
        media_type = media_type or mimetypes.guess_type(filename)[0]
    else:
        raise TypeError(
            f"image must be a path, bytes, an open file or a `data:` URL, not "
            f"{type(source).__name__}."
        )

    media_type = (media_type or _sniff(data) or "").lower()
    if media_type == "image/jpg":
        # `mimetypes` never produces this, but a caller passing it through from
        # a browser upload does. The API accepts it; normalising is free.
        media_type = "image/jpeg"
    return filename, data, media_type or "application/octet-stream"


def _image_parts(
    image: ImageInput,
    media_type: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(files, json) for one image, whichever encoding suits what was passed.

    Bytes and paths go up as `multipart/form-data`, which is the cheaper of the
    two encodings by a third and the one the API documents first. A `data:` URL
    is already base64 and is sent as JSON rather than decoded and re-encoded.
    """
    if isinstance(image, str) and image.startswith("data:"):
        payload, embedded = _decode_data_url(image)
        return None, _body(image=payload, media_type=media_type or embedded or None)
    return {"file": read_file(image, media_type)}, None


# --------------------------------------------------------------------------- #
# Documents — is this the card it claims to be, and is the photograph usable?


def _document_detect(
    route: str,
    image: ImageInput | None,
    *,
    object: str | None,
    thresholds: Thresholds | None,
    media_type: str | None,
) -> Call:
    cuts = _cuts(thresholds)
    has_object = bool((object or "").strip())
    if (image is None) == (not has_object):
        # The API refuses both and neither with a 422; saying so here costs no
        # request and names the arguments as this package spells them.
        raise ValueError(
            "pass exactly one of `image` (a path, bytes, an open file or a "
            "`data:` URL) or `object` (a key from `client.storage.upload`)."
        )
    if has_object:
        return Call(
            "POST",
            f"/v1/ai/documents/{route}",
            json=_body(object=str(object).strip(), thresholds=cuts),
            parse=_parser(DocumentVerdict),
        )

    files, body = _image_parts(image, media_type)
    if files is not None:
        # A multipart part is text, so `thresholds` travels as a JSON string and
        # is validated by the same spec the JSON body goes through.
        form = _body(thresholds=json.dumps(cuts) if cuts else None)
        return Call(
            "POST",
            f"/v1/ai/documents/{route}",
            files=files,
            data=form,
            parse=_parser(DocumentVerdict),
        )
    return Call(
        "POST",
        f"/v1/ai/documents/{route}",
        json={**(body or {}), **_body(thresholds=cuts)},
        parse=_parser(DocumentVerdict),
    )


def ktp_detect(
    image: ImageInput | None = None,
    *,
    object: str | None = None,
    thresholds: Thresholds | None = None,
    media_type: str | None = None,
) -> Call:
    return _document_detect(
        "ktp-detection", image, object=object, thresholds=thresholds, media_type=media_type
    )


def npwp_detect(
    image: ImageInput | None = None,
    *,
    object: str | None = None,
    thresholds: Thresholds | None = None,
    media_type: str | None = None,
) -> Call:
    return _document_detect(
        "npwp-detection", image, object=object, thresholds=thresholds, media_type=media_type
    )


# --------------------------------------------------------------------------- #
# Guardrails — what must not reach the model, and what must not leave it.


def prompt_injection_detect(
    text: str,
    *,
    thresholds: Thresholds | None = None,
) -> Call:
    return Call(
        "POST",
        "/v1/ai/guardrails/prompt-injection",
        json=_body(text=text, thresholds=_cuts(thresholds)),
        parse=_parser(PromptInjectionVerdict),
    )


def pii_mask(
    text: str,
    *,
    mask: bool = True,
    thresholds: Thresholds | None = None,
) -> Call:
    return Call(
        "POST",
        "/v1/ai/guardrails/pii-detection",
        json=_body(text=text, mask=mask, thresholds=_cuts(thresholds)),
        parse=_parser(PIIResult),
    )


# --------------------------------------------------------------------------- #
# Fraud — is this content trying to deceive the person reading it?


def email_phishing_detect(
    body: str,
    *,
    subject: str | None = None,
    sender: str | None = None,
    reply_to: str | None = None,
    display_name: str | None = None,
    thresholds: Thresholds | None = None,
) -> Call:
    return Call(
        "POST",
        "/v1/ai/fraud/email-phishing",
        json=_body(
            body=body,
            subject=subject,
            sender=sender,
            reply_to=reply_to,
            display_name=display_name,
            thresholds=_cuts(thresholds),
        ),
        parse=_parser(PhishingVerdict),
    )


def chat_scam_detect(
    message: str,
    *,
    history: Sequence[str] | None = None,
    sender: str | None = None,
    thresholds: Thresholds | None = None,
) -> Call:
    return Call(
        "POST",
        "/v1/ai/fraud/chat-scam",
        json=_body(
            message=message,
            history=_history(history),
            sender=sender,
            thresholds=_cuts(thresholds),
        ),
        parse=_parser(ScamVerdict),
    )


# --------------------------------------------------------------------------- #
# Chat — what is this message asking for?


def _intent_classify(
    route: str,
    message: str,
    *,
    history: Sequence[str] | None,
    channel: str | None,
    thresholds: Thresholds | None,
) -> Call:
    return Call(
        "POST",
        f"/v1/ai/chat/{route}",
        json=_body(
            message=message,
            history=_history(history),
            channel=channel,
            thresholds=_cuts(thresholds),
        ),
        parse=_parser(IntentResult),
    )


def commerce_intent_classify(
    message: str,
    *,
    history: Sequence[str] | None = None,
    channel: str | None = None,
    thresholds: Thresholds | None = None,
) -> Call:
    return _intent_classify(
        "commerce-intent",
        message,
        history=history,
        channel=channel,
        thresholds=thresholds,
    )


def software_support_intent_classify(
    message: str,
    *,
    history: Sequence[str] | None = None,
    channel: str | None = None,
    thresholds: Thresholds | None = None,
) -> Call:
    return _intent_classify(
        "software-support-intent",
        message,
        history=history,
        channel=channel,
        thresholds=thresholds,
    )


# --------------------------------------------------------------------------- #
# Storage — put a file in the bucket, send its key instead of its bytes.


def upload_url(content_type: str, *, filename: str = "", expires_in: int | None = None) -> Call:
    """`POST /v1/platform/storage/uploads` — a signed PUT for one file.

    Answers with the `SignedUrlData` dict: `object`, `url`, `method`, `headers`,
    `expires_at`. The file itself goes to `url`, never through the API.
    """
    return Call(
        "POST",
        "/v1/platform/storage/uploads",
        json=_body(content_type=content_type, filename=filename or None, expires_in=expires_in),
        parse=_raw,
    )


# --------------------------------------------------------------------------- #
# Models — the builds the hosted API serves, for a deployment of your own.


def _model_list(data: Any, request_id: str | None) -> list[ModelInfo]:
    return [ModelInfo.from_data(entry) for entry in (data or [])]


def _model_download(data: Any, request_id: str | None) -> ModelDownload:
    return ModelDownload.from_data(data or {})


def models_list() -> Call:
    """`GET /v1/platform/downloads/models` — every build you may download."""
    return Call("GET", "/v1/platform/downloads/models", parse=_model_list)


def models_download(slug: str) -> Call:
    """`POST /v1/platform/downloads/models/{slug}/download` — signed GETs.

    The API signs; it never streams a byte. Each file is then fetched straight
    from Cloud Storage, without the API key.
    """
    name = slug.strip().strip("/")
    if not name or "/" in name:
        raise ValueError(f"{slug!r} is not a model slug, e.g. 'guardrails-pii-detection-id'.")
    return Call(
        "POST",
        f"/v1/platform/downloads/models/{name}/download",
        parse=_model_download,
    )


# --------------------------------------------------------------------------- #
# Metadata.


def info(path: str) -> Call:
    """`GET <endpoint>/info` — what this deployment loaded, and what you may send.

    Authenticated like anything under `/v1/ai` and priced at nothing: `/info`
    declares no endpoint id, and an endpoint id is what a price row joins on.
    """
    return Call("GET", f"{path}/info", parse=_raw)


def health() -> Call:
    """`GET /health` — liveness. Dependency-free and never rate limited."""
    return Call("GET", "/health", parse=_raw)


# Route path per endpoint, in taxonomy order. The CLI and `Intura.endpoints()`
# read this rather than keeping a second list that can drift.
ENDPOINTS: dict[str, str] = {
    "documents/ktp-detection": "/v1/ai/documents/ktp-detection",
    "documents/npwp-detection": "/v1/ai/documents/npwp-detection",
    "guardrails/prompt-injection": "/v1/ai/guardrails/prompt-injection",
    "guardrails/pii-detection": "/v1/ai/guardrails/pii-detection",
    "fraud/email-phishing": "/v1/ai/fraud/email-phishing",
    "fraud/chat-scam": "/v1/ai/fraud/chat-scam",
    "chat/commerce-intent": "/v1/ai/chat/commerce-intent",
    "chat/software-support-intent": "/v1/ai/chat/software-support-intent",
}


__all__ = [
    "ACCEPTED_MEDIA_TYPES",
    "ENDPOINTS",
    "Call",
    "ImageInput",
    "chat_scam_detect",
    "commerce_intent_classify",
    "email_phishing_detect",
    "health",
    "info",
    "ktp_detect",
    "models_download",
    "models_list",
    "npwp_detect",
    "pii_mask",
    "prompt_injection_detect",
    "read_file",
    "software_support_intent_classify",
    "upload_url",
]
