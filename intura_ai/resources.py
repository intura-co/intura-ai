"""The endpoint tree, as attributes.

The path in the docs is the path in the code, so a `curl` and a call in Python
are the same string read twice:

    POST /v1/ai/guardrails/prompt-injection
    client.guardrails.prompt_injection.detect(text=...)

Every leaf has the endpoint's verb — `detect`, `mask`, `classify` — and `info()`,
which reports what this deployment accepts and can answer — the languages, the
cuts you may move. `info()` is authenticated like anything under `/v1/ai` and
priced at nothing.

Sync and async are the same tree twice. The methods are one line each because
the request they build lives in `_calls`; what differs between the two classes
is `await`, and nothing else. Keeping them explicit rather than generated is
what makes the signatures visible to an editor, which is where most people read
an SDK.
"""

from __future__ import annotations

import builtins
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from . import _calls, _files
from ._calls import ENDPOINTS, ImageInput, Thresholds
from ._errors import TransferError
from .types import (
    DocumentVerdict,
    IntentResult,
    ModelDownload,
    ModelInfo,
    PhishingVerdict,
    PIIResult,
    PromptInjectionVerdict,
    ScamVerdict,
)


class _Resource:
    def __init__(self, client: Any) -> None:
        self._client = client


class _AsyncResource:
    def __init__(self, client: Any) -> None:
        self._client = client


# --------------------------------------------------------------------------- #
# Documents — is this the card it claims to be, and is the photograph usable?
#
# Two independent verdicts and a control — and, when the frame is the card, what
# is printed on it (`verdict.extraction`): the raw values, a masked copy to store
# instead, and what the NIK encodes. Log the masked copy, never `fields`.
#
# A photograph goes up as `image` (a path, bytes, an open file, a `data:` URL) or
# as `object`, the key `client.storage.upload` returned. The second is for a
# capture already in the bucket — the bytes do not travel through the API twice.


class KTPDetection(_Resource):
    """`/v1/ai/documents/ktp-detection` — is this an Indonesian KTP?"""

    _path = ENDPOINTS["documents/ktp-detection"]

    def detect(
        self,
        image: ImageInput | None = None,
        *,
        object: str | None = None,
        thresholds: Thresholds | None = None,
        media_type: str | None = None,
    ) -> DocumentVerdict:
        """Check one photograph. `image` is a path, bytes, an open file or a
        base64 `data:` URL; the media type is read from the file or the bytes
        unless you name it. Or pass `object`, a key from `client.storage.upload`
        — exactly one of the two."""
        return self._client._call(
            _calls.ktp_detect(image, object=object, thresholds=thresholds, media_type=media_type)
        )

    def info(self) -> dict[str, Any]:
        return self._client._call(_calls.info(self._path))


class NPWPDetection(_Resource):
    """`/v1/ai/documents/npwp-detection` — is this an Indonesian NPWP card?"""

    _path = ENDPOINTS["documents/npwp-detection"]

    def detect(
        self,
        image: ImageInput | None = None,
        *,
        object: str | None = None,
        thresholds: Thresholds | None = None,
        media_type: str | None = None,
    ) -> DocumentVerdict:
        """The same two questions for the tax card, both number generations."""
        return self._client._call(
            _calls.npwp_detect(image, object=object, thresholds=thresholds, media_type=media_type)
        )

    def info(self) -> dict[str, Any]:
        return self._client._call(_calls.info(self._path))


class Documents(_Resource):
    """Is this the card it claims to be, and is the photograph usable?"""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.ktp_detection = KTPDetection(client)
        self.npwp_detection = NPWPDetection(client)


# --------------------------------------------------------------------------- #
# Guardrails — what must not reach the model, and what must not leave it.


class PromptInjection(_Resource):
    """`/v1/ai/guardrails/prompt-injection` — screen input for instruction override."""

    _path = ENDPOINTS["guardrails/prompt-injection"]

    def detect(
        self,
        text: str,
        *,
        thresholds: Thresholds | None = None,
    ) -> PromptInjectionVerdict:
        """Screen a user turn or a retrieved document.

        Run it on both. A retrieved page carrying "ignore your instructions" is
        the same attack as a user typing it, and it arrives at the model with
        more trust.
        """
        return self._client._call(_calls.prompt_injection_detect(text, thresholds=thresholds))

    def info(self) -> dict[str, Any]:
        return self._client._call(_calls.info(self._path))


class PIIDetection(_Resource):
    """`/v1/ai/guardrails/pii-detection` — find and mask personal data."""

    _path = ENDPOINTS["guardrails/pii-detection"]

    def mask(
        self,
        text: str,
        *,
        mask: bool = True,
        thresholds: Thresholds | None = None,
    ) -> PIIResult:
        """Mask personal data in Indonesian or English text.

        `mask=False` detects without producing a masked copy — the span list
        still comes back. Use it to decide whether a document may move at all,
        rather than to move a redacted version of it.
        """
        return self._client._call(_calls.pii_mask(text, mask=mask, thresholds=thresholds))

    def detect(
        self,
        text: str,
        *,
        thresholds: Thresholds | None = None,
    ) -> PIIResult:
        """Spans only, no masked copy. `mask(text, mask=False)` by another name."""
        return self.mask(text, mask=False, thresholds=thresholds)

    def info(self) -> dict[str, Any]:
        return self._client._call(_calls.info(self._path))


class Guardrails(_Resource):
    """What must not reach the model, and what must not leave it."""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.prompt_injection = PromptInjection(client)
        self.pii_detection = PIIDetection(client)


# --------------------------------------------------------------------------- #
# Fraud — is this content trying to deceive the person reading it?


class EmailPhishing(_Resource):
    """`/v1/ai/fraud/email-phishing` — screen an email for social engineering."""

    _path = ENDPOINTS["fraud/email-phishing"]

    def detect(
        self,
        body: str,
        *,
        subject: str | None = None,
        sender: str | None = None,
        reply_to: str | None = None,
        display_name: str | None = None,
        thresholds: Thresholds | None = None,
    ) -> PhishingVerdict:
        """Screen one email. Send the HTML body as-is rather than flattening it:
        an anchor whose text and href name different domains is one of the
        strongest signals available, and stripping tags destroys it. Every
        header you have raises recall."""
        return self._client._call(
            _calls.email_phishing_detect(
                body,
                subject=subject,
                sender=sender,
                reply_to=reply_to,
                display_name=display_name,
                thresholds=thresholds,
            )
        )

    def info(self) -> dict[str, Any]:
        return self._client._call(_calls.info(self._path))


class ChatScam(_Resource):
    """`/v1/ai/fraud/chat-scam` — classify a chat message into seven kinds of fraud."""

    _path = ENDPOINTS["fraud/chat-scam"]

    def detect(
        self,
        message: str,
        *,
        history: Sequence[str] | None = None,
        sender: str | None = None,
        thresholds: Thresholds | None = None,
    ) -> ScamVerdict:
        """Screen one message, with the turns before it if you have them.

        `history` is worth sending: half of chat fraud is an arc rather than a
        message, and none of the lines that build it is a scam on its own.
        """
        return self._client._call(
            _calls.chat_scam_detect(message, history=history, sender=sender, thresholds=thresholds)
        )

    def info(self) -> dict[str, Any]:
        return self._client._call(_calls.info(self._path))


class Fraud(_Resource):
    """Is this content trying to deceive the person reading it?"""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.email_phishing = EmailPhishing(client)
        self.chat_scam = ChatScam(client)


# --------------------------------------------------------------------------- #
# Chat — what is this message asking for?


class CommerceIntent(_Resource):
    """`/v1/ai/chat/commerce-intent` — sort a customer message into six classes."""

    _path = ENDPOINTS["chat/commerce-intent"]

    def classify(
        self,
        message: str,
        *,
        history: Sequence[str] | None = None,
        channel: str | None = None,
        thresholds: Thresholds | None = None,
    ) -> IntentResult:
        """Classify one message and say whether it is confident enough to route.

        Branch on `result.action`: `route` sends it to the class handler,
        `escalate` sends it to a larger model or a person.
        """
        return self._client._call(
            _calls.commerce_intent_classify(
                message, history=history, channel=channel, thresholds=thresholds
            )
        )

    def info(self) -> dict[str, Any]:
        return self._client._call(_calls.info(self._path))


class SoftwareSupportIntent(_Resource):
    """`/v1/ai/chat/software-support-intent` — sort a support ticket into five classes."""

    _path = ENDPOINTS["chat/software-support-intent"]

    def classify(
        self,
        message: str,
        *,
        history: Sequence[str] | None = None,
        channel: str | None = None,
        thresholds: Thresholds | None = None,
    ) -> IntentResult:
        """Classify one ticket — broken, missing, identity, money, or a question —
        and say whether it is confident enough to file on. Send the whole ticket,
        log paste included."""
        return self._client._call(
            _calls.software_support_intent_classify(
                message, history=history, channel=channel, thresholds=thresholds
            )
        )

    def info(self) -> dict[str, Any]:
        return self._client._call(_calls.info(self._path))


class Chat(_Resource):
    """What is this message asking for?"""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.commerce_intent = CommerceIntent(client)
        self.software_support_intent = SoftwareSupportIntent(client)


# --------------------------------------------------------------------------- #
# Storage — put a file in the bucket, send its key instead of its bytes.


class Storage(_Resource):
    """`/v1/platform/storage` — upload a file straight to Cloud Storage."""

    def upload(
        self,
        file: ImageInput,
        *,
        content_type: str | None = None,
        expires_in: int | None = None,
    ) -> str:
        """Upload one file and return its `object` key.

        The API signs a PUT; the bytes go straight to the bucket, without your
        API key. Pass the key to `documents.*.detect(object=...)`. Free: the
        endpoint that reads the object is the one that counts a request.
        """
        filename, data, media_type = _calls.read_file(file, content_type)
        signed = self._client._call(
            _calls.upload_url(media_type, filename=filename, expires_in=expires_in)
        )
        try:
            response = self._client._http.request(**_files.put_request(signed, data))
        except httpx.HTTPError as exc:
            raise TransferError(f"could not reach Cloud Storage: {exc}") from exc
        _files.check_put(response)
        return str(signed["object"])


# --------------------------------------------------------------------------- #
# Models — download the builds the hosted API serves, to run them yourself.
#
# The API key is used to list and to sign. The files come straight from Cloud
# Storage on links that live fifteen minutes; the key is never sent there.
# Downloading is free and does not count against your requests.


class Models(_Resource):
    """`/v1/platform/downloads/models` — the models you may self-host."""

    def list(self) -> builtins.list[ModelInfo]:
        """Every build you may download: one per endpoint and language."""
        return self._client._call(_calls.models_list())

    def links(self, slug: str) -> ModelDownload:
        """Signed Cloud Storage links to every file of `slug`, without
        downloading anything. For a downloader of your own."""
        return self._client._call(_calls.models_download(slug))

    def download(
        self,
        slug: str,
        dest: str | os.PathLike[str] = ".",
        *,
        overwrite: bool = False,
    ) -> Path:
        """Download every file of `slug` into `<dest>/<slug>/<version>/`.

        Returns that directory. A file already there at the right size is kept
        unless `overwrite`, so an interrupted download picks up where it
        stopped. Raises `NotFound` for a slug that is not in `list()`, and
        `TransferError` when Cloud Storage refuses or drops a file.
        """
        links = self.links(slug)
        root = _files.bundle_dir(dest, links)
        resigned = False
        for name in [f.name for f in _files.pending(root, links.files, overwrite)]:
            file = next(f for f in links.files if f.name == name)
            target = _files.target_path(root, name)
            status = _files.fetch(self._client._http, file, target)
            if status is None:
                continue
            if resigned:
                raise _files.refused(file, status)
            # The links expired mid-bundle. Sign the rest once more; every
            # file after this one uses the fresh links too.
            links, resigned = self.links(slug), True
            file = next((f for f in links.files if f.name == name), file)
            if _files.fetch(self._client._http, file, target) is not None:
                raise _files.refused(file, status)
        return root


# --------------------------------------------------------------------------- #
# The same tree, awaited.


class AsyncKTPDetection(_AsyncResource):
    """`/v1/ai/documents/ktp-detection` — is this an Indonesian KTP?"""

    _path = ENDPOINTS["documents/ktp-detection"]

    async def detect(
        self,
        image: ImageInput | None = None,
        *,
        object: str | None = None,
        thresholds: Thresholds | None = None,
        media_type: str | None = None,
    ) -> DocumentVerdict:
        return await self._client._call(
            _calls.ktp_detect(image, object=object, thresholds=thresholds, media_type=media_type)
        )

    async def info(self) -> dict[str, Any]:
        return await self._client._call(_calls.info(self._path))


class AsyncNPWPDetection(_AsyncResource):
    """`/v1/ai/documents/npwp-detection` — is this an Indonesian NPWP card?"""

    _path = ENDPOINTS["documents/npwp-detection"]

    async def detect(
        self,
        image: ImageInput | None = None,
        *,
        object: str | None = None,
        thresholds: Thresholds | None = None,
        media_type: str | None = None,
    ) -> DocumentVerdict:
        return await self._client._call(
            _calls.npwp_detect(image, object=object, thresholds=thresholds, media_type=media_type)
        )

    async def info(self) -> dict[str, Any]:
        return await self._client._call(_calls.info(self._path))


class AsyncDocuments(_AsyncResource):
    """Is this the card it claims to be, and is the photograph usable?"""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.ktp_detection = AsyncKTPDetection(client)
        self.npwp_detection = AsyncNPWPDetection(client)


class AsyncPromptInjection(_AsyncResource):
    """`/v1/ai/guardrails/prompt-injection` — screen input for instruction override."""

    _path = ENDPOINTS["guardrails/prompt-injection"]

    async def detect(
        self,
        text: str,
        *,
        thresholds: Thresholds | None = None,
    ) -> PromptInjectionVerdict:
        return await self._client._call(_calls.prompt_injection_detect(text, thresholds=thresholds))

    async def info(self) -> dict[str, Any]:
        return await self._client._call(_calls.info(self._path))


class AsyncPIIDetection(_AsyncResource):
    """`/v1/ai/guardrails/pii-detection` — find and mask personal data."""

    _path = ENDPOINTS["guardrails/pii-detection"]

    async def mask(
        self,
        text: str,
        *,
        mask: bool = True,
        thresholds: Thresholds | None = None,
    ) -> PIIResult:
        return await self._client._call(_calls.pii_mask(text, mask=mask, thresholds=thresholds))

    async def detect(
        self,
        text: str,
        *,
        thresholds: Thresholds | None = None,
    ) -> PIIResult:
        return await self.mask(text, mask=False, thresholds=thresholds)

    async def info(self) -> dict[str, Any]:
        return await self._client._call(_calls.info(self._path))


class AsyncGuardrails(_AsyncResource):
    """What must not reach the model, and what must not leave it."""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.prompt_injection = AsyncPromptInjection(client)
        self.pii_detection = AsyncPIIDetection(client)


class AsyncEmailPhishing(_AsyncResource):
    """`/v1/ai/fraud/email-phishing` — screen an email for social engineering."""

    _path = ENDPOINTS["fraud/email-phishing"]

    async def detect(
        self,
        body: str,
        *,
        subject: str | None = None,
        sender: str | None = None,
        reply_to: str | None = None,
        display_name: str | None = None,
        thresholds: Thresholds | None = None,
    ) -> PhishingVerdict:
        return await self._client._call(
            _calls.email_phishing_detect(
                body,
                subject=subject,
                sender=sender,
                reply_to=reply_to,
                display_name=display_name,
                thresholds=thresholds,
            )
        )

    async def info(self) -> dict[str, Any]:
        return await self._client._call(_calls.info(self._path))


class AsyncChatScam(_AsyncResource):
    """`/v1/ai/fraud/chat-scam` — classify a chat message into seven kinds of fraud."""

    _path = ENDPOINTS["fraud/chat-scam"]

    async def detect(
        self,
        message: str,
        *,
        history: Sequence[str] | None = None,
        sender: str | None = None,
        thresholds: Thresholds | None = None,
    ) -> ScamVerdict:
        return await self._client._call(
            _calls.chat_scam_detect(message, history=history, sender=sender, thresholds=thresholds)
        )

    async def info(self) -> dict[str, Any]:
        return await self._client._call(_calls.info(self._path))


class AsyncFraud(_AsyncResource):
    """Is this content trying to deceive the person reading it?"""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.email_phishing = AsyncEmailPhishing(client)
        self.chat_scam = AsyncChatScam(client)


class AsyncCommerceIntent(_AsyncResource):
    """`/v1/ai/chat/commerce-intent` — sort a customer message into six classes."""

    _path = ENDPOINTS["chat/commerce-intent"]

    async def classify(
        self,
        message: str,
        *,
        history: Sequence[str] | None = None,
        channel: str | None = None,
        thresholds: Thresholds | None = None,
    ) -> IntentResult:
        return await self._client._call(
            _calls.commerce_intent_classify(
                message, history=history, channel=channel, thresholds=thresholds
            )
        )

    async def info(self) -> dict[str, Any]:
        return await self._client._call(_calls.info(self._path))


class AsyncSoftwareSupportIntent(_AsyncResource):
    """`/v1/ai/chat/software-support-intent` — sort a support ticket into five classes."""

    _path = ENDPOINTS["chat/software-support-intent"]

    async def classify(
        self,
        message: str,
        *,
        history: Sequence[str] | None = None,
        channel: str | None = None,
        thresholds: Thresholds | None = None,
    ) -> IntentResult:
        return await self._client._call(
            _calls.software_support_intent_classify(
                message, history=history, channel=channel, thresholds=thresholds
            )
        )

    async def info(self) -> dict[str, Any]:
        return await self._client._call(_calls.info(self._path))


class AsyncChat(_AsyncResource):
    """What is this message asking for?"""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.commerce_intent = AsyncCommerceIntent(client)
        self.software_support_intent = AsyncSoftwareSupportIntent(client)


class AsyncStorage(_AsyncResource):
    """`/v1/platform/storage` — upload a file straight to Cloud Storage."""

    async def upload(
        self,
        file: ImageInput,
        *,
        content_type: str | None = None,
        expires_in: int | None = None,
    ) -> str:
        filename, data, media_type = _calls.read_file(file, content_type)
        signed = await self._client._call(
            _calls.upload_url(media_type, filename=filename, expires_in=expires_in)
        )
        try:
            response = await self._client._http.request(**_files.put_request(signed, data))
        except httpx.HTTPError as exc:
            raise TransferError(f"could not reach Cloud Storage: {exc}") from exc
        _files.check_put(response)
        return str(signed["object"])


class AsyncModels(_AsyncResource):
    """`/v1/platform/downloads/models` — the models you may self-host."""

    async def list(self) -> builtins.list[ModelInfo]:
        return await self._client._call(_calls.models_list())

    async def links(self, slug: str) -> ModelDownload:
        return await self._client._call(_calls.models_download(slug))

    async def download(
        self,
        slug: str,
        dest: str | os.PathLike[str] = ".",
        *,
        overwrite: bool = False,
    ) -> Path:
        links = await self.links(slug)
        root = _files.bundle_dir(dest, links)
        resigned = False
        for name in [f.name for f in _files.pending(root, links.files, overwrite)]:
            file = next(f for f in links.files if f.name == name)
            target = _files.target_path(root, name)
            status = await _files.afetch(self._client._http, file, target)
            if status is None:
                continue
            if resigned:
                raise _files.refused(file, status)
            # The links expired mid-bundle. Sign the rest once more; every
            # file after this one uses the fresh links too.
            links, resigned = await self.links(slug), True
            file = next((f for f in links.files if f.name == name), file)
            if await _files.afetch(self._client._http, file, target) is not None:
                raise _files.refused(file, status)
        return root


__all__ = [
    "AsyncChat",
    "AsyncDocuments",
    "AsyncFraud",
    "AsyncGuardrails",
    "AsyncModels",
    "AsyncStorage",
    "Chat",
    "Documents",
    "Fraud",
    "Guardrails",
    "Models",
    "Storage",
]
