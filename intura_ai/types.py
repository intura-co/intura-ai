"""The response contract, as Python objects.

Every `/v1/ai` endpoint answers with the same four fields, so every result here
is the same four attributes:

    label        what it decided           "pii_found", "scam", "benign"
    score        the number it decided ON  0.94
    thresholds   the cuts it decided BY    {"block": 0.7, "review": 0.4}
    result       the answer, or None       {"masked_text": "…", "entities": […]}

`label` is `score` compared against `thresholds`. Reading the three together is
the whole explanation of any answer this API gives: *0.94 is at or above your
block cut of 0.7, so this is an injection.*

There is one number, and it is `score`
--------------------------------------
Not `confidence`. The API carried both once — how sure it was of the label,
beside the number the thresholds actually cut — and every integration in front
of two 0..1 floats thresholded on whichever it read first. Only `score` ships,
under the name the cuts are stated in. "How far from the line" is arithmetic you
can do here: `score - thresholds["block"]`.

Thresholds are yours
--------------------
Intura ships a default; you move it in the console for every call your keys
make, or per call with `thresholds=` on any method here. What comes back in
`.thresholds` is what was actually applied after both, which is why it is on
every response: a verdict cannot be compared with an older one unless you know
the bar each was held to.

Why the subclasses are thin
---------------------------
Each endpoint adds properties over the same `result` dict rather than a parallel
set of fields, so `.result` is always exactly what the API sent and nothing here
can quietly disagree with it. A field the service adds tomorrow is readable the
day it ships, through `.result`, without a release of this package.

Branch on `.action` where a route has one — `block`/`review`/`allow`,
`accept`/`retry`/`reject`, `route`/`escalate`. It is the endpoint's own policy
decision, already taken against your cuts, and re-deriving it from `score` in
each caller is how two services end up disagreeing about the same verdict.

None of these is truth-testable, deliberately. `if verdict:` would have to mean
"an attack was found" on the guardrail trees and "the photograph is fine" on the
document ones, and a reader cannot see which from the call site. The named
property says it: `.injection`, `.found`, `.scam`, `.phishing`, `.routed`,
`.accepted`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DocumentVerdict",
    "EntitySpan",
    "Extraction",
    "Inference",
    "IntentResult",
    "ModelDownload",
    "ModelFile",
    "ModelInfo",
    "PIIResult",
    "PhishingVerdict",
    "PromptInjectionVerdict",
    "ScamVerdict",
]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@dataclass(frozen=True)
class Inference:
    """The four fields every `/v1/ai` endpoint returns.

    There was a fifth, `model`, naming the artifact that answered. The API
    stopped sending it on 2026-09-24 and stopped accepting a `model` on the
    request in the same change: a caller cannot act on the name — the score and
    the cuts are the same contract whichever build ran — and reading it invited
    integrations to branch on Intura's internals. Which build serves a call is
    answered by the deployment's own operator surfaces, not by this object.
    """

    label: str
    score: float
    thresholds: dict[str, float] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    request_id: str | None = None

    @classmethod
    def from_data(cls, data: dict[str, Any], *, request_id: str | None = None) -> Any:
        """Build from the `data` object of a success envelope.

        Tolerant on purpose: an endpoint that adds a field, or a deployment
        running a build older than this package, must not raise here. Missing
        keys take their documented empty value and `result` keeps whatever the
        service sent.
        """
        return cls(
            label=str(data.get("label", "")),
            score=float(data.get("score") or 0.0),
            thresholds={str(k): float(v) for k, v in _mapping(data.get("thresholds")).items()},
            result=data.get("result") if isinstance(data.get("result"), dict) else None,
            request_id=request_id,
        )

    @property
    def action(self) -> str | None:
        """The decision to branch on, or None on a route that has none."""
        return (self.result or {}).get("action")

    @property
    def languages(self) -> list[str]:
        """The languages the input was actually read in.

        Every supported language when none could be identified: reading in one
        language too many costs a forward pass, one too few is a hole.
        """
        return list((self.result or {}).get("languages") or [])

    def get(self, key: str, default: Any = None) -> Any:
        """A key out of `result`, without the `or {}` dance at the call site."""
        return (self.result or {}).get(key, default)


@dataclass(frozen=True)
class PromptInjectionVerdict(Inference):
    """`POST /v1/ai/guardrails/prompt-injection`.

    `label` is `injection`, `suspicious` or `benign`; `score` is P(injection).
    """

    @property
    def injection(self) -> bool:
        """True when the score crossed your block cut."""
        return bool(self.get("injection", False))


@dataclass(frozen=True)
class EntitySpan:
    """One detected PII span. Never carries the matched value.

    Offsets index the text you submitted, so `text[span.start:span.end]` is the
    value — which is the point: the span list can be logged and the document
    cannot.
    """

    type: str
    placeholder: str
    start: int
    end: int
    score: float

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> EntitySpan:
        return cls(
            type=str(data.get("type", "")),
            placeholder=str(data.get("placeholder", "")),
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            score=float(data.get("score") or 0.0),
        )


@dataclass(frozen=True)
class PIIResult(Inference):
    """`POST /v1/ai/guardrails/pii-detection`.

    `label` is `pii_found` or `clean`.
    """

    @property
    def masked_text(self) -> str | None:
        """Your document with every span replaced by its typed placeholder.

        Byte-identical to what you sent everywhere else. None when the call
        passed `mask=False`.
        """
        return self.get("masked_text")

    @property
    def entities(self) -> list[EntitySpan]:
        """Every span found, in document order."""
        return [EntitySpan.from_data(e) for e in (self.get("entities") or [])]

    @property
    def counts(self) -> dict[str, int]:
        """Spans per entity type."""
        return {str(k): int(v) for k, v in _mapping(self.get("counts")).items()}

    @property
    def risk(self) -> float:
        """How much of a problem this document is, 0..1.

        A different question from `score`: it rises with the number of distinct
        entity types and with their sensitivity — one national ID outranks one
        phone number, and an ID beside a name outranks either alone.
        """
        return float(self.get("risk") or 0.0)

    @property
    def found(self) -> bool:
        return self.label == "pii_found"


@dataclass(frozen=True)
class PhishingVerdict(Inference):
    """`POST /v1/ai/fraud/email-phishing`.

    `label` is `phishing`, `suspicious` or `benign`.
    """

    @property
    def phishing(self) -> bool:
        return bool(self.get("phishing", False))

    @property
    def fields_screened(self) -> list[str]:
        """Which of `subject`, `body`, `sender`, `reply_to`, `display_name` were
        sent. The header checks cannot fire on what was not sent, so this is how
        a body-only verdict is told from a full one."""
        return list(self.get("fields_screened") or [])


@dataclass(frozen=True)
class ScamVerdict(Inference):
    """`POST /v1/ai/fraud/chat-scam`.

    `label` is `benign` or which kind of scam: `otp_theft`, `phishing_link`,
    `impersonation`, `investment_scam`, `job_scam`, `prize_scam`, `loan_scam`.
    """

    @property
    def scam(self) -> bool:
        return bool(self.get("scam", False))

    @property
    def scores(self) -> dict[str, float]:
        """Probability mass per class. Answers *which kind*, not *whether*."""
        return {str(k): float(v) for k, v in _mapping(self.get("scores")).items()}

    @property
    def guidance(self) -> str:
        """One line to show the person who received the message. Empty for benign."""
        return str(self.get("guidance") or "")

    @property
    def history_used(self) -> int:
        """How many previous turns were read. Two verdicts with different values
        here did not see the same evidence."""
        return int(self.get("history_used") or 0)


@dataclass(frozen=True)
class IntentResult(Inference):
    """`POST /v1/ai/chat/commerce-intent` and `/chat/software-support-intent`.

    `action` is `route` or `escalate`, and it is the field to branch on: a class
    that cleared both halves of the check can go to its handler, one that did
    not goes to a larger model or a person. `label` is `other` whenever the
    action is `escalate`, so routing on `label` alone silently sends every
    unsure message to the `other` queue.
    """

    @property
    def routed(self) -> bool:
        return self.action == "route"

    @property
    def scores(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in _mapping(self.get("scores")).items()}

    @property
    def candidate(self) -> str:
        """The class preferred before the confidence check. For logging and
        threshold tuning — never route on it."""
        return str(self.get("candidate") or "")

    @property
    def margin(self) -> float:
        """The top class's lead over the runner-up. The second half of the check:
        a top class that barely leads is a coin flip however high it scored."""
        return float(self.get("margin") or 0.0)

    @property
    def handling(self) -> str:
        """One line on what this class means and what a sensible handler does."""
        return str(self.get("handling") or "")

    @property
    def history_used(self) -> int:
        return int(self.get("history_used") or 0)


@dataclass(frozen=True)
class DocumentVerdict(Inference):
    """`POST /v1/ai/documents/ktp-detection` and `/documents/npwp-detection`.

    Two independent verdicts and a control. `action` is `accept` (right card,
    usable photo), `retry` (ask for another capture, show `advice`) or `reject`
    (not the document you asked for).

    When the frame is the card, `extraction` carries what is printed on it —
    the raw values, a masked copy to store or log instead, and what the NIK
    encodes. It is None for any other label.
    """

    @property
    def extraction(self) -> Extraction | None:
        """The card's printed values, three ways. None unless `label` is this
        card and text was read."""
        data = self.get("extraction")
        return Extraction.from_data(data) if isinstance(data, dict) else None

    @property
    def document(self) -> dict[str, Any]:
        """Is this the card it claims to be — evidence and its own score."""
        return _mapping(self.get("document"))

    @property
    def quality(self) -> dict[str, Any]:
        """Is the photograph usable — capture metrics and advice."""
        return _mapping(self.get("quality"))

    @property
    def image(self) -> dict[str, Any]:
        """What was decoded: dimensions, media type, bytes read."""
        return _mapping(self.get("image"))

    @property
    def accepted(self) -> bool:
        return self.action == "accept"

    @property
    def advice(self) -> list[str]:
        """What to tell the person holding the camera, one line per issue.

        Written for them rather than for a developer — these are the strings a
        capture UI shows. Empty when there is nothing to fix.
        """
        return [str(line) for line in (self.quality.get("advice") or [])]

    @property
    def issues(self) -> list[str]:
        """What is wrong with the capture, named for your logs rather than for
        the person holding the phone."""
        return [str(issue) for issue in (self.quality.get("issues") or [])]


@dataclass(frozen=True)
class Extraction:
    """`result.extraction` on a document verdict.

    `fields` is every printed value, tidied — digits as digits, dates as ISO —
    with None for a field that was not read. `masked` has the same keys with the
    identifying values replaced by the placeholders `guardrails/pii-detection`
    writes (`[NATIONAL_ID]`, `[NAME]`, `[DATE_OF_BIRTH]`, `[ADDRESS]`,
    `[TAX_ID]`): the copy to store or log where the raw values do not belong.
    `derived` is what the NIK encodes — region codes, date of birth, gender —
    and None unless the number passes its format check.
    """

    fields: dict[str, str | None] = field(default_factory=dict)
    masked: dict[str, str | None] = field(default_factory=dict)
    derived: dict[str, str] | None = None

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> Extraction:
        derived = data.get("derived")
        return cls(
            fields=_mapping(data.get("fields")),
            masked=_mapping(data.get("masked")),
            derived=_mapping(derived) if isinstance(derived, dict) else None,
        )


# --------------------------------------------------------------------------- #
# Models — `client.models`. Not inference results: these describe a build.


@dataclass(frozen=True)
class ModelInfo:
    """One model you may download: the same build the hosted API serves for
    `endpoint` in `language`. `version` is the build id — the string to quote
    when you ask about what you are running."""

    slug: str
    name: str
    tree: str
    endpoint: str
    language: str
    version: str
    files: tuple[str, ...] = ()

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> ModelInfo:
        return cls(
            slug=str(data.get("slug", "")),
            name=str(data.get("name", "")),
            tree=str(data.get("tree", "")),
            endpoint=str(data.get("endpoint", "")),
            language=str(data.get("language", "")),
            version=str(data.get("version", "")),
            files=tuple(str(f) for f in (data.get("files") or [])),
        )


@dataclass(frozen=True)
class ModelFile:
    """One file of a build, and a signed Cloud Storage link to it."""

    name: str
    size: int
    url: str = field(repr=False)

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> ModelFile:
        return cls(
            name=str(data.get("name", "")),
            size=int(data.get("size") or 0),
            url=str(data.get("url", "")),
        )


@dataclass(frozen=True)
class ModelDownload:
    """Signed links to every file of one build, valid until `expires_at`.

    The links are short-lived and cannot be revoked, so treat them like a
    credential: `url` is left out of the repr for that reason.
    """

    slug: str
    version: str
    files: tuple[ModelFile, ...] = ()
    expires_at: str | None = None

    @property
    def size(self) -> int:
        """Total bytes."""
        return sum(f.size for f in self.files)

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> ModelDownload:
        return cls(
            slug=str(data.get("slug", "")),
            version=str(data.get("version", "")),
            files=tuple(ModelFile.from_data(f) for f in (data.get("files") or [])),
            expires_at=str(data["expires_at"]) if data.get("expires_at") else None,
        )
