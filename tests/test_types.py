"""Reading a response: the four fields, and the properties over `result`."""

from __future__ import annotations

from intura_ai.types import (
    DocumentVerdict,
    Inference,
    IntentResult,
    PhishingVerdict,
    PIIResult,
    PromptInjectionVerdict,
    ScamVerdict,
)


class TestTheFourFields:
    def test_every_answer_is_the_same_four_fields(self):
        verdict = Inference.from_data(
            {
                "label": "injection",
                "score": 0.97,
                "thresholds": {"block": 0.7, "review": 0.4},
                "result": {"injection": True, "action": "block", "languages": ["id", "en"]},
            },
            request_id="req_1",
        )
        assert verdict.label == "injection"
        assert verdict.score == 0.97
        assert verdict.thresholds["block"] == 0.7
        assert verdict.action == "block"
        assert verdict.languages == ["id", "en"]
        assert verdict.request_id == "req_1"

    def test_the_distance_to_the_line_is_arithmetic(self):
        """Nothing was lost by dropping `confidence`: the cut is on the response
        beside the score."""
        verdict = Inference.from_data(
            {"label": "injection", "score": 0.9, "thresholds": {"block": 0.7}}
        )
        assert round(verdict.score - verdict.thresholds["block"], 2) == 0.2

    def test_a_field_this_package_has_never_heard_of_is_still_readable(self):
        """An endpoint that adds a key must be usable the day it ships, not the
        day this package catches up."""
        verdict = Inference.from_data(
            {"label": "benign", "score": 0.1, "result": {"something_new": 42}}
        )
        assert verdict.get("something_new") == 42

    def test_a_null_result_does_not_raise(self):
        """`result` is null when the input was rejected and there is nothing to
        return."""
        verdict = Inference.from_data({"label": "unreadable", "score": 0.0, "result": None})
        assert verdict.result is None
        assert verdict.action is None
        assert verdict.languages == []

    def test_a_sparse_payload_takes_documented_empties(self):
        verdict = Inference.from_data({"label": "benign"})
        assert verdict.score == 0.0
        assert verdict.thresholds == {}
        assert verdict.result is None

    def test_a_model_key_on_the_wire_is_ignored(self):
        """A deployment older than this package still sends `model`. It must not
        reappear as an attribute, and it must not raise."""
        verdict = Inference.from_data({"label": "benign", "score": 0.1, "model": "whatever"})
        assert not hasattr(verdict, "model")
        assert verdict.label == "benign"


class TestPromptInjection:
    def test_the_boolean_and_the_action_agree(self):
        verdict = PromptInjectionVerdict.from_data(
            {
                "label": "injection",
                "score": 0.97,
                "thresholds": {"block": 0.7},
                "result": {"injection": True, "action": "block", "languages": ["id"]},
            }
        )
        assert verdict.injection is True
        assert verdict.action == "block"

    def test_review_is_neither_blocked_nor_ignored(self):
        """The band exists so one cut does not have to catch attacks and also
        not block a curious user."""
        verdict = PromptInjectionVerdict.from_data(
            {
                "label": "suspicious",
                "score": 0.55,
                "thresholds": {"block": 0.7, "review": 0.4},
                "result": {"injection": False, "action": "review", "languages": ["en"]},
            }
        )
        assert verdict.injection is False
        assert verdict.action == "review"


class TestPII:
    def _result(self, **overrides):
        data = {
            "label": "pii_found",
            "score": 0.94,
            "thresholds": {"min_score": 0.3},
            "result": {
                "masked_text": "Halo, saya [NAME], NIK [NATIONAL_ID].",
                "entities": [
                    {
                        "type": "PERSON",
                        "placeholder": "[NAME]",
                        "start": 11,
                        "end": 23,
                        "score": 0.78,
                    },
                    {
                        "type": "NATIONAL_ID",
                        "placeholder": "[NATIONAL_ID]",
                        "start": 29,
                        "end": 45,
                        "score": 1.0,
                    },
                ],
                "n_entities": 2,
                "counts": {"PERSON": 1, "NATIONAL_ID": 1},
                "risk": 0.62,
                "languages": ["id"],
            },
        }
        data["result"].update(overrides)
        return PIIResult.from_data(data)

    def test_spans_come_back_typed(self):
        result = self._result()
        assert [e.type for e in result.entities] == ["PERSON", "NATIONAL_ID"]
        assert result.entities[0].placeholder == "[NAME]"
        assert result.counts == {"PERSON": 1, "NATIONAL_ID": 1}

    def test_offsets_index_the_text_you_sent(self):
        text = "Halo, saya Budi Santoso, NIK 3174012345678901."
        span = self._result().entities[0]
        assert text[span.start : span.end] == "Budi Santoso"

    def test_a_format_verified_span_scores_one(self):
        """A Luhn-checked card or a structurally valid NIK cannot be a lookalike."""
        assert self._result().entities[1].score == 1.0

    def test_risk_is_not_score(self):
        """Two different questions: how sure, and how much of a problem."""
        result = self._result()
        assert result.score == 0.94
        assert result.risk == 0.62

    def test_detect_only_returns_no_masked_copy(self):
        assert self._result(masked_text=None).masked_text is None


class TestFraudAndChat:
    def test_scam_names_the_kind_and_what_to_tell_the_recipient(self):
        verdict = ScamVerdict.from_data(
            {
                "label": "otp_theft",
                "score": 0.91,
                "thresholds": {"block": 0.6},
                "result": {
                    "scam": True,
                    "action": "block",
                    "scores": {"otp_theft": 0.91, "benign": 0.02},
                    "guidance": "Jangan pernah membagikan kode OTP.",
                    "history_used": 2,
                    "languages": ["id"],
                },
            }
        )
        assert verdict.label == "otp_theft"
        assert verdict.scores["otp_theft"] == 0.91
        assert verdict.guidance.startswith("Jangan")
        assert verdict.history_used == 2

    def test_phishing_says_which_fields_it_could_read(self):
        verdict = PhishingVerdict.from_data(
            {
                "label": "phishing",
                "score": 0.88,
                "result": {
                    "phishing": True,
                    "action": "block",
                    "fields_screened": ["body", "subject"],
                    "languages": ["id"],
                },
            }
        )
        assert verdict.phishing is True
        assert verdict.fields_screened == ["body", "subject"]

    def test_an_escalated_intent_is_labelled_other(self):
        """Routing on `label` alone would send every unsure message to the
        `other` queue; `action` is the field to branch on."""
        result = IntentResult.from_data(
            {
                "label": "other",
                "score": 0.52,
                "thresholds": {"route": 0.8, "margin": 0.15},
                "result": {
                    "action": "escalate",
                    "scores": {"product": 0.52, "order": 0.44},
                    "candidate": "product",
                    "margin": 0.08,
                    "handling": "Send to a person.",
                    "history_used": 0,
                    "languages": ["id"],
                },
            }
        )
        assert result.routed is False
        assert result.candidate == "product"
        assert result.margin == 0.08

    def test_a_routed_intent_matches_its_candidate(self):
        result = IntentResult.from_data(
            {
                "label": "order",
                "score": 0.93,
                "result": {"action": "route", "candidate": "order", "margin": 0.6},
            }
        )
        assert result.routed is True
        assert result.candidate == result.label


class TestDocuments:
    def test_the_card_values_come_back_three_ways(self):
        """Raw values, a masked copy to store instead, and what the NIK encodes."""
        verdict = DocumentVerdict.from_data(
            {
                "label": "ktp",
                "score": 0.88,
                "thresholds": {"accept": 0.6, "quality_accept": 0.65},
                "result": {
                    "action": "accept",
                    "document": {"matched_fields": ["nik", "nama"], "number_valid": True},
                    "extraction": {
                        "fields": {"nik": "3273014501900001", "nama": "SITI", "agama": None},
                        "masked": {"nik": "[NATIONAL_ID]", "nama": "[NAME]", "agama": None},
                        "derived": {"jenis_kelamin": "PEREMPUAN", "kode_provinsi": "32"},
                    },
                    "quality": {"score": 0.81, "advice": []},
                    "image": {"width": 1280, "height": 800, "media_type": "image/jpeg"},
                },
            }
        )
        assert verdict.accepted is True
        assert verdict.document["matched_fields"] == ["nik", "nama"]
        extraction = verdict.extraction
        assert extraction is not None
        assert extraction.fields["nik"] == "3273014501900001"
        assert extraction.fields["agama"] is None
        assert extraction.masked["nik"] == "[NATIONAL_ID]"
        assert extraction.derived == {"jenis_kelamin": "PEREMPUAN", "kode_provinsi": "32"}

    def test_no_extraction_unless_the_frame_is_the_card(self):
        verdict = DocumentVerdict.from_data(
            {
                "label": "other",
                "score": 0.1,
                "thresholds": {},
                "result": {"action": "reject", "extraction": None},
            }
        )
        assert verdict.extraction is None

    def test_derived_is_none_when_the_nik_fails_its_check(self):
        verdict = DocumentVerdict.from_data(
            {
                "label": "ktp",
                "score": 0.9,
                "thresholds": {},
                "result": {"extraction": {"fields": {}, "masked": {}, "derived": None}},
            }
        )
        assert verdict.extraction is not None
        assert verdict.extraction.derived is None

    def test_retry_carries_the_advice_to_show_the_camera_holder(self):
        verdict = DocumentVerdict.from_data(
            {
                "label": "unreadable",
                "score": 0.31,
                "result": {
                    "action": "retry",
                    "document": {},
                    "quality": {
                        "score": 0.22,
                        "issues": ["blur"],
                        "advice": ["Hold the camera still and let it focus before capturing."],
                    },
                    "image": {},
                },
            }
        )
        assert verdict.accepted is False
        assert verdict.issues == ["blur"]
        assert verdict.advice == ["Hold the camera still and let it focus before capturing."]


class TestNoTruthTesting:
    def test_a_verdict_is_not_truth_testable(self):
        """`if verdict:` would mean "an attack was found" on a guardrail and "the
        photograph is fine" on a document gate. The named property says which."""
        for model in (PromptInjectionVerdict, PIIResult, ScamVerdict, DocumentVerdict):
            assert "__bool__" not in vars(model)
