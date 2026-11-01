"""Correction feedback must be useful without echoing untrusted error data."""
from types import SimpleNamespace

from pydantic import ValidationError
from pydantic_core import PydanticCustomError
import pytest

from cache import CacheStore
from pipeline import RouterPipeline, RunStats
from schemas import Decision, validation_error_text

CANARY = "FAKE_PRIVATE_VALUE /Users/private/feedback.txt"


def payload(**changes):
    result = dict(action="notify", message_type="personal", reason="A direct message.",
                  confidence=0.85, evidence_message_ids=[])
    result.update(changes)
    return result


@pytest.mark.parametrize("changes,expected", [
    ({"action": CANARY}, "action: literal_error"),
    ({"confidence": 2}, "confidence: less_than_equal"),
    ({"reason": CANARY * 20}, "reason: string_too_long"),
    ({"evidence_message_ids": [CANARY] * 3}, "evidence_message_ids: too_long"),
    ({"evidence_message_ids": [None]}, "evidence_message_ids[0]: string_type"),
    ({CANARY: CANARY}, "unknown_field: extra_forbidden"),
])
def test_feedback_has_allowlisted_field_and_error(changes, expected):
    with pytest.raises(ValidationError) as caught:
        Decision.model_validate(payload(**changes))
    feedback = validation_error_text(caught.value)
    assert feedback == expected
    assert CANARY not in feedback


def test_custom_error_type_message_and_location_are_not_trusted():
    error = ValidationError.from_exception_data("Decision", [{
        "type": PydanticCustomError(CANARY, CANARY),
        "loc": (CANARY, "confidence"), "input": CANARY,
    }])
    assert validation_error_text(error) == "unknown_field: invalid_value"


def test_feedback_is_bounded_and_deduplicated():
    error = ValidationError.from_exception_data("Decision", [
        {"type": "extra_forbidden", "loc": (f"{CANARY}{n}",), "input": CANARY}
        for n in range(100)
    ])
    assert validation_error_text(error) == "unknown_field: extra_forbidden"


@pytest.mark.parametrize("changes,feedback", [
    ({"action": CANARY}, "action: literal_error"),
    ({"evidence_message_ids": [CANARY] * 3}, "evidence_message_ids: too_long"),
    ({"evidence_message_ids": [CANARY]}, "evidence_message_ids: not_in_candidate_set"),
])
def test_pipeline_recovers_using_safe_specific_feedback(tmp_path, dataset, changes, feedback):
    pipeline = object.__new__(RouterPipeline)
    pipeline.cache = CacheStore(tmp_path)
    pipeline.stats = RunStats()
    pipeline.data = dataset
    pipeline.examples = ""
    pipeline._decision_fingerprint = lambda row: "test-fingerprint"
    pipeline._media_result = lambda *args, **kwargs: None
    calls = []

    def route(system, prompt, schema):
        calls.append(prompt)
        if len(calls) == 1:
            return payload(**changes)
        assert feedback in prompt
        assert CANARY not in prompt
        return payload()

    pipeline.provider = SimpleNamespace(route=route)
    row = next(row for row in dataset.messages if not row.get("media_type"))
    result = pipeline.process_message(row)
    assert len(calls) == 2
    assert result["action"] == "notify"
    assert pipeline.stats.validation_retries == 1
    assert pipeline.stats.model_decisions == 1
    assert pipeline.stats.fallback_decisions == 0
    record = pipeline.cache.get_decision(row["message_id"], "test-fingerprint", False)
    assert record["source"] == "model"
