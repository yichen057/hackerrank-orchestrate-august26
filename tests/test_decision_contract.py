"""The model's output contract.

Every assertion here states a property the pipeline must hold no matter what the
model returns. These are the guardrails, so they are tested directly.
"""
from __future__ import annotations

import pytest

from schemas import Decision, MESSAGE_TYPES, decision_schema, validate_decision


def _payload(**overrides) -> dict:
    base = {
        "action": "notify",
        "message_type": "personal",
        "reason": "A direct message from a known contact.",
        "confidence": 0.87,
        "evidence_message_ids": ["message_0001"],
    }
    base.update(overrides)
    return base


CANDIDATES = ["message_0001", "message_0002", "message_0003"]


def test_accepts_a_well_formed_decision():
    decision = validate_decision(_payload(), CANDIDATES)
    assert decision.action == "notify"
    assert decision.evidence_message_ids == ["message_0001"]


def test_rejects_evidence_id_outside_the_candidate_set():
    """The core anti-fabrication property: a cited ID the retrieval layer never
    offered must not survive validation, even though it is a well-formed string."""
    with pytest.raises(ValueError, match="not in runtime candidate set"):
        validate_decision(_payload(evidence_message_ids=["message_9999"]), CANDIDATES)


def test_rejects_unknown_fields():
    with pytest.raises(Exception):
        Decision.model_validate(_payload(priority="high"))


def test_rejects_more_than_two_evidence_ids():
    with pytest.raises(Exception):
        Decision.model_validate(_payload(evidence_message_ids=CANDIDATES))


def test_rejects_duplicate_evidence_ids():
    with pytest.raises(Exception):
        Decision.model_validate(
            _payload(evidence_message_ids=["message_0001", "message_0001"])
        )


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("inf"), float("nan")])
def test_rejects_confidence_outside_zero_to_one_or_non_finite(bad):
    with pytest.raises(Exception):
        Decision.model_validate(_payload(confidence=bad))


@pytest.mark.parametrize("field,bad", [("action", "snooze"), ("message_type", "invoice")])
def test_rejects_values_outside_the_enum(field, bad):
    with pytest.raises(Exception):
        Decision.model_validate(_payload(**{field: bad}))


def test_schema_restricts_evidence_to_the_candidate_ids():
    """The candidate set is compiled into the request schema as an enum, so an
    out-of-set ID is refused by the provider before it ever reaches this code."""
    schema = decision_schema(CANDIDATES)
    evidence = schema["properties"]["evidence_message_ids"]
    assert evidence["items"]["enum"] == CANDIDATES
    assert evidence["maxItems"] == 2
    assert schema["additionalProperties"] is False


def test_empty_candidate_set_forbids_citing_anything():
    """With no history there is nothing legal to cite, so the schema must allow
    zero items rather than inviting the model to invent a placeholder."""
    schema = decision_schema([])
    assert schema["properties"]["evidence_message_ids"]["maxItems"] == 0
    assert "enum" not in schema["properties"]["evidence_message_ids"]["items"]


def test_message_id_is_not_a_field_the_model_can_set():
    """`message_id` is copied in by code. A hallucinated id cannot create a row."""
    assert "message_id" not in Decision.model_fields
    assert "message_id" not in decision_schema(CANDIDATES)["properties"]


def test_every_message_type_in_the_enum_is_accepted():
    for message_type in MESSAGE_TYPES:
        assert Decision.model_validate(_payload(message_type=message_type))
