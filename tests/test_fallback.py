"""The deterministic fallback.

When the provider cannot be reached the system must still emit a valid, *safe*
decision without calling anything. These constants are the fail-safe, so they
are pinned.
"""
from __future__ import annotations

from fallback import conservative_fallback
from schemas import Decision


def test_fallback_is_a_fixed_conservative_decision():
    decision = conservative_fallback()
    assert decision.action == "digest"
    assert decision.message_type == "unknown"
    assert decision.confidence == 0.20
    assert decision.evidence_message_ids == []


def test_fallback_never_claims_high_confidence():
    assert conservative_fallback().confidence < 0.5


def test_fallback_satisfies_the_same_contract_as_a_model_decision():
    decision = conservative_fallback()
    assert Decision.model_validate(decision.model_dump()) == decision


def test_fallback_is_deterministic():
    assert conservative_fallback() == conservative_fallback()
