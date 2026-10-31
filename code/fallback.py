from __future__ import annotations

from schemas import Decision


def conservative_fallback() -> Decision:
    return Decision(
        action="digest",
        message_type="unknown",
        reason="Model unavailable; saved for later review.",
        confidence=0.20,
        evidence_message_ids=[],
    )
