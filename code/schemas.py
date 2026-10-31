from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


ACTIONS = ["notify", "digest", "mute"]
MESSAGE_TYPES = [
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
]
SCHEMA_VERSION = "decision-v2-max-two-evidence"


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["notify", "digest", "mute"]
    message_type: Literal[
        "personal",
        "urgent",
        "event",
        "payment",
        "business_update",
        "promotion",
        "greeting",
        "forward",
        "spam",
        "scam",
        "unknown",
    ]
    reason: str = Field(min_length=1, max_length=200)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_message_ids: list[str] = Field(max_length=2)

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence must be finite")
        return value

    @field_validator("evidence_message_ids")
    @classmethod
    def evidence_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("evidence_message_ids must be unique")
        return value


def decision_schema(candidate_ids: list[str]) -> dict:
    items: dict = {"type": "string"}
    if candidate_ids:
        items["enum"] = candidate_ids
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action",
            "message_type",
            "reason",
            "confidence",
            "evidence_message_ids",
        ],
        "properties": {
            "action": {"type": "string", "enum": ACTIONS},
            "message_type": {"type": "string", "enum": MESSAGE_TYPES},
            "reason": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_message_ids": {
                "type": "array",
                "maxItems": 2 if candidate_ids else 0,
                "items": items,
            },
        },
    }


def validate_decision(payload: object, candidate_ids: list[str]) -> Decision:
    decision = Decision.model_validate(payload)
    invalid = [
        message_id
        for message_id in decision.evidence_message_ids
        if message_id not in candidate_ids
    ]
    if invalid:
        raise ValueError(
            f"evidence_message_ids not in runtime candidate set: {invalid}"
        )
    return decision


def validation_error_text(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return error.json(include_url=False)
    return str(error)
