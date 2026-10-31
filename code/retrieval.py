from __future__ import annotations

import math
import re
from datetime import datetime

from data_io import Dataset, as_bool


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(text.casefold()) if len(token) >= 2}


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _conversation_key(row: dict[str, str]) -> tuple[str, str]:
    kind = row.get("conversation_type", "")
    if kind == "group":
        return "group_id", row.get("group_id", "")
    if kind == "business":
        return "business_id", row.get("business_id", "")
    return "sender_user_id", row.get("sender_user_id", "")


def _event_flags(event: dict[str, str]) -> dict[str, bool | int | None]:
    reaction = event.get("reaction_time_minutes", "")
    return {
        "opened": as_bool(event.get("message_opened")),
        "replied": as_bool(event.get("message_replied")),
        "reaction_time_minutes": int(reaction) if reaction else None,
        "dismissed": as_bool(event.get("notification_dismissed")),
        "muted_after_message": as_bool(event.get("muted_after_message")),
        "reported": as_bool(event.get("message_reported")),
    }


def retrieve_candidates(
    data: Dataset, incoming: dict[str, str], content_text: str, limit: int = 8
) -> list[dict]:
    incoming_time = datetime.fromisoformat(incoming["created_at"])
    key_name, key_value = _conversation_key(incoming)
    ranked: list[tuple[float, datetime, str, dict]] = []

    for history in data.history_by_user.get(incoming["user_id"], []):
        history_time = datetime.fromisoformat(history["created_at"])
        if history_time >= incoming_time:
            continue

        same_type = history["conversation_type"] == incoming["conversation_type"]
        same_entity = bool(key_value) and history.get(key_name, "") == key_value
        similarity = _jaccard(content_text, history.get("message_text", ""))
        raw_event = data.events.get((incoming["user_id"], history["message_id"]), {})
        event = _event_flags(raw_event)
        has_event = any(
            bool(event[name])
            for name in (
                "opened",
                "replied",
                "dismissed",
                "muted_after_message",
                "reported",
            )
        )
        if not (same_entity or similarity >= 0.15 or (same_type and has_event)):
            continue

        age_days = max(0, (incoming_time - history_time).days)
        recency = max(0, 10 - math.floor(age_days / 30))
        score = (
            (100 if same_entity else 0)
            + (20 if same_type else 0)
            + (30 * similarity)
            + (12 if event["reported"] else 0)
            + (10 if event["muted_after_message"] else 0)
            + (6 if event["replied"] else 0)
            + (3 if event["dismissed"] else 0)
            + (2 if event["opened"] else 0)
            + recency
        )
        candidate = {
            "message_id": history["message_id"],
            "conversation_type": history["conversation_type"],
            "group_id": history["group_id"] or None,
            "business_id": history["business_id"] or None,
            "sender_user_id": history["sender_user_id"] or None,
            "created_at": history["created_at"],
            "message_text": history["message_text"],
            "media_type": history["media_type"] or None,
            "media_id": history["media_id"] or None,
            "forwarded_count": int(history["forwarded_count"] or 0),
            "event": event,
            "relation": {
                "same_conversation_type": same_type,
                "same_conversation_entity": same_entity,
                "text_jaccard": round(similarity, 4),
                "age_days": age_days,
                "retrieval_score": round(score, 4),
            },
        }
        ranked.append((score, history_time, history["message_id"], candidate))

    ranked.sort(key=lambda item: (-item[0], -item[1].timestamp(), item[2]))
    return [item[3] for item in ranked[:limit]]
