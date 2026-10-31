from __future__ import annotations

from datetime import date, datetime, time

from data_io import Dataset, as_bool, as_int
from retrieval import retrieve_candidates


def _nullable(row: dict[str, str] | None) -> dict[str, str] | None:
    return row if row else None


def _in_do_not_disturb(created_at: str, window: str) -> bool | None:
    if not created_at or not window or "-" not in window:
        return None
    start_text, end_text = window.split("-", 1)
    current = datetime.fromisoformat(created_at).time()
    start = time.fromisoformat(start_text)
    end = time.fromisoformat(end_text)
    if start <= end:
        return start <= current < end
    return current >= start or current < end


def _domain(value: str) -> str:
    return value.strip().casefold().rstrip(".")


def deterministic_signals(data: Dataset, row: dict[str, str]) -> dict:
    user = data.users.get(row["user_id"])
    group_id = row.get("group_id", "")
    business_id = row.get("business_id", "")
    recipient_member = data.group_members.get((group_id, row["user_id"])) if group_id else None
    sender_member = data.group_members.get((group_id, row.get("sender_user_id", ""))) if group_id else None
    business = data.businesses.get(business_id) if business_id else None
    business_history = data.user_business.get((row["user_id"], business_id)) if business_id else None

    domain_mismatch = None
    sender_domain_age = None
    business_age = None
    verified = None
    reports = None
    if business:
        domain_mismatch = _domain(business["domain_used_by_sender"]) != _domain(
            business["official_domain"]
        )
        sender_domain_age = as_int(business["domain_used_by_sender_age_days"])
        business_age = as_int(business["account_age_days"])
        verified = as_bool(business["verified"])
        reports = as_int(business["user_reports_30d"])

    promotions_opted_out = None
    if business_history:
        promotions_opted_out = bool(
            business_history["promotions_opted_out_at"].strip()
            or business_history["allows_promotions"] == "0"
        )

    return {
        "domain_mismatch": domain_mismatch,
        "domain_used_by_sender_age_days": sender_domain_age,
        "business_account_age_days": business_age,
        "domain_much_newer_than_business": None,
        "business_verified": verified,
        "business_user_reports_30d": reports,
        "promotions_opted_out": promotions_opted_out,
        "group_muted_by_user": (
            as_bool(recipient_member["group_muted_by_user"])
            if recipient_member
            else None
        ),
        "recipient_group_role": recipient_member["role"] if recipient_member else None,
        "sender_group_role": sender_member["role"] if sender_member else None,
        "in_do_not_disturb": (
            _in_do_not_disturb(row["created_at"], user["do_not_disturb_window"])
            if user
            else None
        ),
    }


def _notification_load(data: Dataset, row: dict[str, str]) -> dict:
    incoming_date = datetime.fromisoformat(row["created_at"]).date()
    prior = [
        item
        for item in data.daily_by_user.get(row["user_id"], [])
        if date.fromisoformat(item["date"]) <= incoming_date
    ]
    sent = sum(as_int(item["notifications_sent"]) for item in prior)
    dismissed = sum(as_int(item["notifications_dismissed"]) for item in prior)
    return {
        "rows_newest_first": list(reversed(prior)),
        "notifications_sent_total": sent,
        "notifications_dismissed_total": dismissed,
        "dismissal_rate": round(dismissed / sent, 4) if sent else 0.0,
    }


def build_context(
    data: Dataset, row: dict[str, str], media_result: dict | None
) -> dict:
    extracted = media_result.get("content", "") if media_result else ""
    content_text = "\n".join(part for part in (row.get("message_text", ""), extracted) if part)
    evidence = retrieve_candidates(data, row, content_text)
    group_id = row.get("group_id", "")
    business_id = row.get("business_id", "")
    return {
        "incoming": dict(row),
        "user": _nullable(data.users.get(row["user_id"])),
        "group": _nullable(data.groups.get(group_id)) if group_id else None,
        "recipient_membership": _nullable(
            data.group_members.get((group_id, row["user_id"]))
        ) if group_id else None,
        "sender_membership": _nullable(
            data.group_members.get((group_id, row.get("sender_user_id", "")))
        ) if group_id else None,
        "business": _nullable(data.businesses.get(business_id)) if business_id else None,
        "business_history": _nullable(
            data.user_business.get((row["user_id"], business_id))
        ) if business_id else None,
        "notification_load": _notification_load(data, row),
        "media": media_result,
        "evidence_candidates": evidence,
        "signals": deterministic_signals(data, row),
    }


def decisive_signal_summary(data: Dataset, sample: dict[str, str]) -> str:
    signals = deterministic_signals(data, sample)
    parts = [f"dnd={signals['in_do_not_disturb']}"]
    if sample["conversation_type"] == "group":
        parts.extend(
            [
                f"sender_role={signals['sender_group_role']}",
                f"group_muted={signals['group_muted_by_user']}",
            ]
        )
    elif sample["conversation_type"] == "business":
        parts.extend(
            [
                f"verified={signals['business_verified']}",
                f"domain_mismatch={signals['domain_mismatch']}",
                f"opted_out={signals['promotions_opted_out']}",
                f"reports={signals['business_user_reports_30d']}",
            ]
        )
    else:
        parts.append(f"forwarded_count={sample.get('forwarded_count', '0') or '0'}")

    evidence_ids = [] if sample["evidence_message_ids"] == "none" else sample["evidence_message_ids"].split(";")
    event_totals = {name: 0 for name in ("opened", "replied", "dismissed", "muted", "reported")}
    field_map = {
        "opened": "message_opened",
        "replied": "message_replied",
        "dismissed": "notification_dismissed",
        "muted": "muted_after_message",
        "reported": "message_reported",
    }
    for message_id in evidence_ids:
        event = data.events.get((sample["user_id"], message_id), {})
        for name, field in field_map.items():
            event_totals[name] += int(event.get(field, "0") or 0)
    if evidence_ids:
        parts.append(
            "evidence_events="
            + "/".join(f"{name}:{count}" for name, count in event_totals.items())
        )
    return "; ".join(parts)
