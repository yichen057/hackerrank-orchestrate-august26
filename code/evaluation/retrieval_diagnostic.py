from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_io import Dataset  # noqa: E402
from retrieval import (  # noqa: E402
    _conversation_key,
    _event_flags,
    _jaccard,
    retrieve_candidates,
)


EVALUATION_DIR = Path(__file__).resolve().parent
EVALUATION_CACHE = EVALUATION_DIR / ".cache" / "leave_one_out"
LOO_RESULTS = EVALUATION_DIR / "leave_one_out_v3_original.json"
RESULTS_PATH = EVALUATION_DIR / "retrieval_diagnostic_v3_original.json"


def _evidence_set(value: str | list[str]) -> set[str]:
    if isinstance(value, list):
        return set(value)
    return set() if value == "none" else set(value.split(";"))


def _media_content(media_id: str) -> str:
    if not media_id:
        return ""
    path = EVALUATION_CACHE / "media" / f"{media_id}.json"
    if not path.exists():
        raise RuntimeError(f"Missing local evaluation media cache: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    result = record.get("result", {})
    if result.get("status") != "ok":
        raise RuntimeError(f"Evaluation media is not readable: {media_id}")
    return result.get("content", "")


def _relation_text(detail: dict) -> str:
    event_names = [
        name
        for name in ("opened", "replied", "dismissed", "muted_after_message", "reported")
        if detail["event"].get(name)
    ]
    return (
        f"same_entity={detail['same_entity']}, "
        f"same_type={detail['same_type']}, "
        f"jaccard={detail['text_jaccard']:.4f}, "
        f"age_days={detail['age_days']}, "
        f"events={','.join(event_names) if event_names else 'none'}"
    )


def _all_history_details(
    data: Dataset,
    incoming: dict[str, str],
    content_text: str,
) -> tuple[list[dict], dict[str, dict]]:
    incoming_time = datetime.fromisoformat(incoming["created_at"])
    key_name, key_value = _conversation_key(incoming)
    details: list[dict] = []

    for history in data.history_by_user.get(incoming["user_id"], []):
        history_time = datetime.fromisoformat(history["created_at"])
        same_type = history["conversation_type"] == incoming["conversation_type"]
        same_entity = bool(key_value) and history.get(key_name, "") == key_value
        similarity = _jaccard(content_text, history.get("message_text", ""))
        event = _event_flags(
            data.events.get((incoming["user_id"], history["message_id"]), {})
        )
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
        earlier = history_time < incoming_time
        filter_passed = earlier and (
            same_entity or similarity >= 0.15 or (same_type and has_event)
        )
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
        reasons = []
        if not earlier:
            reasons.append("history timestamp is not earlier than incoming timestamp")
        if earlier and not filter_passed:
            reasons.extend(
                [
                    "same conversation entity is false",
                    f"text Jaccard {similarity:.4f} is below 0.1500",
                    (
                        "same conversation type plus an event is false "
                        f"(same_type={same_type}, has_event={has_event})"
                    ),
                ]
            )
        details.append(
            {
                "message_id": history["message_id"],
                "created_at": history["created_at"],
                "history_text": history["message_text"],
                "earlier": earlier,
                "filter_passed": filter_passed,
                "filter_failure_reasons": reasons,
                "score": round(score, 4),
                "score_raw": score,
                "same_entity": same_entity,
                "same_type": same_type,
                "text_jaccard": round(similarity, 4),
                "age_days": age_days,
                "event": event,
                "history_time": history_time,
            }
        )

    eligible = [item for item in details if item["filter_passed"]]
    eligible.sort(
        key=lambda item: (
            -item["score_raw"],
            -item["history_time"].timestamp(),
            item["message_id"],
        )
    )
    for rank, item in enumerate(eligible, start=1):
        item["eligible_rank"] = rank
    by_id = {item["message_id"]: item for item in details}
    return eligible, by_id


def _clean_detail(detail: dict | None) -> dict | None:
    if detail is None:
        return None
    return {
        key: value
        for key, value in detail.items()
        if key not in {"history_time", "score_raw"}
    }


def main() -> int:
    data = Dataset.load(REPO_ROOT / "dataset")
    loo = json.loads(LOO_RESULTS.read_text(encoding="utf-8"))
    predictions = {item["message_id"]: item for item in loo["all_results"]}
    input_fields = tuple(data.messages[0].keys())
    audits = []

    for sample in data.samples:
        incoming = {field: sample[field] for field in input_fields}
        media_text = _media_content(incoming["media_id"]) if incoming["media_id"] else ""
        content_text = "\n".join(
            part for part in (incoming["message_text"], media_text) if part
        )
        eligible, by_id = _all_history_details(data, incoming, content_text)
        production_top = retrieve_candidates(data, incoming, content_text, limit=8)
        diagnostic_top_ids = [item["message_id"] for item in eligible[:8]]
        production_top_ids = [item["message_id"] for item in production_top]
        if diagnostic_top_ids != production_top_ids:
            raise RuntimeError(
                f"Diagnostic ranking diverged for {sample['message_id']}: "
                f"{diagnostic_top_ids} != {production_top_ids}"
            )

        expected_ids = _evidence_set(sample["evidence_message_ids"])
        chosen_ids = set(
            predictions[sample["message_id"]]["predicted"]["evidence_message_ids"]
        )
        expected_details = []
        for expected_id in sorted(expected_ids):
            detail = by_id.get(expected_id)
            if detail is None:
                expected_details.append(
                    {
                        "message_id": expected_id,
                        "status": "not_in_same_user_history",
                        "detail": None,
                        "outranked_by": [],
                    }
                )
                continue
            rank = detail.get("eligible_rank")
            if not detail["filter_passed"]:
                status = "filtered_out"
            elif rank and rank <= 8:
                status = "candidate"
            else:
                status = "ranked_below_eight"
            expected_details.append(
                {
                    "message_id": expected_id,
                    "status": status,
                    "detail": _clean_detail(detail),
                    "outranked_by": [
                        {
                            "message_id": candidate["message_id"],
                            "score": candidate["score"],
                            "relation": _relation_text(candidate),
                        }
                        for candidate in eligible
                        if rank is not None and candidate["eligible_rank"] < rank
                    ],
                }
            )

        chosen_details = []
        for chosen_id in sorted(chosen_ids):
            detail = by_id.get(chosen_id)
            chosen_details.append(
                {
                    "message_id": chosen_id,
                    "detail": _clean_detail(detail),
                    "relation": _relation_text(detail) if detail else None,
                }
            )

        exact = expected_ids == chosen_ids
        expected_in_top = {
            item["message_id"]
            for item in expected_details
            if item["status"] == "candidate"
        }
        if exact:
            cause = "exact"
        elif not expected_ids:
            cause = "no_expected_evidence"
        elif expected_in_top != expected_ids:
            cause = "A_retrieval"
        else:
            cause = "B_model_selection"

        audits.append(
            {
                "message_id": sample["message_id"],
                "expected_ids": sorted(expected_ids),
                "chosen_ids": sorted(chosen_ids),
                "exact_set_match": exact,
                "top_eight_ids": diagnostic_top_ids,
                "top_ranked_id": diagnostic_top_ids[0] if diagnostic_top_ids else None,
                "top_ranked_is_expected": bool(expected_ids)
                and bool(diagnostic_top_ids)
                and diagnostic_top_ids[0] in expected_ids,
                "all_expected_in_top_eight": expected_in_top == expected_ids,
                "any_expected_in_top_eight": bool(expected_in_top),
                "expected_details": expected_details,
                "chosen_details": chosen_details,
                "diagnostic_cause": cause,
            }
        )

    with_expected = [item for item in audits if item["expected_ids"]]
    nonexact_with_expected = [
        item for item in with_expected if not item["exact_set_match"]
    ]
    expected_detail_rows = [
        detail
        for item in with_expected
        for detail in item["expected_details"]
    ]
    rank_buckets = Counter()
    for detail in expected_detail_rows:
        rank = (detail.get("detail") or {}).get("eligible_rank")
        if detail["status"] != "candidate":
            continue
        if rank == 1:
            rank_buckets["first"] += 1
        elif rank <= 4:
            rank_buckets["second_to_fourth"] += 1
        else:
            rank_buckets["fifth_to_eighth"] += 1

    summary = {
        "samples_total": len(audits),
        "samples_with_expected_evidence": len(with_expected),
        "samples_expected_none": len(audits) - len(with_expected),
        "sample_presence": {
            "all_expected_ids_present": sum(
                item["all_expected_in_top_eight"] for item in with_expected
            ),
            "some_but_not_all_expected_ids_present": sum(
                item["any_expected_in_top_eight"]
                and not item["all_expected_in_top_eight"]
                for item in with_expected
            ),
            "no_expected_ids_present": sum(
                not item["any_expected_in_top_eight"] for item in with_expected
            ),
        },
        "expected_id_presence": {
            "expected_ids_total": len(expected_detail_rows),
            "present_in_top_eight": sum(
                item["status"] == "candidate" for item in expected_detail_rows
            ),
            "filtered_out": sum(
                item["status"] == "filtered_out" for item in expected_detail_rows
            ),
            "ranked_below_eight": sum(
                item["status"] == "ranked_below_eight"
                for item in expected_detail_rows
            ),
            "not_in_same_user_history": sum(
                item["status"] == "not_in_same_user_history"
                for item in expected_detail_rows
            ),
        },
        "present_expected_id_rank_buckets": dict(sorted(rank_buckets.items())),
        "top_ranked_is_expected": {
            "count": sum(item["top_ranked_is_expected"] for item in with_expected),
            "denominator_with_expected_evidence": len(with_expected),
        },
        "nonexact_evidence_cause": dict(
            sorted(Counter(item["diagnostic_cause"] for item in nonexact_with_expected).items())
        ),
        "retrieval_implementation_match": True,
        "api_calls": 0,
    }
    payload = {"summary": summary, "samples": audits}
    temporary = RESULTS_PATH.with_name(f".{RESULTS_PATH.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(RESULTS_PATH)
    print("RETRIEVAL_DIAGNOSTIC=" + json.dumps(summary, sort_keys=True))
    print(f"RESULTS_PATH={RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
