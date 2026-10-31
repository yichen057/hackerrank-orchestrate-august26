from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from cache import CacheStore  # noqa: E402
from config import load_config  # noqa: E402
from context import deterministic_signals  # noqa: E402
from data_io import Dataset  # noqa: E402
from pipeline import RouterPipeline  # noqa: E402
from prompts import PROMPT_VERSION, compressed_examples  # noqa: E402
from provider import GeminiProvider  # noqa: E402


ACTION_VALUES = ("notify", "digest", "mute")
MESSAGE_TYPE_VALUES = (
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
)
LABEL_FIELDS = {
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
}
RESULTS_PATH = Path(__file__).resolve().parent / "leave_one_out_v4_submitted.json"
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "leave_one_out"


def _evidence_set(value: str | list[str]) -> set[str]:
    if isinstance(value, list):
        return set(value)
    return set() if value == "none" else set(value.split(";"))


def _matrix(expected_values: tuple[str, ...], pairs: list[tuple[str, str]]) -> dict:
    return {
        expected: {
            predicted: sum(
                1
                for expected_value, predicted_value in pairs
                if expected_value == expected and predicted_value == predicted
            )
            for predicted in expected_values
        }
        for expected in expected_values
    }


def _confidence_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
    }


def _failure_category(result: dict) -> str:
    expected_action = result["expected"]["action"]
    predicted_action = result["predicted"]["action"]
    expected_type = result["expected"]["message_type"]
    predicted_type = result["predicted"]["message_type"]
    safety_types = {"scam", "spam"}

    if (
        predicted_action == "mute"
        and predicted_type in safety_types
        and not (expected_action == "mute" and expected_type in safety_types)
    ):
        return "safety_over_trigger"
    if (
        expected_action == "mute"
        and expected_type in safety_types
        and not (predicted_action == "mute" and predicted_type in safety_types)
    ):
        return "safety_under_trigger"
    if predicted_action == "notify" and expected_action != "notify":
        return "priority_over_trigger"
    if expected_action == "notify" and predicted_action != "notify":
        return "priority_under_trigger"
    if predicted_action == expected_action and predicted_type != expected_type:
        return "category_boundary"
    if predicted_action != expected_action:
        return "other_action_mismatch"
    return "other_type_mismatch"


def _write_results(payload: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULTS_PATH.with_name(f".{RESULTS_PATH.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(RESULTS_PATH)


def main() -> int:
    started = time.perf_counter()
    base_config = load_config()
    evaluation_config = replace(base_config, cache_dir=CACHE_DIR)
    data = Dataset.load(base_config.dataset_dir)
    if len(data.samples) != 30:
        raise RuntimeError(f"Expected 30 solved samples, found {len(data.samples)}")

    input_fields = tuple(data.messages[0].keys())
    if LABEL_FIELDS & set(input_fields):
        raise RuntimeError("Input schema unexpectedly contains scoring-label fields")

    provider = GeminiProvider(evaluation_config)
    pipeline = RouterPipeline(
        evaluation_config,
        data,
        provider,
        CacheStore(CACHE_DIR),
    )
    results: list[dict] = []

    for index, held_out in enumerate(data.samples, start=1):
        held_out_id = held_out["message_id"]
        other_samples = [
            sample for sample in data.samples if sample["message_id"] != held_out_id
        ]
        if len(other_samples) != 29:
            raise RuntimeError(
                f"Leave-one-out exclusion failed for {held_out_id}: "
                f"found {len(other_samples)} prompt examples"
            )
        prompt_data = replace(data, samples=other_samples)
        pipeline.examples = compressed_examples(prompt_data)

        incoming = {field: held_out[field] for field in input_fields}
        if LABEL_FIELDS & set(incoming):
            raise RuntimeError(f"Label leakage into incoming row {held_out_id}")

        predicted = pipeline.process_message(incoming)
        signals = deterministic_signals(data, incoming)
        expected_evidence = _evidence_set(held_out["evidence_message_ids"])
        predicted_evidence = _evidence_set(predicted["evidence_message_ids"])
        result = {
            "message_id": held_out_id,
            "expected": {
                "action": held_out["action"],
                "message_type": held_out["message_type"],
                "reason": held_out["reason"],
                "confidence": float(held_out["confidence"]),
                "evidence_message_ids": sorted(expected_evidence),
            },
            "predicted": {
                "action": predicted["action"],
                "message_type": predicted["message_type"],
                "reason": predicted["reason"],
                "confidence": float(predicted["confidence"]),
                "evidence_message_ids": sorted(predicted_evidence),
            },
            "action_correct": predicted["action"] == held_out["action"],
            "message_type_correct": (
                predicted["message_type"] == held_out["message_type"]
            ),
            "evidence_exact": predicted_evidence == expected_evidence,
            "evidence_partial_overlap": bool(
                predicted_evidence & expected_evidence
            )
            and predicted_evidence != expected_evidence,
            "true_signals": sorted(
                name for name, value in signals.items() if value is True
            ),
        }
        if not (result["action_correct"] and result["message_type_correct"]):
            result["failure_category"] = _failure_category(result)
        results.append(result)
        print(
            f"LOO_PROGRESS completed={index}/30 message_id={held_out_id} "
            f"action_correct={result['action_correct']} "
            f"type_correct={result['message_type_correct']}",
            flush=True,
        )

    action_pairs = [
        (item["expected"]["action"], item["predicted"]["action"])
        for item in results
    ]
    type_pairs = [
        (item["expected"]["message_type"], item["predicted"]["message_type"])
        for item in results
    ]
    fully_correct = [
        item for item in results if item["action_correct"] and item["message_type_correct"]
    ]
    mismatches = [
        item for item in results if not (item["action_correct"] and item["message_type_correct"])
    ]
    evidence_exact_count = sum(item["evidence_exact"] for item in results)
    evidence_partial_count = sum(item["evidence_partial_overlap"] for item in results)
    failure_categories = Counter(
        item["failure_category"] for item in mismatches
    )
    expected_promotion_or_update_predicted_safety = Counter(
        f"{item['expected']['message_type']}->{item['predicted']['message_type']}"
        for item in results
        if item["expected"]["message_type"] in {"promotion", "business_update"}
        and item["predicted"]["message_type"] in {"scam", "spam"}
    )

    cache_sources = Counter()
    for path in CACHE_DIR.joinpath("decisions").glob("*.json"):
        cache_sources[json.loads(path.read_text(encoding="utf-8"))["source"]] += 1

    elapsed = time.perf_counter() - started
    summary = {
        "prompt_version": PROMPT_VERSION,
        "evaluation_mode": "leave-one-out",
        "rows": len(results),
        "few_shot_examples_per_row": 29,
        "action_accuracy": {
            "correct": sum(item["action_correct"] for item in results),
            "total": len(results),
            "rate": round(sum(item["action_correct"] for item in results) / len(results), 4),
        },
        "message_type_accuracy": {
            "correct": sum(item["message_type_correct"] for item in results),
            "total": len(results),
            "rate": round(sum(item["message_type_correct"] for item in results) / len(results), 4),
        },
        "action_confusion_matrix": _matrix(ACTION_VALUES, action_pairs),
        "message_type_confusion_matrix": _matrix(MESSAGE_TYPE_VALUES, type_pairs),
        "promotion_or_business_update_predicted_as_safety": dict(
            sorted(expected_promotion_or_update_predicted_safety.items())
        ),
        "evidence": {
            "exact": evidence_exact_count,
            "partial_overlap": evidence_partial_count,
            "no_overlap": len(results) - evidence_exact_count - evidence_partial_count,
        },
        "confidence": {
            "expected_all": _confidence_summary(
                [item["expected"]["confidence"] for item in results]
            ),
            "predicted_all": _confidence_summary(
                [item["predicted"]["confidence"] for item in results]
            ),
            "predicted_fully_correct": _confidence_summary(
                [item["predicted"]["confidence"] for item in fully_correct]
            ),
            "predicted_mismatched": _confidence_summary(
                [item["predicted"]["confidence"] for item in mismatches]
            ),
        },
        "failure_categories": dict(sorted(failure_categories.items())),
        "cache_sources": dict(sorted(cache_sources.items())),
        "pipeline_stats": pipeline.stats.to_dict(),
        "provider_stats": provider.stats.to_dict(),
        "elapsed_seconds": round(elapsed, 3),
    }
    payload = {
        "summary": summary,
        "mismatches": mismatches,
        "all_results": results,
    }
    _write_results(payload)

    print(f"MODEL_NAME={evaluation_config.model}")
    print("EVALUATION_MODE=leave-one-out")
    print("HELD_OUT_LABELS_USED_ONLY_FOR_SCORING=true")
    print(f"CACHE_DIR={CACHE_DIR}")
    print(f"RESULTS_PATH={RESULTS_PATH}")
    print("SUMMARY=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
