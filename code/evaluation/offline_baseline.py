from __future__ import annotations

import json
import sys
import time
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_io import Dataset  # noqa: E402
from fallback import conservative_fallback  # noqa: E402


RESULTS_PATH = Path(__file__).resolve().parent / "offline_fallback_baseline.json"


def _evidence_set(value: str) -> set[str]:
    return set() if value == "none" else set(value.split(";"))


def main() -> int:
    started = time.perf_counter()
    data = Dataset.load(REPO_ROOT / "dataset")
    fallback = conservative_fallback()
    predicted_evidence = set(fallback.evidence_message_ids)
    rows = []
    for sample in data.samples:
        expected_evidence = _evidence_set(sample["evidence_message_ids"])
        rows.append(
            {
                "message_id": sample["message_id"],
                "expected": {
                    "action": sample["action"],
                    "message_type": sample["message_type"],
                    "evidence_message_ids": sorted(expected_evidence),
                },
                "predicted": {
                    "action": fallback.action,
                    "message_type": fallback.message_type,
                    "reason": fallback.reason,
                    "confidence": fallback.confidence,
                    "evidence_message_ids": sorted(predicted_evidence),
                },
                "action_correct": fallback.action == sample["action"],
                "message_type_correct": fallback.message_type
                == sample["message_type"],
                "both_correct": fallback.action == sample["action"]
                and fallback.message_type == sample["message_type"],
                "evidence_exact": predicted_evidence == expected_evidence,
            }
        )

    summary = {
        "configuration": "forced-offline-conservative-fallback",
        "rows": len(rows),
        "action_accuracy": sum(row["action_correct"] for row in rows),
        "message_type_accuracy": sum(
            row["message_type_correct"] for row in rows
        ),
        "both_correct": sum(row["both_correct"] for row in rows),
        "evidence_exact": sum(row["evidence_exact"] for row in rows),
        "api_calls": 0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    payload = {"summary": summary, "all_results": rows}
    temporary = RESULTS_PATH.with_name(f".{RESULTS_PATH.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(RESULTS_PATH)
    print("OFFLINE_BASELINE=" + json.dumps(summary, sort_keys=True))
    print(f"RESULTS_PATH={RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
