from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from cache import CacheStore  # noqa: E402
from config import load_config  # noqa: E402
from context import build_context  # noqa: E402
from data_io import Dataset  # noqa: E402
from pipeline import RouterPipeline  # noqa: E402
from prompts import SYSTEM_PROMPT, compressed_examples, routing_prompt  # noqa: E402
from provider import GeminiProvider  # noqa: E402
from schemas import decision_schema, validate_decision  # noqa: E402


CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "gemini_3_5_smoke"
RESULTS_PATH = Path(__file__).resolve().parent / "gemini_3_5_smoke.json"
TEXT_MESSAGE_ID = "msg_023"
IMAGE_ID = "img_002"
VOICE_ID = "vn_007"


def _primary_decision(message_id: str) -> dict:
    path = REPO_ROOT / "code" / ".cache" / "decisions" / f"{message_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))["decision"]


def _primary_media(media_id: str) -> dict:
    path = REPO_ROOT / "code" / ".cache" / "media" / f"{media_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))["result"]


def _media_comparison(primary: dict, fallback: dict) -> dict:
    primary_content = primary.get("content", "")
    fallback_content = fallback.get("content", "")
    return {
        "primary_status": primary.get("status"),
        "fallback_status": fallback.get("status"),
        "primary_content": primary_content,
        "fallback_content": fallback_content,
        "primary_characters": len(primary_content),
        "fallback_characters": len(fallback_content),
        "exact_content_match": primary_content == fallback_content,
        "fallback_error": fallback.get("error_category"),
    }


def _write(payload: dict) -> None:
    temporary = RESULTS_PATH.with_name(f".{RESULTS_PATH.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(RESULTS_PATH)


def main() -> int:
    started = time.perf_counter()
    base_config = load_config()
    if base_config.model != "gemini-3.5-flash":
        raise RuntimeError(
            f"Expected GEMINI_MODEL=gemini-3.5-flash, got {base_config.model!r}"
        )
    config = replace(base_config, cache_dir=CACHE_DIR)
    data = Dataset.load(config.dataset_dir)
    provider = GeminiProvider(config)
    provider.RETRY_WAITS_SECONDS = ()
    cache = CacheStore(CACHE_DIR)
    pipeline = RouterPipeline(config, data, provider, cache)
    results: dict = {
        "configuration": {
            "model": config.model,
            "cache_dir": str(CACHE_DIR),
            "maximum_http_attempts_per_call": 1,
        }
    }

    row = next(item for item in data.messages if item["message_id"] == TEXT_MESSAGE_ID)
    if row.get("media_type"):
        raise RuntimeError(f"{TEXT_MESSAGE_ID} is not text-only")
    context = build_context(data, row, None)
    candidates = [
        item["message_id"] for item in context["evidence_candidates"]
    ]
    schema = decision_schema(candidates)
    try:
        payload = provider.route(
            SYSTEM_PROMPT,
            routing_prompt(context, compressed_examples(data)),
            schema,
        )
        decision = validate_decision(payload, candidates).model_dump()
        results["routing"] = {
            "status": "ok",
            "message_id": TEXT_MESSAGE_ID,
            "fallback_model_decision": decision,
            "primary_model_decision": _primary_decision(TEXT_MESSAGE_ID),
            "exact_decision_match": decision == _primary_decision(TEXT_MESSAGE_ID),
        }
    except Exception as error:
        results["routing"] = {
            "status": "failed",
            "message_id": TEXT_MESSAGE_ID,
            "error": f"{type(error).__name__}: {error}",
        }
        results["api_stats"] = provider.stats.to_dict()
        results["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        _write(results)
        print("FALLBACK_MODEL_SMOKE=" + json.dumps(results, ensure_ascii=False, sort_keys=True))
        return 1

    image_result = pipeline.smoke_media("image", IMAGE_ID)
    results["image"] = {
        "media_id": IMAGE_ID,
        **_media_comparison(_primary_media(IMAGE_ID), image_result),
    }
    voice_result = pipeline.smoke_media("voice", VOICE_ID)
    results["voice"] = {
        "media_id": VOICE_ID,
        **_media_comparison(_primary_media(VOICE_ID), voice_result),
    }
    results["api_stats"] = provider.stats.to_dict()
    results["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    _write(results)
    print("FALLBACK_MODEL_SMOKE=" + json.dumps(results, ensure_ascii=False, sort_keys=True))
    print(f"RESULTS_PATH={RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
