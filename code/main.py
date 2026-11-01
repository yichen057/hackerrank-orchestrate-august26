from __future__ import annotations

import argparse
import json
import time

from cache import CacheStore
from config import load_config, load_forced_offline_config
from data_io import Dataset
from pipeline import RouterPipeline
from provider import GeminiProvider, ProviderFailure, ProviderStats


class ForcedOfflineProvider:
    """Provider-shaped object that cannot make an API call."""

    def __init__(self) -> None:
        self.stats = ProviderStats()

    @staticmethod
    def _blocked() -> None:
        raise ProviderFailure("offline")

    def route(self, *_args, **_kwargs):
        self._blocked()

    def describe_image(self, *_args, **_kwargs):
        self._blocked()

    def transcribe_audio(self, *_args, **_kwargs):
        self._blocked()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route WhatsApp messages.")
    parser.add_argument("--output", help="Override the output CSV path.")
    parser.add_argument(
        "--refresh-fallbacks",
        action="store_true",
        help="Rerun only cached fallback decisions; preserve validated model decisions.",
    )
    parser.add_argument(
        "--message-id",
        help="Process one message for controlled vertical-slice verification.",
    )
    parser.add_argument(
        "--print-prompt",
        metavar="MESSAGE_ID",
        help="Print the exact prompts for one text message and exit without an API call.",
    )
    parser.add_argument(
        "--smoke-media",
        action="store_true",
        help="Test exactly the approved AVIF image and M4A audio through Gemini.",
    )
    parser.add_argument(
        "--force-offline",
        action="store_true",
        help=(
            "Opt in to the deterministic no-API fallback path. Requires --output "
            "and uses a separate evaluation cache."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    if args.force_offline and (
        args.refresh_fallbacks or args.message_id or args.print_prompt or args.smoke_media
    ):
        raise RuntimeError(
            "--force-offline cannot be combined with other execution-mode flags."
        )
    config = (
        load_forced_offline_config(args.output)
        if args.force_offline
        else load_config(args.output)
    )
    data = Dataset.load(config.dataset_dir)
    provider = ForcedOfflineProvider() if args.force_offline else GeminiProvider(config)
    pipeline = RouterPipeline(config, data, provider, CacheStore(config.cache_dir))
    print(f"MODEL_NAME={config.model}")
    print(
        "CONFIGURED_MIN_CALL_INTERVAL_SECONDS="
        f"{config.min_call_interval_seconds:.3f}"
    )

    if args.print_prompt:
        system_prompt, user_prompt, _ = pipeline.render_text_prompt(args.print_prompt)
        print("=== SYSTEM PROMPT ===")
        print(system_prompt)
        print("=== ROUTING PROMPT ===")
        print(user_prompt)
        print("=== APPROXIMATE EXAMPLE TOKENS ===")
        print(json.dumps(pipeline.example_token_estimates, sort_keys=True))
        print("=== PROMPT PREVIEW TIMING ===")
        print(f"elapsed_seconds={time.perf_counter() - started:.3f}")
        return 0

    if args.smoke_media:
        smoke_results = {}
        for media_kind, media_id in (("image", "img_020"), ("voice", "vn_007")):
            item_started = time.perf_counter()
            result = pipeline.smoke_media(media_kind, media_id)
            smoke_results[media_id] = result
            print(
                f"SMOKE_RESULT_{media_id}="
                + json.dumps(result, ensure_ascii=False, sort_keys=True)
            )
            print(
                f"SMOKE_SECONDS_{media_id}="
                f"{time.perf_counter() - item_started:.3f}"
            )
        print("API_STATS=" + json.dumps(provider.stats.to_dict(), sort_keys=True))
        print(f"ELAPSED_SECONDS={time.perf_counter() - started:.3f}")
        return 0 if all(item.get("status") == "ok" for item in smoke_results.values()) else 1

    rows = pipeline.run(
        config.output_path,
        message_id=args.message_id,
        refresh_fallbacks=args.refresh_fallbacks,
    )
    elapsed = time.perf_counter() - started
    if args.message_id:
        print("OUTPUT_ROW=" + json.dumps(rows[0], ensure_ascii=False, sort_keys=True))
    print(f"OUTPUT_PATH={config.output_path}")
    print("RUN_STATS=" + json.dumps(pipeline.stats.to_dict(), sort_keys=True))
    print("API_STATS=" + json.dumps(provider.stats.to_dict(), sort_keys=True))
    print(f"ELAPSED_SECONDS={elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
