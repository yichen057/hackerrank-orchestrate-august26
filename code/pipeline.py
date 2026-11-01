from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cache import CacheStore
from config import Config
from context import build_context
from data_io import Dataset, write_output
from fallback import conservative_fallback
from errors import local_error_category, safe_category
if TYPE_CHECKING:
    from media import PreparedMedia
from prompts import (
    MEDIA_PROMPT_VERSION,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    approximate_example_tokens,
    compressed_examples,
    correction_prompt,
    routing_prompt,
)
from provider import GeminiProvider, ProviderFailure
from schemas import (
    SCHEMA_VERSION,
    Decision,
    decision_schema,
    validate_decision,
    validation_error_text,
)


@dataclass
class RunStats:
    rows_requested: int = 0
    decision_cache_hits: int = 0
    media_cache_hits: int = 0
    model_decisions: int = 0
    fallback_decisions: int = 0
    validation_retries: int = 0
    media_unreadable: int = 0

    def to_dict(self) -> dict:
        return vars(self).copy()


class RouterPipeline:
    def __init__(
        self,
        config: Config,
        data: Dataset,
        provider: GeminiProvider,
        cache: CacheStore,
    ) -> None:
        self.config = config
        self.data = data
        self.provider = provider
        self.cache = cache
        self.examples = compressed_examples(data)
        self.example_token_estimates = approximate_example_tokens(data, self.examples)
        self.stats = RunStats()

    def _message(self, message_id: str) -> dict[str, str]:
        for row in self.data.messages:
            if row["message_id"] == message_id:
                return row
        raise KeyError(f"Unknown message_id: {message_id}")

    def _media_path(self, row: dict[str, str]) -> Path | None:
        if row.get("media_type") == "image":
            return self.data.images.get(row.get("media_id", ""))
        if row.get("media_type") == "voice":
            return self.data.voices.get(row.get("media_id", ""))
        return None

    def _decision_fingerprint(self, row: dict[str, str]) -> str:
        media_path = self._media_path(row)
        media_hash = (
            hashlib.sha256(media_path.read_bytes()).hexdigest()
            if media_path and media_path.exists()
            else None
        )
        payload = {
            "message": row,
            "dataset": self.data.source_fingerprint,
            "media_hash": media_hash,
            "model": self.config.model,
            "prompt": PROMPT_VERSION,
            "media_prompt": MEDIA_PROMPT_VERSION,
            "schema": SCHEMA_VERSION,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def _prepare(self, row: dict[str, str]) -> PreparedMedia | None:
        from media import prepare_audio, prepare_image

        media_type = row.get("media_type", "")
        media_id = row.get("media_id", "")
        if not media_type:
            return None
        path = self._media_path(row)
        if not media_id or not path or not path.exists():
            raise FileNotFoundError(
                f"Missing media mapping or file for message {row['message_id']}"
            )
        if media_type == "image":
            return prepare_image(media_id, path, self.cache)
        if media_type == "voice":
            return prepare_audio(media_id, path, self.cache)
        raise ValueError(f"Unsupported media_type: {media_type}")

    def _media_fingerprint(self, prepared: PreparedMedia) -> str:
        from media import CONVERSION_VERSION

        value = {
            "source_sha256": prepared.source_sha256,
            "detected_format": prepared.detected_format,
            "conversion": CONVERSION_VERSION,
            "model": self.config.model,
            "prompt": MEDIA_PROMPT_VERSION,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def _media_result(
        self,
        row: dict[str, str],
        refresh_unreadable: bool = False,
    ) -> dict | None:
        if not row.get("media_type"):
            return None
        try:
            prepared = self._prepare(row)
            assert prepared is not None
            fingerprint = self._media_fingerprint(prepared)
            cached = self.cache.get_media(prepared.media_id, fingerprint)
            if cached and not (
                refresh_unreadable
                and cached.get("result", {}).get("status") == "unreadable"
            ):
                self.stats.media_cache_hits += 1
                return cached["result"]

            metadata = prepared.context_metadata()
            try:
                if prepared.media_kind == "image":
                    content = self.provider.describe_image(
                        prepared.source_path.name,
                        prepared.prepared_path.read_bytes(),
                        "image/png" if prepared.converted else prepared.mime_type,
                    )
                else:
                    content = self.provider.transcribe_audio(
                        prepared.source_path.name,
                        prepared.prepared_path.read_bytes(),
                        prepared.api_format or "wav",
                    )
                result = {**metadata, "status": "ok", "content": content}
            except ProviderFailure as error:
                self.stats.media_unreadable += 1
                result = {
                    **metadata,
                    "status": "unreadable",
                    "content": "",
                    "error_category": safe_category(error.category),
                }
            self.cache.put_media(
                prepared.media_id,
                {"fingerprint": fingerprint, "result": result},
            )
            return result
        except Exception as error:
            self.stats.media_unreadable += 1
            return {
                "media_id": row.get("media_id") or None,
                "media_kind": row.get("media_type") or None,
                "source_file_name": None,
                "status": "unreadable",
                "content": "",
                "error_category": local_error_category(error),
            }

    def render_text_prompt(self, message_id: str) -> tuple[str, str, dict]:
        row = self._message(message_id)
        if row.get("media_type"):
            raise ValueError("Prompt preview requires a text-only message")
        context = build_context(self.data, row, None)
        return SYSTEM_PROMPT, routing_prompt(context, self.examples), context

    def smoke_media(self, media_kind: str, media_id: str) -> dict:
        if media_kind not in {"image", "voice"}:
            raise ValueError(f"Unsupported smoke-test media kind: {media_kind}")
        result = self._media_result(
            {
                "message_id": f"smoke_{media_id}",
                "media_type": media_kind,
                "media_id": media_id,
            }
        )
        assert result is not None
        return result

    def process_message(
        self, row: dict[str, str], refresh_fallbacks: bool = False
    ) -> dict[str, str]:
        self.stats.rows_requested += 1
        fingerprint = self._decision_fingerprint(row)
        cached = self.cache.get_decision(
            row["message_id"], fingerprint, refresh_fallbacks
        )
        if cached:
            self.stats.decision_cache_hits += 1
            decision = Decision.model_validate(cached["decision"])
            return self._serialize(row["message_id"], decision)

        media_result = self._media_result(
            row,
            refresh_unreadable=refresh_fallbacks,
        )
        context = build_context(self.data, row, media_result)
        candidates = [
            item["message_id"] for item in context["evidence_candidates"]
        ]
        schema = decision_schema(candidates)
        prompt = routing_prompt(context, self.examples)

        decision: Decision | None = None
        source = "model"
        error_text = ""
        for validation_round in range(2):
            current_prompt = (
                prompt
                if validation_round == 0
                else correction_prompt(prompt, error_text)
            )
            try:
                payload = self.provider.route(SYSTEM_PROMPT, current_prompt, schema)
            except ProviderFailure:
                break
            except Exception as error:
                error_text = validation_error_text(error)
                if validation_round == 0:
                    self.stats.validation_retries += 1
                    continue
                break
            try:
                decision = validate_decision(payload, candidates)
                break
            except Exception as error:
                error_text = validation_error_text(error)
                if validation_round == 0:
                    self.stats.validation_retries += 1

        if decision is None:
            decision = conservative_fallback()
            source = "fallback"
            self.stats.fallback_decisions += 1
        else:
            self.stats.model_decisions += 1

        self.cache.put_decision(
            row["message_id"],
            {
                "fingerprint": fingerprint,
                "source": source,
                "decision": decision.model_dump(),
            },
        )
        return self._serialize(row["message_id"], decision)

    @staticmethod
    def _serialize(message_id: str, decision: Decision) -> dict[str, str]:
        return {
            "message_id": message_id,
            "action": decision.action,
            "message_type": decision.message_type,
            "reason": decision.reason,
            "confidence": f"{decision.confidence:.2f}",
            "evidence_message_ids": (
                ";".join(decision.evidence_message_ids)
                if decision.evidence_message_ids
                else "none"
            ),
        }

    def run(
        self,
        output_path: Path,
        message_id: str | None = None,
        refresh_fallbacks: bool = False,
    ) -> list[dict[str, str]]:
        rows = (
            [self._message(message_id)]
            if message_id
            else self.data.messages
        )
        output_rows = []
        total = len(rows)
        for index, row in enumerate(rows, start=1):
            output_rows.append(
                self.process_message(row, refresh_fallbacks=refresh_fallbacks)
            )
            print(
                f"PROGRESS completed={index}/{total} message_id={row['message_id']} "
                f"model={self.stats.model_decisions} fallback={self.stats.fallback_decisions} "
                f"decision_cache_hits={self.stats.decision_cache_hits}",
                flush=True,
            )
        write_output(output_rows, output_path)
        return output_rows
