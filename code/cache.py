from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from errors import safe_category


class CacheStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.decisions = root / "decisions"
        self.media = root / "media"
        self.converted = root / "converted"
        for directory in (self.decisions, self.media, self.converted):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def get_decision(
        self, message_id: str, fingerprint: str, refresh_fallbacks: bool
    ) -> dict[str, Any] | None:
        record = self._read(self.decisions / f"{message_id}.json")
        if not record or record.get("fingerprint") != fingerprint:
            return None
        if refresh_fallbacks and record.get("source") == "fallback":
            return None
        return record

    def put_decision(self, message_id: str, record: dict[str, Any]) -> None:
        self._write(self.decisions / f"{message_id}.json", record)

    def get_media(self, media_id: str, fingerprint: str) -> dict[str, Any] | None:
        record = self._read(self.media / f"{media_id}.json")
        if record and self.sanitize_media_record(record):
            self.put_media(media_id, record)
        if not record or record.get("fingerprint") != fingerprint:
            return None
        return record

    def put_media(self, media_id: str, record: dict[str, Any]) -> None:
        self.sanitize_media_record(record)
        self._write(self.media / f"{media_id}.json", record)

    @staticmethod
    def sanitize_media_record(record: dict[str, Any]) -> bool:
        result = record.get("result")
        if not isinstance(result, dict):
            return False
        before = result.copy()
        if result.get("status") == "unreadable":
            result["content"] = ""
            result["error_category"] = safe_category(result.get("error_category"))
        elif "error_category" in result:
            result["error_category"] = safe_category(result["error_category"])
        return result != before
