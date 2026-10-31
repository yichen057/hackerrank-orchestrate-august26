from __future__ import annotations

import csv
import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def as_bool(value: str | None) -> bool:
    return str(value).strip() == "1"


@dataclass
class Dataset:
    dataset_dir: Path
    messages: list[dict[str, str]]
    samples: list[dict[str, str]]
    users: dict[str, dict[str, str]]
    groups: dict[str, dict[str, str]]
    group_members: dict[tuple[str, str], dict[str, str]]
    businesses: dict[str, dict[str, str]]
    user_business: dict[tuple[str, str], dict[str, str]]
    history_by_user: dict[str, list[dict[str, str]]]
    events: dict[tuple[str, str], dict[str, str]]
    images: dict[str, Path]
    voices: dict[str, Path]
    daily_by_user: dict[str, list[dict[str, str]]]
    source_fingerprint: str

    @classmethod
    def load(cls, dataset_dir: Path) -> "Dataset":
        names = [
            "messages.csv",
            "sample_messages.csv",
            "users.csv",
            "groups.csv",
            "group_members.csv",
            "business_accounts.csv",
            "user_business_history.csv",
            "message_history.csv",
            "message_events.csv",
            "images.csv",
            "voice_notes.csv",
            "daily_notification_summary.csv",
            "output.csv",
        ]
        paths = {name: dataset_dir / name for name in names}
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing dataset files: {missing}")

        with paths["output.csv"].open(newline="", encoding="utf-8") as handle:
            template_header = next(csv.reader(handle))
        if template_header != OUTPUT_COLUMNS:
            raise ValueError(
                f"dataset/output.csv header mismatch: {template_header!r}"
            )

        rows = {name: read_csv(path) for name, path in paths.items() if name != "output.csv"}
        history_by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows["message_history.csv"]:
            history_by_user[row["user_id"]].append(row)
        for user_rows in history_by_user.values():
            user_rows.sort(key=lambda row: (row["created_at"], row["message_id"]))

        daily_by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows["daily_notification_summary.csv"]:
            daily_by_user[row["user_id"]].append(row)
        for user_rows in daily_by_user.values():
            user_rows.sort(key=lambda row: row["date"])

        digest = hashlib.sha256()
        for name in sorted(names):
            digest.update(name.encode("utf-8"))
            digest.update(paths[name].read_bytes())

        return cls(
            dataset_dir=dataset_dir,
            messages=rows["messages.csv"],
            samples=rows["sample_messages.csv"],
            users={row["user_id"]: row for row in rows["users.csv"]},
            groups={row["group_id"]: row for row in rows["groups.csv"]},
            group_members={
                (row["group_id"], row["user_id"]): row
                for row in rows["group_members.csv"]
            },
            businesses={
                row["business_id"]: row for row in rows["business_accounts.csv"]
            },
            user_business={
                (row["user_id"], row["business_id"]): row
                for row in rows["user_business_history.csv"]
            },
            history_by_user=dict(history_by_user),
            events={
                (row["user_id"], row["message_id"]): row
                for row in rows["message_events.csv"]
            },
            images={
                row["image_id"]: dataset_dir / row["file_path"]
                for row in rows["images.csv"]
            },
            voices={
                row["voice_note_id"]: dataset_dir / row["file_path"]
                for row in rows["voice_notes.csv"]
            },
            daily_by_user=dict(daily_by_user),
            source_fingerprint=digest.hexdigest(),
        )


def write_output(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output_path)
