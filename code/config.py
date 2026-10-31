from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MIN_CALL_INTERVAL_SECONDS = 2.0


@dataclass(frozen=True)
class Config:
    repo_root: Path
    dataset_dir: Path
    cache_dir: Path
    output_path: Path
    api_key: str
    base_url: str
    model: str
    min_call_interval_seconds: float
    timeout_seconds: float = 180.0


def _output_path(output: str | None) -> Path:
    output_path = Path(output) if output else REPO_ROOT / "submission" / "output.csv"
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    return output_path


def load_config(output: str | None = None) -> Config:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Run through a fresh shell that reads your profile."
        )

    base_url = os.environ.get("GEMINI_BASE_URL", DEFAULT_BASE_URL).strip()
    model = os.environ.get("GEMINI_MODEL", "").strip()
    if not base_url:
        raise RuntimeError("GEMINI_BASE_URL cannot be empty.")
    if not model:
        raise RuntimeError(
            "GEMINI_MODEL is not set. Export it in the shell profile or run command."
        )
    interval_text = os.environ.get(
        "GEMINI_MIN_CALL_INTERVAL_SECONDS",
        str(DEFAULT_MIN_CALL_INTERVAL_SECONDS),
    )
    try:
        min_call_interval_seconds = float(interval_text)
    except ValueError as error:
        raise RuntimeError(
            "GEMINI_MIN_CALL_INTERVAL_SECONDS must be greater than 0 and at most 60."
        ) from error
    if not 0 < min_call_interval_seconds <= 60:
        raise RuntimeError(
            "GEMINI_MIN_CALL_INTERVAL_SECONDS must be greater than 0 and at most 60."
        )

    return Config(
        repo_root=REPO_ROOT,
        dataset_dir=REPO_ROOT / "dataset",
        cache_dir=REPO_ROOT / "code" / ".cache",
        output_path=_output_path(output),
        api_key=api_key,
        base_url=base_url,
        model=model,
        min_call_interval_seconds=min_call_interval_seconds,
    )


def load_forced_offline_config(output: str | None) -> Config:
    if not output:
        raise RuntimeError("--force-offline requires an explicit --output path.")
    output_path = _output_path(output)
    production_output = REPO_ROOT / "submission" / "output.csv"
    if output_path.resolve() == production_output.resolve():
        raise RuntimeError("Forced-offline output cannot be submission/output.csv.")
    return Config(
        repo_root=REPO_ROOT,
        dataset_dir=REPO_ROOT / "dataset",
        cache_dir=REPO_ROOT / "code" / "evaluation" / ".cache" / "offline_baseline",
        output_path=output_path,
        api_key="",
        base_url="",
        model="forced-offline",
        min_call_interval_seconds=DEFAULT_MIN_CALL_INTERVAL_SECONDS,
    )
