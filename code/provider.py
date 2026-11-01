from __future__ import annotations

import base64
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, TypeVar

import openai
from openai import OpenAI

from config import Config
from errors import ERROR_CATEGORIES, ProviderFailure, safe_category
from prompts import MEDIA_SYSTEM_PROMPT, UNTRUSTED_END, UNTRUSTED_START


T = TypeVar("T")


def provider_error_category(error: Exception) -> str:
    if isinstance(error, ProviderFailure):
        return safe_category(error.category)
    if isinstance(error, openai.APITimeoutError):
        return "timeout"
    if isinstance(error, openai.APIConnectionError):
        return "connection_failed"
    if isinstance(error, openai.APIStatusError):
        status = error.status_code
        if status == 429:
            return "rate_limited"
        if status in (401, 403):
            return "auth_rejected"
        if status == 408:
            return "timeout"
        if 400 <= status <= 499:
            return "bad_request"
        if 500 <= status <= 599:
            return "server_error"
    return "unknown"


@dataclass
class ProviderStats:
    logical_calls: int = 0
    http_requests: int = 0
    transport_retries: int = 0
    backoff_seconds: float = 0.0
    routing_calls: int = 0
    image_calls: int = 0
    audio_calls: int = 0
    pacing_seconds: float = 0.0
    configured_min_interval_seconds: float = 0.0
    final_min_interval_seconds: float = 0.0
    interval_increases: int = 0
    interval_reductions: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class GeminiProvider:
    RETRY_WAITS_SECONDS = (15.0, 30.0, 60.0, 90.0, 120.0)
    MAX_INTERVAL_SECONDS = 60.0

    def __init__(self, config: Config) -> None:
        self.model = config.model
        self.default_interval = config.min_call_interval_seconds
        self.current_interval = config.min_call_interval_seconds
        self.last_logical_call_started: float | None = None
        self.consecutive_successes = 0
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        self.stats = ProviderStats(
            configured_min_interval_seconds=self.default_interval,
            final_min_interval_seconds=self.current_interval,
        )

    def _pace(self) -> None:
        now = time.monotonic()
        if self.last_logical_call_started is not None:
            remaining = self.current_interval - (
                now - self.last_logical_call_started
            )
            if remaining > 0:
                print(
                    f"PACING_WAIT seconds={remaining:.3f} "
                    f"interval={self.current_interval:.3f}",
                    file=sys.stderr,
                    flush=True,
                )
                self.stats.pacing_seconds += remaining
                time.sleep(remaining)
        self.last_logical_call_started = time.monotonic()

    def _increase_interval(self) -> None:
        old = self.current_interval
        self.current_interval = min(
            self.MAX_INTERVAL_SECONDS,
            self.current_interval * 1.5,
        )
        self.consecutive_successes = 0
        self.stats.final_min_interval_seconds = self.current_interval
        if self.current_interval != old:
            self.stats.interval_increases += 1
            print(
                f"PACING_CHANGE reason=HTTP_429 old_seconds={old:.3f} "
                f"new_seconds={self.current_interval:.3f}",
                file=sys.stderr,
                flush=True,
            )

    def _record_success(self) -> None:
        self.consecutive_successes += 1
        if self.consecutive_successes < 10:
            return
        self.consecutive_successes = 0
        old = self.current_interval
        self.current_interval = max(
            self.default_interval,
            self.current_interval * 0.9,
        )
        self.stats.final_min_interval_seconds = self.current_interval
        if self.current_interval != old:
            self.stats.interval_reductions += 1
            print(
                f"PACING_CHANGE reason=10_SUCCESSES old_seconds={old:.3f} "
                f"new_seconds={self.current_interval:.3f}",
                file=sys.stderr,
                flush=True,
            )

    @staticmethod
    def _retryable(error: Exception) -> tuple[bool, str]:
        category = provider_error_category(error)
        return category in {
            "rate_limited", "server_error", "timeout", "connection_failed"
        }, category

    def _with_backoff(self, kind: str, operation: Callable[[], T]) -> T:
        self._pace()
        self.stats.logical_calls += 1
        if kind == "routing":
            self.stats.routing_calls += 1
        elif kind == "image":
            self.stats.image_calls += 1
        elif kind == "audio":
            self.stats.audio_calls += 1

        total_attempts = len(self.RETRY_WAITS_SECONDS) + 1
        for attempt in range(1, total_attempts + 1):
            self.stats.http_requests += 1
            try:
                result = operation()
                self._record_success()
                return result
            except Exception as error:  # classified immediately below
                retryable, label = self._retryable(error)
                if label == "rate_limited":
                    print(
                        f"TRANSPORT_ERROR kind={kind} attempt={attempt}/"
                        f"{total_attempts} category={label}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._increase_interval()
                else:
                    self.consecutive_successes = 0
                if not retryable:
                    raise ProviderFailure(label) from None
                if attempt == total_attempts:
                    raise ProviderFailure(label) from None
                wait_seconds = self.RETRY_WAITS_SECONDS[attempt - 1]
                self.stats.transport_retries += 1
                self.stats.backoff_seconds += wait_seconds
                print(
                    f"BACKOFF kind={kind} error={label} "
                    f"attempt={attempt}/{total_attempts} "
                    f"wait_seconds={wait_seconds:.0f}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait_seconds)
        raise AssertionError("unreachable")

    def route(self, system_prompt: str, user_prompt: str, schema: dict) -> object:
        def request():
            return self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "message_routing_decision",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )

        response = self._with_backoff("routing", request)
        content = response.choices[0].message.content
        if not content:
            raise ProviderFailure("empty_response")
        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("invalid_response") from None

    def describe_image(self, file_name: str, media_bytes: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(media_bytes).decode("ascii")
        prompt = (
            "Describe the message content in this image for a notification router. "
            "Read all visible text accurately and note visual context, sender claims, "
            "amounts, dates, deadlines, links, urgency, and safety concerns.\n"
            f"{UNTRUSTED_START}\nfile_name={file_name}\n{UNTRUSTED_END}"
        )

        def request():
            return self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": MEDIA_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{encoded}"
                                },
                            },
                        ],
                    },
                ],
            )

        response = self._with_backoff("image", request)
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ProviderFailure("empty_response")
        return content.strip()

    def transcribe_audio(self, file_name: str, media_bytes: bytes, audio_format: str) -> str:
        encoded = base64.b64encode(media_bytes).decode("ascii")
        prompt = (
            "Transcribe this voice message accurately for a notification router. "
            "Include spoken names, amounts, dates, deadlines, links, urgency, and "
            "safety concerns. Mention important non-speech sounds briefly.\n"
            f"{UNTRUSTED_START}\nfile_name={file_name}\n{UNTRUSTED_END}"
        )

        def request():
            return self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": MEDIA_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded,
                                    "format": audio_format,
                                },
                            },
                        ],
                    },
                ],
            )

        response = self._with_backoff("audio", request)
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ProviderFailure("empty_response")
        return content.strip()
