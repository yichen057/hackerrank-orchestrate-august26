"""Exception text is untrusted: it must not become persisted or sent diagnostics."""
from types import SimpleNamespace
import json
import traceback

import httpx
import openai
import pytest

from cache import CacheStore
from errors import ERROR_CATEGORIES, ConversionFailure, ProviderFailure, UnsupportedMediaError
from pipeline import RouterPipeline, RunStats
from provider import GeminiProvider, ProviderStats, provider_error_category
from schemas import Decision, validation_error_text

SECRET = "FAKE_SECRET_DO_NOT_TRANSMIT /Users/private/example.txt"


def sdk_error(status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.invalid"))
    return openai.APIStatusError(SECRET, response=response, body={"message": SECRET})


@pytest.mark.parametrize("status,category", [
    (429, "rate_limited"), (401, "auth_rejected"), (403, "auth_rejected"),
    (400, "bad_request"), (408, "timeout"), (422, "bad_request"), (503, "server_error"),
])
def test_http_errors_use_status_not_message(status, category):
    assert provider_error_category(sdk_error(status)) == category


@pytest.mark.parametrize("kind,category", [
    (openai.APITimeoutError, "timeout"),
    (openai.APIConnectionError, "connection_failed"),
])
def test_connection_errors_use_sdk_types(kind, category):
    error = kind(request=httpx.Request("POST", "https://example.invalid"))
    assert provider_error_category(error) == category


def test_message_cannot_spoof_category():
    assert provider_error_category(RuntimeError("401 timeout rate limit " + SECRET)) == "unknown"
    failure = ProviderFailure(SECRET)
    assert failure.category == "unknown"
    assert SECRET not in str(failure)


@pytest.mark.parametrize("status", [400, 429, 503])
def test_transport_failure_never_logs_or_raises_raw_body(status, capsys):
    provider = object.__new__(GeminiProvider)
    provider.stats = ProviderStats()
    provider._pace = lambda: None
    provider._increase_interval = lambda: None
    provider.RETRY_WAITS_SECONDS = ()

    def operation():
        raise sdk_error(status)

    with pytest.raises(ProviderFailure) as caught:
        provider._with_backoff("image", operation)
    assert caught.value.category in ERROR_CATEGORIES
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err


def pipeline_stub(tmp_path):
    pipeline = object.__new__(RouterPipeline)
    pipeline.cache = CacheStore(tmp_path)
    pipeline.stats = RunStats()
    pipeline._media_fingerprint = lambda prepared: "fingerprint"
    return pipeline


@pytest.mark.parametrize("failure,category", [
    (FileNotFoundError(SECRET), "missing_media"),
    (PermissionError(SECRET), "local_io_error"),
    (ValueError(SECRET), "unknown"),
    (UnsupportedMediaError(SECRET), "unsupported_media"),
    (ConversionFailure(SECRET), "conversion_failed"),
])
def test_local_media_errors_are_closed_categories(tmp_path, failure, category):
    pipeline = pipeline_stub(tmp_path)

    def fail(row):
        raise failure

    pipeline._prepare = fail
    result = pipeline._media_result({"media_type": "image", "media_id": "test"})
    assert result["error_category"] == category
    assert SECRET not in json.dumps(result)


def test_provider_media_failure_is_safe_in_result_and_cache(tmp_path):
    pipeline = pipeline_stub(tmp_path)
    path = tmp_path / "image.png"
    path.write_bytes(b"test fixture")
    prepared = SimpleNamespace(
        media_id="test", media_kind="image", source_path=path,
        prepared_path=path, converted=False, mime_type="image/png",
        context_metadata=lambda: {"media_id": "test"},
    )
    pipeline._prepare = lambda row: prepared

    def fail(*args):
        raise ProviderFailure(SECRET)

    pipeline.provider = SimpleNamespace(describe_image=fail)
    result = pipeline._media_result({"media_type": "image", "media_id": "test"})
    assert result["error_category"] in ERROR_CATEGORIES
    assert SECRET not in json.dumps(result)
    assert SECRET not in (tmp_path / "media/test.json").read_text()


def test_legacy_cache_is_sanitized_without_losing_fingerprint(tmp_path):
    cache = CacheStore(tmp_path)
    path = cache.media / "test.json"
    cache._write(path, {"fingerprint": "fp", "result": {
        "status": "unreadable", "content": SECRET, "error_category": SECRET,
    }})
    record = cache.get_media("test", "fp")
    assert record["fingerprint"] == "fp"
    assert record["result"]["error_category"] == "unknown"
    assert record["result"]["content"] == ""
    assert SECRET not in path.read_text()


def test_cache_write_normalizes_category_and_preserves_success(tmp_path):
    cache = CacheStore(tmp_path)
    cache.put_media("test", {"fingerprint": "fp", "result": {
        "status": "ok", "content": "Successful description", "error_category": SECRET,
    }})
    result = cache.get_media("test", "fp")["result"]
    assert result["content"] == "Successful description"
    assert result["error_category"] in ERROR_CATEGORIES


def test_pydantic_error_does_not_echo_input():
    with pytest.raises(ValueError) as caught:
        Decision.model_validate({"action": SECRET})
    assert validation_error_text(caught.value) == "invalid_decision"
    assert validation_error_text(RuntimeError(SECRET)) == "invalid_response"


@pytest.mark.parametrize("bad_payload", [False, True])
def test_correction_prompt_never_contains_exception_text(tmp_path, dataset, bad_payload):
    pipeline = pipeline_stub(tmp_path)
    pipeline.data = dataset
    pipeline.examples = ""
    pipeline._decision_fingerprint = lambda row: "fp"
    pipeline._media_result = lambda *args, **kwargs: None
    prompts = []

    def route(system, prompt, schema):
        prompts.append(prompt)
        if bad_payload:
            return {"action": SECRET}
        raise RuntimeError(SECRET)

    pipeline.provider = SimpleNamespace(route=route)
    row = next(row for row in dataset.messages if not row.get("media_type"))
    result = pipeline.process_message(row)
    assert len(prompts) == 2
    assert all(SECRET not in prompt for prompt in prompts)
    assert SECRET not in json.dumps(result)
    assert SECRET not in (tmp_path / "decisions" / f"{row['message_id']}.json").read_text()
