"""Closed diagnostic vocabulary. Never classify by untrusted exception text."""

ERROR_CATEGORIES = frozenset({
    "rate_limited", "timeout", "auth_rejected", "connection_failed",
    "bad_request", "server_error", "empty_response", "unsupported_media",
    "conversion_failed", "missing_media", "local_io_error", "offline", "unknown",
})


def safe_category(value: object) -> str:
    return value if isinstance(value, str) and value in ERROR_CATEGORIES else "unknown"


class ProviderFailure(RuntimeError):
    def __init__(self, category: str = "unknown") -> None:
        self.category = safe_category(category)
        super().__init__(self.category)


class UnsupportedMediaError(ValueError):
    pass


class ConversionFailure(RuntimeError):
    pass


def local_error_category(error: Exception) -> str:
    if isinstance(error, ProviderFailure):
        return safe_category(error.category)
    if isinstance(error, FileNotFoundError):
        return "missing_media"
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, UnsupportedMediaError):
        return "unsupported_media"
    if isinstance(error, ConversionFailure):
        return "conversion_failed"
    if isinstance(error, OSError):
        return "local_io_error"
    return "unknown"
