"""Custom exception hierarchy for the LLM serving stack.

All exceptions map cleanly to HTTP status codes and the unified
ErrorResponse schema (type, message, param, code) defined in schemas.py.
"""

from __future__ import annotations


class LLMServingError(Exception):
    """Base exception for the serving stack."""

    status_code: int = 500
    error_type: str = "server_error"
    error_code: str | None = None
    param: str | None = None

    def __init__(self, message: str, **kwargs: object) -> None:
        self.message = message
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        super().__init__(message)


# ── Client errors (4xx) ─────────────────────────────────────────────────────


class InvalidRequestError(LLMServingError):
    """400 -- malformed or invalid request."""

    status_code = 400
    error_type = "invalid_request_error"


class AuthenticationError(LLMServingError):
    """401 -- missing or invalid API key."""

    status_code = 401
    error_type = "authentication_error"
    error_code = "invalid_api_key"


class PermissionError(LLMServingError):
    """403 -- valid key but insufficient permissions."""

    status_code = 403
    error_type = "permission_error"
    error_code = "insufficient_permissions"


class ModelNotFoundError(LLMServingError):
    """404 -- requested model does not exist in the registry."""

    status_code = 404
    error_type = "invalid_request_error"
    error_code = "model_not_found"
    param = "model"

    def __init__(self, model: str, available: list[str]) -> None:
        self.model = model
        self.available = available
        super().__init__(
            f"Model '{model}' not found. Available: {available}",
            param="model",
            error_code="model_not_found",
        )


class RateLimitError(LLMServingError):
    """429 -- rate limit exceeded."""

    status_code = 429
    error_type = "rate_limit_error"
    error_code = "rate_limit_exceeded"

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(message)


# ── Server errors (5xx) ─────────────────────────────────────────────────────


class ProviderError(LLMServingError):
    """502 -- upstream provider returned an error."""

    status_code = 502
    error_type = "provider_error"

    def __init__(
        self,
        message: str,
        provider: str,
        upstream_status: int | None = None,
    ) -> None:
        self.provider = provider
        self.upstream_status = upstream_status
        super().__init__(message, error_code=f"{provider}_error")


class ModelNotLoadedError(LLMServingError):
    """503 -- model exists in registry but is not loaded / available."""

    status_code = 503
    error_type = "server_error"
    error_code = "model_not_loaded"


class InternalError(LLMServingError):
    """500 -- catch-all internal error."""

    status_code = 500
    error_type = "server_error"
    error_code = "internal_error"
