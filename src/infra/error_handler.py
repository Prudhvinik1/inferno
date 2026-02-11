"""Global error normalization for the FastAPI application.

Catches all exceptions (custom, provider, validation, unexpected) and
returns a consistent JSON error body matching the OpenAI error shape:

    {"error": {"message": "...", "type": "...", "param": null, "code": "..."}}

Install via ``install_error_handlers(app)`` in the app factory.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.exceptions import LLMServingError, RateLimitError
from src.core.schemas import ErrorDetail, ErrorResponse
from src.providers.external_api import (
    ProviderAPIError,
    ProviderAuthError,
    ProviderRateLimitError,
)

logger = structlog.get_logger()


def install_error_handlers(app: FastAPI) -> None:
    """Register exception handlers that normalize all errors to ErrorResponse."""

    # ── Our custom exceptions ────────────────────────────────────────────

    @app.exception_handler(LLMServingError)
    async def handle_llm_serving_error(
        request: Request, exc: LLMServingError
    ) -> JSONResponse:
        body = ErrorResponse(
            error=ErrorDetail(
                message=exc.message,
                type=exc.error_type,
                param=exc.param,
                code=exc.error_code,
            )
        )
        headers = {}
        if isinstance(exc, RateLimitError) and exc.retry_after is not None:
            headers["Retry-After"] = str(int(exc.retry_after))

        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(),
            headers=headers or None,
        )

    # ── Provider API errors ──────────────────────────────────────────────

    @app.exception_handler(ProviderAPIError)
    async def handle_provider_api_error(
        request: Request, exc: ProviderAPIError
    ) -> JSONResponse:
        # Map provider errors to appropriate HTTP status codes
        if isinstance(exc, ProviderAuthError):
            status = 502  # our upstream's auth failed -- that's a bad gateway
        elif isinstance(exc, ProviderRateLimitError):
            status = 429
        else:
            status = 502

        body = ErrorResponse(
            error=ErrorDetail(
                message=str(exc),
                type=exc.error_type,
                code=exc.error_code,
            )
        )
        logger.warning(
            "provider.api_error",
            provider=exc.provider,
            upstream_status=exc.status_code,
            message=str(exc),
        )
        return JSONResponse(status_code=status, content=body.model_dump())

    # ── Pydantic validation errors ───────────────────────────────────────

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        # Build a human-readable message from the first error
        first = errors[0] if errors else {}
        loc = " -> ".join(str(l) for l in first.get("loc", []))
        msg = first.get("msg", "Validation error")

        body = ErrorResponse(
            error=ErrorDetail(
                message=f"{msg} at {loc}" if loc else msg,
                type="invalid_request_error",
                param=loc or None,
                code="validation_error",
            )
        )
        return JSONResponse(status_code=400, content=body.model_dump())

    # ── Starlette HTTP exceptions (from HTTPException raises) ────────────

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        # If detail is already a dict (from our manual HTTPException raises),
        # wrap it in the standard shape
        if isinstance(exc.detail, dict) and "message" in exc.detail:
            body = ErrorResponse(error=ErrorDetail(**exc.detail))
        else:
            body = ErrorResponse(
                error=ErrorDetail(
                    message=str(exc.detail),
                    type="error",
                    code=str(exc.status_code),
                )
            )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    # ── Catch-all for unhandled exceptions ───────────────────────────────

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("unhandled_error", path=request.url.path)
        body = ErrorResponse(
            error=ErrorDetail(
                message="Internal server error",
                type="server_error",
                code="internal_error",
            )
        )
        return JSONResponse(status_code=500, content=body.model_dump())
