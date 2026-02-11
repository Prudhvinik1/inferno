"""POST /v1/chat/completions -- OpenAI-compatible chat completion endpoint."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from src.core.router import ModelRouter
from src.core.schemas import ChatRequest, ChatResponse, ErrorResponse
from src.main import get_router

logger = structlog.get_logger()

router = APIRouter(tags=["chat"])


@router.post(
    "/chat/completions",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        502: {"model": ErrorResponse, "description": "Provider error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def chat_completions(
    body: ChatRequest,
    request: Request,
    model_router: ModelRouter = Depends(get_router),
) -> ChatResponse | StreamingResponse:
    """Create a chat completion.

    Supports both streaming (SSE) and non-streaming responses,
    controlled by the ``stream`` field in the request body.

    Errors are caught by the global error handler middleware and
    returned in the unified ErrorResponse format.
    """
    if body.stream:
        return _stream_response(body, request, model_router)
    return await model_router.generate(body)


def _stream_response(
    body: ChatRequest,
    request: Request,
    model_router: ModelRouter,
) -> StreamingResponse:
    """Build an SSE StreamingResponse that yields chat completion chunks.

    Handles client disconnect by checking ``request.is_disconnected()``
    between chunks so we can cancel generation early.
    """

    async def event_generator():
        try:
            async for chunk in model_router.stream(body):
                # Check if client has disconnected
                if await request.is_disconnected():
                    logger.info("chat.stream.client_disconnected", model=body.model)
                    break

                data = chunk.model_dump_json(exclude_none=True)
                yield f"data: {data}\n\n"

            # Send the [DONE] sentinel (OpenAI convention)
            yield "data: [DONE]\n\n"

        except Exception as exc:
            # For streaming, errors mid-stream are sent as SSE error events
            from src.core.schemas import ErrorDetail, ErrorResponse as ErrResp

            logger.exception("chat.stream.error", model=body.model)
            error = ErrResp(
                error=ErrorDetail(
                    message=str(exc),
                    type="server_error",
                    code="stream_error",
                )
            )
            yield f"data: {error.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
