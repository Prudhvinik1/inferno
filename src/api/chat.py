"""POST /v1/chat/completions -- OpenAI-compatible chat completion endpoint."""

from __future__ import annotations

import json

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.core.router import ModelNotFoundError, ModelRouter
from src.core.schemas import ChatRequest, ChatResponse, ErrorDetail, ErrorResponse
from src.main import get_router

logger = structlog.get_logger()

router = APIRouter(tags=["chat"])


@router.post(
    "/chat/completions",
    response_model=ChatResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
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
    """
    try:
        if body.stream:
            return _stream_response(body, request, model_router)
        return await model_router.generate(body)

    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                message=str(exc),
                type="invalid_request_error",
                param="model",
                code="model_not_found",
            ).model_dump(),
        ) from exc
    except RuntimeError as exc:
        # Covers "model not loaded" from LocalLLMProvider
        raise HTTPException(
            status_code=503,
            detail=ErrorDetail(
                message=str(exc),
                type="server_error",
                code="model_not_loaded",
            ).model_dump(),
        ) from exc
    except Exception as exc:
        logger.exception("chat.completions.error", model=body.model)
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                message="Internal server error",
                type="server_error",
                code="internal_error",
            ).model_dump(),
        ) from exc


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

        except ModelNotFoundError as exc:
            error = ErrorResponse(
                error=ErrorDetail(
                    message=str(exc),
                    type="invalid_request_error",
                    param="model",
                    code="model_not_found",
                )
            )
            yield f"data: {error.model_dump_json()}\n\n"

        except RuntimeError as exc:
            error = ErrorResponse(
                error=ErrorDetail(
                    message=str(exc),
                    type="server_error",
                    code="model_not_loaded",
                )
            )
            yield f"data: {error.model_dump_json()}\n\n"

        except Exception as exc:
            logger.exception("chat.stream.error", model=body.model)
            error = ErrorResponse(
                error=ErrorDetail(
                    message="Internal server error during streaming",
                    type="server_error",
                    code="internal_error",
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
