"""POST /v1/embeddings -- OpenAI-compatible embeddings endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.core.router import ModelRouter
from src.core.schemas import EmbeddingRequest, EmbeddingResponse, ErrorResponse
from src.main import get_router

router = APIRouter(tags=["embeddings"])


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        502: {"model": ErrorResponse, "description": "Provider error"},
    },
)
async def create_embeddings(
    body: EmbeddingRequest,
    model_router: ModelRouter = Depends(get_router),
) -> EmbeddingResponse:
    """Generate embeddings for the given input text(s).

    Accepts a single string or a list of strings. Returns embedding
    vectors in the OpenAI-compatible response format.

    Errors are handled by the global error middleware.
    """
    return await model_router.embed(body)
