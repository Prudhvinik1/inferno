"""POST /v1/search -- RAG-powered search endpoint."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.core.rag import RAGPipeline, RAGRequest
from src.core.schemas import ErrorResponse
from src.main import get_rag_pipeline

logger = structlog.get_logger()

router = APIRouter(tags=["search"])


# ── Request / response schemas ───────────────────────────────────────────────


class SearchRequestBody(BaseModel):
    """Search request body."""

    query: str = Field(description="The search query")
    model: str = Field(default="", description="LLM model for answer generation")
    embedding_model: str = Field(default="", description="Model for query embedding")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to retrieve")
    metadata_filter: dict[str, Any] | None = Field(
        default=None, description="Filter results by metadata"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = Field(default=False, description="Stream the generated answer")
    system_prompt: str = Field(default="", description="Custom system prompt")


class IngestRequestBody(BaseModel):
    """Ingest documents into the vector store."""

    documents: list[IngestDocument]
    embedding_model: str = Field(default="", description="Model for embedding")


class IngestDocument(BaseModel):
    """A single document to ingest."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponseBody(BaseModel):
    """Search response with answer and sources."""

    answer: str
    sources: list[SourceBody]
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)


class SourceBody(BaseModel):
    """A source document."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponseBody(BaseModel):
    """Ingest response."""

    ingested: int
    message: str = "Documents ingested successfully"


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post(
    "/search",
    response_model=SearchResponseBody,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        502: {"model": ErrorResponse, "description": "Provider error"},
    },
)
async def search(
    body: SearchRequestBody,
    request: Request,
    rag: RAGPipeline = Depends(get_rag_pipeline),
) -> SearchResponseBody | StreamingResponse:
    """Search the knowledge base and generate an answer using RAG.

    Embeds the query, retrieves relevant documents from the vector store,
    augments the LLM prompt with context, and generates an answer.
    """
    rag_request = RAGRequest(
        query=body.query,
        model=body.model,
        embedding_model=body.embedding_model,
        top_k=body.top_k,
        metadata_filter=body.metadata_filter,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        stream=body.stream,
        system_prompt=body.system_prompt,
    )

    if body.stream:
        return await _stream_search(rag_request, request, rag)

    result = await rag.query(rag_request)

    return SearchResponseBody(
        answer=result.answer,
        sources=[
            SourceBody(id=s.id, text=s.text, score=s.score, metadata=s.metadata)
            for s in result.sources
        ],
        model=result.model,
        usage=result.usage,
    )


@router.post(
    "/search/ingest",
    response_model=IngestResponseBody,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def ingest_documents(
    body: IngestRequestBody,
    rag: RAGPipeline = Depends(get_rag_pipeline),
) -> IngestResponseBody:
    """Ingest documents into the vector store for later retrieval.

    Each document is embedded and stored. Documents can then be
    retrieved via the /v1/search endpoint.
    """
    await rag.ingest_batch(
        ids=[d.id for d in body.documents],
        texts=[d.text for d in body.documents],
        embedding_model=body.embedding_model,
        metadatas=[d.metadata for d in body.documents],
    )

    return IngestResponseBody(ingested=len(body.documents))


# ── Streaming helper ─────────────────────────────────────────────────────────


async def _stream_search(
    rag_request: RAGRequest,
    request: Request,
    rag: RAGPipeline,
) -> StreamingResponse:
    """Stream the RAG answer as SSE, prefixed with a sources event."""
    import json

    sources, stream_iter = await rag.stream_query(rag_request)

    async def event_generator():
        # First, emit sources as a custom SSE event
        sources_data = [
            {"id": s.id, "text": s.text, "score": s.score, "metadata": s.metadata}
            for s in sources
        ]
        yield f"event: sources\ndata: {json.dumps(sources_data)}\n\n"

        # Then stream the LLM response chunks
        try:
            async for chunk in stream_iter:
                if await request.is_disconnected():
                    break
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.exception("search.stream.error")
            from src.core.schemas import ErrorDetail, ErrorResponse as ErrResp

            error = ErrResp(
                error=ErrorDetail(
                    message=str(exc), type="server_error", code="stream_error"
                )
            )
            yield f"data: {error.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
