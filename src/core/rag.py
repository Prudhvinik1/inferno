"""Retrieval-augmented generation (RAG) pipeline.

Flow: query -> embed -> retrieve top-k from vector store -> augment
system prompt with context -> stream/generate via LLM provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.core.router import ModelRouter
from src.core.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    EmbeddingRequest,
    Message,
    Role,
)
from src.store.vector import SearchResult, VectorStore

logger = structlog.get_logger()


# ── RAG request / response types ────────────────────────────────────────────


@dataclass
class RAGRequest:
    """A search/RAG query."""

    query: str
    model: str = ""
    embedding_model: str = ""
    top_k: int = 5
    metadata_filter: dict[str, Any] | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False
    system_prompt: str = ""


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""

    answer: str
    sources: list[RAGSource] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class RAGSource:
    """A source document used to generate the answer."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ── RAG pipeline ─────────────────────────────────────────────────────────────


class RAGPipeline:
    """Orchestrates retrieval-augmented generation.

    1. Embed the user query via the model router.
    2. Search the vector store for relevant documents.
    3. Build an augmented prompt with retrieved context.
    4. Generate a response via the model router.
    """

    def __init__(
        self,
        router: ModelRouter,
        vector_store: VectorStore,
    ) -> None:
        self._router = router
        self._store = vector_store

    async def query(self, request: RAGRequest) -> RAGResponse:
        """Run the full RAG pipeline (non-streaming)."""
        # 1. Embed the query
        query_embedding = await self._embed_query(request.query, request.embedding_model)

        # 2. Retrieve relevant documents
        results = await self._store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )

        logger.info("rag.retrieved", query=request.query[:80], results=len(results))

        # 3. Build augmented chat request
        chat_request = self._build_chat_request(request, results)

        # 4. Generate response
        response = await self._router.generate(chat_request)

        answer = response.choices[0].message.content or "" if response.choices else ""

        return RAGResponse(
            answer=answer,
            sources=[
                RAGSource(
                    id=r.id, text=r.text, score=r.score, metadata=r.metadata
                )
                for r in results
            ],
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        )

    async def stream_query(
        self, request: RAGRequest
    ) -> tuple[list[RAGSource], AsyncIterator[ChatStreamChunk]]:
        """Run the RAG pipeline with streaming generation.

        Returns the sources immediately and an async iterator of stream chunks.
        """
        # 1. Embed the query
        query_embedding = await self._embed_query(request.query, request.embedding_model)

        # 2. Retrieve
        results = await self._store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )

        logger.info("rag.stream_retrieved", query=request.query[:80], results=len(results))

        # 3. Build augmented request
        chat_request = self._build_chat_request(request, results)
        chat_request.stream = True

        # 4. Return sources + stream iterator
        sources = [
            RAGSource(id=r.id, text=r.text, score=r.score, metadata=r.metadata)
            for r in results
        ]
        return sources, self._router.stream(chat_request)

    # ── Ingest documents into the vector store ───────────────────────────

    async def ingest(
        self,
        id: str,
        text: str,
        embedding_model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Embed a document and add it to the vector store."""
        embedding = await self._embed_query(text, embedding_model)
        await self._store.add(id=id, text=text, embedding=embedding, metadata=metadata)
        logger.debug("rag.ingested", id=id, text_len=len(text))

    async def ingest_batch(
        self,
        ids: list[str],
        texts: list[str],
        embedding_model: str = "",
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Embed and add multiple documents to the vector store."""
        embeddings = []
        for text in texts:
            emb = await self._embed_query(text, embedding_model)
            embeddings.append(emb)

        await self._store.add_batch(
            ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas
        )
        logger.info("rag.batch_ingested", count=len(ids))

    # ── Internal helpers ─────────────────────────────────────────────────

    async def _embed_query(self, text: str, model: str) -> list[float]:
        """Embed a single text string via the model router."""
        request = EmbeddingRequest(model=model, input=text)
        response = await self._router.embed(request)
        return response.data[0].embedding

    def _build_chat_request(
        self, request: RAGRequest, results: list[SearchResult]
    ) -> ChatRequest:
        """Build a ChatRequest with retrieved context injected into the system prompt."""
        # Format retrieved context
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[{i}] (score: {r.score:.3f})\n{r.text}")

        context_block = "\n\n---\n\n".join(context_parts)

        system_content = (
            f"{request.system_prompt}\n\n" if request.system_prompt else ""
        )
        system_content += (
            "Use the following retrieved context to answer the user's question. "
            "If the context doesn't contain relevant information, say so. "
            "Cite sources using [n] notation.\n\n"
            f"## Retrieved Context\n\n{context_block}"
        )

        messages = [
            Message(role=Role.SYSTEM, content=system_content),
            Message(role=Role.USER, content=request.query),
        ]

        return ChatRequest(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
        )
