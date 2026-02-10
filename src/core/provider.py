"""Abstract base class for LLM providers.

Every inference backend (local llama.cpp, OpenAI API, Anthropic API)
implements this interface so the router and API layer stay provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from src.core.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    EmbeddingRequest,
    EmbeddingResponse,
)


class Provider(ABC):
    """Unified provider interface for LLM inference.

    Lifecycle:
        provider = SomeProvider(settings)
        await provider.startup()   # load models, open connections
        ...
        await provider.shutdown()  # release resources
    """

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """Initialise provider resources (load model, open HTTP pool, etc.)."""

    async def shutdown(self) -> None:
        """Release provider resources."""

    # ── Inference ────────────────────────────────────────────────────────

    @abstractmethod
    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Run a non-streaming chat completion.

        Args:
            request: Unified chat completion request.

        Returns:
            Complete chat response with usage stats.
        """
        ...

    @abstractmethod
    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Run a streaming chat completion.

        Args:
            request: Unified chat completion request (stream=True).

        Yields:
            Successive stream chunks until finish_reason is set.
        """
        ...
        # Make this an async generator so subclasses using `yield` work.
        # (Python requires at least one `yield` for the type to resolve.)
        if False:  # pragma: no cover
            yield  # type: ignore[misc]

    # ── Embeddings ───────────────────────────────────────────────────────

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for the given input.

        Default implementation raises NotImplementedError so providers that
        don't support embeddings can omit it.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")

    # ── Health ───────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Return provider health status.

        Returns:
            Dict with at least {"status": "ok" | "degraded" | "down"}.
        """
        return {"status": "ok"}

    # ── Metadata ─────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier for logging and metrics (e.g. 'local', 'openai')."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_name}>"
