"""Model router: registry + routing policy + provider lifecycle.

The ModelRouter owns the mapping from model names to provider instances
and handles request routing, fallback on provider failure, and default
model selection.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import structlog

from src.config import ModelConfig, Settings
from src.core.exceptions import ModelNotFoundError, ProviderError
from src.core.provider import Provider
from src.core.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    EmbeddingRequest,
    EmbeddingResponse,
)

logger = structlog.get_logger()

MAX_FALLBACK_DEPTH = 5


class ModelRegistry:
    """Maps model names/aliases to their configuration and provider instances.

    The registry is built from the ``models`` list in Settings and stores
    the provider instance once created by the router.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ModelConfig] = {}
        self._providers: dict[str, Provider] = {}  # provider_type -> Provider instance
        self._model_to_provider: dict[str, str] = {}  # model_name -> provider_type
        self._default_model: str | None = None

    def register(self, config: ModelConfig, provider: Provider) -> None:
        """Register a model with its config and provider."""
        self._entries[config.name] = config
        self._model_to_provider[config.name] = config.provider

        # Store provider instance (shared across models of same type)
        if config.provider not in self._providers:
            self._providers[config.provider] = provider

        if config.is_default:
            self._default_model = config.name

        logger.debug(
            "registry.model_registered",
            model=config.name,
            provider=config.provider,
            model_id=config.model_id,
            is_default=config.is_default,
            fallback=config.fallback_model,
        )

    def resolve(self, model_name: str) -> tuple[ModelConfig, Provider]:
        """Resolve a model name to its config and provider.

        Falls back to default model if model_name is empty.

        Raises:
            ModelNotFoundError: If model is not registered.
        """
        if not model_name and self._default_model:
            model_name = self._default_model

        if model_name not in self._entries:
            raise ModelNotFoundError(model_name, list(self._entries.keys()))

        config = self._entries[model_name]
        provider = self._providers[config.provider]
        return config, provider

    def get_fallback_chain(self, model_name: str) -> list[tuple[ModelConfig, Provider]]:
        """Build the ordered fallback chain for a model.

        Returns a list of (config, provider) starting with the requested
        model and continuing through each declared ``fallback_model``.
        Stops at ``MAX_FALLBACK_DEPTH`` or if a cycle is detected.
        """
        chain: list[tuple[ModelConfig, Provider]] = []
        seen: set[str] = set()
        current = model_name

        while current and current not in seen and len(chain) < MAX_FALLBACK_DEPTH:
            if current not in self._entries:
                break
            seen.add(current)
            config = self._entries[current]
            provider = self._providers[config.provider]
            chain.append((config, provider))
            current = config.fallback_model  # type: ignore[assignment]

        return chain

    @property
    def default_model(self) -> str | None:
        return self._default_model

    @property
    def model_names(self) -> list[str]:
        return list(self._entries.keys())

    @property
    def providers(self) -> dict[str, Provider]:
        """All registered provider instances keyed by provider type."""
        return dict(self._providers)


class ModelRouter:
    """Routes chat/embedding requests to the appropriate provider.

    Responsibilities:
        1. Build the model registry from settings.
        2. Instantiate and manage provider lifecycles.
        3. Route requests based on model name with fallback on failure.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._registry = ModelRegistry()

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """Create providers and register models."""
        providers_cache: dict[str, Provider] = {}

        for model_cfg in self._settings.models:
            provider_type = model_cfg.provider

            # Reuse provider instance for same provider type
            if provider_type not in providers_cache:
                provider = self._create_provider(provider_type)
                await provider.startup()
                providers_cache[provider_type] = provider

            self._registry.register(model_cfg, providers_cache[provider_type])

        logger.info(
            "router.started",
            models=self._registry.model_names,
            default=self._registry.default_model,
        )

    async def shutdown(self) -> None:
        """Shut down all providers."""
        for name, provider in self._registry.providers.items():
            try:
                await provider.shutdown()
                logger.info("router.provider_stopped", provider=name)
            except Exception:
                logger.exception("router.provider_shutdown_error", provider=name)

    # ── Routing with fallback ────────────────────────────────────────────

    def _resolve_model_name(self, model: str) -> str:
        """Normalise model name, falling back to default when empty."""
        if not model and self._registry.default_model:
            return self._registry.default_model
        return model

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Route a non-streaming chat request with fallback on failure."""
        model_name = self._resolve_model_name(request.model)
        chain = self._registry.get_fallback_chain(model_name)

        if not chain:
            raise ModelNotFoundError(model_name, self._registry.model_names)

        last_exc: Exception | None = None
        for config, provider in chain:
            try:
                routed = request.model_copy(update={"model": config.model_id})
                logger.debug(
                    "router.generate",
                    model=config.name,
                    provider=provider.provider_name,
                    is_fallback=config.name != model_name,
                )
                response = await provider.generate(routed)
                return response.model_copy(update={"model": config.name})
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "router.generate_failed",
                    model=config.name,
                    provider=provider.provider_name,
                    error=str(exc),
                    has_fallback=config.fallback_model is not None,
                )

        raise ProviderError(
            message=f"All providers failed for model '{model_name}': {last_exc}",
            provider=model_name,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Route a streaming chat request with fallback on failure.

        Fallback is attempted only if the provider raises before yielding
        the first chunk. Once streaming has started, errors propagate
        directly to the caller.
        """
        model_name = self._resolve_model_name(request.model)
        chain = self._registry.get_fallback_chain(model_name)

        if not chain:
            raise ModelNotFoundError(model_name, self._registry.model_names)

        last_exc: Exception | None = None
        for config, provider in chain:
            try:
                routed = request.model_copy(update={"model": config.model_id})
                logger.debug(
                    "router.stream",
                    model=config.name,
                    provider=provider.provider_name,
                    is_fallback=config.name != model_name,
                )
                async for chunk in provider.stream(routed):
                    yield chunk.model_copy(update={"model": config.name})
                # Stream completed successfully — stop fallback chain.
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "router.stream_failed",
                    model=config.name,
                    provider=provider.provider_name,
                    error=str(exc),
                    has_fallback=config.fallback_model is not None,
                )

        raise ProviderError(
            message=f"All providers failed for model '{model_name}': {last_exc}",
            provider=model_name,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Route an embedding request with fallback on failure."""
        model_name = self._resolve_model_name(request.model)
        chain = self._registry.get_fallback_chain(model_name)

        if not chain:
            raise ModelNotFoundError(model_name, self._registry.model_names)

        last_exc: Exception | None = None
        for config, provider in chain:
            try:
                routed = request.model_copy(update={"model": config.model_id})
                response = await provider.embed(routed)
                return response.model_copy(update={"model": config.name})
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "router.embed_failed",
                    model=config.name,
                    provider=provider.provider_name,
                    error=str(exc),
                    has_fallback=config.fallback_model is not None,
                )

        raise ProviderError(
            message=f"All providers failed for model '{model_name}': {last_exc}",
            provider=model_name,
        )

    # ── Provider factory ─────────────────────────────────────────────────

    def _create_provider(self, provider_type: str) -> Provider:
        """Instantiate a provider by type string.

        Imports are deferred so we don't pull in heavy dependencies
        (llama_cpp, httpx) at import time.
        """
        if provider_type == "local":
            from src.providers.local_llm import LocalLLMProvider

            return LocalLLMProvider(self._settings.local)

        if provider_type == "openai":
            from src.providers.external_api import OpenAIProvider

            return OpenAIProvider(self._settings.openai)

        if provider_type == "anthropic":
            from src.providers.external_api import AnthropicProvider

            return AnthropicProvider(self._settings.anthropic)

        raise ValueError(f"Unknown provider type: {provider_type}")

    # ── Public accessors ─────────────────────────────────────────────────

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def providers(self) -> dict[str, Provider]:
        return self._registry.providers

    @property
    def model_names(self) -> list[str]:
        return self._registry.model_names

    def get_provider_health(self) -> dict[str, Any]:
        """Aggregate health from all providers."""
        return {name: "registered" for name in self._registry.providers}
