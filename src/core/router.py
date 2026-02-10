"""Model router: registry + routing policy + provider lifecycle.

The ModelRouter owns the mapping from model names to provider instances
and handles request routing, alias resolution, and default model selection.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import structlog

from src.config import ModelConfig, Settings
from src.core.provider import Provider
from src.core.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    EmbeddingRequest,
    EmbeddingResponse,
)

logger = structlog.get_logger()


class ModelNotFoundError(Exception):
    """Raised when a requested model is not in the registry."""

    def __init__(self, model: str, available: list[str]) -> None:
        self.model = model
        self.available = available
        super().__init__(f"Model '{model}' not found. Available: {available}")


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
        )

    def resolve(self, model_name: str) -> tuple[ModelConfig, Provider]:
        """Resolve a model name to its config and provider.

        Falls back to default model if model_name is empty.

        Raises:
            ModelNotFoundError: If model is not registered.
        """
        # Use default if no model specified
        if not model_name and self._default_model:
            model_name = self._default_model

        if model_name not in self._entries:
            raise ModelNotFoundError(model_name, list(self._entries.keys()))

        config = self._entries[model_name]
        provider = self._providers[config.provider]
        return config, provider

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
        3. Route requests based on model name.
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

    # ── Routing ──────────────────────────────────────────────────────────

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Route a non-streaming chat request to the appropriate provider."""
        _config, provider = self._registry.resolve(request.model)
        logger.debug("router.generate", model=request.model, provider=provider.provider_name)
        return await provider.generate(request)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Route a streaming chat request to the appropriate provider."""
        _config, provider = self._registry.resolve(request.model)
        logger.debug("router.stream", model=request.model, provider=provider.provider_name)
        async for chunk in provider.stream(request):
            yield chunk

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Route an embedding request to the appropriate provider."""
        _config, provider = self._registry.resolve(request.model)
        return await provider.embed(request)

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
    def providers(self) -> dict[str, Provider]:
        return self._registry.providers

    @property
    def model_names(self) -> list[str]:
        return self._registry.model_names

    def get_provider_health(self) -> dict[str, Any]:
        """Aggregate health from all providers."""
        return {name: "registered" for name in self._registry.providers}
