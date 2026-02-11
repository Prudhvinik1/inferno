"""Tests for ModelRouter routing policy and fallback behaviour."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import pytest

from src.config import ModelConfig
from src.core.provider import Provider
from src.core.router import (
    MAX_FALLBACK_DEPTH,
    ModelNotFoundError,
    ModelRegistry,
    ModelRouter,
    ProviderError,
)
from src.core.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Choice,
    DeltaContent,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    Message,
    Role,
    StreamChoice,
    Usage,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


class FakeProvider(Provider):
    """A provider that records calls and can be configured to fail."""

    def __init__(self, name: str = "fake", *, fail: bool = False) -> None:
        self._name = name
        self._fail = fail
        self.generate_calls: list[ChatRequest] = []
        self.stream_calls: list[ChatRequest] = []
        self.embed_calls: list[EmbeddingRequest] = []

    @property
    def provider_name(self) -> str:
        return self._name

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.generate_calls.append(request)
        if self._fail:
            raise RuntimeError(f"{self._name} provider failed")
        return ChatResponse(
            id="resp-1",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    message=Message(role=Role.ASSISTANT, content=f"hello from {self._name}"),
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        self.stream_calls.append(request)
        if self._fail:
            raise RuntimeError(f"{self._name} provider failed")
        yield ChatStreamChunk(
            id="chunk-1",
            created=int(time.time()),
            model=request.model,
            choices=[StreamChoice(delta=DeltaContent(content=f"chunk from {self._name}"))],
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.embed_calls.append(request)
        if self._fail:
            raise RuntimeError(f"{self._name} provider failed")
        return EmbeddingResponse(
            model=request.model,
            data=[EmbeddingData(embedding=[0.1, 0.2, 0.3])],
            usage=Usage(prompt_tokens=1, completion_tokens=0, total_tokens=1),
        )


def _make_config(
    name: str,
    provider: str = "openai",
    model_id: str = "gpt-4o",
    is_default: bool = False,
    fallback_model: str | None = None,
) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider=provider,
        model_id=model_id,
        is_default=is_default,
        fallback_model=fallback_model,
    )


def _chat_request(model: str = "primary") -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=[Message(role=Role.USER, content="hi")],
    )


def _embed_request(model: str = "primary") -> EmbeddingRequest:
    return EmbeddingRequest(model=model, input="hello")


# ── ModelRegistry tests ──────────────────────────────────────────────────────


class TestModelRegistry:
    def test_register_and_resolve(self) -> None:
        reg = ModelRegistry()
        provider = FakeProvider("openai")
        cfg = _make_config("gpt-4o", provider="openai")
        reg.register(cfg, provider)

        resolved_cfg, resolved_prov = reg.resolve("gpt-4o")
        assert resolved_cfg.name == "gpt-4o"
        assert resolved_prov is provider

    def test_resolve_unknown_model_raises(self) -> None:
        reg = ModelRegistry()
        with pytest.raises(ModelNotFoundError) as exc_info:
            reg.resolve("nope")
        assert exc_info.value.model == "nope"

    def test_resolve_empty_uses_default(self) -> None:
        reg = ModelRegistry()
        provider = FakeProvider("openai")
        cfg = _make_config("gpt-4o", provider="openai", is_default=True)
        reg.register(cfg, provider)

        resolved_cfg, _ = reg.resolve("")
        assert resolved_cfg.name == "gpt-4o"

    def test_fallback_chain_single(self) -> None:
        reg = ModelRegistry()
        provider = FakeProvider("openai")
        reg.register(_make_config("model-a", provider="openai"), provider)

        chain = reg.get_fallback_chain("model-a")
        assert len(chain) == 1
        assert chain[0][0].name == "model-a"

    def test_fallback_chain_multi(self) -> None:
        reg = ModelRegistry()
        prov_a = FakeProvider("openai")
        prov_b = FakeProvider("anthropic")
        reg.register(
            _make_config("model-a", provider="openai", fallback_model="model-b"), prov_a
        )
        reg.register(_make_config("model-b", provider="anthropic"), prov_b)

        chain = reg.get_fallback_chain("model-a")
        assert len(chain) == 2
        assert chain[0][0].name == "model-a"
        assert chain[1][0].name == "model-b"

    def test_fallback_chain_cycle_detection(self) -> None:
        reg = ModelRegistry()
        provider = FakeProvider("openai")
        reg.register(
            _make_config("a", provider="openai", fallback_model="b"), provider
        )
        reg.register(
            _make_config("b", provider="openai", fallback_model="a"), provider
        )

        chain = reg.get_fallback_chain("a")
        assert len(chain) == 2  # a -> b, then stops (b -> a would be a cycle)

    def test_fallback_chain_depth_limit(self) -> None:
        reg = ModelRegistry()
        provider = FakeProvider("openai")
        # Build chain longer than MAX_FALLBACK_DEPTH
        for i in range(MAX_FALLBACK_DEPTH + 3):
            fb = f"m{i + 1}" if i < MAX_FALLBACK_DEPTH + 2 else None
            reg.register(
                _make_config(f"m{i}", provider="openai", fallback_model=fb), provider
            )

        chain = reg.get_fallback_chain("m0")
        assert len(chain) == MAX_FALLBACK_DEPTH

    def test_fallback_chain_missing_fallback_stops(self) -> None:
        reg = ModelRegistry()
        provider = FakeProvider("openai")
        reg.register(
            _make_config("a", provider="openai", fallback_model="nonexistent"), provider
        )

        chain = reg.get_fallback_chain("a")
        assert len(chain) == 1  # stops because "nonexistent" is not registered

    def test_model_names_and_providers(self) -> None:
        reg = ModelRegistry()
        prov = FakeProvider("openai")
        reg.register(_make_config("m1", provider="openai"), prov)
        reg.register(_make_config("m2", provider="openai"), prov)

        assert set(reg.model_names) == {"m1", "m2"}
        assert "openai" in reg.providers


# ── ModelRouter tests (generate) ─────────────────────────────────────────────


class TestModelRouterGenerate:
    @pytest.fixture()
    def setup_router(self) -> tuple[ModelRouter, FakeProvider, FakeProvider]:
        """Create a router with two providers, primary falling back to secondary."""
        from src.config import Settings

        settings = Settings(
            models=[
                ModelConfig(
                    name="primary",
                    provider="openai",
                    model_id="gpt-4o",
                    is_default=True,
                    fallback_model="secondary",
                ),
                ModelConfig(
                    name="secondary",
                    provider="anthropic",
                    model_id="claude-3",
                ),
            ]
        )
        router = ModelRouter(settings)

        primary = FakeProvider("openai")
        secondary = FakeProvider("anthropic")
        router._registry.register(settings.models[0], primary)
        router._registry.register(settings.models[1], secondary)

        return router, primary, secondary

    @pytest.mark.asyncio
    async def test_routes_to_correct_provider(
        self, setup_router: tuple[ModelRouter, FakeProvider, FakeProvider]
    ) -> None:
        router, primary, secondary = setup_router
        resp = await router.generate(_chat_request("primary"))

        assert resp.model == "primary"
        assert len(primary.generate_calls) == 1
        # Request sent to provider uses model_id, not model name
        assert primary.generate_calls[0].model == "gpt-4o"
        assert len(secondary.generate_calls) == 0

    @pytest.mark.asyncio
    async def test_fallback_on_failure(
        self, setup_router: tuple[ModelRouter, FakeProvider, FakeProvider]
    ) -> None:
        router, primary, secondary = setup_router
        primary._fail = True

        resp = await router.generate(_chat_request("primary"))
        assert resp.model == "secondary"
        assert len(primary.generate_calls) == 1
        assert len(secondary.generate_calls) == 1

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises(
        self, setup_router: tuple[ModelRouter, FakeProvider, FakeProvider]
    ) -> None:
        router, primary, secondary = setup_router
        primary._fail = True
        secondary._fail = True

        with pytest.raises(ProviderError) as exc_info:
            await router.generate(_chat_request("primary"))
        assert exc_info.value.model == "primary"

    @pytest.mark.asyncio
    async def test_unknown_model_raises(
        self, setup_router: tuple[ModelRouter, FakeProvider, FakeProvider]
    ) -> None:
        router, _, _ = setup_router
        with pytest.raises(ModelNotFoundError):
            await router.generate(_chat_request("nonexistent"))

    @pytest.mark.asyncio
    async def test_default_model_resolution(
        self, setup_router: tuple[ModelRouter, FakeProvider, FakeProvider]
    ) -> None:
        router, primary, _ = setup_router
        resp = await router.generate(_chat_request(""))
        assert resp.model == "primary"
        assert len(primary.generate_calls) == 1


# ── ModelRouter tests (stream) ───────────────────────────────────────────────


class TestModelRouterStream:
    @pytest.fixture()
    def setup_router(self) -> tuple[ModelRouter, FakeProvider, FakeProvider]:
        from src.config import Settings

        settings = Settings(
            models=[
                ModelConfig(
                    name="primary",
                    provider="openai",
                    model_id="gpt-4o",
                    is_default=True,
                    fallback_model="secondary",
                ),
                ModelConfig(
                    name="secondary",
                    provider="anthropic",
                    model_id="claude-3",
                ),
            ]
        )
        router = ModelRouter(settings)

        primary = FakeProvider("openai")
        secondary = FakeProvider("anthropic")
        router._registry.register(settings.models[0], primary)
        router._registry.register(settings.models[1], secondary)

        return router, primary, secondary

    @pytest.mark.asyncio
    async def test_stream_routes_correctly(
        self, setup_router: tuple[ModelRouter, FakeProvider, FakeProvider]
    ) -> None:
        router, primary, _ = setup_router
        chunks = [c async for c in router.stream(_chat_request("primary"))]
        assert len(chunks) == 1
        assert chunks[0].model == "primary"

    @pytest.mark.asyncio
    async def test_stream_fallback_on_failure(
        self, setup_router: tuple[ModelRouter, FakeProvider, FakeProvider]
    ) -> None:
        router, primary, secondary = setup_router
        primary._fail = True

        chunks = [c async for c in router.stream(_chat_request("primary"))]
        assert len(chunks) == 1
        assert chunks[0].model == "secondary"

    @pytest.mark.asyncio
    async def test_stream_all_fail_raises(
        self, setup_router: tuple[ModelRouter, FakeProvider, FakeProvider]
    ) -> None:
        router, primary, secondary = setup_router
        primary._fail = True
        secondary._fail = True

        with pytest.raises(ProviderError):
            async for _ in router.stream(_chat_request("primary")):
                pass


# ── ModelRouter tests (embed) ────────────────────────────────────────────────


class TestModelRouterEmbed:
    @pytest.fixture()
    def setup_router(self) -> tuple[ModelRouter, FakeProvider, FakeProvider]:
        from src.config import Settings

        settings = Settings(
            models=[
                ModelConfig(
                    name="primary",
                    provider="openai",
                    model_id="gpt-4o",
                    is_default=True,
                    fallback_model="secondary",
                ),
                ModelConfig(
                    name="secondary",
                    provider="anthropic",
                    model_id="claude-3",
                ),
            ]
        )
        router = ModelRouter(settings)

        primary = FakeProvider("openai")
        secondary = FakeProvider("anthropic")
        router._registry.register(settings.models[0], primary)
        router._registry.register(settings.models[1], secondary)

        return router, primary, secondary

    @pytest.mark.asyncio
    async def test_embed_routes_correctly(
        self, setup_router: tuple[ModelRouter, FakeProvider, FakeProvider]
    ) -> None:
        router, primary, _ = setup_router
        resp = await router.embed(_embed_request("primary"))
        assert resp.model == "primary"
        assert len(primary.embed_calls) == 1

    @pytest.mark.asyncio
    async def test_embed_fallback_on_failure(
        self, setup_router: tuple[ModelRouter, FakeProvider, FakeProvider]
    ) -> None:
        router, primary, secondary = setup_router
        primary._fail = True

        resp = await router.embed(_embed_request("primary"))
        assert resp.model == "secondary"
        assert len(secondary.embed_calls) == 1


# ── No fallback configured ──────────────────────────────────────────────────


class TestNoFallback:
    @pytest.mark.asyncio
    async def test_no_fallback_raises_on_failure(self) -> None:
        from src.config import Settings

        settings = Settings(
            models=[
                ModelConfig(
                    name="solo",
                    provider="openai",
                    model_id="gpt-4o",
                    is_default=True,
                ),
            ]
        )
        router = ModelRouter(settings)
        provider = FakeProvider("openai", fail=True)
        router._registry.register(settings.models[0], provider)

        with pytest.raises(ProviderError):
            await router.generate(_chat_request("solo"))
