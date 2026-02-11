"""FastAPI application entrypoint."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.config import Settings, get_settings
from src.core.rag import RAGPipeline
from src.core.router import ModelRouter
from src.store.vector import VectorStore, create_vector_store

logger = structlog.get_logger()

# ── Shared application state ────────────────────────────────────────────────

_startup_time: float = 0.0
_model_router: ModelRouter | None = None
_vector_store: VectorStore | None = None
_rag_pipeline: RAGPipeline | None = None


def get_router() -> ModelRouter:
    """Return the initialised model router. Raises if called before startup."""
    if _model_router is None:
        raise RuntimeError("ModelRouter not initialised – app not started")
    return _model_router


def get_rag_pipeline() -> RAGPipeline:
    """Return the initialised RAG pipeline. Raises if called before startup."""
    if _rag_pipeline is None:
        raise RuntimeError("RAGPipeline not initialised – app not started")
    return _rag_pipeline


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the application."""
    global _startup_time, _model_router, _vector_store, _rag_pipeline

    settings: Settings = get_settings()
    _startup_time = time.time()

    # Initialise model router (registers providers, loads local models)
    _model_router = ModelRouter(settings)
    await _model_router.startup()

    # Initialise vector store and RAG pipeline
    persist_dir = Path(settings.vector_store.persist_dir) if settings.vector_store.persist_dir else None
    _vector_store = create_vector_store(
        backend=settings.vector_store.backend,
        persist_dir=persist_dir,
        dimension=settings.vector_store.dimension,
    )
    await _vector_store.load()
    _rag_pipeline = RAGPipeline(router=_model_router, vector_store=_vector_store)

    logger.info("app.started", models=len(settings.models), vector_backend=settings.vector_store.backend)

    yield

    # Persist vector store before shutdown
    if _vector_store:
        await _vector_store.persist()

    await _model_router.shutdown()
    _model_router = None
    _vector_store = None
    _rag_pipeline = None
    logger.info("app.stopped")


# ── App factory ──────────────────────────────────────────────────────────────

def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="LLM Serving API",
        version="0.1.0",
        description="Minimal LLM serving stack with voice and search",
        lifespan=lifespan,
    )

    # ── Health endpoint ──────────────────────────────────────────────────

    @app.get("/health", tags=["system"])
    async def health() -> JSONResponse:
        """Health check returning status and uptime."""
        uptime = time.time() - _startup_time if _startup_time else 0.0
        router = _model_router

        provider_status: dict[str, Any] = {}
        if router:
            for name, provider in router.providers.items():
                provider_status[name] = await provider.health()

        return JSONResponse(
            content={
                "status": "ok",
                "uptime_seconds": round(uptime, 2),
                "providers": provider_status,
            }
        )

    # ── Error handlers ──────────────────────────────────────────────────

    from src.infra.error_handler import install_error_handlers

    install_error_handlers(app)

    # ── Register routers ────────────────────────────────────────────────

    from src.api.chat import router as chat_router
    from src.api.embeddings import router as embeddings_router
    from src.api.search import router as search_router

    app.include_router(chat_router, prefix="/v1")
    app.include_router(embeddings_router, prefix="/v1")
    app.include_router(search_router, prefix="/v1")

    return app


# ── Uvicorn entrypoint ──────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.server.log_level,
        reload=settings.debug,
    )
