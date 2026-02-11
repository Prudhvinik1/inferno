"""Local LLM provider using llama-cpp-python for CPU/GPU inference."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import structlog

from src.config import LocalProviderSettings
from src.core.provider import Provider
from src.core.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    Choice,
    DeltaContent,
    Message,
    Role,
    StreamChoice,
    Usage,
)

logger = structlog.get_logger()


def _messages_to_prompt(messages: list[Message]) -> str:
    """Convert a list of chat messages to a single text prompt.

    Uses a simple ChatML-style format that works well with most GGUF models.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.role.value
        content = msg.content or ""
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    # Add assistant start tag to prompt the model to respond
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


class LocalLLMProvider(Provider):
    """Provider backed by llama-cpp-python for local GGUF model inference.

    Supports CPU-first inference on Apple Silicon with optional GPU offload.
    The model is loaded on startup() and released on shutdown().
    """

    def __init__(self, settings: LocalProviderSettings) -> None:
        self._settings = settings
        self._llama: Any = None  # Will be llama_cpp.Llama once loaded
        self._model_path: Path | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """Load the GGUF model file.

        If the model file does not exist, the provider starts in degraded
        mode and returns a helpful error on inference requests.
        """
        # Find all .gguf files in the model directory
        model_dir = self._settings.model_dir
        if not model_dir.exists():
            logger.warning("local.model_dir_missing", path=str(model_dir))
            return

        gguf_files = list(model_dir.glob("*.gguf"))
        if not gguf_files:
            logger.warning("local.no_gguf_files", path=str(model_dir))
            return

        # Use the first GGUF file found (or a specific one if configured)
        self._model_path = gguf_files[0]

        try:
            from llama_cpp import Llama

            self._llama = Llama(
                model_path=str(self._model_path),
                n_ctx=self._settings.n_ctx,
                n_threads=self._settings.n_threads,
                n_gpu_layers=self._settings.n_gpu_layers,
                verbose=self._settings.verbose,
            )
            logger.info(
                "local.model_loaded",
                path=str(self._model_path),
                n_ctx=self._settings.n_ctx,
                n_threads=self._settings.n_threads,
                n_gpu_layers=self._settings.n_gpu_layers,
            )
        except ImportError:
            logger.error(
                "local.llama_cpp_not_installed",
                hint="Install with: pip install llama-cpp-python",
            )
        except Exception:
            logger.exception("local.model_load_failed", path=str(self._model_path))

    async def shutdown(self) -> None:
        """Release the loaded model."""
        if self._llama is not None:
            del self._llama
            self._llama = None
            logger.info("local.model_unloaded")

    # ── Inference ────────────────────────────────────────────────────────

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Run a non-streaming chat completion against the local model."""
        self._ensure_loaded()

        prompt = _messages_to_prompt(request.messages)
        stop = request.stop if isinstance(request.stop, list) else (
            [request.stop] if request.stop else ["<|im_end|>"]
        )

        result = self._llama(
            prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=stop,
        )

        text = result["choices"][0]["text"]
        usage = result.get("usage", {})
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        return ChatResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role=Role.ASSISTANT, content=text.strip()),
                    finish_reason=result["choices"][0].get("finish_reason", "stop"),
                )
            ],
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Run a streaming chat completion, yielding token-by-token chunks."""
        self._ensure_loaded()

        prompt = _messages_to_prompt(request.messages)
        stop = request.stop if isinstance(request.stop, list) else (
            [request.stop] if request.stop else ["<|im_end|>"]
        )
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Send initial chunk with role
        yield ChatStreamChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaContent(role=Role.ASSISTANT),
                )
            ],
        )

        # Stream token by token
        stream_iter = self._llama(
            prompt,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=stop,
            stream=True,
        )

        for chunk in stream_iter:
            token = chunk["choices"][0].get("text", "")
            finish_reason = chunk["choices"][0].get("finish_reason")

            yield ChatStreamChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaContent(content=token if token else None),
                        finish_reason=finish_reason,
                    )
                ],
            )

            if finish_reason:
                break

    # ── Health ───────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        if self._llama is not None:
            return {
                "status": "ok",
                "model": str(self._model_path),
                "n_ctx": self._settings.n_ctx,
            }
        return {
            "status": "degraded",
            "reason": "no model loaded",
            "model_dir": str(self._settings.model_dir),
        }

    # ── Metadata ─────────────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "local"

    # ── Internal ─────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Raise if the model is not loaded."""
        if self._llama is None:
            raise RuntimeError(
                f"Local model not loaded. Place a .gguf file in '{self._settings.model_dir}/' "
                f"and restart the server. Install llama-cpp-python if missing."
            )
