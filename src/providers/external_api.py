"""External API providers for OpenAI and Anthropic.

Both providers use httpx async clients and map between their native
response formats and the unified schema defined in src.core.schemas.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from src.config import AnthropicProviderSettings, OpenAIProviderSettings
from src.core.provider import Provider
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
    ToolCall,
    ToolCallFunction,
    Usage,
)

logger = structlog.get_logger()


# ── Provider errors ──────────────────────────────────────────────────────────


class ProviderAPIError(Exception):
    """Raised when an external API returns a non-2xx response."""

    def __init__(
        self,
        provider: str,
        status_code: int,
        message: str,
        error_type: str = "api_error",
        error_code: str | None = None,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        super().__init__(f"[{provider}] {status_code}: {message}")


class ProviderAuthError(ProviderAPIError):
    """Raised on 401/403 from external API."""


class ProviderRateLimitError(ProviderAPIError):
    """Raised on 429 from external API."""


# ═══════════════════════════════════════════════════════════════════════════════
#  OpenAI Provider
# ═══════════════════════════════════════════════════════════════════════════════


class OpenAIProvider(Provider):
    """Provider for the OpenAI Chat Completions API.

    Uses httpx async client. The response shape is already OpenAI-compatible
    so mapping is mostly passthrough.
    """

    def __init__(self, settings: OpenAIProviderSettings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def startup(self) -> None:
        if not self._settings.api_key:
            logger.warning("openai.no_api_key", hint="Set OPENAI__API_KEY in .env")
            return

        self._client = httpx.AsyncClient(
            base_url=self._settings.base_url,
            headers={
                "Authorization": f"Bearer {self._settings.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._settings.timeout),
        )
        logger.info("openai.started", base_url=self._settings.base_url)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("openai.stopped")

    # ── Inference ────────────────────────────────────────────────────────

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self._ensure_client()

        payload = self._build_payload(request, stream=False)
        resp = await self._client.post("/chat/completions", json=payload)
        self._check_response(resp)

        data = resp.json()
        return self._parse_chat_response(data, request.model)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        self._ensure_client()

        payload = self._build_payload(request, stream=True)

        async with self._client.stream(
            "POST", "/chat/completions", json=payload
        ) as resp:
            self._check_response(resp)

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]  # strip "data: "
                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                yield self._parse_stream_chunk(data, request.model)

    # ── Embeddings ───────────────────────────────────────────────────────

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self._ensure_client()

        payload = {
            "model": request.model,
            "input": request.input,
        }
        resp = await self._client.post("/embeddings", json=payload)
        self._check_response(resp)

        data = resp.json()
        return EmbeddingResponse(
            model=data.get("model", request.model),
            data=[
                EmbeddingData(
                    index=item["index"],
                    embedding=item["embedding"],
                )
                for item in data["data"]
            ],
            usage=Usage(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                total_tokens=data.get("usage", {}).get("total_tokens", 0),
            ),
        )

    # ── Health ───────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        if self._client is None:
            return {"status": "degraded", "reason": "no API key configured"}
        try:
            resp = await self._client.get("/models")
            if resp.status_code == 200:
                return {"status": "ok", "base_url": self._settings.base_url}
            return {"status": "degraded", "http_status": resp.status_code}
        except Exception as exc:
            return {"status": "down", "error": str(exc)}

    @property
    def provider_name(self) -> str:
        return "openai"

    # ── Internal helpers ─────────────────────────────────────────────────

    def _ensure_client(self) -> None:
        if self._client is None:
            raise RuntimeError(
                "OpenAI provider not initialised. Set OPENAI__API_KEY in .env and restart."
            )

    def _build_payload(self, request: ChatRequest, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": m.role.value, "content": m.content}
                for m in request.messages
            ],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.tools:
            payload["tools"] = [t.model_dump() for t in request.tools]
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        return payload

    def _check_response(self, resp: httpx.Response) -> None:
        """Raise typed errors for non-2xx responses."""
        if resp.status_code >= 200 and resp.status_code < 300:
            return

        try:
            body = resp.json()
            msg = body.get("error", {}).get("message", resp.text)
            err_type = body.get("error", {}).get("type", "api_error")
            err_code = body.get("error", {}).get("code")
        except Exception:
            msg = resp.text
            err_type = "api_error"
            err_code = None

        if resp.status_code in (401, 403):
            raise ProviderAuthError("openai", resp.status_code, msg, err_type, err_code)
        if resp.status_code == 429:
            raise ProviderRateLimitError("openai", resp.status_code, msg, err_type, err_code)

        raise ProviderAPIError("openai", resp.status_code, msg, err_type, err_code)

    @staticmethod
    def _parse_chat_response(data: dict[str, Any], model: str) -> ChatResponse:
        choices = []
        for c in data.get("choices", []):
            msg = c.get("message", {})
            tool_calls = None
            if msg.get("tool_calls"):
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        function=ToolCallFunction(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                    for tc in msg["tool_calls"]
                ]

            choices.append(
                Choice(
                    index=c.get("index", 0),
                    message=Message(
                        role=Role(msg.get("role", "assistant")),
                        content=msg.get("content"),
                        tool_calls=tool_calls,
                    ),
                    finish_reason=c.get("finish_reason"),
                )
            )

        usage_data = data.get("usage", {})
        return ChatResponse(
            id=data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            created=data.get("created", int(time.time())),
            model=data.get("model", model),
            choices=choices,
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
        )

    @staticmethod
    def _parse_stream_chunk(data: dict[str, Any], model: str) -> ChatStreamChunk:
        choices = []
        for c in data.get("choices", []):
            delta = c.get("delta", {})
            tool_calls = None
            if delta.get("tool_calls"):
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", ""),
                        function=ToolCallFunction(
                            name=tc.get("function", {}).get("name", ""),
                            arguments=tc.get("function", {}).get("arguments", ""),
                        ),
                    )
                    for tc in delta["tool_calls"]
                ]

            choices.append(
                StreamChoice(
                    index=c.get("index", 0),
                    delta=DeltaContent(
                        role=Role(delta["role"]) if "role" in delta else None,
                        content=delta.get("content"),
                        tool_calls=tool_calls,
                    ),
                    finish_reason=c.get("finish_reason"),
                )
            )

        return ChatStreamChunk(
            id=data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            created=data.get("created", int(time.time())),
            model=data.get("model", model),
            choices=choices,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Anthropic Provider
# ═══════════════════════════════════════════════════════════════════════════════


class AnthropicProvider(Provider):
    """Provider for the Anthropic Messages API.

    Maps between Anthropic's message format (system as top-level param,
    content blocks) and our unified OpenAI-compatible schema.
    """

    def __init__(self, settings: AnthropicProviderSettings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def startup(self) -> None:
        if not self._settings.api_key:
            logger.warning("anthropic.no_api_key", hint="Set ANTHROPIC__API_KEY in .env")
            return

        self._client = httpx.AsyncClient(
            base_url=self._settings.base_url,
            headers={
                "x-api-key": self._settings.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self._settings.timeout),
        )
        logger.info("anthropic.started", base_url=self._settings.base_url)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("anthropic.stopped")

    # ── Inference ────────────────────────────────────────────────────────

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self._ensure_client()

        payload = self._build_payload(request, stream=False)
        resp = await self._client.post("/v1/messages", json=payload)
        self._check_response(resp)

        data = resp.json()
        return self._parse_response(data, request.model)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        self._ensure_client()

        payload = self._build_payload(request, stream=True)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Emit initial role chunk
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

        async with self._client.stream(
            "POST", "/v1/messages", json=payload
        ) as resp:
            self._check_response(resp)

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Anthropic sends different event types
                event_type = event.get("type", "")

                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        yield ChatStreamChunk(
                            id=completion_id,
                            created=created,
                            model=request.model,
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=DeltaContent(content=text),
                                )
                            ],
                        )

                elif event_type == "message_stop":
                    yield ChatStreamChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaContent(),
                                finish_reason="stop",
                            )
                        ],
                    )

    # ── Health ───────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        if self._client is None:
            return {"status": "degraded", "reason": "no API key configured"}
        return {"status": "ok", "base_url": self._settings.base_url}

    @property
    def provider_name(self) -> str:
        return "anthropic"

    # ── Internal helpers ─────────────────────────────────────────────────

    def _ensure_client(self) -> None:
        if self._client is None:
            raise RuntimeError(
                "Anthropic provider not initialised. Set ANTHROPIC__API_KEY in .env and restart."
            )

    def _build_payload(self, request: ChatRequest, stream: bool) -> dict[str, Any]:
        """Convert unified ChatRequest to Anthropic Messages API format.

        Key differences from OpenAI:
        - system message is a top-level ``system`` param, not in messages
        - messages only contain user/assistant roles
        - max_tokens is required (default to 1024)
        """
        system_text = ""
        messages = []
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                system_text += (msg.content or "") + "\n"
            else:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content or "",
                })

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()
        if request.stop:
            stop = request.stop if isinstance(request.stop, list) else [request.stop]
            payload["stop_sequences"] = stop

        # Anthropic tool use
        if request.tools:
            payload["tools"] = [
                {
                    "name": t.function.name,
                    "description": t.function.description,
                    "input_schema": t.function.parameters,
                }
                for t in request.tools
            ]

        return payload

    def _check_response(self, resp: httpx.Response) -> None:
        if resp.status_code >= 200 and resp.status_code < 300:
            return

        try:
            body = resp.json()
            msg = body.get("error", {}).get("message", resp.text)
            err_type = body.get("error", {}).get("type", "api_error")
        except Exception:
            msg = resp.text
            err_type = "api_error"

        if resp.status_code in (401, 403):
            raise ProviderAuthError("anthropic", resp.status_code, msg, err_type)
        if resp.status_code == 429:
            raise ProviderRateLimitError("anthropic", resp.status_code, msg, err_type)

        raise ProviderAPIError("anthropic", resp.status_code, msg, err_type)

    @staticmethod
    def _parse_response(data: dict[str, Any], model: str) -> ChatResponse:
        """Convert Anthropic Messages API response to unified ChatResponse."""
        # Extract text from content blocks
        content_parts = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content_parts.append(block.get("text", ""))

        text = "".join(content_parts)
        usage_data = data.get("usage", {})

        return ChatResponse(
            id=data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
            created=int(time.time()),
            model=data.get("model", model),
            choices=[
                Choice(
                    index=0,
                    message=Message(role=Role.ASSISTANT, content=text),
                    finish_reason=_anthropic_stop_reason(data.get("stop_reason")),
                )
            ],
            usage=Usage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=(
                    usage_data.get("input_tokens", 0)
                    + usage_data.get("output_tokens", 0)
                ),
            ),
        )


def _anthropic_stop_reason(reason: str | None) -> str:
    """Map Anthropic stop_reason to OpenAI finish_reason."""
    mapping = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }
    return mapping.get(reason or "", "stop")
