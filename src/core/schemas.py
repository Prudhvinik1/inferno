"""Shared request / response schemas used across providers and API layer."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Chat types ───────────────────────────────────────────────────────────────

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolFunction(BaseModel):
    """Function definition within a tool."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    """Tool available for the model to call."""

    type: Literal["function"] = "function"
    function: ToolFunction


class ToolCallFunction(BaseModel):
    """A function call made by the model."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call within an assistant message."""

    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


# ── Request ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Unified chat completion request (OpenAI-compatible shape)."""

    model: str
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | str | None = None
    tools: list[Tool] | None = None
    tool_choice: Literal["auto", "none"] | str | None = None


# ── Response ─────────────────────────────────────────────────────────────────

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str | None = "stop"


class ChatResponse(BaseModel):
    """Non-streaming chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


# ── Streaming events ────────────────────────────────────────────────────────

class DeltaContent(BaseModel):
    """Content delta in a streaming chunk."""

    role: Role | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: str | None = None


class ChatStreamChunk(BaseModel):
    """A single chunk in a streaming chat completion."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


# ── Embeddings ───────────────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    """Embedding request."""

    model: str
    input: str | list[str]


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int = 0
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    """Embedding response."""

    object: str = "list"
    model: str
    data: list[EmbeddingData]
    usage: Usage = Field(default_factory=Usage)


# ── Errors ───────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """Unified error response (matches OpenAI error shape)."""

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# Resolve forward references
Message.model_rebuild()
