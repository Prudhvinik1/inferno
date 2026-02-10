"""Application configuration using pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """HTTP server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"


class ModelConfig(BaseSettings):
    """Configuration for a single model provider mapping."""

    model_config = SettingsConfigDict(extra="allow")

    name: str = Field(description="Model name exposed via API (e.g. 'gpt-4o-mini')")
    provider: Literal["local", "openai", "anthropic"] = Field(
        description="Which provider backend to use"
    )
    model_id: str = Field(
        description="Provider-specific model identifier (e.g. GGUF path or API model name)"
    )
    is_default: bool = False


class LocalProviderSettings(BaseSettings):
    """Settings for the local llama.cpp provider."""

    model_dir: Path = Field(
        default=Path("models"),
        description="Directory containing GGUF model files",
    )
    n_ctx: int = Field(default=4096, description="Context window size")
    n_threads: int = Field(default=4, description="Number of CPU threads for inference")
    n_gpu_layers: int = Field(
        default=0, description="Number of layers to offload to GPU (0 = CPU only)"
    )
    verbose: bool = False


class OpenAIProviderSettings(BaseSettings):
    """Settings for the OpenAI API provider."""

    api_key: str = Field(default="", description="OpenAI API key")
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL",
    )
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    max_retries: int = Field(default=2, description="Max retries on transient errors")


class AnthropicProviderSettings(BaseSettings):
    """Settings for the Anthropic API provider."""

    api_key: str = Field(default="", description="Anthropic API key")
    base_url: str = Field(
        default="https://api.anthropic.com",
        description="Anthropic API base URL",
    )
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    max_retries: int = Field(default=2, description="Max retries on transient errors")


class AuthSettings(BaseSettings):
    """Authentication and rate limiting configuration."""

    api_keys: list[str] = Field(
        default_factory=list,
        description="Valid API keys. Empty list disables auth.",
    )
    rate_limit_rpm: int = Field(default=60, description="Requests per minute per API key")
    rate_limit_burst: int = Field(default=10, description="Burst allowance above steady rate")


class CacheSettings(BaseSettings):
    """Cache configuration."""

    response_max_size: int = Field(default=256, description="Max cached responses")
    response_ttl: int = Field(default=3600, description="Response cache TTL in seconds")
    embeddings_max_size: int = Field(default=1024, description="Max cached embeddings")
    embeddings_ttl: int = Field(default=86400, description="Embeddings cache TTL in seconds")


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    backend: Literal["numpy", "faiss"] = Field(
        default="numpy", description="Vector store backend"
    )
    persist_dir: Path = Field(
        default=Path("data/vectors"),
        description="Directory for persisting vector index",
    )
    default_top_k: int = Field(default=5, description="Default number of results to return")


class Settings(BaseSettings):
    """Root application settings. Loads from environment and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Sub-configs
    server: ServerSettings = Field(default_factory=ServerSettings)
    local: LocalProviderSettings = Field(default_factory=LocalProviderSettings)
    openai: OpenAIProviderSettings = Field(default_factory=OpenAIProviderSettings)
    anthropic: AnthropicProviderSettings = Field(default_factory=AnthropicProviderSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)

    # Model registry (configured via JSON or env)
    models: list[ModelConfig] = Field(
        default_factory=lambda: [
            ModelConfig(
                name="local-default",
                provider="local",
                model_id="default.gguf",
                is_default=True,
            )
        ],
        description="List of available models and their provider mappings",
    )

    # Application
    debug: bool = False


def get_settings() -> Settings:
    """Create and return application settings. Cached at module level."""
    return Settings()
