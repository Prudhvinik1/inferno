# Inferno

Minimal LLM serving stack with voice and search. Built from scratch in Python to demonstrate production API design, real-time streaming, and multi-provider inference routing.

## What This Is

A unified API gateway that sits in front of local and cloud LLM providers, exposing an OpenAI-compatible `/v1/chat/completions` endpoint with SSE streaming. Designed for Apple Silicon first, cheap cloud VMs second.

**Target applications:** voice assistants, semantic search, RAG pipelines.

## Architecture

```
Client (HTTP/SSE/WebSocket)
  |
  v
FastAPI Gateway
  ├── /v1/chat/completions   (streaming + non-streaming)
  ├── /v1/embeddings          (planned)
  ├── /v1/search              (planned — RAG pipeline)
  ├── /v1/audio/*             (planned — STT/TTS)
  ├── /v1/realtime            (planned — WebSocket voice)
  └── /health
  |
  v
Model Router  (name → provider resolution, fallback policy)
  |
  ├── LocalLLMProvider     (llama.cpp, GGUF models, CPU/GPU)
  ├── OpenAIProvider       (planned — httpx async adapter)
  └── AnthropicProvider    (planned — httpx async adapter)
```

## Quick Start

### Prerequisites

- Python 3.11+
- A GGUF model file (e.g. Phi-3-mini, Llama-3.2-1B)

### Install

```bash
# Clone
git clone <your-repo-url> && cd inferno

# Create venv
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Download a model

Place any GGUF model in the `models/` directory:

```bash
mkdir -p models
# Example: download a small model
# huggingface-cli download TheBloke/phi-2-GGUF phi-2.Q4_K_M.gguf --local-dir models/
```

### Run

```bash
# Start the server
python -m src.main

# Or with uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Test

```bash
# Health check
curl http://localhost:8000/health

# Chat completion (non-streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Chat completion (streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-default",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

## Configuration

All config loads from environment variables or a `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER__HOST` | `0.0.0.0` | Server bind address |
| `SERVER__PORT` | `8000` | Server port |
| `SERVER__LOG_LEVEL` | `info` | Log level |
| `LOCAL__MODEL_DIR` | `models` | Directory containing GGUF files |
| `LOCAL__N_CTX` | `4096` | Context window size |
| `LOCAL__N_THREADS` | `4` | CPU threads for inference |
| `LOCAL__N_GPU_LAYERS` | `0` | GPU layers (0 = CPU only) |
| `OPENAI__API_KEY` | | OpenAI API key |
| `ANTHROPIC__API_KEY` | | Anthropic API key |
| `AUTH__API_KEYS` | `[]` | API keys (empty = auth disabled) |
| `AUTH__RATE_LIMIT_RPM` | `60` | Requests per minute per key |
| `DEBUG` | `false` | Enable debug mode + hot reload |

## Project Structure

```
src/
├── main.py              # FastAPI app, lifespan, health endpoint
├── config.py            # pydantic-settings configuration
├── api/
│   └── chat.py          # POST /v1/chat/completions (SSE + JSON)
├── core/
│   ├── provider.py      # Provider ABC (generate, stream, embed, health)
│   ├── router.py        # ModelRegistry + ModelRouter
│   └── schemas.py       # Request/response schemas (OpenAI-compatible)
├── providers/
│   └── local_llm.py     # llama-cpp-python provider
├── infra/               # (planned) auth, rate limiting, cache, metrics
└── store/               # (planned) vector store for RAG
tests/
pyproject.toml
```

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible chat completion endpoint.

**Request body:**

```json
{
  "model": "local-default",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 512,
  "top_p": 1.0,
  "stop": null,
  "tools": null,
  "tool_choice": null
}
```

**Non-streaming response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1707600000,
  "model": "local-default",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello!"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
}
```

**Streaming response (SSE):**

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1707600000,"model":"local-default","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1707600000,"model":"local-default","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1707600000,"model":"local-default","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}

data: [DONE]
```

### `GET /health`

Returns server status, uptime, and per-provider health.

## Roadmap

- [x] Provider ABC + model registry + router
- [x] Local LLM provider (llama-cpp-python)
- [x] `/v1/chat/completions` with SSE streaming
- [x] Client disconnect cancellation
- [x] Tool/function calling schema
- [ ] External API providers (OpenAI, Anthropic)
- [ ] Unified error normalization middleware
- [ ] `/v1/embeddings` + vector store + RAG pipeline
- [ ] Auth + rate limiting + structured logs + metrics
- [ ] STT (Whisper) + TTS + `/v1/audio/*`
- [ ] Real-time voice WebSocket (`/v1/realtime`)
- [ ] Function calling execution loop
- [ ] Response/embedding cache
- [ ] End-to-end voice search demo

## License

Proprietary.
