# Claude Code Proxy

A universal proxy server that allows **Claude Code CLI** and **Claude Agent SDK** to use **multiple AI backends simultaneously** with intelligent routing. Features native Anthropic format support for Ollama.

```
┌───────────────────┐       ┌─────────────────┐      ┌──────────────────────────┐
│  Claude Code CLI  │       │                 │      │  🚀 Multi-Provider:     │
│                   │─────▶│  Claude Code    │─────▶│  • OpenRouter (400+)     │
│ Claude Agent SDK  │       │     Proxy       │      │  • NVIDIA NIM            │
│                   │◀─────│  (Smart Router) │◀─────│  • Ollama (Native)       │
└───────────────────┘       └─────────────────┘      │  • OpenAI                │
                                                     │  • Google Gemini         │
   Anthropic API            Intelligent              │  • Anthropic             │
     Format                 Routing                   └──────────────────────────┘
```

## ✨ Key Features

- **🔄 Multi-Provider Support** - Configure all providers simultaneously, proxy intelligently routes based on model name
- **⚡ Ollama Native Format** - Zero overhead for Ollama, full Anthropic format support including thinking blocks
- **🧠 Thinking Blocks Support** - Full support for Anthropic's extended thinking responses
- **🎯 Dynamic Model Selection** - Specify any model via URL path
- **🔌 400+ OpenRouter Models** - Access to latest open-weight models
- **🛠️ Tool Use Support** - Full function calling support across providers

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- API keys for your chosen providers (all optional - use what you want!)

### Setup

1. Clone and install:
   ```bash
   git clone https://github.com/Okeysir198/P20260106-claude-code-proxy.git
   cd P20260106-claude-code-proxy
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run the server:
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 4000 --reload
   ```

4. Use with Claude Code:
   ```bash
   # Uses your default provider (configured in .env)
   ANTHROPIC_BASE_URL=http://localhost:4000 claude

   # Or specify any model dynamically via URL
   ANTHROPIC_BASE_URL=http://localhost:4000/openai:stepfun/step-3.5-flash:free claude
   ```

### Docker

```bash
docker compose up -d
```

## Multi-Provider Configuration

### All Providers Simultaneously

The proxy supports **all providers at once** with intelligent routing:

```dotenv
# =============================================================================
# Provider 1: Ollama (Native Anthropic Format)
# =============================================================================
PREFERRED_PROVIDER="ollama"
OLLAMA_API_BASE="http://localhost:11434"  # or http://host.docker.internal:11434 for Docker
OLLAMA_NUM_CTX=60000
BIG_MODEL="gpt-oss:20b"
SMALL_MODEL="gpt-oss:20b"

# =============================================================================
# Provider 2: OpenRouter (400+ models)
# =============================================================================
OPENROUTER_API_KEY="sk-or-v1-..."
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

# =============================================================================
# Provider 3: NVIDIA NIM API
# =============================================================================
NVIDIA_API_KEY="nvapi-..."
NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
NVIDIA_REASONING_BUDGET=16384
NVIDIA_ENABLE_THINKING=true
```

### Intelligent Routing

The proxy automatically routes requests based on model name prefixes:

| Model Prefix | Provider | Example Models |
|-------------|----------|----------------|
| `stepfun/`, `qwen/`, `mistralai/`, `x-ai/` | OpenRouter | `stepfun/step-3.5-flash:free` |
| `nvidia/`, `meta/` | NVIDIA NIM | `nvidia/nemotron-3-nano-30b-a3b` |
| `ollama/`, default | Ollama | `gpt-oss:20b` |

### Usage Examples

```bash
# OpenRouter (free model)
ANTHROPIC_BASE_URL=http://localhost:4000/openai:stepfun/step-3.5-flash:free claude -p "Hello"

# NVIDIA NIM
ANTHROPIC_BASE_URL=http://localhost:4000/openai:nvidia/nemotron-3-nano-30b-a3b claude -p "Hello"

# Ollama (uses default BIG_MODEL)
ANTHROPIC_BASE_URL=http://localhost:4000 claude -p "Hello"
```

## Provider-Specific Configuration

### Ollama (Native Anthropic Format) ⚡

> **Direct Ollama:** Ollama supports the Anthropic API natively! You can use it directly:
> ```bash
> ANTHROPIC_AUTH_TOKEN=ollama ANTHROPIC_BASE_URL=http://localhost:11434 claude --model glm-4.7-flash
> ```
>
> **Via Proxy:** The proxy adds model mapping and multi-provider support:
> ```bash
> ANTHROPIC_BASE_URL=http://localhost:4000 claude  # Maps claude-sonnet-4 → gpt-oss:20b
> ```

**Benefits of Using via Proxy:**
- ✅ **Model Mapping** - Use Claude model names (`claude-sonnet-4`) → Ollama models
- ✅ **Multi-Provider** - Switch between OpenRouter, NVIDIA, Ollama in one config
- ✅ **Zero Overhead** - Direct proxy to Ollama's `/v1/messages`, no conversion
- ✅ **Full Feature Support** - Content blocks, thinking, tools, streaming

**Recommended Coding Models:**
- `gpt-oss:20b` - OpenAI's open-weight model, excellent for coding/tool use, runs in 16GB RAM
- `qwen2.5-coder:32b` - Best for multi-language projects, competitive with GPT-4o
- `qwen3-coder` - Handles massive codebases, comparable to Claude Sonnet
- `devstral-small-2:24b` - Excellent for file operations
- `glm-4.7-flash` - Fast reasoning model

```dotenv
PREFERRED_PROVIDER="ollama"
OLLAMA_API_BASE="http://localhost:11434"
OLLAMA_NUM_CTX=60000
BIG_MODEL="gpt-oss:20b"
SMALL_MODEL="gpt-oss:20b"
```

### OpenRouter (400+ Models)

Get your API key at [https://openrouter.ai/keys](https://openrouter.ai/keys) and browse models at [https://openrouter.ai/models](https://openrouter.ai/models).

**Top Free Models:**
- `stepfun/step-3.5-flash:free` - Fast, free tier
- `google/gemini-2.0-flash-exp:free` - Requires privacy settings enabled

**Top Paid Models:**
- `qwen/qwen3-coder-480b-a35b-instruct` - Best for agentic coding
- `mistralai/devstral-2` - 123B dense, 256K context
- `x-ai/grok-3-beta` - Latest xAI model
- `anthropic/claude-sonnet-4` - Claude Sonnet 4
- `google/gemini-2.5-pro-preview` - Gemini 2.5 Pro

```dotenv
OPENROUTER_API_KEY="sk-or-v1-..."
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

### NVIDIA NIM

Get your free API key at [https://build.nvidia.com/](https://build.nvidia.com/).

```dotenv
NVIDIA_API_KEY="nvapi-..."
NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
NVIDIA_REASONING_BUDGET=16384
NVIDIA_ENABLE_THINKING=true
```

### OpenAI

```dotenv
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api.openai.com/v1"
BIG_MODEL="gpt-4.1"
SMALL_MODEL="gpt-4.1-mini"
```

### Google Gemini

```dotenv
GEMINI_API_KEY="your-key"
BIG_MODEL="gemini-2.5-pro"
SMALL_MODEL="gemini-2.5-flash"
```

## Dynamic Model Selection

Specify any model directly in the URL without modifying `.env`:

### URL Format

```
http://localhost:4000/{provider}:{model}/v1/messages
```

### Examples

```bash
# OpenRouter models
ANTHROPIC_BASE_URL=http://localhost:4000/openai:stepfun/step-3.5-flash:free claude
ANTHROPIC_BASE_URL=http://localhost:4000/openai:qwen/qwen3-coder-480b-a35b-instruct claude
ANTHROPIC_BASE_URL=http://localhost:4000/openai:mistralai/devstral-2 claude

# NVIDIA NIM
ANTHROPIC_BASE_URL=http://localhost:4000/openai:nvidia/nemotron-3-nano-30b-a3b claude

# OpenAI
ANTHROPIC_BASE_URL=http://localhost:4000/openai:gpt-4.1 claude

# Google Gemini
ANTHROPIC_BASE_URL=http://localhost:4000/gemini:gemini-2.5-pro claude

# Anthropic (passthrough)
ANTHROPIC_BASE_URL=http://localhost:4000/anthropic:claude-sonnet-4 claude

# Ollama (with specific model)
ANTHROPIC_BASE_URL=http://localhost:4000/ollama:glm-4.7-flash:latest claude
```

## Supported Providers

| URL Provider | Backend | Auto-Routed Models |
|--------------|---------|-------------------|
| `openai` | OpenAI / OpenRouter / NVIDIA | `stepfun/*`, `qwen/*`, `nvidia/*`, `meta/*`, etc. |
| `ollama` | Local Ollama (Native) | Default provider, `ollama/*` |
| `gemini` / `google` | Google Gemini | `gemini/*` |
| `anthropic` | Anthropic API | Passthrough mode |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| **Multi-Provider** | | |
| `OPENROUTER_API_KEY` | OpenRouter API key | - |
| `OPENROUTER_BASE_URL` | OpenRouter endpoint | `https://openrouter.ai/api/v1` |
| `NVIDIA_API_KEY` | NVIDIA NIM API key | - |
| `NVIDIA_BASE_URL` | NVIDIA NIM endpoint | `https://integrate.api.nvidia.com/v1` |
| **Ollama** | | |
| `OLLAMA_API_BASE` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_NUM_CTX` | Context length | `128000` |
| **OpenAI** | | |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_BASE_URL` | Custom endpoint | - |
| **General** | | |
| `PREFERRED_PROVIDER` | Default provider | `openai` |
| `BIG_MODEL` | Model for sonnet/opus | `gpt-4.1` |
| `SMALL_MODEL` | Model for haiku | `gpt-4.1-mini` |
| `MAX_OUTPUT_TOKENS` | Output token cap (0 = no cap) | `0` |

## Model Mapping

Claude model names are automatically mapped:

| Request contains | Maps to |
|------------------|---------|
| `haiku` | `SMALL_MODEL` |
| `sonnet`, `opus`, `claude` | `BIG_MODEL` |

## Content Block Support

The proxy supports all Anthropic content block types:

| Block Type | Description | Supported By |
|------------|-------------|--------------|
| `text` | Plain text content | ✅ All providers |
| `image` | Image content (base64) | ✅ All providers |
| `tool_use` | Function calling requests | ✅ All providers |
| `tool_result` | Function call results | ✅ All providers |
| `thinking` | Extended reasoning blocks | ✅ Ollama, OpenRouter |

**Note:** Some models (like Ollama's `glm-4.7-flash`) naturally return `thinking` blocks as part of their reasoning process. The proxy fully supports this format.

## Testing

Verify the proxy is running:
```bash
curl http://localhost:4000/
# Expected: {"message":"Claude Code Proxy"}
```

Test with different providers:
```bash
# OpenRouter (free model)
curl -X POST http://localhost:4000/openai:stepfun/step-3.5-flash:free/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -d '{"model":"stepfun/step-3.5-flash:free","max_tokens":100,"messages":[{"role":"user","content":"Say Hello"}]}'

# NVIDIA NIM
curl -X POST http://localhost:4000/openai:nvidia/nemotron-3-nano-30b-a3b/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -d '{"model":"nvidia/nemotron-3-nano-30b-a3b","max_tokens":100,"messages":[{"role":"user","content":"Say Hello"}]}'

# Ollama (native format)
curl -X POST http://localhost:4000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -d '{"model":"claude-sonnet-4","max_tokens":100,"messages":[{"role":"user","content":"Say Hello"}]}'
```

## How It Works

1. **Receives** requests in Anthropic API format
2. **Detects** target provider from model name prefix
3. **Routes** to appropriate backend:
   - **Ollama**: Native Anthropic format (zero conversion)
   - **Others**: Converts to LiteLLM/OpenAI format
4. **Returns** response in Anthropic format

Supports both streaming and non-streaming responses.

## References

This project is based on [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) by 1rgs.
