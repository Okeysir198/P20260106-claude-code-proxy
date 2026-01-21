# Claude Code Proxy

A proxy server that allows Claude Code CLI and [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) to use alternative AI backends via LiteLLM.

```
┌───────────────────┐       ┌─────────────────┐      ┌──────────────────────────┐
│  Claude Code CLI  │       │                 │      │  NVIDIA NIM (Recommended)│
│                   │─────▶│  Claude Code    │─────▶│  Ollama (Local)          │
│ Claude Agent SDK  │       │     Proxy       │      │  OpenAI                  │
│                   │◀─────│   (LiteLLM)     │◀─────│  Google Gemini           │
└───────────────────┘       └─────────────────┘      │  Anthropic               │
                                                     │  OpenRouter              │
                                                     └──────────────────────────┘
   Anthropic API            Translates &                  AI Backends
     Format                 Routes Request
```

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- API key for your chosen provider:
  - [NVIDIA NIM](https://build.nvidia.com/) (Recommended - free tier available)
  - [Ollama](https://ollama.com/) (Local - no API key needed)
  - [OpenAI](https://platform.openai.com/)
  - [Google Gemini](https://aistudio.google.com/)
  - [OpenRouter](https://openrouter.ai/)
  - [Anthropic](https://console.anthropic.com/)

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
   # Edit .env with your API keys and preferences
   ```

3. Run the server:
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 4000 --reload
   ```

4. Use with Claude Code:
   ```bash
   # Option A: Use environment variables (configure .env first)
   ANTHROPIC_BASE_URL=http://localhost:4000 claude

   # Option B: Specify model directly in URL (no .env needed)
   ANTHROPIC_BASE_URL=http://localhost:4000/ollama:gpt-oss:20b claude
   ANTHROPIC_BASE_URL=http://localhost:4000/openai:nvidia/nemotron-3-nano-30b-a3b claude
   ANTHROPIC_BASE_URL=http://localhost:4000/gemini:gemini-2.5-pro claude
   ```

   Quick test:
   ```bash
   # With environment variables
   ANTHROPIC_BASE_URL=http://localhost:4000 claude -p "Say Hello"

   # Or with dynamic model in URL
   ANTHROPIC_BASE_URL=http://localhost:4000/ollama:gpt-oss:20b claude -p "Say Hello"
   ```

5. Use with Claude Agent SDK:
   ```python
   import asyncio
   from claude_code_sdk import query, ClaudeCodeOptions

   async def main():
       options = ClaudeCodeOptions(
           system_prompt="You are an expert Python developer",
           permission_mode='acceptEdits',
           cwd="/your/project/path"
       )

       async for message in query(
           prompt="Create a Python file that says hello",
           options=options
       ):
           print(message)

   asyncio.run(main())
   ```

   Run with:
   ```bash
   # With environment variables
   ANTHROPIC_BASE_URL=http://localhost:4000 python your_agent.py

   # Or with dynamic model in URL
   ANTHROPIC_BASE_URL=http://localhost:4000/openai:gpt-4.1 python your_agent.py
   ```

### Docker

Build and run from source:
```bash
docker build -t claude-code-proxy .
docker run -d --env-file .env -p 4000:4000 claude-code-proxy
```

Or with docker-compose:
```bash
docker compose up -d
```

For development (auto-restart on `.env` changes):
```bash
docker compose watch
```

### Testing

Verify the proxy is running:
```bash
curl http://localhost:4000/
# Expected: {"message":"Claude Code Proxy"}
```

Test with Claude Code:
```bash
# With environment variables
ANTHROPIC_BASE_URL=http://localhost:4000 claude -p "hello"

# Or with dynamic model in URL
ANTHROPIC_BASE_URL=http://localhost:4000/ollama:gpt-oss:20b claude -p "hello"
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PREFERRED_PROVIDER` | Backend provider: `openai`, `google`, `ollama`, or `anthropic` | `openai` |
| `OPENAI_BASE_URL` | Custom OpenAI-compatible endpoint (NVIDIA, OpenRouter, etc.) | - |
| `OPENAI_API_KEY` | OpenAI/NVIDIA/OpenRouter API key | - |
| `BIG_MODEL` | Model for sonnet/opus requests | `gpt-4.1` |
| `SMALL_MODEL` | Model for haiku requests | `gpt-4.1-mini` |
| `NVIDIA_REASONING_BUDGET` | NVIDIA reasoning token budget | `16384` |
| `NVIDIA_ENABLE_THINKING` | Enable NVIDIA thinking mode | `true` |
| `OLLAMA_API_BASE` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_NUM_CTX` | Ollama context length (use 128000 for long context) | `128000` |
| `GEMINI_API_KEY` | Google AI Studio API key | - |
| `USE_VERTEX_AUTH` | Use GCP ADC instead of API key | `false` |
| `VERTEX_PROJECT` | GCP project ID (if using Vertex) | - |
| `VERTEX_LOCATION` | GCP region (if using Vertex) | - |
| `ANTHROPIC_API_KEY` | Anthropic API key (for passthrough mode) | - |
| `MAX_OUTPUT_TOKENS` | Max output tokens cap (0 = no cap) | `0` |

### Example Configurations

**NVIDIA NIM (Recommended - Default):**

Get your free API key at [https://build.nvidia.com/](https://build.nvidia.com/) and browse available models at [https://build.nvidia.com/models](https://build.nvidia.com/models).

```dotenv
PREFERRED_PROVIDER="openai"
OPENAI_API_KEY="nvapi-..."
OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1"
BIG_MODEL="mistralai/devstral-2-123b-instruct-2512"
SMALL_MODEL="mistralai/devstral-2-123b-instruct-2512"
NVIDIA_REASONING_BUDGET=16384
NVIDIA_ENABLE_THINKING=true
```

**Ollama (local):**

> **Important:** Use models with long context window support. Set `OLLAMA_NUM_CTX=128000` for optimal performance.

Recommended coding models:
- `gpt-oss:20b` - OpenAI's open-weight model, excellent for coding/tool use, runs in 16GB RAM
- `qwen2.5-coder:32b` - Best for multi-language projects, competitive with GPT-4o
- `qwen3-coder` - Handles massive codebases, comparable to Claude Sonnet
- `devstral` - Excellent for file operations and large codebase management
- `deepseek-coder-v2` - Supports 300+ languages, great for code generation

```dotenv
PREFERRED_PROVIDER="ollama"
# Without Docker: use localhost
OLLAMA_API_BASE="http://localhost:11434"
# With Docker: use host.docker.internal
# OLLAMA_API_BASE="http://host.docker.internal:11434"
OLLAMA_NUM_CTX=128000
BIG_MODEL="gpt-oss:20b"
SMALL_MODEL="gpt-oss:20b"
```

**OpenAI:**
```dotenv
PREFERRED_PROVIDER="openai"
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api.openai.com/v1"
BIG_MODEL="gpt-4.1"
SMALL_MODEL="gpt-4.1-mini"
```

**Google Gemini:**
```dotenv
PREFERRED_PROVIDER="google"
GEMINI_API_KEY="your-key"
BIG_MODEL="gemini-2.5-pro"
SMALL_MODEL="gemini-2.5-flash"
```

**Anthropic passthrough:**
```dotenv
PREFERRED_PROVIDER="anthropic"
ANTHROPIC_API_KEY="sk-ant-..."
```

**OpenRouter (400+ models):**
```dotenv
PREFERRED_PROVIDER="openai"
OPENAI_API_KEY="sk-or-v1-..."
OPENAI_BASE_URL="https://openrouter.ai/api/v1"
BIG_MODEL="qwen/qwen3-coder-480b-a35b-instruct"
SMALL_MODEL="qwen/qwen3-coder-480b-a35b-instruct"
```

## Dynamic Model Selection via URL

Instead of configuring `BIG_MODEL` and `SMALL_MODEL` environment variables, you can specify the model directly in the URL:

```bash
ANTHROPIC_BASE_URL=http://localhost:4000/{provider}:{model}
```

### URL Format

```
http://localhost:4000/{provider}:{model}/v1/messages
```

### Examples

**Ollama (local):**
```bash
ANTHROPIC_BASE_URL=http://localhost:4000/ollama:gpt-oss:20b claude -p "Hello"
```

**NVIDIA NIM:**
```bash
ANTHROPIC_BASE_URL=http://localhost:4000/openai:nvidia/nemotron-3-nano-30b-a3b claude -p "Hello"
```

**OpenAI:**
```bash
ANTHROPIC_BASE_URL=http://localhost:4000/openai:gpt-4.1 claude -p "Hello"
```

**Google Gemini:**
```bash
ANTHROPIC_BASE_URL=http://localhost:4000/gemini:gemini-2.5-pro claude -p "Hello"
```

**Anthropic (passthrough):**
```bash
ANTHROPIC_BASE_URL=http://localhost:4000/anthropic:claude-sonnet-4 claude -p "Hello"
```

### Supported Providers

| URL Provider | Backend | Notes |
|--------------|---------|-------|
| `openai` | OpenAI API / NVIDIA NIM / OpenRouter | Uses `OPENAI_API_KEY` and `OPENAI_BASE_URL` |
| `ollama` | Local Ollama | Uses `OLLAMA_API_BASE` |
| `gemini` / `google` | Google Gemini | Uses `GEMINI_API_KEY` or Vertex AI auth |
| `anthropic` | Anthropic API | Passthrough mode |

### Backward Compatibility

The existing endpoint `/v1/messages` still works and uses the `BIG_MODEL`/`SMALL_MODEL` environment variables as before.

| Usage | Behavior |
|-------|----------|
| `ANTHROPIC_BASE_URL=http://localhost:4000` | Uses env vars (existing behavior) |
| `ANTHROPIC_BASE_URL=http://localhost:4000/openai:gpt-4.1` | Uses URL model, ignores BIG_MODEL/SMALL_MODEL |

## Model Mapping

The proxy maps Claude model names to your configured backend:

| Request contains | Maps to |
|------------------|---------|
| `haiku` | `SMALL_MODEL` |
| `sonnet`, `opus`, `claude` | `BIG_MODEL` |

Models are automatically prefixed with the provider (`openai/`, `gemini/`, `ollama_chat/`, or `anthropic/`).

## How It Works

1. Receives requests in Anthropic API format
2. Maps model names based on configuration
3. Translates to LiteLLM/OpenAI format
4. Sends to configured backend
5. Converts response back to Anthropic format

Supports both streaming and non-streaming responses.

## References

This project is based on [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) by 1rgs.
