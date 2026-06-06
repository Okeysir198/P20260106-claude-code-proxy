# CLAUDE.md — Claude Code Proxy

A single-file FastAPI proxy that lets **Claude Code CLI** and the **Claude Agent
SDK** (both speak ONLY the Anthropic Messages protocol) drive **OpenAI-compatible
and other backends** — NVIDIA NIM, OpenRouter, OpenAI, Ollama, Google Gemini /
Vertex, or Anthropic passthrough. It receives Anthropic `/v1/messages`, translates
to the target provider via `litellm`, and translates the response (incl. streaming
SSE, tool-calls, and thinking blocks) back to Anthropic format.

- **Entry point:** `server.py` (~2050 lines, everything lives here).
- **Stack:** FastAPI + `uvicorn` + `litellm` + `httpx` + `pydantic`. Python ≥3.10, `uv`.
- **Default port:** `4000`.

## Why it exists

Claude Code only POSTs to `{ANTHROPIC_BASE_URL}/v1/messages` in Anthropic format.
Most models (NVIDIA Nemotron, OpenRouter, etc.) only serve the OpenAI Chat
Completions format. This proxy is the translation seam between the two.

## Two routing modes

### 1. Static (env-driven) — `POST /v1/messages`
Routing comes from environment variables. Claude Code's built-in tiers are mapped:
- `…haiku…` → `SMALL_MODEL`
- `…sonnet…` / `…opus…` / `claude*` → `BIG_MODEL`
- provider chosen by `PREFERRED_PROVIDER` (`openai` | `ollama` | `google` | `anthropic`).

### 2. Dynamic (URL-pinned) — `POST /{provider}:{model}/v1/messages`
The provider **and** model are taken from the URL path, **bypassing** the
tier-mapping above. The body's `model` field is ignored. This is the mode to use
when you want to pin one specific upstream model.

- `parse_provider_model` splits on the **first** colon, so the model may itself
  contain `/` and `:` (e.g. `nvidia/nemotron-3-ultra-550b-a55b`, `qwen2.5-coder:32b`).
- Valid providers: `openai`, `ollama`, `gemini`, `google`, `anthropic`.
- `get_prefixed_model` adds the LiteLLM prefix (`openai/…`, `ollama_chat/…`,
  `gemini/…`, `anthropic/…`).
- A matching `…/v1/messages/count_tokens` dynamic route exists too.

Example (what this repo's sibling `claude_sdk` uses):
```
POST http://localhost:4000/openai:nvidia/nemotron-3-ultra-550b-a55b/v1/messages
  -> provider=openai, model=nvidia/nemotron-3-ultra-550b-a55b
  -> OpenAI-compatible call to NVIDIA build.nvidia.com
```

## Provider resolution (the `openai` provider is overloaded)

Under the `openai` provider, the **model-name prefix** selects which
OpenAI-compatible upstream + credentials are used (`OPENAI_COMPATIBLE_PROVIDERS`):

| Model prefix | Upstream | Key / base env |
|---|---|---|
| `nvidia/`, `meta/` | **NVIDIA NIM** | `NVIDIA_API_KEY` / `NVIDIA_BASE_URL` |
| `qwen/`, `mistralai/`, `deepseek/`, `google/`, `openai/`, `x-ai/`, `meta-llama/`, … | **OpenRouter** | `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` |
| anything else | **default OpenAI** | `OPENAI_API_KEY` / `OPENAI_BASE_URL` |

- If the matched provider's key is unset, it **falls back to the default
  `OPENAI_*` credentials**.
- ⚠️ **Prefix collision:** the dynamic route prefixes the model to
  `openai/<model>` before credential lookup, so a model like
  `nvidia/nemotron-…` is matched as `openai/nvidia/nemotron-…`. Because `openai/`
  is itself an OpenRouter prefix, set the default `OPENAI_BASE_URL` /
  `OPENAI_API_KEY` to your NVIDIA endpoint+key (and leave `OPENROUTER_API_KEY`
  unset) so NVIDIA models resolve cleanly. The `nemotron` reasoning extras are
  still applied (next section).

## NVIDIA Nemotron reasoning

For any model whose name contains `nemotron`, the proxy attaches an
`extra_body` so the NIM emits proper reasoning:
```python
extra_body = {
  "reasoning_budget": NVIDIA_REASONING_BUDGET,          # default 16384
  "chat_template_kwargs": {"enable_thinking": NVIDIA_ENABLE_THINKING},  # default true
}
```
Reasoning is surfaced as Anthropic `thinking` blocks (or merged into text if the
token budget is too small — give it room, e.g. `max_tokens ≥ 256`).

## Environment variables

| Var | Purpose |
|---|---|
| `PREFERRED_PROVIDER` | Static-mode provider (`openai`/`ollama`/`google`/`anthropic`). Default `openai`. |
| `BIG_MODEL` / `SMALL_MODEL` | Static-mode targets for sonnet-or-opus / haiku tiers. |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | Default OpenAI-compatible upstream (point at NVIDIA for Nemotron). |
| `NVIDIA_API_KEY` / `NVIDIA_BASE_URL` | NVIDIA NIM (`https://integrate.api.nvidia.com/v1`). |
| `NVIDIA_REASONING_BUDGET` / `NVIDIA_ENABLE_THINKING` | Nemotron reasoning controls. |
| `OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` | OpenRouter upstream. |
| `OLLAMA_API_BASE` / `OLLAMA_NUM_CTX` | Native-Anthropic Ollama path. |
| `GEMINI_API_KEY`, `USE_VERTEX_AUTH`, `VERTEX_PROJECT`, `VERTEX_LOCATION` | Google / Vertex. |
| `ANTHROPIC_API_KEY` | Anthropic passthrough mode. |
| `MAX_OUTPUT_TOKENS` | Output cap (`0` = no cap). |

Secrets live in `.env` (gitignored) — see `.env.example`.

## Run

```bash
# Dev
uv run uvicorn server:app --host 0.0.0.0 --port 4000 --reload
# Docker
docker compose up -d        # binds :4000, host.docker.internal mapped for Ollama
```

## Point a client at it

```bash
# Static mode (uses PREFERRED_PROVIDER + BIG/SMALL_MODEL)
ANTHROPIC_BASE_URL=http://localhost:4000 ANTHROPIC_API_KEY=dummy claude
# Dynamic mode (pin one model in the URL)
ANTHROPIC_BASE_URL=http://localhost:4000/openai:nvidia/nemotron-3-ultra-550b-a55b \
  ANTHROPIC_API_KEY=dummy claude
```
The proxy holds the real upstream key; the client's `ANTHROPIC_API_KEY` is unused
(send any non-empty value).

## Used by

`P20251204-livekit-voice-agents` → `livekit-backend/src/claude_sdk` routes its
Claude Code subprocess here. Provider `proxy` in `claude_sdk/config.yaml` sets a
literal `base_url: http://localhost:4000/openai:nvidia/nemotron-3-ultra-550b-a55b`
(the dynamic provider:model URL), so the Claude Agent runs on NVIDIA
Nemotron-Ultra. The sibling `deep_agent_sdk` talks to NVIDIA directly (it's
LangChain/OpenAI-native) and does NOT need this proxy.

## Gotchas

- **Single source file** — all logic is in `server.py`; there are two
  `parse_provider_model` definitions and the **second** (line ~1846, used by the
  dynamic routes) is the one in effect.
- **Reasoning leak** — too-small `max_tokens` truncates mid-thought so reasoning
  shows up in the text block. Raise `max_tokens`.
- **One model per dynamic URL** — to switch models, change the URL path (or use a
  second client env), not the request body.
