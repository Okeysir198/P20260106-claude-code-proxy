from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator, ConfigDict
from typing import List, Dict, Any, Optional, Union, Literal
from dotenv import load_dotenv
import logging
import json
import os
import litellm
import uuid
import time
import re
import sys
import traceback
import asyncio
import httpx

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARN,  # Change to INFO level to show more details
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:", 
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

app = FastAPI()

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Get Vertex AI project and location from environment (if set)
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "unset")

# Option to use Gemini API key instead of ADC for Vertex AI
USE_VERTEX_AUTH = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"

# Get OpenAI base URL from environment (if set)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# =============================================================================
# Multi-Provider Support for OpenAI-compatible endpoints
# =============================================================================
# Multiple OpenAI-compatible providers can be configured simultaneously.
# The proxy will intelligently route requests based on model name patterns.

# OpenRouter Configuration (for models like: stepfun/step-3.5-flash:free, qwen/qwen3-coder-480b, etc.)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# NVIDIA NIM Configuration (for models like: nvidia/nemotron-3-nano-30b-a3b, etc.)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Model prefix patterns for routing to different OpenAI-compatible providers
# Format: {prefix: (api_key_var, base_url_var, provider_name)}
OPENAI_COMPATIBLE_PROVIDERS = {
    # OpenRouter - matches models like: stepfun/, qwen/, mistralai/, google/, etc.
    "stepfun/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "qwen/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "mistralai/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "x-ai/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "anthropic/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "google/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "openai/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "meta-llama/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "deepseek/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "microsoft/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "cognitive/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "nousresearch/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "liquid/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "fireworks/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "perplexity/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "sophosympatheia/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),
    "gradual/": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "openrouter"),

    # NVIDIA NIM - matches models like: nvidia/, meta/
    "nvidia/": ("NVIDIA_API_KEY", "NVIDIA_BASE_URL", "nvidia"),
    "meta/": ("NVIDIA_API_KEY", "NVIDIA_BASE_URL", "nvidia"),
}

# Helper function for safe integer parsing from environment
def safe_int_env(name: str, default: int) -> int:
    """Safely parse an integer from environment variable with fallback."""
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        logger.warning(f"Invalid integer value for {name}, using default {default}")
        return default

# Get Ollama endpoint and context size
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_NUM_CTX = safe_int_env("OLLAMA_NUM_CTX", 128000)

# Max output tokens (0 = no cap, let model/provider decide)
MAX_OUTPUT_TOKENS = safe_int_env("MAX_OUTPUT_TOKENS", 0)

# NVIDIA NIM reasoning settings
NVIDIA_REASONING_BUDGET = safe_int_env("NVIDIA_REASONING_BUDGET", 16384)
NVIDIA_ENABLE_THINKING = os.environ.get("NVIDIA_ENABLE_THINKING", "true").lower() == "true"

# Get preferred provider (default to openai)
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

# List of OpenAI models
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",  # Added default big model
    "gpt-4.1-mini" # Added default small model
]

# List of Gemini models
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]

# List of Ollama models
OLLAMA_MODELS = [
    "llama3",
    "llama3.1",
    "llama3:8b",
    "llama3:70b",
    "codellama:34b-instruct",
    "llama2",
    "mistral",
    "mistral:instruct",
    "mixtral",
    "mixtral:instruct",
    "phi3:mini",
    "phi3:medium",
    "qwen2.5-coder:32b",
    "qwen2.5-coder:14b",
    "deepseek-coder-v2",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
    "gpt-oss:20b",
    "nemotron-3-nano:30b",
    "devstral-small-2:24b"
]

# Shared model mapping logic
def map_model_name(model: str) -> str:
    """Map Claude model names to provider-specific models based on configuration."""
    original_model = model

    # Remove provider prefixes for matching
    clean_v = model
    for prefix in ['anthropic/', 'openai/', 'gemini/', 'ollama_chat/', 'ollama/']:
        if clean_v.startswith(prefix):
            clean_v = clean_v[len(prefix):]
            break

    # Anthropic passthrough mode
    if PREFERRED_PROVIDER == "anthropic":
        return f"anthropic/{clean_v}"

    # Map haiku to SMALL_MODEL
    if 'haiku' in clean_v.lower():
        if PREFERRED_PROVIDER == "ollama" and SMALL_MODEL in OLLAMA_MODELS:
            return f"ollama_chat/{SMALL_MODEL}"
        elif PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
            return f"gemini/{SMALL_MODEL}"
        return f"openai/{SMALL_MODEL}"

    # Map sonnet/opus/claude to BIG_MODEL
    if 'sonnet' in clean_v.lower() or 'opus' in clean_v.lower() or clean_v.lower().startswith('claude'):
        if PREFERRED_PROVIDER == "ollama" and BIG_MODEL in OLLAMA_MODELS:
            return f"ollama_chat/{BIG_MODEL}"
        elif PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
            return f"gemini/{BIG_MODEL}"
        return f"openai/{BIG_MODEL}"

    # Add prefixes to known models
    if clean_v in OLLAMA_MODELS and not model.startswith(('ollama/', 'ollama_chat/')):
        return f"ollama_chat/{clean_v}"
    if clean_v in GEMINI_MODELS and not model.startswith('gemini/'):
        return f"gemini/{clean_v}"
    if clean_v in OPENAI_MODELS and not model.startswith('openai/'):
        return f"openai/{clean_v}"

    # No mapping found, return original
    if not model.startswith(('openai/', 'gemini/', 'anthropic/', 'ollama/', 'ollama_chat/')):
        logger.warning(f"No prefix or mapping rule for model: '{original_model}'")
    return model


def parse_provider_model(path: str) -> tuple[str, str] | None:
    """
    Parse dynamic provider and model from URL path.

    Extracts provider and model name from paths like '/openai:gpt-4.1/v1/messages'.
    Supports providers: openai, ollama, gemini, google, anthropic.
    Handles model names containing colons (e.g., 'qwen2.5-coder:32b') and slashes
    (e.g., 'nvidia/nemotron-3-nano-30b-a3b').

    Args:
        path: The URL path to parse (e.g., '/openai:gpt-4.1/v1/messages')

    Returns:
        A tuple of (provider, model) if the path matches the dynamic pattern,
        or None if it doesn't match (e.g., '/v1/messages' with no dynamic model).

    Examples:
        >>> parse_provider_model('/openai:gpt-4.1/v1/messages')
        ('openai', 'gpt-4.1')
        >>> parse_provider_model('/ollama:qwen2.5-coder:32b/v1/messages')
        ('ollama', 'qwen2.5-coder:32b')
        >>> parse_provider_model('/openai:nvidia/nemotron-3-nano-30b-a3b/v1/messages')
        ('openai', 'nvidia/nemotron-3-nano-30b-a3b')
        >>> parse_provider_model('/v1/messages')
        None
    """
    # Supported providers
    supported_providers = {'openai', 'ollama', 'gemini', 'google', 'anthropic'}

    # Remove leading slash if present
    if path.startswith('/'):
        path = path[1:]

    # Check if path starts with a supported provider followed by a colon
    for provider in supported_providers:
        prefix = f"{provider}:"
        if path.startswith(prefix):
            # Extract everything after 'provider:' up to '/v1/' or end
            remainder = path[len(prefix):]

            # Find the '/v1/' boundary to extract the model name
            v1_index = remainder.find('/v1/')
            if v1_index != -1:
                model = remainder[:v1_index]
            else:
                # No /v1/ found, take everything (edge case)
                model = remainder.rstrip('/')

            if model:
                logger.debug(f"Parsed dynamic provider/model from path: provider={provider}, model={model}")
                return (provider, model)

    # No dynamic provider:model pattern found
    return None


def get_credentials_for_provider(provider: str, model: str) -> dict:
    """
    Get credentials and settings for a specific provider.

    Returns a dictionary with the appropriate API keys and settings for each provider.
    The returned dict can be passed directly to LiteLLM completion calls.

    Args:
        provider: The provider name (openai, ollama, gemini, google, anthropic)
        model: The model name (used for provider-specific settings like NVIDIA reasoning)

    Returns:
        dict: Provider-specific credentials and settings including:
            - api_key: The API key for the provider
            - api_base: The base URL (if applicable)
            - Additional provider-specific settings
    """
    provider = provider.lower()
    credentials = {}

    if provider == "openai":
        # Check if model matches a known OpenAI-compatible provider pattern
        matched_provider = None
        for prefix, (api_key_var, base_url_var, provider_name) in OPENAI_COMPATIBLE_PROVIDERS.items():
            if model.lower().startswith(prefix):
                matched_provider = (api_key_var, base_url_var, provider_name)
                break

        if matched_provider:
            # Use the matched provider's credentials
            api_key_var, base_url_var, provider_name = matched_provider
            api_key = globals().get(api_key_var)
            base_url = globals().get(base_url_var)

            if api_key:
                credentials["api_key"] = api_key
                if base_url:
                    credentials["api_base"] = base_url
                    # Add NVIDIA reasoning settings for NVIDIA provider
                    if provider_name == "nvidia" and "nemotron" in model.lower():
                        credentials["extra_body"] = {
                            "reasoning_budget": NVIDIA_REASONING_BUDGET,
                            "chat_template_kwargs": {"enable_thinking": NVIDIA_ENABLE_THINKING}
                        }
                        logger.debug(f"{provider_name.upper()} credentials with reasoning: reasoning_budget={NVIDIA_REASONING_BUDGET}, enable_thinking={NVIDIA_ENABLE_THINKING}")
                    else:
                        logger.debug(f"{provider_name.upper()} credentials retrieved for model: {model}")
                else:
                    logger.warning(f"{provider_name.upper()} base URL not configured")
            else:
                logger.warning(f"{provider_name.upper()} API key not configured, falling back to default OPENAI credentials")
                # Fall through to default credentials - use the else branch below
                matched_provider = None
        else:
            # Use default OpenAI credentials
            credentials["api_key"] = OPENAI_API_KEY
            if OPENAI_BASE_URL:
                credentials["api_base"] = OPENAI_BASE_URL
                # Add NVIDIA reasoning settings only for nemotron models
                if "nemotron" in model.lower():
                    credentials["extra_body"] = {
                        "reasoning_budget": NVIDIA_REASONING_BUDGET,
                        "chat_template_kwargs": {"enable_thinking": NVIDIA_ENABLE_THINKING}
                    }
                    logger.debug(f"NVIDIA NIM credentials: reasoning_budget={NVIDIA_REASONING_BUDGET}, enable_thinking={NVIDIA_ENABLE_THINKING}")
            logger.debug(f"OpenAI credentials retrieved for model: {model}")

    elif provider == "ollama":
        credentials["api_key"] = "ollama"  # Dummy key for LiteLLM
        credentials["api_base"] = OLLAMA_API_BASE
        credentials["num_ctx"] = OLLAMA_NUM_CTX
        logger.debug(f"Ollama credentials: api_base={OLLAMA_API_BASE}, num_ctx={OLLAMA_NUM_CTX}")

    elif provider in ("gemini", "google"):
        if USE_VERTEX_AUTH:
            credentials["vertex_project"] = VERTEX_PROJECT
            credentials["vertex_location"] = VERTEX_LOCATION
            credentials["custom_llm_provider"] = "vertex_ai"
            logger.debug(f"Vertex AI credentials: project={VERTEX_PROJECT}, location={VERTEX_LOCATION}")
        else:
            credentials["api_key"] = GEMINI_API_KEY
            logger.debug(f"Gemini API key credentials retrieved for model: {model}")

    elif provider == "anthropic":
        credentials["api_key"] = ANTHROPIC_API_KEY
        logger.debug(f"Anthropic credentials retrieved for model: {model}")

    else:
        logger.warning(f"Unknown provider '{provider}', returning empty credentials")

    return credentials


# Helper function to clean schema for Gemini
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini schema.")
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()): # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema

# Helper function to fix and parse tool arguments JSON
def parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """
    Parse tool arguments from various formats into a valid dictionary.
    Handles common JSON issues from different models including:
    - Single quotes instead of double quotes
    - Trailing commas
    - Unquoted keys
    - Incomplete/partial JSON
    - Escaped characters
    """
    if arguments is None:
        return {}

    if isinstance(arguments, dict):
        return arguments

    if not isinstance(arguments, str):
        return {"value": str(arguments)}

    # Empty string
    if not arguments.strip():
        return {}

    # Try direct JSON parse first
    try:
        result = json.loads(arguments)
        if isinstance(result, dict):
            return result
        return {"value": result}
    except json.JSONDecodeError:
        pass

    # Try a series of fixes for common JSON issues
    fixed = arguments.strip()

    # Fix 1: Remove trailing commas before } or ]
    try:
        fixed_trailing = re.sub(r',(\s*[}\]])', r'\1', fixed)
        result = json.loads(fixed_trailing)
        if isinstance(result, dict):
            return result
        return {"value": result}
    except (json.JSONDecodeError, re.error):
        pass

    # Fix 2: Single quotes to double quotes
    try:
        # Replace single quotes used as string delimiters
        fixed_quotes = re.sub(r"'([^']*)'(\s*[,:\]}])", r'"\1"\2', fixed)
        fixed_quotes = re.sub(r"(\{|\[|,)\s*'([^']*)'", r'\1"\2"', fixed_quotes)
        # Also handle keys with single quotes
        fixed_quotes = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', fixed_quotes)
        result = json.loads(fixed_quotes)
        if isinstance(result, dict):
            return result
        return {"value": result}
    except (json.JSONDecodeError, re.error):
        pass

    # Fix 3: Unquoted keys (e.g., {key: "value"} -> {"key": "value"})
    try:
        fixed_keys = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
        result = json.loads(fixed_keys)
        if isinstance(result, dict):
            return result
        return {"value": result}
    except (json.JSONDecodeError, re.error):
        pass

    # Fix 4: Combined fixes - trailing commas + single quotes + unquoted keys
    try:
        combined = fixed
        combined = re.sub(r',(\s*[}\]])', r'\1', combined)  # trailing commas
        combined = re.sub(r"'([^']*)'(\s*[,:\]}])", r'"\1"\2', combined)  # single quotes values
        combined = re.sub(r"(\{|\[|,)\s*'([^']*)'", r'\1"\2"', combined)  # single quotes after delimiters
        combined = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', combined)  # single quote keys
        combined = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', combined)  # unquoted keys
        result = json.loads(combined)
        if isinstance(result, dict):
            return result
        return {"value": result}
    except (json.JSONDecodeError, re.error):
        pass

    # Fix 5: Handle incomplete JSON by attempting to close brackets
    try:
        incomplete = fixed
        # Count brackets
        open_braces = incomplete.count('{') - incomplete.count('}')
        open_brackets = incomplete.count('[') - incomplete.count(']')
        # Close unclosed brackets
        if open_braces > 0 or open_brackets > 0:
            incomplete = incomplete.rstrip(',')  # remove trailing comma
            incomplete += ']' * open_brackets + '}' * open_braces
            result = json.loads(incomplete)
            if isinstance(result, dict):
                return result
            return {"value": result}
    except (json.JSONDecodeError, re.error):
        pass

    # Fix 6: Try wrapping as a simple string value
    try:
        result = json.loads(f'{{"value": {json.dumps(arguments)}}}')
        return result
    except json.JSONDecodeError:
        pass

    # Last resort: return empty dict to avoid breaking the tool call
    logger.warning(f"Could not parse tool arguments, returning empty dict: {arguments[:200] if len(arguments) > 200 else arguments}")
    return {}

# Models for Anthropic API requests (allow extra fields for forward compatibility)
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra="allow")


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]
    model_config = ConfigDict(extra="allow")


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]
    cache_control: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra="allow")


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]
    cache_control: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra="allow")


class ContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str
    model_config = ConfigDict(extra="allow")


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str
    model_config = ConfigDict(extra="allow")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult, ContentBlockThinking]]]
    model_config = ConfigDict(extra="allow")


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]
    model_config = ConfigDict(extra="allow")


class ThinkingConfig(BaseModel):
    enabled: bool = True
    model_config = ConfigDict(extra="allow")

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_field(cls, v, info):
        original_model = v
        new_model = map_model_name(v)
        if new_model != v:
            logger.debug(f"MODEL MAPPING: '{original_model}' -> '{new_model}'")
        if isinstance(info.data, dict):
            info.data['original_model'] = original_model
        return new_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_token_count(cls, v, info):
        original_model = v
        new_model = map_model_name(v)
        if new_model != v:
            logger.debug(f"TOKEN COUNT MAPPING: '{original_model}' -> '{new_model}'")
        if isinstance(info.data, dict):
            info.data['original_model'] = original_model
        return new_model

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse, ContentBlockThinking]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path
    
    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")
    
    # Process the request and get the response
    response = await call_next(request)
    
    return response

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"
        
    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except (TypeError, ValueError):
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except Exception:
                    result += "Unparseable content\n"
        return result.strip()
        
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except (TypeError, ValueError):
            return str(content)

    # Fallback for any other type
    try:
        return str(content)
    except Exception:
        return "Unparseable content"


async def call_ollama_native(anthropic_request: MessagesRequest, model: str, raw_request: Request):
    """
    Direct proxy to Ollama's native Anthropic-compatible API at /v1/messages.

    Ollama natively supports the Anthropic API format, so we simply proxy the request
    without any conversion. This is the same as using:
        ANTHROPIC_AUTH_TOKEN=ollama ANTHROPIC_BASE_URL=http://localhost:11434 claude

    The proxy only:
    1. Extracts the clean model name (removes ollama/ prefix)
    2. Forwards the request to Ollama's /v1/messages endpoint
    3. Returns Ollama's response as-is
    """
    # Extract clean model name (remove ollama/ or ollama_chat/ prefix)
    clean_model = model
    for prefix in ["ollama_chat/", "ollama/"]:
        if clean_model.startswith(prefix):
            clean_model = clean_model[len(prefix):]
            break

    # Get the raw request body and update the model
    body = await raw_request.body()
    body_dict = json.loads(body.decode('utf-8'))
    body_dict["model"] = clean_model

    logger.debug(f"🎯 Direct proxy to Ollama Anthropic API: model={clean_model}, stream={anthropic_request.stream}")

    # Prepare headers - pass through relevant headers from original request
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": raw_request.headers.get("anthropic-version", "2023-06-01"),
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if anthropic_request.stream:
                # Streaming response - proxy the stream directly
                async def stream_ollama():
                    async with client.stream(
                        "POST",
                        f"{OLLAMA_API_BASE.rstrip('/')}/v1/messages",
                        json=body_dict,
                        headers=headers
                    ) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes():
                            yield chunk

                return StreamingResponse(
                    stream_ollama(),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming response - return Ollama's response directly
                response = await client.post(
                    f"{OLLAMA_API_BASE.rstrip('/')}/v1/messages",
                    json=body_dict,
                    headers=headers
                )
                response.raise_for_status()
                # Return Ollama's response as-is (already in Anthropic format)
                return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Ollama API error: {e}")
        raise HTTPException(
            status_code=e.response.status_code if hasattr(e, 'response') else 500,
            detail=f"Ollama API error: {str(e)}"
        )


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format

    messages = []
    
    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    
    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            # OpenAI/LiteLLM format expects the assistant to call the tool, 
            # and the user's next message to include the result as plain text
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                # For user messages with tool_result, split into separate messages
                text_content = ""
                
                # Extract all text parts and concatenate them
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            # Add tool result as a message by itself - simulate the normal flow
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            
                            # Handle different formats of tool result content
                            result_content = ""
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    result_content = block.content
                                elif isinstance(block.content, list):
                                    # If content is a list of blocks, extract text from each
                                    for content_block in block.content:
                                        if hasattr(content_block, "type") and content_block.type == "text":
                                            result_content += content_block.text + "\n"
                                        elif isinstance(content_block, dict) and content_block.get("type") == "text":
                                            result_content += content_block.get("text", "") + "\n"
                                        elif isinstance(content_block, dict):
                                            # Handle any dict by trying to extract text or convert to JSON
                                            if "text" in content_block:
                                                result_content += content_block.get("text", "") + "\n"
                                            else:
                                                try:
                                                    result_content += json.dumps(content_block) + "\n"
                                                except (TypeError, ValueError):
                                                    result_content += str(content_block) + "\n"
                                elif isinstance(block.content, dict):
                                    # Handle dictionary content
                                    if block.content.get("type") == "text":
                                        result_content = block.content.get("text", "")
                                    else:
                                        try:
                                            result_content = json.dumps(block.content)
                                        except (TypeError, ValueError):
                                            result_content = str(block.content)
                                else:
                                    # Handle any other type by converting to string
                                    try:
                                        result_content = str(block.content)
                                    except Exception:
                                        result_content = "Unparseable content"
                            
                            # In OpenAI format, tool results come from the user (rather than being content blocks)
                            text_content += f"Tool result for {tool_id}:\n{result_content}\n"
                
                # Add as a single user message with all the content
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                # Regular handling for other message types
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            processed_content.append({"type": "image", "source": block.source})
                        elif block.type == "tool_use":
                            # Handle tool use blocks if needed
                            processed_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif block.type == "tool_result":
                            # Handle different formats of tool result content
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            }
                            
                            # Process the content field properly
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    # If it's a simple string, create a text block for it
                                    processed_content_block["content"] = [{"type": "text", "text": block.content}]
                                elif isinstance(block.content, list):
                                    # If it's already a list of blocks, keep it
                                    processed_content_block["content"] = block.content
                                else:
                                    # Default fallback
                                    processed_content_block["content"] = [{"type": "text", "text": str(block.content)}]
                            else:
                                # Default empty content
                                processed_content_block["content"] = [{"type": "text", "text": ""}]
                                
                            processed_content.append(processed_content_block)
                
                messages.append({"role": msg.role, "content": processed_content})
    
    # Apply max_tokens cap if configured (0 = no cap)
    max_tokens = anthropic_request.max_tokens
    if MAX_OUTPUT_TOKENS > 0:
        max_tokens = min(max_tokens, MAX_OUTPUT_TOKENS)
        if max_tokens != anthropic_request.max_tokens:
            logger.debug(f"Capping max_tokens to {MAX_OUTPUT_TOKENS} (original: {anthropic_request.max_tokens})")
    
    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,  # it understands "anthropic/claude-x" format
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    # Only include thinking field for Anthropic models
    if anthropic_request.thinking and anthropic_request.model.startswith("anthropic/"):
        litellm_request["thinking"] = anthropic_request.thinking

    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/")

        for tool in anthropic_request.tools:
            # Convert to dict if it's a pydantic model
            if hasattr(tool, 'dict'):
                tool_dict = tool.dict()
            else:
                # Ensure tool_dict is a dictionary, handle potential errors if 'tool' isn't dict-like
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                     logger.error(f"Could not convert tool to dict: {tool}")
                     continue # Skip this tool if conversion fails

            # Clean the schema if targeting a Gemini model
            input_schema = tool_dict.get("input_schema", {})
            if is_gemini_model:
                 logger.debug(f"Cleaning schema for Gemini tool: {tool_dict.get('name')}")
                 input_schema = clean_gemini_schema(input_schema)

            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema # Use potentially cleaned schema
                }
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools
    
    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'dict'):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice
            
        # Handle Anthropic's tool_choice format
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            # OpenAI doesn't support "any", use "required" as equivalent
            litellm_request["tool_choice"] = "required"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        else:
            # Default to auto if we can't determine
            litellm_request["tool_choice"] = "auto"
    
    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], 
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    
    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Check if this is a Claude model (which supports content blocks)
        is_claude_model = clean_model.startswith("claude-")
        
        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__ 
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }
                    
            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")
        
        # Create content list for Anthropic format
        content = []
        
        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})
        
        # Add tool calls if present (tool_use in Anthropic format) - only for Claude models
        if tool_calls and is_claude_model:
            logger.debug(f"Processing tool calls: {tool_calls}")
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")
                
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Parse arguments using the helper function
                arguments = parse_tool_arguments(arguments)

                logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")

                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })
        elif tool_calls and not is_claude_model:
            # For non-Claude models, convert tool calls to text format
            logger.debug(f"Converting tool calls to text for non-Claude model: {clean_model}")
            
            # We'll append tool info to the text content
            tool_text = "\n\nTool usage:\n"
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        args_dict = json.loads(arguments)
                        arguments_str = json.dumps(args_dict, indent=2)
                    except json.JSONDecodeError:
                        arguments_str = arguments
                else:
                    arguments_str = json.dumps(arguments, indent=2)
                
                tool_text += f"Tool: {name}\nArguments: {arguments_str}\n\n"
            
            # Add or append tool text to content
            if content and content[0]["type"] == "text":
                content[0]["text"] += tool_text
            else:
                content.append({"type": "text", "text": tool_text})
        
        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        
        # Map OpenAI finish_reason to Anthropic stop_reason
        STOP_REASON_MAP = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "function_call": "tool_use",  # Legacy OpenAI
            "content_filter": "end_turn",  # Content filtered
        }
        stop_reason = STOP_REASON_MAP.get(finish_reason, "end_turn")
        
        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )
        
        return anthropic_response
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error converting response: {str(e)}\n{error_traceback}")

        # Raise HTTPException instead of returning error in content (would be HTTP 200)
        raise HTTPException(
            status_code=500,
            detail=f"Error converting response: {str(e)}"
        )

# Streaming constants
STREAMING_CHUNK_TIMEOUT = 60.0  # Timeout for individual chunks (seconds)
MAX_ACCUMULATED_TEXT_SIZE = 10 * 1024 * 1024  # 10MB max accumulated text
ANTHROPIC_ID_LENGTH = 24


async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:ANTHROPIC_ID_LENGTH]}"  # Format similar to Anthropic's IDs
        
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        
        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        
        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0
        
        # Process each chunk with timeout and memory protection
        async def get_next_chunk():
            """Get next chunk with timeout."""
            return await asyncio.wait_for(
                response_generator.__anext__(),
                timeout=STREAMING_CHUNK_TIMEOUT
            )

        while True:
            try:
                chunk = await get_next_chunk()
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for chunk after {STREAMING_CHUNK_TIMEOUT}s")
                break

            try:
                # Check memory limit
                if len(accumulated_text) > MAX_ACCUMULATED_TEXT_SIZE:
                    logger.warning(f"Accumulated text exceeded {MAX_ACCUMULATED_TEXT_SIZE} bytes, truncating")
                    accumulated_text = accumulated_text[-MAX_ACCUMULATED_TEXT_SIZE:]

                # Check if this is the end of the response with usage data
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                
                # Handle text content
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Get the delta from the choice
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, 'message', {})
                    
                    # Check for finish_reason to know when we're done
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    # Process text content
                    delta_content = None
                    
                    # Handle different formats of delta content
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']
                    
                    # Accumulate text content
                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content
                        
                        # Always emit text deltas if no tool calls started
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"
                    
                    # Process tool calls
                    delta_tool_calls = None
                    
                    # Handle different formats of tool calls
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']
                    
                    # Process tool calls if any
                    if delta_tool_calls:
                        # First tool call we've seen - need to handle text properly
                        if tool_index is None:
                            # If we've been streaming text, close that text block
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # If we've accumulated text but not sent it, we need to emit it now
                            # This handles the case where the first delta has both text and a tool call
                            elif accumulated_text and not text_sent and not text_block_closed:
                                # Send the accumulated text
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                # Close the text block
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # Close text block even if we haven't sent anything - models sometimes emit empty text blocks
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                
                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]
                        
                        for tool_call in delta_tool_calls:
                            # Get the index of this tool call (for multiple tools)
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0
                            
                            # Check if this is a new tool or a continuation
                            if tool_index is None or current_index != tool_index:
                                # New tool call - create a new tool_use block
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index
                                
                                # Extract function info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    name = function.get('name', '') if isinstance(function, dict) else ""
                                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    name = getattr(function, 'name', '') if function else ''
                                    tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")
                                
                                # Start a new tool_use block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""
                            
                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''

                            # If we have arguments, send them as a delta
                            if arguments:
                                # In streaming mode, arguments come as incremental JSON fragments
                                # that should be passed through directly to the client.
                                # The client will accumulate these fragments into complete JSON.
                                if isinstance(arguments, dict):
                                    # Already a dict, convert to JSON string
                                    args_json = json.dumps(arguments)
                                elif isinstance(arguments, str):
                                    # Pass through string fragments directly - they are partial JSON
                                    # that will be assembled by the client
                                    args_json = arguments
                                else:
                                    args_json = str(arguments)

                                # Add to accumulated tool content
                                tool_content += args_json if isinstance(args_json, str) else ""

                                # Send the update
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"
                    
                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        
                        # Close any open tool call blocks
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                        
                        # If we accumulated text but never sent or closed text block, do it now
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                # Send the accumulated text
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            # Close the text block
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        
                        # Map OpenAI finish_reason to Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"
                        
                        # Send message_delta with stop reason and usage
                        usage = {"output_tokens": output_tokens}
                        
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"
                        
                        # Send message_stop event
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        
                        # Send final [DONE] marker to match Anthropic's behavior
                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            # Close any open tool call blocks
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
            
            # Close the text content block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            
            # Send final message_delta with usage
            usage = {"output_tokens": output_tokens}
            
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"
            
            # Send message_stop event
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            
            # Send final [DONE] marker to match Anthropic's behavior
            yield "data: [DONE]\n\n"
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in streaming: {str(e)}\n{error_traceback}")

        # Send error as text content block and use valid stop_reason
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': f'[Streaming error: {str(e)}]'}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        # Ensure response generator is properly closed
        if hasattr(response_generator, 'aclose'):
            try:
                await response_generator.aclose()
            except Exception:
                pass  # Ignore cleanup errors

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    try:
        # print the body here
        body = await raw_request.body()
    
        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")

        # Determine provider from model prefix
        provider = "openai"  # Default provider for most models
        if request.model.startswith("gemini/"):
            provider = "gemini"
        elif request.model.startswith("ollama/") or request.model.startswith("ollama_chat/"):
            provider = "ollama"
        elif request.model.startswith("anthropic/"):
            provider = "anthropic"

        # For Ollama models, use native API (supports Anthropic format directly)
        if provider == "ollama":
            logger.debug(f"🎯 Using native Ollama Anthropic API for model: {request.model}")
            ollama_response = await call_ollama_native(request, request.model, raw_request)

            # If it's a StreamingResponse, return it directly
            if isinstance(ollama_response, StreamingResponse):
                return ollama_response

            # Otherwise, it's already in Anthropic format, return directly
            return ollama_response

        # For other providers, convert to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)

        # Get credentials for this provider (with intelligent routing for openai)
        credentials = get_credentials_for_provider(provider, request.model)

        # Apply credentials to the request
        litellm_request.update(credentials)
        
        # For OpenAI models - modify request format to work with limitations
        if "openai" in litellm_request["model"] and "messages" in litellm_request:
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")
            
            # For OpenAI models, we need to convert content blocks to simple strings
            # and handle other requirements
            for i, msg in enumerate(litellm_request["messages"]):
                # Special case - handle message content directly when it's a list of tool_result
                # This is a specific case we're seeing in the error
                if "content" in msg and isinstance(msg["content"], list):
                    is_only_tool_result = True
                    for block in msg["content"]:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            is_only_tool_result = False
                            break
                    
                    if is_only_tool_result and len(msg["content"]) > 0:
                        logger.warning(f"Found message with only tool_result content - special handling required")
                        # Extract the content from all tool_result blocks
                        all_text = ""
                        for block in msg["content"]:
                            all_text += "Tool Result:\n"
                            result_content = block.get("content", [])
                            
                            # Handle different formats of content
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        all_text += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        # Fall back to string representation of any dict
                                        try:
                                            item_text = item.get("text", json.dumps(item))
                                            all_text += item_text + "\n"
                                        except (TypeError, ValueError):
                                            all_text += str(item) + "\n"
                            elif isinstance(result_content, str):
                                all_text += result_content + "\n"
                            else:
                                try:
                                    all_text += json.dumps(result_content) + "\n"
                                except (TypeError, ValueError):
                                    all_text += str(result_content) + "\n"
                        
                        # Replace the list with extracted text
                        litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                        logger.warning(f"Converted tool_result to plain text: {all_text.strip()[:200]}...")
                        continue  # Skip normal processing for this message
                
                # 1. Handle content field - normal case
                if "content" in msg:
                    # Check if content is a list (content blocks)
                    if isinstance(msg["content"], list):
                        # Convert complex content blocks to simple string
                        text_content = ""
                        for block in msg["content"]:
                            if isinstance(block, dict):
                                # Handle different content block types
                                if block.get("type") == "text":
                                    text_content += block.get("text", "") + "\n"
                                
                                # Handle tool_result content blocks - extract nested text
                                elif block.get("type") == "tool_result":
                                    tool_id = block.get("tool_use_id", "unknown")
                                    text_content += f"[Tool Result ID: {tool_id}]\n"
                                    
                                    # Extract text from the tool_result content
                                    result_content = block.get("content", [])
                                    if isinstance(result_content, list):
                                        for item in result_content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                text_content += item.get("text", "") + "\n"
                                            elif isinstance(item, dict):
                                                # Handle any dict by trying to extract text or convert to JSON
                                                if "text" in item:
                                                    text_content += item.get("text", "") + "\n"
                                                else:
                                                    try:
                                                        text_content += json.dumps(item) + "\n"
                                                    except (TypeError, ValueError):
                                                        text_content += str(item) + "\n"
                                    elif isinstance(result_content, dict):
                                        # Handle dictionary content
                                        if result_content.get("type") == "text":
                                            text_content += result_content.get("text", "") + "\n"
                                        else:
                                            try:
                                                text_content += json.dumps(result_content) + "\n"
                                            except (TypeError, ValueError):
                                                text_content += str(result_content) + "\n"
                                    elif isinstance(result_content, str):
                                        text_content += result_content + "\n"
                                    else:
                                        try:
                                            text_content += json.dumps(result_content) + "\n"
                                        except (TypeError, ValueError):
                                            text_content += str(result_content) + "\n"
                                
                                # Handle tool_use content blocks
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    tool_id = block.get("id", "unknown")
                                    tool_input = json.dumps(block.get("input", {}))
                                    text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                                
                                # Handle image content blocks
                                elif block.get("type") == "image":
                                    text_content += "[Image content - not displayed in text format]\n"
                        
                        # Make sure content is never empty for OpenAI models
                        if not text_content.strip():
                            text_content = "..."
                        
                        litellm_request["messages"][i]["content"] = text_content.strip()
                    # Also check for None or empty string content
                    elif msg["content"] is None:
                        litellm_request["messages"][i]["content"] = "..." # Empty content not allowed
                
                # 2. Remove any fields OpenAI doesn't support in messages
                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        logger.warning(f"Removing unsupported field from message: {key}")
                        del msg[key]
            
            # 3. Final validation - check for any remaining invalid values and dump full message details
            for i, msg in enumerate(litellm_request["messages"]):
                # Log the message format for debugging
                logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")
                
                # If content is still a list or None, replace with placeholder
                if isinstance(msg.get("content"), list):
                    logger.warning(f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'))}")
                    # Last resort - stringify the entire content as JSON
                    litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
                elif msg.get("content") is None:
                    logger.warning(f"Message {i} has None content - replacing with placeholder")
                    litellm_request["messages"][i]["content"] = "..." # Fallback placeholder
        
        # Only log basic info about the request, not the full details
        logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")

        # Ensure last message is from user (required by OpenAI-compatible APIs like NVIDIA)
        # This fixes the "Cannot set add_generation_prompt to True" error
        if litellm_request["messages"] and litellm_request["messages"][-1].get("role") == "assistant":
            logger.debug("Last message is from assistant - adding continuation prompt")
            litellm_request["messages"].append({
                "role": "user",
                "content": "Continue."
            })

        # Handle streaming mode
        if request.stream:
            # Use LiteLLM for streaming
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            # Ensure we use the async version for streaming
            response_generator = await litellm.acompletion(**litellm_request)
            
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")
            
            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            
            return anthropic_response
                
    except Exception as e:
        error_traceback = traceback.format_exc()

        # Capture as much info as possible about the error (don't expose traceback to client)
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
        }
        
        # Check for LiteLLM-specific attributes
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)
        
        # Check for additional exception details in dictionaries
        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__']:
                    error_details[key] = str(value)
        
        def sanitize_for_json(obj):
            """Recursively sanitize objects for JSON serialization."""
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return sanitize_for_json(obj.__dict__)
            elif hasattr(obj, 'text'):
                return str(obj.text)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # Log all error details with safe serialization (include traceback for server logs)
        sanitized_details = sanitize_for_json(error_details)
        logger.error(f"Error processing request: {json.dumps(sanitized_details, indent=2)}\n{error_traceback}")
        
        # Format error for response
        error_message = f"Error: {str(e)}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\nMessage: {error_details['message']}"
        if 'response' in error_details and error_details['response']:
            error_message += f"\nResponse: {error_details['response']}"
        
        # Return detailed error
        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)

@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    try:
        # Log the incoming token count request
        original_model = request.original_model or request.model
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter
            
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            
            # Count tokens (token_counter only needs model and messages)
            token_count = token_counter(
                model=converted_request["model"],
                messages=converted_request["messages"],
            )
            
            # Return Anthropic-style response
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            return TokenCountResponse(input_tokens=1000)  # Default fallback
            
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


# =============================================================================
# Dynamic Routing Endpoints - Allow specifying provider:model in the URL path
# =============================================================================

VALID_PROVIDERS = {"openai", "ollama", "gemini", "google", "anthropic"}

def parse_provider_model(provider_model: str) -> tuple[str, str]:
    """
    Parse provider_model string to extract provider and model.
    Format: "provider:model" where model may contain colons (e.g., "ollama:qwen2.5-coder:32b")

    Returns: (provider, model) tuple
    Raises: HTTPException if provider is invalid
    """
    if ":" not in provider_model:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider_model format: '{provider_model}'. Expected format: 'provider:model'"
        )

    # Split on first colon only - model may contain additional colons
    parts = provider_model.split(":", 1)
    provider = parts[0].lower()
    model = parts[1]

    if provider not in VALID_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider: '{provider}'. Valid providers are: {', '.join(sorted(VALID_PROVIDERS))}"
        )

    return provider, model


def get_prefixed_model(provider: str, model: str) -> str:
    """
    Add the correct LiteLLM prefix based on the provider.

    Returns: Model name with appropriate prefix for LiteLLM
    """
    if provider == "openai":
        return f"openai/{model}"
    elif provider == "ollama":
        return f"ollama_chat/{model}"
    elif provider in ("gemini", "google"):
        return f"gemini/{model}"
    elif provider == "anthropic":
        return f"anthropic/{model}"
    else:
        # Shouldn't reach here due to validation, but just in case
        return model


@app.post("/{provider_model:path}/v1/messages/count_tokens")
async def count_tokens_dynamic(
    provider_model: str,
    request: TokenCountRequest,
    raw_request: Request
):
    """
    Token counting endpoint with dynamic model selection via URL path.

    URL format: /{provider}:{model}/v1/messages/count_tokens
    Examples:
        - /openai:gpt-4.1/v1/messages/count_tokens
        - /ollama:qwen2.5-coder:32b/v1/messages/count_tokens
        - /gemini:gemini-2.0-flash/v1/messages/count_tokens
    """
    try:
        # Parse the provider and model from URL path
        provider, model = parse_provider_model(provider_model)
        prefixed_model = get_prefixed_model(provider, model)

        logger.debug(f"Dynamic token count: provider={provider}, model={model}, prefixed={prefixed_model}")

        # Create a new request with the extracted model (bypassing map_model_name)
        modified_request = TokenCountRequest(
            model=prefixed_model,
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            tool_choice=request.tool_choice,
            thinking=request.thinking,
            original_model=model  # Store original for logging
        )

        # Delegate to the main count_tokens function
        return await count_tokens(modified_request, raw_request)

    except HTTPException:
        raise
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in dynamic token counting: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


@app.post("/{provider_model:path}/v1/messages")
async def create_message_dynamic(
    provider_model: str,
    request: MessagesRequest,
    raw_request: Request
):
    """
    Messages endpoint with dynamic model selection via URL path.

    URL format: /{provider}:{model}/v1/messages
    Examples:
        - /openai:gpt-4.1/v1/messages
        - /ollama:qwen2.5-coder:32b/v1/messages
        - /gemini:gemini-2.0-flash/v1/messages
        - /anthropic:claude-sonnet-4-20250514/v1/messages
    """
    try:
        # Parse the provider and model from URL path
        provider, model = parse_provider_model(provider_model)
        prefixed_model = get_prefixed_model(provider, model)

        logger.debug(f"Dynamic message: provider={provider}, model={model}, prefixed={prefixed_model}")

        # Create a modified request with the extracted model
        # We need to override the model field to bypass map_model_name
        modified_request = MessagesRequest(
            model=prefixed_model,
            max_tokens=request.max_tokens,
            messages=request.messages,
            system=request.system,
            stop_sequences=request.stop_sequences,
            stream=request.stream,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            metadata=request.metadata,
            tools=request.tools,
            tool_choice=request.tool_choice,
            thinking=request.thinking
        )

        # Delegate to the main create_message function
        return await create_message(modified_request, raw_request)

    except HTTPException:
        raise
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in dynamic message creation: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error creating message: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Claude Code Proxy"}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"
    
    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    # Format status code
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"

    # Put it all together
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"

    # Log to console (use print for ANSI colors since logger may strip them)
    sys.stdout.write(f"{log_line}\n{model_line}\n")
    sys.stdout.flush()

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Claude Code Proxy Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=4000, help="Port to bind to (default: 4000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    parser.add_argument("--log-level", default="error", choices=["debug", "info", "warning", "error"], help="Log level (default: error)")

    args = parser.parse_args()

    print(f"{Colors.BOLD}Claude Code Proxy Server{Colors.RESET}")
    print(f"Running on {Colors.CYAN}http://{args.host}:{args.port}{Colors.RESET}")

    if args.reload:
        uvicorn.run("server:app", host=args.host, port=args.port, log_level=args.log_level, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
