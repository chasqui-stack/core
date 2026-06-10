"""Chat model factory — one place where the LLM is chosen.

Everything else (orchestrator, tools, memory) talks to a `BaseChatModel`;
swapping provider/model is a .env change, never a code change:

    LLM_PROVIDER=google     LLM_MODEL=gemini-3-flash-preview
    LLM_PROVIDER=anthropic  LLM_MODEL=claude-sonnet-4-6
    LLM_PROVIDER=openai     LLM_MODEL=gpt-5-mini
    LLM_PROVIDER=ollama     LLM_MODEL=llama3.3          # local, no API key
    LLM_PROVIDER=openrouter LLM_MODEL=qwen/qwen3-coder  # OpenAI-compatible

Modality support (vision/audio) is resolved separately by
app/core/llm_capabilities.py using the same provider/model pair.
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.core.config import settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# settings.llm_provider → init_chat_model's model_provider
_PROVIDER_MAP = {
    "google": "google_genai",
    "openrouter": "openai",  # OpenAI-compatible API
}


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    **kwargs,
) -> BaseChatModel:
    """Build the configured chat model (args default to settings)."""
    provider = provider or settings.llm_provider
    model = model or settings.llm_model

    init_kwargs: dict = {
        "model_provider": _PROVIDER_MAP.get(provider, provider),
        "temperature": settings.llm_temperature,
    }

    if provider == "google" and settings.google_api_key:
        init_kwargs["api_key"] = settings.google_api_key
    elif provider == "anthropic" and settings.anthropic_api_key:
        init_kwargs["api_key"] = settings.anthropic_api_key
    elif provider == "openai" and settings.openai_api_key:
        init_kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:  # any OpenAI-compatible server
            init_kwargs["base_url"] = settings.openai_base_url
    elif provider == "openrouter":
        init_kwargs["api_key"] = settings.openrouter_api_key
        init_kwargs["base_url"] = OPENROUTER_BASE_URL
    elif provider == "ollama" and settings.ollama_base_url:
        init_kwargs["base_url"] = settings.ollama_base_url

    init_kwargs.update(kwargs)
    return init_chat_model(model, **init_kwargs)
