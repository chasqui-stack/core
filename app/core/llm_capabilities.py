"""Per-model modality support (vision/audio input).

The gateway normalizes text/audio/image/button into the canonical contract,
but whether the LLM can actually SEE an image or HEAR an audio depends on the
configured model. The orchestrator (Sprint 3) checks these flags to degrade
gracefully — e.g. reply "envíamelo como texto" instead of failing the turn.

Resolution order:
1. Explicit env overrides (LLM_SUPPORTS_VISION / LLM_SUPPORTS_AUDIO) — for
   models we don't know about (ollama fine-tunes, openrouter routes, etc.).
2. Longest-prefix match against the registry below.
3. Unknown model → assume TEXT-ONLY and log a warning (safe default).
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCapabilities:
    vision: bool = False  # can take image input
    audio: bool = False   # can take audio input natively (no STT step)


# Keyed by "<llm_provider>:<model-prefix>" — longest prefix wins.
# Conservative on purpose: only modalities the provider's chat API accepts
# natively. (e.g. plain OpenAI gpt-4o chat does NOT take audio input — that's
# the audio-preview/realtime variants.)
_REGISTRY: dict[str, ModelCapabilities] = {
    # Google Gemini — natively multimodal across the family
    "google:gemini-3": ModelCapabilities(vision=True, audio=True),
    "google:gemini-2.5": ModelCapabilities(vision=True, audio=True),
    "google:gemini-2.0": ModelCapabilities(vision=True, audio=True),
    # Anthropic Claude — vision yes; no native audio input
    "anthropic:claude": ModelCapabilities(vision=True, audio=False),
    # OpenAI — vision yes on the chat models; audio only on special variants
    "openai:gpt-5": ModelCapabilities(vision=True, audio=False),
    "openai:gpt-4o-audio": ModelCapabilities(vision=True, audio=True),
    "openai:gpt-4o": ModelCapabilities(vision=True, audio=False),
    "openai:gpt-4.1": ModelCapabilities(vision=True, audio=False),
    # ollama / openrouter / others: no safe assumption — use env overrides
}


def resolve_capabilities(
    provider: str,
    model: str,
    *,
    vision_override: bool | None = None,
    audio_override: bool | None = None,
) -> ModelCapabilities:
    """Capabilities for the configured model (overrides > registry > text-only)."""
    key = f"{provider}:{model}"

    matched: tuple[int, ModelCapabilities] | None = None
    for prefix, caps in _REGISTRY.items():
        if key.startswith(prefix) and (matched is None or len(prefix) > matched[0]):
            matched = (len(prefix), caps)

    if matched is None and (vision_override is None or audio_override is None):
        logger.warning(
            "Unknown LLM '%s' — assuming text-only. Set LLM_SUPPORTS_VISION / "
            "LLM_SUPPORTS_AUDIO to override.",
            key,
        )

    base = matched[1] if matched else ModelCapabilities()
    return ModelCapabilities(
        vision=vision_override if vision_override is not None else base.vision,
        audio=audio_override if audio_override is not None else base.audio,
    )
