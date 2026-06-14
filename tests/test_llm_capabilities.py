"""Unit tests for the per-model capability registry (no DB)."""

from app.core.llm_capabilities import resolve_capabilities


def test_gemini_is_fully_multimodal():
    caps = resolve_capabilities("google", "gemini-2.5-flash")
    assert caps.vision and caps.audio


def test_gemini_3_preview_matches_prefix():
    caps = resolve_capabilities("google", "gemini-3-flash-preview")
    assert caps.vision and caps.audio


def test_claude_has_vision_but_no_audio():
    caps = resolve_capabilities("anthropic", "claude-sonnet-4-6")
    assert caps.vision and not caps.audio


def test_longest_prefix_wins():
    # gpt-4o-audio must match its own entry, not the plain gpt-4o one
    assert resolve_capabilities("openai", "gpt-4o-audio-preview").audio
    assert not resolve_capabilities("openai", "gpt-4o-mini").audio


def test_unknown_model_defaults_to_text_only(caplog):
    caps = resolve_capabilities("ollama", "llama3.3")
    assert not caps.vision and not caps.audio
    assert any("text-only" in r.message for r in caplog.records)


def test_env_overrides_beat_registry():
    caps = resolve_capabilities(
        "ollama", "llava", vision_override=True, audio_override=False
    )
    assert caps.vision and not caps.audio


def test_partial_override_resolves_caps_and_stays_quiet(caplog):
    # Unknown model + only vision set (the OpenRouter/Minimax case): vision must
    # be on, and the "text-only" warning must NOT fire — it's misleading once
    # an override is configured.
    caps = resolve_capabilities(
        "openrouter", "minimax/minimax-m3", vision_override=True
    )
    assert caps.vision and not caps.audio
    assert not any("text-only" in r.message for r in caplog.records)
