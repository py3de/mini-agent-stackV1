"""Model-agnostic LLM layer.

Every tool in the stack gets its model from `get_llm()`, never from a provider SDK directly.
Switching providers is a config change (LLM_PROVIDER / --provider), not a code change, and
adding a provider means adding one entry to PROVIDERS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from agent_stack.config import settings


@dataclass(frozen=True)
class ProviderSpec:
    """How to reach one provider through LangChain's `init_chat_model`."""

    langchain_provider: str  # provider id understood by init_chat_model
    default_model: str  # sensible default, overridable via MODEL_<NAME>
    api_key_env: str | None  # env var that must be set, or None (e.g. local Ollama)


PROVIDERS: dict[str, ProviderSpec] = {
    "google": ProviderSpec("google_genai", "gemini-3.8-flash", "GOOGLE_API_KEY"),
    "openai": ProviderSpec("openai", "gpt-4.1-mini", "OPENAI_API_KEY"),
    "anthropic": ProviderSpec("anthropic", "claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    "ollama": ProviderSpec("ollama", "llama3.1", None),
}


class ProviderNotConfigured(RuntimeError):
    """Raised when a provider is requested but its API key is missing."""


def model_name_for(provider: str) -> str:
    spec = PROVIDERS[provider]
    return os.getenv(f"MODEL_{provider.upper()}", spec.default_model)


def available_providers() -> list[str]:
    """Providers that can be used right now (key present, or no key needed)."""
    return [
        name
        for name, spec in PROVIDERS.items()
        if spec.api_key_env is None or os.getenv(spec.api_key_env)
    ]


# Backup models used automatically when the primary model fails (e.g. "503 high demand").
# Override with LLM_FALLBACKS="google:gemini-3.5-flash-lite,openai:gpt-4.1-mini" (provider:model, comma-separated).
DEFAULT_FALLBACKS: dict[str, list[str]] = {
    "google": ["google:gemini-3.5-flash-lite"],
}

TRANSIENT_MARKERS = (
    "503", "unavailable", "504", "deadline_exceeded", "deadline expired", "429", "resource_exhausted",
    "rate limit", "overloaded", "high demand", "timeout", "timed out", "500 internal",
)


def is_transient_error(exc: BaseException) -> bool:
    """True for errors worth retrying: overload, rate limits, timeouts. Not for bad keys or wrong model names."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def get_fallback_llms(provider: str | None = None, temperature: float | None = None) -> list[BaseChatModel]:
    """Configured backup models for a provider. Backups whose API key is missing are skipped."""
    provider = (provider or settings.llm_provider).lower()
    raw = os.getenv("LLM_FALLBACKS")
    entries = [e.strip() for e in raw.split(",") if e.strip()] if raw is not None else DEFAULT_FALLBACKS.get(provider, [])
    fallbacks = []
    for entry in entries:
        fb_provider, _, fb_model = entry.partition(":")
        try:
            fallbacks.append(get_llm(fb_provider, fb_model or None, temperature))
        except (ProviderNotConfigured, ValueError):
            continue
    return fallbacks


def get_llm(provider: str | None = None, model: str | None = None, temperature: float | None = None) -> BaseChatModel:
    """Return a chat model for the given provider, behind one common interface.

    All returned models support `.bind_tools()` and `.with_structured_output()`, which is
    what the agent relies on, so the agent code never needs to know which provider it runs on.
    """
    provider = (provider or settings.llm_provider).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}")

    spec = PROVIDERS[provider]
    if spec.api_key_env and not os.getenv(spec.api_key_env):
        raise ProviderNotConfigured(
            f"Provider '{provider}' needs {spec.api_key_env} to be set (see .env.example)."
        )

    kwargs: dict = {}
    # Keep SDK retries low and let our own fallback/retry layer take over: SDK defaults (e.g. 6 silent retries
    # on Google) made an overloaded model cost minutes before the backup model was even tried.
    if provider != "ollama":
        kwargs |= {"max_retries": int(os.getenv("LLM_MAX_RETRIES", "1")), "timeout": float(os.getenv("LLM_TIMEOUT_S", "120"))}
    # Only send a temperature when one is set: newer models (e.g. Gemini 3.x) use fixed sampling and warn otherwise.
    temperature = temperature if temperature is not None else (float(t) if (t := os.getenv("LLM_TEMPERATURE")) else None)
    if temperature is not None:
        kwargs["temperature"] = temperature

    return init_chat_model(model or model_name_for(provider), model_provider=spec.langchain_provider, **kwargs)
