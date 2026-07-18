"""Wire application settings and ``tasks.json`` into a ready LLM client.

This is the one place that maps a provider name to its adapter and its API key.
It builds only the adapters the registry actually references, so a deployment
that uses only Gemini needs no OpenRouter key.
"""

from pathlib import Path

from lexme.config import Settings
from lexme.llm.client import RoutingLlmClient
from lexme.llm.gemini_adapter import GeminiAdapter
from lexme.llm.openrouter_adapter import OpenRouterAdapter
from lexme.llm.protocol import LlmClient
from lexme.llm.provider import ProviderAdapter
from lexme.llm.registry import DEFAULT_TASKS_PATH, LlmConfigError, load_task_registry


def build_llm_client(settings: Settings, tasks_path: Path = DEFAULT_TASKS_PATH) -> LlmClient:
    """Return a :class:`LlmClient` routing every task to its configured provider.

    Reads the task registry from ``tasks_path`` and builds one adapter per
    referenced provider using the API keys in ``settings``. Raises
    :class:`LlmConfigError` if the registry names an unknown provider.
    """
    registry = load_task_registry(tasks_path)
    adapters = {provider: _build_adapter(provider, settings) for provider in registry.providers()}
    return RoutingLlmClient(registry, adapters)


def _build_adapter(provider: str, settings: Settings) -> ProviderAdapter:
    """Build the adapter for one provider from settings, or fail loudly."""
    if provider == "gemini":
        return GeminiAdapter(settings.gemini_api_key)
    if provider == "openrouter":
        return OpenRouterAdapter(settings.openrouter_api_key)
    raise LlmConfigError(f"no adapter is registered for provider {provider!r}")
