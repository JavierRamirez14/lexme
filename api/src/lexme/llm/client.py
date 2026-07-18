"""The routing client: resolve a task, dispatch to its provider, parse the reply.

This is the real :class:`~lexme.llm.protocol.LlmClient`. It holds the task
registry and one adapter per provider; it never imports a provider SDK.
"""

from collections.abc import Mapping, Sequence

from lexme.llm.protocol import Message, ModelT, TaskNotConfiguredError
from lexme.llm.provider import ProviderAdapter, ProviderRequest
from lexme.llm.registry import LlmConfigError, TaskModel, TaskRegistry
from lexme.llm.structured import json_schema_instruction, parse_model


class RoutingLlmClient:
    """Route each call to the provider its task names in the registry.

    ``adapters`` maps a provider name to the adapter that speaks to it. Every
    provider referenced by the registry must be present, or construction fails.
    """

    def __init__(self, registry: TaskRegistry, adapters: Mapping[str, ProviderAdapter]) -> None:
        """Build the client, checking every configured provider has an adapter."""
        missing = registry.providers() - adapters.keys()
        if missing:
            raise LlmConfigError(f"no adapter supplied for provider(s): {sorted(missing)}")
        self._registry = registry
        self._adapters = dict(adapters)

    def complete(self, task: str, messages: Sequence[Message]) -> str:
        """Run ``task`` over ``messages`` and return the reply text."""
        spec = self._resolve(task)
        return self._adapter_for(spec).complete(
            ProviderRequest(model=spec.model, temperature=spec.temperature, messages=messages)
        )

    def complete_structured(
        self, task: str, messages: Sequence[Message], response_model: type[ModelT]
    ) -> ModelT:
        """Run ``task`` and parse the reply into ``response_model``."""
        spec = self._resolve(task)
        instructed = (*messages, Message("system", json_schema_instruction(response_model)))
        raw = self._adapter_for(spec).complete(
            ProviderRequest(
                model=spec.model,
                temperature=spec.temperature,
                messages=instructed,
                json_mode=True,
            )
        )
        return parse_model(raw, response_model)

    def _resolve(self, task: str) -> TaskModel:
        """Look up ``task`` in the registry or raise a caller-facing error."""
        if not self._registry.has(task):
            raise TaskNotConfiguredError(
                f"task '{task}' is not in the registry; known tasks: {self._registry.tasks()}"
            )
        return self._registry.resolve(task)

    def _adapter_for(self, spec: TaskModel) -> ProviderAdapter:
        """Return the adapter for a resolved task's provider."""
        return self._adapters[spec.provider]
