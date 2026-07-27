"""The routing client: resolve a task, dispatch to its provider, parse the reply.

This is the real :class:`~lexme.llm.protocol.LlmClient`. It holds the task
registry and one adapter per provider; it never imports a provider SDK.

A structured call that comes back unparseable is resampled a bounded number of
times. The free-tier models this interface targets occasionally corrupt an
otherwise well-formed object -- a stray token after the closing bracket, a bad
escape mid-string -- and the same prompt asked again usually comes back clean.
One such reply would otherwise cost a whole harness run, or a user's answer.
"""

import logging
from collections.abc import Mapping, Sequence

from lexme.llm.protocol import Message, ModelT, StructuredOutputError, TaskNotConfiguredError
from lexme.llm.provider import ProviderAdapter, ProviderRequest
from lexme.llm.registry import LlmConfigError, TaskModel, TaskRegistry
from lexme.llm.structured import json_schema_instruction, parse_model

logger = logging.getLogger(__name__)

#: How many times a structured call is asked before its reply is given up on.
STRUCTURED_ATTEMPTS = 3


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
        """Run ``task`` and parse the reply into ``response_model``.

        Resamples up to :data:`STRUCTURED_ATTEMPTS` times while the reply does not
        parse, then raises the last
        :class:`~lexme.llm.protocol.StructuredOutputError`.
        """
        spec = self._resolve(task)
        request = ProviderRequest(
            model=spec.model,
            temperature=spec.temperature,
            messages=_instructed(messages, response_model),
            json_mode=True,
        )
        adapter = self._adapter_for(spec)
        for attempt in range(1, STRUCTURED_ATTEMPTS + 1):
            try:
                return parse_model(adapter.complete(request), response_model)
            except StructuredOutputError as error:
                if attempt == STRUCTURED_ATTEMPTS:
                    raise
                logger.warning(
                    "task '%s' returned an unparseable reply (attempt %d/%d): %s",
                    task,
                    attempt,
                    STRUCTURED_ATTEMPTS,
                    error,
                )
        raise AssertionError("unreachable: the loop returns or raises on the last attempt")

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


def _instructed(messages: Sequence[Message], response_model: type[ModelT]) -> tuple[Message, ...]:
    """Fold ``response_model``'s schema instruction into the task's system message.

    It is merged into the last system message rather than appended as one of its
    own after the user's turn. A provider that concatenates system messages sees
    the same text either way, but an OpenAI-compatible provider keeps the turns in
    order and reads a system message that follows the user's turn as background
    rather than as the thing to do, and answers in prose instead of JSON. Merging
    also keeps the instruction out of the user turn, which the flash models are
    sensitive to on long prompts.
    """
    instruction = json_schema_instruction(response_model)
    for index in reversed(range(len(messages))):
        if messages[index].role == "system":
            merged = Message("system", f"{messages[index].content}\n\n{instruction}")
            return (*messages[:index], merged, *messages[index + 1 :])
    return (Message("system", instruction), *messages)
