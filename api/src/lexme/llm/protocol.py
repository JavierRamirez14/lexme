"""Public surface of the LLM interface: messages, the client protocol, errors.

Graphs and nodes depend only on the names in this module. The concrete client is
built by :func:`lexme.llm.factory.build_llm_client`; tests inject
:class:`lexme.llm.fake.FakeLlmClient`. Both satisfy :class:`LlmClient`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class Message:
    """One turn of a prompt: a role and its text content."""

    role: Role
    content: str


class LlmError(RuntimeError):
    """Base class for every error raised by the LLM interface."""


class TaskNotConfiguredError(LlmError):
    """Raised when a task name has no entry in the task registry."""


class StructuredOutputError(LlmError):
    """Raised when a model reply cannot be parsed into the requested schema."""


class ProviderResponseError(LlmError):
    """Raised when a provider returns a body without a usable completion."""


@runtime_checkable
class LlmClient(Protocol):
    """A model invoked by task name, with optional Pydantic-typed output.

    The task name selects the provider, pinned model and temperature; callers
    never name a provider. ``complete`` returns free text; ``complete_structured``
    returns a validated instance of ``response_model``.
    """

    def complete(self, task: str, messages: Sequence[Message]) -> str:
        """Run ``task`` over ``messages`` and return the reply text.

        Raises :class:`TaskNotConfiguredError` if ``task`` is unknown.
        """
        ...

    def complete_structured(
        self, task: str, messages: Sequence[Message], response_model: type[ModelT]
    ) -> ModelT:
        """Run ``task`` and parse the reply into ``response_model``.

        Raises :class:`TaskNotConfiguredError` if ``task`` is unknown, and
        :class:`StructuredOutputError` if the reply does not fit the schema.
        """
        ...
