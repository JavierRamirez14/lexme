"""The deterministic fake used as the whole suite's single substitution point.

Tests program replies per task; nothing here touches the network. Real retrieval,
citation verification, the checklist cross-check and the code gates all run for
real against the fake, so a test that swaps this in still exercises the system's
own logic. Replies can be Pydantic instances, dicts or raw strings, so a test can
also feed malformed output (a corrupt citation, invalid JSON) on purpose.
"""

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from lexme.llm.protocol import LlmError, Message, ModelT
from lexme.llm.structured import parse_model


@dataclass(frozen=True)
class RecordedCall:
    """One call captured by the fake, for assertions about what was asked."""

    task: str
    messages: tuple[Message, ...]
    response_model: type[BaseModel] | None


class FakeLlmClient:
    """A deterministic :class:`~lexme.llm.protocol.LlmClient` for tests.

    Replies are drained in FIFO order per task. Program them at construction with
    a ``{task: [reply, ...]}`` mapping or later with :meth:`queue`.
    """

    def __init__(self, responses: Mapping[str, Sequence[object]] | None = None) -> None:
        """Build the fake, optionally pre-loading a per-task reply queue."""
        self._queues: dict[str, deque[object]] = {}
        self.calls: list[RecordedCall] = []
        for task, replies in (responses or {}).items():
            self.queue(task, *replies)

    def queue(self, task: str, *responses: object) -> None:
        """Append ``responses`` to ``task``'s reply queue, in order."""
        self._queues.setdefault(task, deque()).extend(responses)

    def complete(self, task: str, messages: Sequence[Message]) -> str:
        """Return the next reply for ``task`` as text."""
        self.calls.append(RecordedCall(task, tuple(messages), None))
        reply = self._next(task)
        if not isinstance(reply, str):
            raise LlmError(f"fake reply for task '{task}' is {type(reply).__name__}, not str")
        return reply

    def complete_structured(
        self, task: str, messages: Sequence[Message], response_model: type[ModelT]
    ) -> ModelT:
        """Return the next reply for ``task`` coerced into ``response_model``.

        A reply may be an instance of ``response_model``, a ``dict`` validated
        into it, or a raw JSON string parsed by the interface's own validator.
        """
        self.calls.append(RecordedCall(task, tuple(messages), response_model))
        reply = self._next(task)
        if isinstance(reply, response_model):
            return reply
        if isinstance(reply, BaseModel):
            raise LlmError(
                f"fake reply for task '{task}' is {type(reply).__name__}, "
                f"not {response_model.__name__}"
            )
        if isinstance(reply, dict):
            return response_model.model_validate(reply)
        if isinstance(reply, str):
            return parse_model(reply, response_model)
        raise LlmError(
            f"fake reply for task '{task}' is {type(reply).__name__}; "
            "expected a model instance, dict or JSON string"
        )

    def _next(self, task: str) -> object:
        """Pop the next queued reply for ``task`` or raise if none remain."""
        queue = self._queues.get(task)
        if not queue:
            raise LlmError(f"fake has no queued reply for task '{task}'")
        return queue.popleft()
