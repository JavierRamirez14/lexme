"""Mode 1's entry point: one consultation, addressable across HTTP requests.

A run can pause mid-graph to ask the user one disambiguating question, so it is
no longer a single function call: it is a thread the caller can start, stream and
later resume. :class:`Mode1Run` wraps the compiled graph and its checkpoint, so
callers work in terms of "answer this question" and "here is the user's reply"
and never in terms of LangGraph threads, commands or interrupts. The model
proposes; the gate, in code, disposes.
"""

from collections.abc import Iterator
from datetime import date

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from lexme.mode1.graph.builder import build_mode1_graph
from lexme.mode1.graph.deps import Mode1Deps
from lexme.mode1.graph.presenter import build_clarification_response, build_response
from lexme.mode1.graph.state import Mode1State
from lexme.mode1.models import AskResponse, Clarification

_INTERRUPT_KEY = "__interrupt__"


class RunNotPausedError(LookupError):
    """Raised when a reply arrives for a thread that is not waiting on a question."""


class Mode1Run:
    """One Mode 1 consultation, identified by the thread its checkpoints live under.

    Start it with :meth:`answer` (or :meth:`stream_answer` to narrate it node by
    node) and, if it paused to ask something, continue it with :meth:`resume`.
    :meth:`response` reads the current outcome off the checkpoint, so the streamed
    and the blocking paths end at the same value.
    """

    def __init__(
        self,
        *,
        thread_id: str,
        vertical: str,
        today: date,
        deps: Mode1Deps,
        checkpointer: BaseCheckpointSaver,
    ) -> None:
        """Compile the graph for this run and bind it to ``thread_id``'s checkpoint."""
        self._graph = build_mode1_graph(deps, checkpointer)
        self._config = {"configurable": {"thread_id": thread_id}}
        self._thread_id = thread_id
        self._vertical = vertical
        self._today = today

    def answer(self, question: str) -> AskResponse:
        """Run the graph over ``question`` and return the outcome it reached.

        The outcome is a cited answer, a partial answer, an honest abstention, a
        scope rejection, or a pause carrying the one question the run needs
        answered before it can continue.
        """
        self._graph.invoke(self._initial_state(question), self._config)
        return self.response()

    def resume(self, answer: str) -> AskResponse:
        """Continue a paused run with the user's ``answer`` to its question.

        Raises :class:`RunNotPausedError` when the thread is unknown or is not
        waiting on an answer, so a stale reply cannot silently start a new run.
        """
        self._require_paused()
        self._graph.invoke(Command(resume=answer), self._config)
        return self.response()

    def stream_answer(self, question: str) -> Iterator[Mode1State]:
        """Yield the state after each node while running ``question``.

        The first state yielded is the input and the last is the state the
        response is built from; a pause emits no extra state, since the run stops
        without changing it.
        """
        return self._stream(self._initial_state(question))

    def stream_resume(self, answer: str) -> Iterator[Mode1State]:
        """Yield the state after each node while continuing a paused run.

        Raises :class:`RunNotPausedError` before any streaming begins, so the
        caller can still answer with an error status rather than a half-sent
        stream.
        """
        self._require_paused()
        return self._stream(Command(resume=answer))

    def response(self) -> AskResponse:
        """Read the run's current outcome from its checkpoint."""
        snapshot = self._graph.get_state(self._config)
        state = Mode1State.model_validate(snapshot.values)
        if snapshot.interrupts:
            clarification = Clarification.model_validate(snapshot.interrupts[0].value)
            return build_clarification_response(state, clarification, self._thread_id)
        return build_response(state, self._thread_id)

    def _stream(self, payload: Mode1State | Command) -> Iterator[Mode1State]:
        """Stream the graph's values, skipping the frame that only reports a pause."""
        for values in self._graph.stream(payload, self._config, stream_mode="values"):
            if _INTERRUPT_KEY in values:
                continue
            yield Mode1State.model_validate(values)

    def _require_paused(self) -> None:
        """Fail loudly when this thread is not waiting on an answer."""
        if not self._graph.get_state(self._config).interrupts:
            raise RunNotPausedError(f"run '{self._thread_id}' is not waiting on an answer")

    def _initial_state(self, question: str) -> Mode1State:
        """Seed the graph state with the run's inputs, anchored at today by default."""
        return Mode1State(
            question=question,
            vertical=self._vertical,
            today=self._today,
            target_date=self._today,
        )
