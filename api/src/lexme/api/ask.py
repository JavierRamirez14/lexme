"""The Mode 1 query endpoints: a blocking answer, an SSE narration, and the resume.

``POST /ask`` runs the agentic graph to completion and returns the final
:class:`AskResponse` -- the shape the eval harness reads. ``POST /ask/stream``
runs the same graph but emits a Server-Sent Event after each node, so the UI can
show the process live (planning, retrieving, iterating n/3, synthesizing) and then
a final ``result`` event with the full response.

Either can come back paused on a single disambiguating question instead of an
answer; ``POST /ask/resume`` and ``POST /ask/resume/stream`` carry the user's
reply back to the same ``thread_id`` and continue the run where it stopped, which
is what makes the pause survive across requests. All four only wire the request's
collaborators and pick the vertical; the whole pipeline lives in the Mode 1
package.
"""

import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.base import BaseCheckpointSaver

from lexme.api.dependencies import get_checkpointer, get_deps, get_today, get_vertical
from lexme.mode1 import (
    AskRequest,
    AskResponse,
    Mode1Deps,
    Mode1Run,
    ResumeRequest,
    RunNotPausedError,
    build_agentic_trace,
)
from lexme.mode1.graph.state import Mode1State

router = APIRouter()

RunFactory = Callable[[str], Mode1Run]


def get_run_factory(
    deps: Mode1Deps = Depends(get_deps),
    checkpointer: BaseCheckpointSaver = Depends(get_checkpointer),
    vertical: str = Depends(get_vertical),
    today: date = Depends(get_today),
) -> RunFactory:
    """Return a factory binding this request's collaborators to a run's thread id."""

    def build(thread_id: str) -> Mode1Run:
        return Mode1Run(
            thread_id=thread_id,
            vertical=vertical,
            today=today,
            deps=deps,
            checkpointer=checkpointer,
        )

    return build


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, build_run: RunFactory = Depends(get_run_factory)) -> AskResponse:
    """Answer a Mode 1 question, decline honestly, or pause to ask one question."""
    return build_run(_new_thread_id()).answer(request.question)


@router.post("/ask/resume", response_model=AskResponse)
def resume(request: ResumeRequest, build_run: RunFactory = Depends(get_run_factory)) -> AskResponse:
    """Continue a paused run with the user's reply to its disambiguating question."""
    run = build_run(request.thread_id)
    with _reporting_unknown_thread():
        return run.resume(request.answer)


@router.post("/ask/stream")
def ask_stream(
    request: AskRequest, build_run: RunFactory = Depends(get_run_factory)
) -> StreamingResponse:
    """Stream the run node by node as SSE, ending with the final response."""
    run = build_run(_new_thread_id())
    return _sse_response(run, run.stream_answer(request.question))


@router.post("/ask/resume/stream")
def resume_stream(
    request: ResumeRequest, build_run: RunFactory = Depends(get_run_factory)
) -> StreamingResponse:
    """Stream the continuation of a paused run, ending with the final response."""
    run = build_run(request.thread_id)
    with _reporting_unknown_thread():
        states = run.stream_resume(request.answer)
    return _sse_response(run, states)


def _new_thread_id() -> str:
    """Mint the identifier a client uses to reply to this run."""
    return str(uuid.uuid4())


@contextmanager
def _reporting_unknown_thread() -> Iterator[None]:
    """Translate a reply to an unknown or already-finished run into a 404."""
    try:
        yield
    except RunNotPausedError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error


def _sse_response(run: Mode1Run, states: Iterator[Mode1State]) -> StreamingResponse:
    """Narrate ``states`` as ``step`` events, then the run's outcome as ``result``.

    Each ``step`` carries the stage name and the agentic trace snapshot after one
    node; the single ``result`` is read from the run once the graph has stopped,
    whether it finished or paused on a question.
    """

    def events() -> Iterator[str]:
        for state in states:
            yield _sse(
                "step",
                {
                    "step": state.current_step,
                    "agentic": build_agentic_trace(state).model_dump(mode="json"),
                },
            )
        yield _sse("result", run.response().model_dump(mode="json"))

    return StreamingResponse(events(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event with a named type and a JSON data payload."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
