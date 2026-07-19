"""The Mode 1 query endpoints: a blocking answer, and an SSE narration of the run.

``POST /ask`` runs the agentic graph to completion and returns the final
:class:`AskResponse` -- the shape the eval harness reads. ``POST /ask/stream``
runs the same graph but emits a Server-Sent Event after each node, so the UI can
show the process live (planning, retrieving, iterating n/3, synthesizing) and then
a final ``result`` event with the full response. Both only wire the request's
collaborators and pick the vertical; the whole pipeline lives in the Mode 1 package.
"""

import json
from collections.abc import Iterator
from datetime import date

import psycopg
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from lexme.api.dependencies import (
    get_corpus_reader,
    get_db_connection,
    get_embedder,
    get_llm_client,
    get_target_date,
)
from lexme.llm import LlmClient
from lexme.mode1 import (
    AskRequest,
    AskResponse,
    answer_question,
    build_agentic_trace,
    build_response,
    stream_states,
)
from lexme.retrieval import QueryEmbedder
from lexme.verification import CorpusReader

DEFAULT_VERTICAL = "vivienda"

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    connection: psycopg.Connection = Depends(get_db_connection),
    embedder: QueryEmbedder = Depends(get_embedder),
    corpus: CorpusReader = Depends(get_corpus_reader),
    llm: LlmClient = Depends(get_llm_client),
    target_date: date = Depends(get_target_date),
) -> AskResponse:
    """Answer a Mode 1 question over the vivienda corpus, or decline honestly."""
    return answer_question(
        request.question,
        connection=connection,
        embedder=embedder,
        corpus=corpus,
        llm=llm,
        vertical=DEFAULT_VERTICAL,
        target_date=target_date,
    )


@router.post("/ask/stream")
def ask_stream(
    request: AskRequest,
    connection: psycopg.Connection = Depends(get_db_connection),
    embedder: QueryEmbedder = Depends(get_embedder),
    corpus: CorpusReader = Depends(get_corpus_reader),
    llm: LlmClient = Depends(get_llm_client),
    target_date: date = Depends(get_target_date),
) -> StreamingResponse:
    """Stream the run node by node as SSE, ending with the final response.

    Emits a ``step`` event carrying the stage name and the agentic trace snapshot
    after each node, then a single ``result`` event with the full response.
    """

    def events() -> Iterator[str]:
        final = None
        for state in stream_states(
            request.question,
            connection=connection,
            embedder=embedder,
            corpus=corpus,
            llm=llm,
            vertical=DEFAULT_VERTICAL,
            target_date=target_date,
        ):
            final = state
            yield _sse(
                "step",
                {
                    "step": state.current_step,
                    "agentic": build_agentic_trace(state).model_dump(mode="json"),
                },
            )
        if final is not None:
            yield _sse("result", build_response(final).model_dump(mode="json"))

    return StreamingResponse(events(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event with a named type and a JSON data payload."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
