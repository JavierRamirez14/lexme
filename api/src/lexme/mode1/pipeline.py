"""Mode 1 entry points: run the agentic graph to a response, or stream its states.

The graph is the orchestrator -- router, planner, the retrieval/self-critique loop,
synthesis and the deterministic gate. These helpers build the graph with the run's
collaborators and either invoke it to a final :class:`AskResponse` or stream the
intermediate states so the SSE endpoint can narrate the run node by node. The model
proposes; the gate, in code, disposes.
"""

from collections.abc import Iterator
from datetime import date

import psycopg

from lexme.llm import LlmClient
from lexme.mode1.graph.builder import build_mode1_graph
from lexme.mode1.graph.deps import Mode1Deps
from lexme.mode1.graph.presenter import build_response
from lexme.mode1.graph.state import Mode1State
from lexme.mode1.models import AskResponse
from lexme.retrieval import QueryEmbedder
from lexme.verification import CorpusReader


def answer_question(
    question: str,
    *,
    connection: psycopg.Connection,
    embedder: QueryEmbedder,
    corpus: CorpusReader,
    llm: LlmClient,
    vertical: str,
    target_date: date,
) -> AskResponse:
    """Run the agentic graph over ``question`` and return its terminal response.

    Classifies scope, decomposes the question, retrieves and self-critiques the
    sub-queries, synthesizes and verifies, then lets the gate pick the outcome:
    a full answer, a partial answer, an honest abstention or a scope rejection.
    """
    graph = build_mode1_graph(_deps(connection, embedder, corpus, llm))
    final = graph.invoke(_initial_state(question, vertical, target_date))
    return build_response(Mode1State.model_validate(final))


def stream_states(
    question: str,
    *,
    connection: psycopg.Connection,
    embedder: QueryEmbedder,
    corpus: CorpusReader,
    llm: LlmClient,
    vertical: str,
    target_date: date,
) -> Iterator[Mode1State]:
    """Yield the graph state after each node, for the SSE narration of a run.

    The first yielded state is the input; each subsequent one reflects one node's
    update. The last state yielded is the terminal state the response is built from.
    """
    graph = build_mode1_graph(_deps(connection, embedder, corpus, llm))
    for values in graph.stream(
        _initial_state(question, vertical, target_date), stream_mode="values"
    ):
        yield Mode1State.model_validate(values)


def _deps(
    connection: psycopg.Connection,
    embedder: QueryEmbedder,
    corpus: CorpusReader,
    llm: LlmClient,
) -> Mode1Deps:
    """Bundle the run's collaborators for the graph nodes."""
    return Mode1Deps(connection=connection, embedder=embedder, corpus=corpus, llm=llm)


def _initial_state(question: str, vertical: str, target_date: date) -> Mode1State:
    """Seed the graph state with the run's inputs."""
    return Mode1State(question=question, vertical=vertical, target_date=target_date)
