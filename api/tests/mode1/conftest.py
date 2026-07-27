"""Fixtures and builders for Mode 1 graph tests: a seeded corpus plus fake replies.

The builders assemble the per-task replies the agentic graph drains -- router,
planner, critique and synthesis -- so a test can program a whole run in a few
readable lines and let real retrieval, verification and the gate run against them.
:func:`ask_server` does the same for the HTTP surface, substituting only the LLM,
the embedder, the clock, the checkpoint store and the vertical's branch package.
"""

import json
import os
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import httpx
import psycopg
import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pgvector.psycopg import register_vector

from lexme.api.dependencies import (
    get_branches,
    get_checkpointer,
    get_db_connection,
    get_embedder,
    get_llm_client,
    get_today,
)
from lexme.blocks import BlockRef
from lexme.ingestion import repository
from lexme.ingestion.xml_parsing import parse_norm_xml
from lexme.llm import FakeLlmClient, LlmClient
from lexme.main import app
from lexme.mode1 import (
    CriticalBranch,
    Mode1Deps,
    Mode1Run,
    PsycopgVersionHistory,
    load_branches,
)
from lexme.mode1.graph.critique import CRITIQUE_TASK, Critique, SubQueryCritique
from lexme.mode1.graph.planner import PLANNING_TASK, Plan, PlannedSubQuery
from lexme.mode1.graph.router import ROUTER_TASK, RouterDecision
from lexme.mode1.models import (
    Mode1Synthesis,
    QueryType,
    RouterScope,
    SubQueryVerdict,
)
from lexme.mode1.synthesis import SYNTHESIS_TASK
from lexme.retrieval import QueryEmbedder
from lexme.verification import CorpusReader, ProposedCitation, PsycopgCorpusReader
from tests.conftest import DeterministicEmbedder, seed_norm

REPO_ROOT = Path(__file__).resolve().parents[3]
LAU_XML = (
    Path(__file__).resolve().parents[1] / "ingestion" / "fixtures" / "lau_sample.xml"
).read_text(encoding="utf-8")
LAU_NORM_ID = "BOE-A-1994-26003"
AS_OF = date(2020, 1, 1)
VERTICAL = "vivienda"


@pytest.fixture
def vivienda_branches() -> tuple[CriticalBranch, ...]:
    """The shipped vivienda branch package, so tests disambiguate on real data."""
    return load_branches(REPO_ROOT / "verticales" / VERTICAL / "disambiguation.json")


@pytest.fixture
def checkpointer() -> BaseCheckpointSaver:
    """A checkpoint store scoped to one test, so paused runs never leak between them."""
    return InMemorySaver()


@pytest.fixture
def server_branches() -> tuple[CriticalBranch, ...]:
    """The branch package the server under test loads; none, unless a module overrides it."""
    return ()


@pytest.fixture
def ask_server(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
    server_branches: tuple[CriticalBranch, ...],
    live_server: str,
) -> Iterator[tuple[str, FakeLlmClient]]:
    """Override the endpoint's collaborators and yield the base URL and fake LLM."""
    dsn = os.environ["LEXME_TEST_DATABASE_URL"]

    def override_connection() -> Iterator[psycopg.Connection]:
        with psycopg.connect(dsn) as connection:
            register_vector(connection)
            yield connection

    app.dependency_overrides[get_db_connection] = override_connection
    app.dependency_overrides[get_embedder] = lambda: deterministic_embedder
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_today] = lambda: AS_OF
    app.dependency_overrides[get_checkpointer] = lambda: checkpointer
    app.dependency_overrides[get_branches] = lambda: server_branches
    try:
        yield live_server, fake_llm
    finally:
        app.dependency_overrides.clear()


def build_run(
    connection: psycopg.Connection,
    embedder: QueryEmbedder,
    corpus: CorpusReader,
    llm: LlmClient,
    checkpointer: BaseCheckpointSaver,
    *,
    thread_id: str = "test-thread",
    today: date = AS_OF,
    branches: tuple[CriticalBranch, ...] = (),
) -> Mode1Run:
    """A Mode 1 run wired to the test's corpus, fake LLM and checkpoint store."""
    return Mode1Run(
        thread_id=thread_id,
        vertical=VERTICAL,
        today=today,
        deps=Mode1Deps(
            connection=connection,
            embedder=embedder,
            corpus=corpus,
            history=PsycopgVersionHistory(connection),
            llm=llm,
            branches=branches,
        ),
        checkpointer=checkpointer,
    )


@pytest.fixture
def seeded_corpus(
    corpus_db: psycopg.Connection, deterministic_embedder: DeterministicEmbedder
) -> psycopg.Connection:
    """A corpus_db with the trimmed LAU norm loaded."""
    seed_norm(corpus_db, parse_norm_xml(LAU_XML), deterministic_embedder)
    return corpus_db


@pytest.fixture
def corpus_reader(seeded_corpus: psycopg.Connection) -> CorpusReader:
    """A corpus reader over the seeded LAU corpus."""
    return PsycopgCorpusReader(seeded_corpus)


def read_sse(base_url: str, question: str) -> list[tuple[str, dict]]:
    """Ask over ``/ask/stream`` and collect the ``(event, data)`` pairs in order."""
    return _read_sse(f"{base_url}/ask/stream", {"question": question})


def read_resume_sse(base_url: str, thread_id: str, answer: str) -> list[tuple[str, dict]]:
    """Reply over ``/ask/resume/stream`` and collect the ``(event, data)`` pairs in order."""
    return _read_sse(f"{base_url}/ask/resume/stream", {"thread_id": thread_id, "answer": answer})


def _read_sse(url: str, payload: dict) -> list[tuple[str, dict]]:
    """POST ``payload`` to an SSE endpoint and decode every event it emits."""
    events: list[tuple[str, dict]] = []
    with httpx.stream("POST", url, json=payload, timeout=10) as response:
        event_name = "message"
        for line in response.iter_lines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                events.append((event_name, json.loads(line[len("data:") :].strip())))
    return events


def in_force_text(conn: psycopg.Connection, block_id: str, as_of: date = AS_OF) -> str:
    """The plain text of a block's redaction in force at ``as_of``."""
    version = repository.get_version_in_force(conn, LAU_NORM_ID, block_id, as_of)
    assert version is not None
    return version.text_content


def in_scope(query_type: QueryType = QueryType.INFORMATIONAL) -> RouterDecision:
    """A router decision keeping the question in scope."""
    return RouterDecision(scope=RouterScope.IN_SCOPE, query_type=query_type)


def out_of_scope(message: str) -> RouterDecision:
    """A router decision rejecting the question as out of scope, with a message."""
    return RouterDecision(
        scope=RouterScope.OUT_OF_SCOPE,
        query_type=QueryType.INFORMATIONAL,
        rejection_message=message,
    )


def plan_of(
    *subqueries: tuple[str, str, bool],
    assumptions: tuple[str, ...] = (),
    target_date_reference: str = "",
    unresolved: tuple[str, ...] = (),
) -> Plan:
    """A plan from ``(text, purpose, is_critical)`` tuples and explicit assumptions.

    ``target_date_reference`` is the temporal anchor the planner claims to have
    read in the question; ``unresolved`` the branch ids it says are still open.
    """
    return Plan(
        subqueries=[
            PlannedSubQuery(text=text, purpose=purpose, is_critical=is_critical)
            for text, purpose, is_critical in subqueries
        ],
        assumptions=list(assumptions),
        target_date_reference=target_date_reference,
        unresolved_branch_ids=list(unresolved),
    )


def critique_of(*rulings: tuple[str, SubQueryVerdict, str]) -> Critique:
    """A critique from ``(subquery_id, verdict, reformulation)`` tuples."""
    return Critique(
        verdicts=[
            SubQueryCritique(subquery_id=subquery_id, verdict=verdict, reformulation=reformulation)
            for subquery_id, verdict, reformulation in rulings
        ]
    )


def block_ref(block_id: str, norm_id: str = LAU_NORM_ID) -> str:
    """The norm-qualified reference a citation names a corpus block by."""
    return str(BlockRef(norm_id=norm_id, block_id=block_id))


def synthesis_of(
    *citations: tuple[str, str], explicacion: str = "", accion: tuple[str, ...] = ()
) -> Mode1Synthesis:
    """A synthesis proposal from ``(block_id, text)`` citation tuples.

    The block id is qualified with the seeded corpus's norm, since a citation
    names a block by its ``norm:block`` reference.
    """
    return Mode1Synthesis(
        fundamento=[
            ProposedCitation(block_ref=block_ref(block_id), text=text)
            for block_id, text in citations
        ],
        explicacion=explicacion,
        accion=list(accion),
    )


def program_single_sufficient(
    fake: FakeLlmClient,
    *,
    query_text: str,
    citation: tuple[str, str],
    explicacion: str = "El plazo mínimo se pacta libremente.",
    accion: tuple[str, ...] = (),
    assumptions: tuple[str, ...] = (),
    query_type: QueryType = QueryType.INFORMATIONAL,
    target_date_reference: str = "",
    unresolved: tuple[str, ...] = (),
) -> None:
    """Queue a one-critical-sub-query run that grounds on the first pass.

    Router in scope, a single critical sub-query, one sufficient critique pass and
    a synthesis proposing ``citation`` -- the common happy path a test builds on.
    """
    fake.queue(ROUTER_TASK, in_scope(query_type))
    fake.queue(
        PLANNING_TASK,
        plan_of(
            (query_text, "Responder la pregunta.", True),
            assumptions=assumptions,
            target_date_reference=target_date_reference,
            unresolved=unresolved,
        ),
    )
    fake.queue(CRITIQUE_TASK, critique_of(("sq1", SubQueryVerdict.SUFFICIENT, "")))
    fake.queue(SYNTHESIS_TASK, synthesis_of(citation, explicacion=explicacion, accion=accion))
