"""Fixtures and builders for Mode 1 graph tests: a seeded corpus plus fake replies.

The builders assemble the per-task replies the agentic graph drains -- router,
planner, critique and synthesis -- so a test can program a whole run in a few
readable lines and let real retrieval, verification and the gate run against them.
"""

from datetime import date
from pathlib import Path

import psycopg
import pytest

from lexme.ingestion import repository
from lexme.ingestion.xml_parsing import parse_norm_xml
from lexme.llm import FakeLlmClient
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
from lexme.verification import CorpusReader, ProposedCitation, PsycopgCorpusReader
from tests.conftest import DeterministicEmbedder, seed_norm

LAU_XML = (
    Path(__file__).resolve().parents[1] / "ingestion" / "fixtures" / "lau_sample.xml"
).read_text(encoding="utf-8")
LAU_NORM_ID = "BOE-A-1994-26003"
AS_OF = date(2020, 1, 1)


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


def in_force_text(conn: psycopg.Connection, block_id: str) -> str:
    """The plain text of a block's redaction in force at the test's target date."""
    version = repository.get_version_in_force(conn, LAU_NORM_ID, block_id, AS_OF)
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
) -> Plan:
    """A plan from ``(text, purpose, is_critical)`` tuples and explicit assumptions."""
    return Plan(
        subqueries=[
            PlannedSubQuery(text=text, purpose=purpose, is_critical=is_critical)
            for text, purpose, is_critical in subqueries
        ],
        assumptions=list(assumptions),
    )


def critique_of(*rulings: tuple[str, SubQueryVerdict, str]) -> Critique:
    """A critique from ``(subquery_id, verdict, reformulation)`` tuples."""
    return Critique(
        verdicts=[
            SubQueryCritique(subquery_id=subquery_id, verdict=verdict, reformulation=reformulation)
            for subquery_id, verdict, reformulation in rulings
        ]
    )


def synthesis_of(
    *citations: tuple[str, str], explicacion: str = "", accion: tuple[str, ...] = ()
) -> Mode1Synthesis:
    """A synthesis proposal from ``(block_id, text)`` citation tuples."""
    return Mode1Synthesis(
        fundamento=[ProposedCitation(block_id=block_id, text=text) for block_id, text in citations],
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
) -> None:
    """Queue a one-critical-sub-query run that grounds on the first pass.

    Router in scope, a single critical sub-query, one sufficient critique pass and
    a synthesis proposing ``citation`` -- the common happy path a test builds on.
    """
    fake.queue(ROUTER_TASK, in_scope())
    fake.queue(
        PLANNING_TASK,
        plan_of((query_text, "Responder la pregunta.", True), assumptions=assumptions),
    )
    fake.queue(CRITIQUE_TASK, critique_of(("sq1", SubQueryVerdict.SUFFICIENT, "")))
    fake.queue(SYNTHESIS_TASK, synthesis_of(citation, explicacion=explicacion, accion=accion))
