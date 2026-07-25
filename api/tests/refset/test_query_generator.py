"""Tests for the Mode 1 query generator and its by-construction guarantees."""

from datetime import UTC, date, datetime

import pytest

from lexme.llm import FakeLlmClient
from lexme.mode1.models import Outcome
from lexme.refset.models import CaseKind, ReviewStatus
from lexme.refset.query_generator import (
    QUERY_GENERATION_TASK,
    GeneratedQuery,
    GenerationError,
    QuerySeed,
    QueryStyle,
    generate_query_case,
)
from tests.refset.conftest import FakeCorpus

_NORM = "BOE-A-1994-26003"
_NOW = datetime(2026, 7, 25, tzinfo=UTC)
_DATE = date(2024, 1, 1)


def _generate(seed: QuerySeed, corpus: FakeCorpus, llm: FakeLlmClient):
    return generate_query_case(
        seed, corpus=corpus, llm=llm, norm_id=_NORM, now=_NOW, resolve_date=_DATE, model="pin-1"
    )


def test_gold_blocks_come_from_the_seed_not_the_model() -> None:
    corpus = FakeCorpus({"a9": "duración mínima de cinco años"})
    llm = FakeLlmClient(
        {
            QUERY_GENERATION_TASK: [
                GeneratedQuery(
                    question="¿cuánto puedo quedarme?",
                    key_points=[{"claim": "cinco años", "block_id": "a9"}],
                )
            ]
        }
    )
    seed = QuerySeed(id="plazo", block_ids=["a9"], expected_outcome=Outcome.ANSWER)

    candidate = _generate(seed, corpus, llm)

    assert candidate.kind is CaseKind.MODE1
    assert candidate.status is ReviewStatus.PENDING
    assert candidate.mode1 is not None
    assert candidate.mode1.gold_block_ids == ["a9"]
    assert candidate.mode1.key_points[0].claim == "cinco años"
    assert candidate.provenance.model == "pin-1"
    assert candidate.provenance.sources == ["a9"]


def test_a_key_point_citing_a_block_outside_the_seed_is_rejected() -> None:
    corpus = FakeCorpus({"a9": "texto"})
    llm = FakeLlmClient(
        {
            QUERY_GENERATION_TASK: [
                GeneratedQuery(question="q", key_points=[{"claim": "x", "block_id": "a10"}])
            ]
        }
    )
    seed = QuerySeed(id="plazo", block_ids=["a9"], expected_outcome=Outcome.ANSWER)

    with pytest.raises(GenerationError, match="outside the seed"):
        _generate(seed, corpus, llm)


def test_an_answer_seed_needs_at_least_one_key_point() -> None:
    corpus = FakeCorpus({"a9": "texto"})
    llm = FakeLlmClient({QUERY_GENERATION_TASK: [GeneratedQuery(question="q", key_points=[])]})
    seed = QuerySeed(id="plazo", block_ids=["a9"], expected_outcome=Outcome.ANSWER)

    with pytest.raises(GenerationError, match="at least one key point"):
        _generate(seed, corpus, llm)


def test_an_out_of_scope_seed_needs_no_blocks_and_carries_no_reference() -> None:
    corpus = FakeCorpus({})
    llm = FakeLlmClient(
        {QUERY_GENERATION_TASK: [GeneratedQuery(question="¿cómo recurro una multa?")]}
    )
    seed = QuerySeed(
        id="multa",
        expected_outcome=Outcome.ROUTER_REJECTION,
        style=QueryStyle.OUT_OF_SCOPE,
        topic="multas de tráfico",
    )

    candidate = _generate(seed, corpus, llm)

    assert candidate.mode1 is not None
    assert candidate.mode1.gold_block_ids == []
    assert candidate.mode1.key_points == []


def test_an_out_of_scope_seed_with_blocks_is_rejected() -> None:
    seed = QuerySeed(
        id="bad",
        block_ids=["a9"],
        expected_outcome=Outcome.ROUTER_REJECTION,
        style=QueryStyle.OUT_OF_SCOPE,
        topic="x",
    )

    with pytest.raises(GenerationError, match="no gold blocks"):
        _generate(seed, FakeCorpus({"a9": "t"}), FakeLlmClient())


def test_a_seed_block_that_does_not_resolve_is_rejected() -> None:
    seed = QuerySeed(id="plazo", block_ids=["a99"], expected_outcome=Outcome.ANSWER)

    with pytest.raises(GenerationError, match="does not resolve"):
        _generate(seed, FakeCorpus({"a9": "t"}), FakeLlmClient())


def test_an_out_of_scope_reference_cannot_carry_key_points() -> None:
    corpus = FakeCorpus({})
    llm = FakeLlmClient(
        {
            QUERY_GENERATION_TASK: [
                GeneratedQuery(question="q", key_points=[{"claim": "x", "block_id": "a9"}])
            ]
        }
    )
    seed = QuerySeed(
        id="multa",
        expected_outcome=Outcome.ROUTER_REJECTION,
        style=QueryStyle.OUT_OF_SCOPE,
        topic="multas",
    )

    with pytest.raises(GenerationError, match="outside the seed"):
        _generate(seed, corpus, llm)
