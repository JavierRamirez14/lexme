"""Integration tests for the Mode 1 agentic graph over the seeded corpus.

Real hybrid retrieval, real citation verification and the real deterministic gate
run against the seeded LAU corpus; only the LLM is faked, so router, planner,
self-critique and synthesis are programmed while every decision that matters --
what to retrieve, what verifies, which outcome to return -- is the system's own.
"""

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver

from lexme.llm import FakeLlmClient
from lexme.mode1 import AbstentionReason, Outcome, SubQueryVerdict
from lexme.mode1.graph.critique import CRITIQUE_TASK
from lexme.mode1.graph.planner import PLANNING_TASK
from lexme.mode1.graph.router import ROUTER_TASK
from lexme.mode1.synthesis import SYNTHESIS_TASK
from lexme.verification import CitationVerdict, CorpusReader
from tests.conftest import DeterministicEmbedder
from tests.mode1.conftest import (
    LAU_NORM_ID,
    build_run,
    critique_of,
    in_force_text,
    in_scope,
    out_of_scope,
    plan_of,
    program_single_sufficient,
    synthesis_of,
)

QUESTION = "¿cuál es el plazo mínimo del arrendamiento de vivienda?"
QUOTE = "La duración del arrendamiento será libremente pactada por las partes"
FABRICATED = "El arrendador podrá desalojar al inquilino en cualquier momento sin preaviso."


def _run(
    conn: psycopg.Connection,
    embedder: DeterministicEmbedder,
    corpus: CorpusReader,
    llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
    question: str = QUESTION,
):
    """Run the agentic graph with the test's wiring and its clock fixed."""
    return build_run(conn, embedder, corpus, llm, checkpointer).answer(question)


def test_a_grounded_question_yields_a_cited_answer_with_assumptions(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
) -> None:
    assert QUOTE in in_force_text(seeded_corpus, "a9")
    program_single_sufficient(
        fake_llm,
        query_text=QUESTION,
        citation=("a9", QUOTE),
        assumptions=("Asumo vivienda habitual y no de temporada.",),
    )

    response = _run(seeded_corpus, deterministic_embedder, corpus_reader, fake_llm, checkpointer)

    assert response.outcome is Outcome.ANSWER
    assert response.answer is not None
    assert response.answer.fundamento[0].block_ref == f"{LAU_NORM_ID}:a9"
    assert response.answer.fundamento[0].verdict is CitationVerdict.VERIFIED_DIRECT
    assert response.answer.asunciones == ["Asumo vivienda habitual y no de temporada."]
    assert response.agentic is not None
    assert len(response.agentic.subqueries) == 1
    assert response.agentic.subqueries[0].verdict is SubQueryVerdict.SUFFICIENT


def test_an_out_of_scope_question_is_rejected_before_retrieval(
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
) -> None:
    fake_llm.queue(ROUTER_TASK, out_of_scope("Eso es tráfico, no alquiler de vivienda."))

    response = build_run(None, None, None, fake_llm, checkpointer).answer(
        "¿cómo recurro una multa de tráfico?"
    )

    assert response.outcome is Outcome.ROUTER_REJECTION
    assert response.rejection is not None
    assert "tráfico" in response.rejection.message
    assert response.agentic is None
    assert [call.task for call in fake_llm.calls] == [ROUTER_TASK]


def test_a_fabricated_citation_forces_an_honest_abstention(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
) -> None:
    program_single_sufficient(fake_llm, query_text=QUESTION, citation=("a9", FABRICATED))

    response = _run(seeded_corpus, deterministic_embedder, corpus_reader, fake_llm, checkpointer)

    assert response.outcome is Outcome.ABSTENTION
    assert response.abstention is not None
    assert response.abstention.reason is AbstentionReason.NO_VERIFIABLE_CITATION
    assert response.answer is None


def test_a_discarded_citation_never_appears_beside_a_valid_one(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
) -> None:
    fake_llm.queue(ROUTER_TASK, in_scope())
    fake_llm.queue(PLANNING_TASK, plan_of((QUESTION, "Responder la pregunta.", True)))
    fake_llm.queue(CRITIQUE_TASK, critique_of(("sq1", SubQueryVerdict.SUFFICIENT, "")))
    fake_llm.queue(SYNTHESIS_TASK, synthesis_of(("a9", QUOTE), ("a9", FABRICATED)))

    response = _run(seeded_corpus, deterministic_embedder, corpus_reader, fake_llm, checkpointer)

    assert response.outcome is Outcome.ANSWER
    assert response.answer is not None
    assert len(response.answer.fundamento) == 1
    assert all(FABRICATED not in citation.text for citation in response.answer.fundamento)
    assert response.citation_verdicts[CitationVerdict.DISCARDED.value] == 1


def test_a_compound_question_decomposes_into_judged_subqueries(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
) -> None:
    fake_llm.queue(ROUTER_TASK, in_scope())
    fake_llm.queue(
        PLANNING_TASK,
        plan_of(
            ("duración plazo mínimo arrendamiento vivienda", "Plazo mínimo.", True),
            ("fianza depósito arrendamiento vivienda", "Importe de la fianza.", True),
        ),
    )
    fake_llm.queue(
        CRITIQUE_TASK,
        critique_of(
            ("sq1", SubQueryVerdict.SUFFICIENT, ""),
            ("sq2", SubQueryVerdict.SUFFICIENT, ""),
        ),
    )
    fake_llm.queue(SYNTHESIS_TASK, synthesis_of(("a9", QUOTE)))

    response = _run(seeded_corpus, deterministic_embedder, corpus_reader, fake_llm, checkpointer)

    assert response.agentic is not None
    assert [sub.id for sub in response.agentic.subqueries] == ["sq1", "sq2"]
    assert all(sub.verdict is SubQueryVerdict.SUFFICIENT for sub in response.agentic.subqueries)
    assert all(sub.retrieval is not None for sub in response.agentic.subqueries)


def test_the_self_critique_loop_iterates_then_stops_when_grounded(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
) -> None:
    fake_llm.queue(ROUTER_TASK, in_scope())
    fake_llm.queue(PLANNING_TASK, plan_of(("plazo arrendamiento", "Plazo mínimo.", True)))
    fake_llm.queue(
        CRITIQUE_TASK,
        critique_of(("sq1", SubQueryVerdict.INSUFFICIENT, QUESTION)),
        critique_of(("sq1", SubQueryVerdict.SUFFICIENT, "")),
    )
    fake_llm.queue(SYNTHESIS_TASK, synthesis_of(("a9", QUOTE)))

    response = _run(seeded_corpus, deterministic_embedder, corpus_reader, fake_llm, checkpointer)

    assert response.outcome is Outcome.ANSWER
    assert response.agentic is not None
    assert len(response.agentic.passes) == 2
    assert response.agentic.first_pass_sufficient == 0
    assert response.agentic.final_sufficient == 1
    assert response.agentic.agentic_delta == 1


def test_the_self_critique_loop_stops_at_three_passes(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
) -> None:
    fake_llm.queue(ROUTER_TASK, in_scope())
    fake_llm.queue(PLANNING_TASK, plan_of((QUESTION, "Plazo mínimo.", True)))
    fake_llm.queue(
        CRITIQUE_TASK,
        *[critique_of(("sq1", SubQueryVerdict.INSUFFICIENT, QUESTION)) for _ in range(3)],
    )
    fake_llm.queue(SYNTHESIS_TASK, synthesis_of(("a9", QUOTE)))

    response = _run(seeded_corpus, deterministic_embedder, corpus_reader, fake_llm, checkpointer)

    assert response.outcome is Outcome.ABSTENTION
    assert response.abstention is not None
    assert response.abstention.reason is AbstentionReason.INSUFFICIENT_CORE
    assert response.agentic is not None
    assert len(response.agentic.passes) == 3
    assert [call.task for call in fake_llm.calls].count(CRITIQUE_TASK) == 3


def test_a_peripheral_gap_yields_a_partial_answer_with_declared_gaps(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
) -> None:
    fake_llm.queue(ROUTER_TASK, in_scope())
    fake_llm.queue(
        PLANNING_TASK,
        plan_of(
            ("duración plazo mínimo arrendamiento vivienda", "Plazo mínimo.", True),
            ("subvenciones ayudas alquiler joven", "Ayudas al alquiler.", False),
        ),
    )
    fake_llm.queue(
        CRITIQUE_TASK,
        *[
            critique_of(
                ("sq1", SubQueryVerdict.SUFFICIENT, ""),
                ("sq2", SubQueryVerdict.INSUFFICIENT, "ayudas alquiler"),
            )
            for _ in range(3)
        ],
    )
    fake_llm.queue(SYNTHESIS_TASK, synthesis_of(("a9", QUOTE)))

    response = _run(seeded_corpus, deterministic_embedder, corpus_reader, fake_llm, checkpointer)

    assert response.outcome is Outcome.PARTIAL_ANSWER
    assert response.answer is not None
    assert response.answer.huecos_declarados == ["Ayudas al alquiler."]
