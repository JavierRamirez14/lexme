"""Integration tests for the Mode 1 deterministic gate.

Real retrieval, real citation verification and the real code gate run against the
seeded LAU corpus; only the LLM is faked, so every outcome is produced by the
system's own logic over a programmed synthesis.
"""

import psycopg

from lexme.llm import FakeLlmClient
from lexme.mode1 import AbstentionReason, Outcome, answer_question
from lexme.mode1.models import Mode1Synthesis
from lexme.mode1.synthesis import SYNTHESIS_TASK
from lexme.verification import CitationVerdict, CorpusReader, ProposedCitation
from tests.conftest import DeterministicEmbedder
from tests.mode1.conftest import AS_OF, in_force_text

QUESTION = "¿cuál es el plazo mínimo del arrendamiento de vivienda?"
FABRICATED = "El arrendador podrá desalojar al inquilino en cualquier momento sin preaviso."


def _run(
    conn: psycopg.Connection,
    embedder: DeterministicEmbedder,
    corpus: CorpusReader,
    llm: FakeLlmClient,
):
    """Run the pipeline with the test's wiring and today's target date fixed."""
    return answer_question(
        QUESTION,
        connection=conn,
        embedder=embedder,
        corpus=corpus,
        llm=llm,
        vertical="vivienda",
        target_date=AS_OF,
    )


def test_a_verbatim_citation_yields_an_answer(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
) -> None:
    quote = "La duración del arrendamiento será libremente pactada por las partes"
    assert quote in in_force_text(seeded_corpus, "a9")
    fake_llm.queue(
        SYNTHESIS_TASK,
        Mode1Synthesis(
            fundamento=[ProposedCitation(block_id="a9", text=quote)],
            explicacion="El plazo mínimo se pacta libremente, con una duración mínima protegida.",
            accion=["Revisa la fecha de tu contrato", "Habla con tu arrendador"],
        ),
    )

    response = _run(seeded_corpus, deterministic_embedder, corpus_reader, fake_llm)

    assert response.outcome is Outcome.ANSWER
    assert response.answer is not None
    assert response.answer.fundamento[0].block_id == "a9"
    assert response.answer.fundamento[0].verdict is CitationVerdict.VERIFIED_DIRECT
    assert response.answer.fundamento[0].anchor.eli.endswith("/eli/es/l/1994/11/24/29")


def test_a_fabricated_citation_forces_an_honest_abstention(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
) -> None:
    fake_llm.queue(
        SYNTHESIS_TASK,
        Mode1Synthesis(
            fundamento=[ProposedCitation(block_id="a9", text=FABRICATED)],
            explicacion="...",
            accion=[],
        ),
    )

    response = _run(seeded_corpus, deterministic_embedder, corpus_reader, fake_llm)

    assert response.outcome is Outcome.ABSTENTION
    assert response.abstention is not None
    assert response.abstention.reason is AbstentionReason.NO_VERIFIABLE_CITATION
    assert response.answer is None


def test_a_discarded_citation_never_appears_beside_a_valid_one(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
) -> None:
    quote = "La duración del arrendamiento será libremente pactada por las partes"
    fake_llm.queue(
        SYNTHESIS_TASK,
        Mode1Synthesis(
            fundamento=[
                ProposedCitation(block_id="a9", text=quote),
                ProposedCitation(block_id="a9", text=FABRICATED),
            ],
            explicacion="El plazo mínimo se pacta libremente.",
            accion=[],
        ),
    )

    response = _run(seeded_corpus, deterministic_embedder, corpus_reader, fake_llm)

    assert response.outcome is Outcome.ANSWER
    assert response.answer is not None
    assert len(response.answer.fundamento) == 1
    assert all(FABRICATED not in citation.text for citation in response.answer.fundamento)
    assert response.citation_verdicts[CitationVerdict.DISCARDED.value] == 1


def test_empty_evidence_abstains_before_calling_the_model(
    corpus_db: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    fake_llm: FakeLlmClient,
) -> None:
    reader = _EmptyCorpusReader()

    response = answer_question(
        QUESTION,
        connection=corpus_db,
        embedder=deterministic_embedder,
        corpus=reader,
        llm=fake_llm,
        vertical="vivienda",
        target_date=AS_OF,
    )

    assert response.outcome is Outcome.ABSTENTION
    assert response.abstention is not None
    assert response.abstention.reason is AbstentionReason.NO_EVIDENCE
    assert fake_llm.calls == []


class _EmptyCorpusReader:
    """A corpus reader that resolves nothing, standing in for an empty corpus."""

    def resolve_block(self, norm_id: str, block_id: str, target_date: object) -> None:
        return None
