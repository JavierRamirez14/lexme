"""Mode 1 tracer bullet: retrieve, synthesize, verify, then decide in code.

The path is deliberately un-agentic: one retrieval, one synthesis, one
verification pass, and a deterministic gate. The model proposes citations; code
decides whether there is enough verified ground to answer. Abstention is a
first-class outcome, chosen by an explicit rule -- empty evidence, or no citation
surviving verification -- never by the model judging its own sufficiency.
"""

from datetime import date

import psycopg

from lexme.llm import LlmClient
from lexme.mode1.models import (
    Abstention,
    AbstentionReason,
    Answer,
    AskResponse,
    Mode1Synthesis,
    Outcome,
    RankedBlockRef,
    RetrievalTrace,
    VerifiedCitation,
)
from lexme.mode1.synthesis import synthesize
from lexme.retrieval import QueryEmbedder, hybrid_retrieve
from lexme.retrieval.models import BlockKey, HybridResult
from lexme.verification import (
    CitationResult,
    CitationVerdict,
    CorpusReader,
    EvidenceBlock,
    summarize,
    verify_citations,
)

SCOPE_REMINDER = (
    "Solo cubro la Ley de Arrendamientos Urbanos estatal (vivienda). "
    "No cubro normativa autonómica ni otros ámbitos."
)
_NO_EVIDENCE_MESSAGE = (
    "No he encontrado ninguna base en la LAU para responder a tu pregunta. "
    "Prueba a reformularla centrándote en el arrendamiento de vivienda."
)
_NO_VERIFIABLE_CITATION_MESSAGE = (
    "He encontrado texto relacionado, pero no he podido respaldar la respuesta "
    "con una cita literal verificada, así que prefiero no responder."
)


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
    """Answer ``question`` in Mode 1, or abstain, deciding the outcome in code.

    Retrieves evidence, synthesizes a three-layer answer, verifies every proposed
    citation against the corpus at ``target_date`` and keeps only the citations
    that hold. Returns an answer when at least one citation survives, otherwise an
    honest abstention.
    """
    retrieval = hybrid_retrieve(
        connection, embedder, query=question, vertical=vertical, target_date=target_date
    )
    trace = _build_trace(retrieval)

    if not retrieval.evidence:
        return _abstain(AbstentionReason.NO_EVIDENCE, _NO_EVIDENCE_MESSAGE, trace, {})

    synthesis = synthesize(llm, question, retrieval.evidence)
    results = _verify(synthesis, retrieval, corpus, target_date)
    verdicts = summarize(results)
    verified = [_to_verified(result) for result in results if _held(result)]

    if not verified:
        return _abstain(
            AbstentionReason.NO_VERIFIABLE_CITATION,
            _NO_VERIFIABLE_CITATION_MESSAGE,
            trace,
            verdicts,
        )

    answer = Answer(
        fundamento=verified,
        explicacion=synthesis.explicacion,
        accion=synthesis.accion,
    )
    return AskResponse(
        outcome=Outcome.ANSWER,
        answer=answer,
        retrieval=trace,
        citation_verdicts=verdicts,
    )


def _verify(
    synthesis: Mode1Synthesis,
    retrieval: HybridResult,
    corpus: CorpusReader,
    target_date: date,
) -> list[CitationResult]:
    """Verify the proposed citations against the retrieved evidence."""
    evidence = [
        EvidenceBlock(norm_id=block.norm_id, block_id=block.block_id)
        for block in retrieval.evidence
    ]
    return verify_citations(synthesis.fundamento, evidence, target_date, corpus)


def _held(result: CitationResult) -> bool:
    """Whether a citation survived verification (was not discarded)."""
    return result.verdict is not CitationVerdict.DISCARDED


def _to_verified(result: CitationResult) -> VerifiedCitation:
    """Map a surviving citation result to the display citation.

    A non-discarded result always carries an anchor, so it is safe to require one.
    """
    if result.anchor is None:
        raise ValueError(f"citation {result.block_id} held with no anchor")
    return VerifiedCitation(
        block_id=result.block_id,
        text=result.text,
        verdict=result.verdict,
        anchor=result.anchor,
    )


def _abstain(
    reason: AbstentionReason,
    message: str,
    trace: RetrievalTrace,
    verdicts: dict[str, int],
) -> AskResponse:
    """Build an abstention response with the scope reminder attached."""
    return AskResponse(
        outcome=Outcome.ABSTENTION,
        abstention=Abstention(reason=reason, message=message, scope_reminder=SCOPE_REMINDER),
        retrieval=trace,
        citation_verdicts=verdicts,
    )


def _build_trace(retrieval: HybridResult) -> RetrievalTrace:
    """Project the hybrid rankings into the serializable retrieval trace."""
    return RetrievalTrace(
        dense=[_ref(key) for key in retrieval.dense_ranking],
        lexical=[_ref(key) for key in retrieval.lexical_ranking],
        fused=[_ref(key) for key, _ in retrieval.fused_ranking],
    )


def _ref(key: BlockKey) -> RankedBlockRef:
    """Build a ranked block reference from a ``(norm_id, block_id)`` key."""
    norm_id, block_id = key
    return RankedBlockRef(norm_id=norm_id, block_id=block_id)
