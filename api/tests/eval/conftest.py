"""Builders for eval-harness tests: a network-free corpus and canned responses.

The guardrail, metrics and compare tests need no database: a small in-memory
corpus reader and hand-built :class:`AskResponse` values exercise them
deterministically. The end-to-end harness test reuses the Mode 1 fixtures and the
real system behind the fake LLM.
"""

from datetime import date

from lexme.mode1 import (
    AgenticTrace,
    Answer,
    AskResponse,
    Outcome,
    RankedBlockRef,
    SubQueryReport,
    VerifiedCitation,
)
from lexme.verification import CitationVerdict, ResolvedBlock, VerifiedAnchor
from tests.mode1.conftest import (  # noqa: F401  re-exported fixtures for the integration test
    checkpointer,
    corpus_reader,
    seeded_corpus,
)

AS_OF = date(2020, 1, 1)
NORM_ID = "BOE-A-1994-26003"


class InMemoryCorpus:
    """A :class:`~lexme.verification.CorpusReader` over a fixed block-text mapping.

    Programmed with ``{(norm_id, block_id): text}``; resolves any date to that text
    and a stub anchor, and returns ``None`` for an unknown block.
    """

    def __init__(self, blocks: dict[tuple[str, str], str]) -> None:
        """Store the block texts this corpus will resolve."""
        self._blocks = blocks

    def resolve_block(self, norm_id: str, block_id: str, target_date: date) -> ResolvedBlock | None:
        """Return the stored redaction for the block, or ``None`` if unknown."""
        text = self._blocks.get((norm_id, block_id))
        if text is None:
            return None
        return ResolvedBlock(text=text, anchor=_anchor(block_id))


class StubRunner:
    """A :class:`CaseRunner` that replays a fixed response per question, in order."""

    def __init__(self, *responses: AskResponse) -> None:
        """Queue the responses to return across successive :meth:`run` calls."""
        self._responses = list(responses)
        self.questions: list[str] = []

    def run(self, question: str) -> AskResponse:
        """Record ``question`` and return the next queued response."""
        self.questions.append(question)
        return self._responses.pop(0)


def _anchor(block_id: str) -> VerifiedAnchor:
    """A filled-in anchor for a block, enough for a resolved response."""
    return VerifiedAnchor(
        eli="https://www.boe.es/eli/es/l/1994/11/24/29",
        consolidated_html_url="https://www.boe.es/buscar/act.php?id=" + NORM_ID,
        block_id=block_id,
        title="Artículo",
        effective_date=AS_OF,
    )


def answer_response(
    *citations: tuple[str, str],
    evidence: tuple[tuple[str, str], ...],
    verdict: CitationVerdict = CitationVerdict.VERIFIED_DIRECT,
) -> AskResponse:
    """An answer response citing ``(block_id, text)`` over the given evidence blocks.

    ``evidence`` is the ``(norm_id, block_id)`` retrieval trace the guardrail reads
    to find each citation's norm; ``verdict`` stamps every displayed citation, so a
    test can hand out a verdict the corpus will not back up.
    """
    fundamento = [
        VerifiedCitation(block_id=block_id, text=text, verdict=verdict, anchor=_anchor(block_id))
        for block_id, text in citations
    ]
    subquery = SubQueryReport(
        id="sq1",
        text="q",
        purpose="p",
        is_critical=True,
        evidence=[
            RankedBlockRef(norm_id=norm_id, block_id=block_id) for norm_id, block_id in evidence
        ],
    )
    return AskResponse(
        outcome=Outcome.ANSWER,
        thread_id="t",
        answer=Answer(fundamento=fundamento, explicacion="e", accion=[], fecha_objetivo=AS_OF),
        agentic=AgenticTrace(subqueries=[subquery], agentic_delta=1),
        citation_verdicts={verdict.value: len(fundamento)},
    )
