"""Builders for eval-harness tests: a network-free corpus and canned responses.

The guardrail, metrics and compare tests need no database: a small in-memory
corpus reader and hand-built :class:`AskResponse` values exercise them
deterministically. The end-to-end harness test reuses the Mode 1 fixtures and the
real system behind the fake LLM.
"""

from collections.abc import Sequence
from datetime import date

from lexme.eval.cases import ClarificationAnswer
from lexme.eval.metrics import Disambiguation
from lexme.eval.runner import CaseRun
from lexme.mode1 import (
    AgenticTrace,
    Answer,
    AskResponse,
    Clarification,
    Outcome,
    PassReport,
    RankedBlockRef,
    RetrievalTrace,
    SubQueryReport,
    VerifiedCitation,
)
from lexme.mode1.branches import AnswerKind
from lexme.verification import CitationVerdict, ResolvedBlock, VerifiedAnchor
from tests.mode1.conftest import (  # noqa: F401  re-exported fixtures for the integration test
    checkpointer,
    corpus_reader,
    seeded_corpus,
    vivienda_branches,
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
    """A :class:`CaseRunner` that replays a fixed run per question, in order.

    A queued :class:`AskResponse` stands for a case answered without pausing; queue
    a :class:`CaseRun` instead to replay a resumed or an unanswered pause.
    """

    def __init__(self, *runs: AskResponse | CaseRun) -> None:
        """Queue the runs to return across successive :meth:`run` calls."""
        self._runs = [_as_case_run(run) for run in runs]
        self.questions: list[str] = []
        self.dates: list[date] = []
        self.pinned_answers: list[tuple[ClarificationAnswer, ...]] = []

    def run(
        self,
        question: str,
        target_date: date,
        clarification_answers: Sequence[ClarificationAnswer] = (),
    ) -> CaseRun:
        """Record what the harness handed over and return the next queued run."""
        self.questions.append(question)
        self.dates.append(target_date)
        self.pinned_answers.append(tuple(clarification_answers))
        return self._runs.pop(0)


def _as_case_run(run: AskResponse | CaseRun) -> CaseRun:
    """Read a queued run, treating a bare response as a case answered directly."""
    if isinstance(run, CaseRun):
        return run
    return CaseRun(response=run, disambiguation=Disambiguation.DIRECT)


def _anchor(block_id: str) -> VerifiedAnchor:
    """A filled-in anchor for a block, enough for a resolved response."""
    return VerifiedAnchor(
        eli="https://www.boe.es/eli/es/l/1994/11/24/29",
        consolidated_html_url="https://www.boe.es/buscar/act.php?id=" + NORM_ID,
        block_id=block_id,
        title="Artículo",
        effective_date=AS_OF,
    )


def _layer_refs(keys: tuple[tuple[str, str], ...]) -> list[RankedBlockRef]:
    """Build a ranking of block references from ``(norm_id, block_id)`` keys."""
    return [RankedBlockRef(norm_id=norm_id, block_id=block_id) for norm_id, block_id in keys]


def traced_response(
    *,
    citations: tuple[tuple[str, str], ...] = (),
    dense: tuple[tuple[str, str], ...],
    lexical: tuple[tuple[str, str], ...],
    fused: tuple[tuple[str, str], ...],
    evidence: tuple[tuple[str, str], ...],
    first_pass_evidence: tuple[str, ...] = (),
    explicacion: str = "e",
    outcome: Outcome = Outcome.ANSWER,
) -> AskResponse:
    """A response with one sub-query's per-layer rankings and a two-pass history.

    ``dense``/``lexical``/``fused``/``evidence`` are that sub-query's rankings as
    ``(norm_id, block_id)`` keys; ``first_pass_evidence`` are the block ids the
    first pass had accumulated, so the agentic recall delta is measurable.
    """
    subquery = SubQueryReport(
        id="sq1",
        text="q",
        purpose="p",
        is_critical=True,
        evidence=_layer_refs(evidence),
        retrieval=RetrievalTrace(
            dense=_layer_refs(dense),
            lexical=_layer_refs(lexical),
            fused=_layer_refs(fused),
        ),
    )
    passes = [
        PassReport(
            pass_number=1,
            sufficient_ids=[],
            insufficient_ids=["sq1"],
            evidence_count=len(first_pass_evidence),
            evidence_block_ids=list(first_pass_evidence),
        ),
        PassReport(
            pass_number=2,
            sufficient_ids=["sq1"],
            insufficient_ids=[],
            evidence_count=len(evidence),
            evidence_block_ids=[block_id for _, block_id in evidence],
        ),
    ]
    answer = Answer(
        fundamento=[
            VerifiedCitation(
                block_id=block_id,
                text=text,
                verdict=CitationVerdict.VERIFIED_DIRECT,
                anchor=_anchor(block_id),
            )
            for block_id, text in citations
        ],
        explicacion=explicacion,
        accion=[],
        fecha_objetivo=AS_OF,
    )
    return AskResponse(
        outcome=outcome,
        thread_id="t",
        answer=answer,
        agentic=AgenticTrace(subqueries=[subquery], passes=passes, agentic_delta=1),
        citation_verdicts={CitationVerdict.VERIFIED_DIRECT.value: len(citations)},
    )


def clarification_response(branch_id: str = "fecha_firma") -> AskResponse:
    """A paused response carrying the disambiguating question and no answer."""
    return AskResponse(
        outcome=Outcome.CLARIFICATION,
        thread_id="t",
        clarification=Clarification(
            branch_id=branch_id, question="¿cuándo firmaste?", answer_kind=AnswerKind.DATE
        ),
        agentic=AgenticTrace(),
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
