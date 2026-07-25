"""Quality metrics for a Mode 1 eval run: per-case results and their aggregate.

These are the soft numbers compared against a baseline with tolerance, distinct
from the citation guardrail's hard invariant. Retrieval recall is measured over
the accumulated evidence set the agentic run finished with -- what actually
conditioned the answer -- not the first search; the agentic delta records what
iterating earned. Judge-based end-to-end metrics arrive in a later ticket.
"""

from statistics import mean

from pydantic import BaseModel

from lexme.eval.cases import EvalCase
from lexme.eval.guardrail import GuardrailViolation
from lexme.mode1 import AskResponse
from lexme.verification import CitationVerdict


class CaseResult(BaseModel):
    """The measured outcome of one case: what it retrieved, cited and recalled.

    ``recall`` is ``None`` when the case declares no gold blocks, so it is excluded
    from the mean rather than counted as zero. ``outcome_as_expected`` is ``None``
    when the case pins no expected outcome. ``violations`` carries this case's
    guardrail failures, empty on a clean case.
    """

    id: str
    question: str
    outcome: str
    expected_outcome: str | None
    outcome_as_expected: bool | None
    displayed_citations: int
    retrieved_block_ids: list[str]
    gold_block_ids: list[str]
    recall: float | None
    agentic_delta: int
    citation_verdicts: dict[str, int]
    violations: list[GuardrailViolation]


class SuiteMetrics(BaseModel):
    """The aggregate over every case in a run: the numbers a baseline compares.

    ``mean_recall`` averages only the cases that declared gold blocks and is
    ``None`` when none did. ``outcome_match_rate`` is the fraction of cases that
    pinned an expected outcome and reached it, ``None`` when none pinned one.
    ``citation_verdicts`` sums the per-verdict counts across the whole set.
    """

    cases: int
    outcomes: dict[str, int]
    mean_recall: float | None
    outcome_match_rate: float | None
    mean_agentic_delta: float
    citation_verdicts: dict[str, int]


def build_case_result(
    case: EvalCase,
    response: AskResponse,
    violations: list[GuardrailViolation],
) -> CaseResult:
    """Measure one case's response against its gold blocks and record its verdicts."""
    retrieved = _accumulated_evidence(response)
    return CaseResult(
        id=case.id,
        question=case.question,
        outcome=response.outcome.value,
        expected_outcome=case.expected_outcome,
        outcome_as_expected=_outcome_as_expected(case.expected_outcome, response.outcome.value),
        displayed_citations=len(response.answer.fundamento) if response.answer else 0,
        retrieved_block_ids=retrieved,
        gold_block_ids=list(case.gold_block_ids),
        recall=_recall(retrieved, case.gold_block_ids),
        agentic_delta=response.agentic.agentic_delta if response.agentic else 0,
        citation_verdicts=dict(response.citation_verdicts),
        violations=violations,
    )


def aggregate(results: list[CaseResult]) -> SuiteMetrics:
    """Fold per-case results into the run's aggregate metrics."""
    recalls = [result.recall for result in results if result.recall is not None]
    matches = [
        result.outcome_as_expected for result in results if result.outcome_as_expected is not None
    ]
    return SuiteMetrics(
        cases=len(results),
        outcomes=_count_outcomes(results),
        mean_recall=mean(recalls) if recalls else None,
        outcome_match_rate=(sum(matches) / len(matches)) if matches else None,
        mean_agentic_delta=mean([result.agentic_delta for result in results]) if results else 0.0,
        citation_verdicts=_sum_verdicts(results),
    )


def _accumulated_evidence(response: AskResponse) -> list[str]:
    """The distinct block ids retrieved across the run, first occurrence order."""
    if response.agentic is None:
        return []
    seen: dict[str, None] = {}
    for subquery in response.agentic.subqueries:
        for block in subquery.evidence:
            seen.setdefault(block.block_id, None)
    return list(seen)


def _outcome_as_expected(expected: str | None, actual: str) -> bool | None:
    """Whether the case reached its pinned outcome, or ``None`` if it pinned none."""
    if expected is None:
        return None
    return expected == actual


def _recall(retrieved: list[str], gold: tuple[str, ...]) -> float | None:
    """Fraction of gold blocks present in ``retrieved``, or ``None`` when no gold."""
    if not gold:
        return None
    hits = sum(1 for block_id in gold if block_id in retrieved)
    return hits / len(gold)


def _count_outcomes(results: list[CaseResult]) -> dict[str, int]:
    """Count how many cases ended in each outcome."""
    counts: dict[str, int] = {}
    for result in results:
        counts[result.outcome] = counts.get(result.outcome, 0) + 1
    return counts


def _sum_verdicts(results: list[CaseResult]) -> dict[str, int]:
    """Sum the per-verdict citation counts across every case, keyed by verdict value."""
    totals = {verdict.value: 0 for verdict in CitationVerdict}
    for result in results:
        for verdict, count in result.citation_verdicts.items():
            totals[verdict] = totals.get(verdict, 0) + count
    return totals
