"""Quality metrics for a Mode 1 eval run: per-case results and their aggregate.

These are the soft numbers compared against a baseline with tolerance, distinct
from the citation guardrail's hard invariant. Retrieval recall is measured by
layer (dense, lexical, fused, and the evidence the run finished with) and
attributed per sub-query, so a failure is attributable ("retrieval recalls 0.92
but synthesis drops a citation"). The agentic delta is reported as a first-pass to
final recall pair -- what iterating earned. The judge's end-to-end numbers ride
along on every answered case that carries reference key points. How each case got
past the disambiguation gate is carried alongside them, because a case that never
got past a pause was never measured and must not be averaged in as if it had been.
"""

from collections.abc import Iterable
from enum import StrEnum
from statistics import mean

from pydantic import BaseModel

from lexme.eval.cases import EvalCase
from lexme.eval.guardrail import GuardrailViolation
from lexme.eval.judge import JudgeMetrics, JudgeVerdict, build_judge_metrics
from lexme.mode1 import AskResponse, Outcome, RankedBlockRef, SubQueryReport
from lexme.verification import CitationVerdict


class Disambiguation(StrEnum):
    """How a case got past the disambiguation gate, if it met it at all.

    ``DIRECT`` never paused. ``RESUMED`` paused and was continued with the reply
    the case pins, so its numbers describe a final answer. ``UNANSWERED`` paused
    with no pinned reply for the branch asked, so the case was never measured past
    the pause and must not be read as either a hit or a generic failure.
    """

    DIRECT = "directo"
    RESUMED = "reanudado"
    UNANSWERED = "sin_respuesta"


LAYER_DENSE = "dense"
LAYER_LEXICAL = "lexical"
LAYER_FUSED = "fused"
LAYER_EVIDENCE = "evidence"
RETRIEVAL_LAYERS = (LAYER_DENSE, LAYER_LEXICAL, LAYER_FUSED, LAYER_EVIDENCE)


class LayerRecall(BaseModel):
    """The gold blocks one retrieval layer recovered, accumulated across sub-queries.

    ``recovered_gold`` are the case's gold blocks present in this layer's rankings;
    ``recall`` is their fraction of the gold, or ``None`` when the case has no gold.
    """

    layer: str
    recovered_gold: list[str]
    recall: float | None


class SubQueryRecall(BaseModel):
    """The gold blocks one sub-query's evidence recovered, attributing recall by plan step."""

    id: str
    recovered_gold: list[str]
    recall: float | None


class RetrievalRecall(BaseModel):
    """A case's retrieval recall by layer and its per-sub-query attribution."""

    layers: list[LayerRecall]
    subqueries: list[SubQueryRecall]


class CaseResult(BaseModel):
    """The measured outcome of one case: what it retrieved, cited, recalled and judged.

    ``recall`` is the final accumulated-evidence recall, ``None`` when the case
    declares no gold blocks so it is excluded from the mean rather than counted as
    zero. ``first_pass_recall`` is the recall the first retrieval pass reached and
    ``recall_delta`` is the gain the agentic loop earned over it. ``retrieval``
    carries the per-layer and per-sub-query breakdown; ``judge`` the end-to-end
    numbers, present only for an answered case with reference key points.
    ``outcome_as_expected`` is ``None`` when the case pins no expected outcome.
    ``disambiguation`` says whether the case answered directly, was resumed with
    the reply it pins, or stopped at a pause it brought no reply for; it is ``None``
    only in an artifact written before the harness measured that.
    """

    id: str
    question: str
    outcome: str
    expected_outcome: str | None
    outcome_as_expected: bool | None
    disambiguation: str | None = None
    displayed_citations: int
    retrieved_block_ids: list[str]
    gold_block_ids: list[str]
    recall: float | None
    first_pass_recall: float | None
    recall_delta: float | None
    agentic_delta: int
    retrieval: RetrievalRecall | None
    citation_verdicts: dict[str, int]
    judge: JudgeMetrics | None
    violations: list[GuardrailViolation]


class JudgeAggregate(BaseModel):
    """The run's aggregate judge numbers over every judged case.

    ``mean_completeness`` averages the per-case completeness. ``unsupported_claim_rate``
    is pooled -- total unsupported claims over total claims across the run -- so a
    verbose case cannot dominate the hallucination number. ``mean_clarity`` averages
    the rubric score.
    """

    judged_cases: int
    mean_completeness: float | None
    unsupported_claim_rate: float | None
    mean_clarity: float | None


class SuiteMetrics(BaseModel):
    """The aggregate over every case in a run: the numbers a baseline compares.

    ``mean_recall`` and ``mean_first_pass_recall`` are the ends of the agentic
    recall pair, and ``mean_recall_delta`` the gain between them, over the cases
    that declared gold blocks. ``retrieval_layer_recall`` is the mean recall of
    each retrieval layer. ``outcome_match_rate`` is the fraction of cases that
    pinned an outcome and reached it; ``abstention_rate`` the fraction that
    abstained and ``expected_abstention_recall`` how many of the cases meant to
    abstain did, so abstention is read next to precision, never alone.
    ``disambiguation`` counts the cases by how they got past the disambiguation
    gate and ``disambiguation_rate`` is the fraction of cases the gate stopped,
    resumed or not, so how many cases a run really measured end to end is never
    ambiguous; both are empty in an artifact written before the harness measured
    that. ``judge`` is ``None`` when no case was judged.
    """

    cases: int
    outcomes: dict[str, int]
    disambiguation: dict[str, int] = {}
    disambiguation_rate: float | None = None
    mean_recall: float | None
    mean_first_pass_recall: float | None
    mean_recall_delta: float | None
    retrieval_layer_recall: dict[str, float]
    outcome_match_rate: float | None
    abstention_rate: float | None
    expected_abstention_recall: float | None
    mean_agentic_delta: float
    judge: JudgeAggregate | None
    citation_verdicts: dict[str, int]


def build_case_result(
    case: EvalCase,
    response: AskResponse,
    violations: list[GuardrailViolation],
    verdict: JudgeVerdict | None = None,
    disambiguation: Disambiguation = Disambiguation.DIRECT,
) -> CaseResult:
    """Measure one case's response against its gold blocks, key points and verdicts."""
    retrieved = _accumulated_evidence(response)
    recall = _recall(retrieved, case.gold_block_ids)
    first_pass = _first_pass_recall(response, case.gold_block_ids)
    return CaseResult(
        id=case.id,
        question=case.question,
        outcome=response.outcome.value,
        expected_outcome=case.expected_outcome,
        outcome_as_expected=_outcome_as_expected(case.expected_outcome, response.outcome.value),
        disambiguation=disambiguation.value,
        displayed_citations=len(response.answer.fundamento) if response.answer else 0,
        retrieved_block_ids=retrieved,
        gold_block_ids=list(case.gold_block_ids),
        recall=recall,
        first_pass_recall=first_pass,
        recall_delta=_delta(first_pass, recall),
        agentic_delta=response.agentic.agentic_delta if response.agentic else 0,
        retrieval=_retrieval_recall(response, case.gold_block_ids),
        citation_verdicts=dict(response.citation_verdicts),
        judge=build_judge_metrics(verdict) if verdict is not None else None,
        violations=violations,
    )


def aggregate(results: list[CaseResult]) -> SuiteMetrics:
    """Fold per-case results into the run's aggregate metrics."""
    recalls = [result.recall for result in results if result.recall is not None]
    first_recalls = [
        result.first_pass_recall for result in results if result.first_pass_recall is not None
    ]
    deltas = [result.recall_delta for result in results if result.recall_delta is not None]
    matches = [
        result.outcome_as_expected for result in results if result.outcome_as_expected is not None
    ]
    return SuiteMetrics(
        cases=len(results),
        outcomes=_count_outcomes(results),
        disambiguation=_count_disambiguation(results),
        disambiguation_rate=_disambiguation_rate(results),
        mean_recall=mean(recalls) if recalls else None,
        mean_first_pass_recall=mean(first_recalls) if first_recalls else None,
        mean_recall_delta=mean(deltas) if deltas else None,
        retrieval_layer_recall=_layer_means(results),
        outcome_match_rate=(sum(matches) / len(matches)) if matches else None,
        abstention_rate=_abstention_rate(results),
        expected_abstention_recall=_expected_abstention_recall(results),
        mean_agentic_delta=mean([result.agentic_delta for result in results]) if results else 0.0,
        judge=_aggregate_judge(results),
        citation_verdicts=_sum_verdicts(results),
    )


def _accumulated_evidence(response: AskResponse) -> list[str]:
    """The distinct block ids retrieved across the run, first occurrence order."""
    if response.agentic is None:
        return []
    return _union_block_ids(sub.evidence for sub in response.agentic.subqueries)


def _retrieval_recall(response: AskResponse, gold: tuple[str, ...]) -> RetrievalRecall | None:
    """The per-layer and per-sub-query recall breakdown, or ``None`` without a trace."""
    if response.agentic is None:
        return None
    subs = response.agentic.subqueries
    layers = [
        LayerRecall(
            layer=name,
            recovered_gold=_recovered(_layer_union(subs, name), gold),
            recall=_recall(_layer_union(subs, name), gold),
        )
        for name in RETRIEVAL_LAYERS
    ]
    subqueries = [
        SubQueryRecall(
            id=sub.id,
            recovered_gold=_recovered([block.block_id for block in sub.evidence], gold),
            recall=_recall([block.block_id for block in sub.evidence], gold),
        )
        for sub in subs
    ]
    return RetrievalRecall(layers=layers, subqueries=subqueries)


def _layer_union(subqueries: list[SubQueryReport], layer: str) -> list[str]:
    """The distinct block ids a retrieval layer surfaced across every sub-query."""
    if layer == LAYER_EVIDENCE:
        return _union_block_ids(sub.evidence for sub in subqueries)
    rankings = [_ranking(sub, layer) for sub in subqueries if sub.retrieval is not None]
    return _union_block_ids(rankings)


def _ranking(sub: SubQueryReport, layer: str) -> list[RankedBlockRef]:
    """One sub-query's ranking for a retriever layer, by explicit name not attribute lookup."""
    assert sub.retrieval is not None  # guarded by the caller
    rankings = {
        LAYER_DENSE: sub.retrieval.dense,
        LAYER_LEXICAL: sub.retrieval.lexical,
        LAYER_FUSED: sub.retrieval.fused,
    }
    return rankings[layer]


def _union_block_ids(groups: Iterable[Iterable[RankedBlockRef]]) -> list[str]:
    """Flatten groups of block references into distinct block ids, first-seen order."""
    seen: dict[str, None] = {}
    for group in groups:
        for ref in group:
            seen.setdefault(ref.block_id, None)
    return list(seen)


def _first_pass_recall(response: AskResponse, gold: tuple[str, ...]) -> float | None:
    """Recall over the evidence the first self-critique pass had accumulated."""
    if response.agentic is None or not response.agentic.passes or not gold:
        return None
    return _recall(response.agentic.passes[0].evidence_block_ids, gold)


def _recovered(retrieved: list[str], gold: tuple[str, ...]) -> list[str]:
    """The gold blocks present in ``retrieved``, in gold order."""
    return [block_id for block_id in gold if block_id in retrieved]


def _delta(first: float | None, final: float | None) -> float | None:
    """The recall gain from first pass to final, or ``None`` if either is absent."""
    if first is None or final is None:
        return None
    return final - first


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


def _layer_means(results: list[CaseResult]) -> dict[str, float]:
    """Mean recall per retrieval layer over the cases that measured it."""
    by_layer: dict[str, list[float]] = {}
    for result in results:
        if result.retrieval is None:
            continue
        for layer in result.retrieval.layers:
            if layer.recall is not None:
                by_layer.setdefault(layer.layer, []).append(layer.recall)
    return {layer: mean(values) for layer, values in by_layer.items()}


def _abstention_rate(results: list[CaseResult]) -> float | None:
    """The fraction of cases that ended in an honest abstention."""
    if not results:
        return None
    abstained = sum(1 for result in results if result.outcome == Outcome.ABSTENTION.value)
    return abstained / len(results)


def _expected_abstention_recall(results: list[CaseResult]) -> float | None:
    """Of the cases meant to abstain, the fraction that actually did."""
    expected = [result for result in results if result.expected_outcome == Outcome.ABSTENTION.value]
    if not expected:
        return None
    honored = sum(1 for result in expected if result.outcome == Outcome.ABSTENTION.value)
    return honored / len(expected)


def _aggregate_judge(results: list[CaseResult]) -> JudgeAggregate | None:
    """Aggregate the per-case judge numbers, or ``None`` when no case was judged."""
    judged = [result.judge for result in results if result.judge is not None]
    if not judged:
        return None
    completeness = [j.completeness for j in judged if j.completeness is not None]
    clarities = [j.clarity for j in judged]
    total_claims = sum(j.claims_total for j in judged)
    total_unsupported = sum(j.claims_unsupported for j in judged)
    return JudgeAggregate(
        judged_cases=len(judged),
        mean_completeness=mean(completeness) if completeness else None,
        unsupported_claim_rate=(total_unsupported / total_claims) if total_claims else None,
        mean_clarity=mean(clarities) if clarities else None,
    )


def _count_outcomes(results: list[CaseResult]) -> dict[str, int]:
    """Count how many cases ended in each outcome."""
    counts: dict[str, int] = {}
    for result in results:
        counts[result.outcome] = counts.get(result.outcome, 0) + 1
    return counts


def _count_disambiguation(results: list[CaseResult]) -> dict[str, int]:
    """Count the cases by how they got past the disambiguation gate, all states listed."""
    counts = {state.value: 0 for state in Disambiguation}
    for result in results:
        if result.disambiguation is not None:
            counts[result.disambiguation] += 1
    return counts


def _disambiguation_rate(results: list[CaseResult]) -> float | None:
    """The fraction of cases the gate stopped, whether or not they were resumed."""
    if not results:
        return None
    paused = sum(1 for result in results if result.disambiguation != Disambiguation.DIRECT.value)
    return paused / len(results)


def _sum_verdicts(results: list[CaseResult]) -> dict[str, int]:
    """Sum the per-verdict citation counts across every case, keyed by verdict value."""
    totals = {verdict.value: 0 for verdict in CitationVerdict}
    for result in results:
        for verdict, count in result.citation_verdicts.items():
            totals[verdict] = totals.get(verdict, 0) + count
    return totals
