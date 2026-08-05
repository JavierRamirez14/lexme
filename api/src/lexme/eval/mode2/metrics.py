"""Mode 2 quality metrics: the asymmetric error the whole design exists to kill.

The grave error a tenancy-risk tool can make is telling a tenant a genuinely
illegal or worse-than-default clause is fine. So the headline numbers are put on
that error: the recall of problematic (red/orange) clauses, and its named
counterpart the false-tranquility rate -- the share of real red/orange clauses the
system let pass as reassuring (green or silence). Precision on the flags is always
published next to the abstention rate, because precision read alone can be inflated
by abstaining. A full per-level confusion matrix, with the abstention classes as
their own columns, is the diagnostic underneath. Every level number is computed over
the clauses the segmentation layer delimited correctly, so it measures
classification, not segmentation. The absence whites are scored separately, by set
recall and precision of the omitted checklist items.
"""

from collections.abc import Iterable, Sequence

from pydantic import BaseModel

from lexme.eval.guardrail import GuardrailViolation
from lexme.eval.mode2.cases import Mode2EvalCase
from lexme.eval.mode2.segmentation import (
    DEFAULT_IOU_THRESHOLD,
    LabeledSpan,
    SpanMatch,
    match_spans,
)
from lexme.eval.repetition import MetricDirection
from lexme.mode2.models import ContractAnalysis
from lexme.mode2.risk import ClauseFinding, CoverageStatus, RiskLevel

CLASS_INFORMATIVA = CoverageStatus.INFORMATIVA.value
CLASS_FUERA_DE_AMBITO = CoverageStatus.FUERA_DE_AMBITO.value
CLASS_NO_CONCLUYENTE = CoverageStatus.NO_CONCLUYENTE.value

PROBLEMATIC_LEVELS: frozenset[RiskLevel] = frozenset({RiskLevel.ILEGAL, RiskLevel.PEOR_QUE_DEFAULT})
PROBLEMATIC_CLASSES: frozenset[str] = frozenset(level.value for level in PROBLEMATIC_LEVELS)
REASSURING_CLASSES: frozenset[str] = frozenset({RiskLevel.CORRECTO.value, CLASS_INFORMATIVA})
ABSTENTION_CLASSES: frozenset[str] = frozenset({CLASS_NO_CONCLUYENTE, CLASS_FUERA_DE_AMBITO})

GOLD_LEVELS: tuple[str, ...] = (
    RiskLevel.ILEGAL.value,
    RiskLevel.PEOR_QUE_DEFAULT.value,
    RiskLevel.NEGOCIABLE.value,
    RiskLevel.CORRECTO.value,
)
PREDICTED_CLASSES: tuple[str, ...] = (
    RiskLevel.ILEGAL.value,
    RiskLevel.PEOR_QUE_DEFAULT.value,
    RiskLevel.NEGOCIABLE.value,
    RiskLevel.CORRECTO.value,
    CLASS_INFORMATIVA,
    CLASS_NO_CONCLUYENTE,
    CLASS_FUERA_DE_AMBITO,
)


class ClausePrediction(BaseModel):
    """One reference clause's verdict: what it was, what the system called it.

    ``predicted_class`` is the class the matched finding placed the clause in, or
    ``None`` when segmentation delimited no span for it (``matched`` false). ``iou``
    is the best character overlap the reference clause found. A reader can attribute
    a miss to segmentation (``matched`` false) or to classification (``matched``
    true, ``predicted_class`` not the expected level).
    """

    clause_id: str
    expected_level: str
    predicted_class: str | None
    iou: float
    matched: bool


class SegmentationLayer(BaseModel):
    """The segmentation layer's own score, kept apart from the level metrics.

    ``delimited`` counts the reference clauses matched at or above ``threshold``
    IoU; ``delimited_rate`` is their fraction, or ``None`` for a contract with no
    clauses. ``matches`` is the per-clause pairing so a drop or a merge is visible.
    """

    threshold: float
    reference_clauses: int
    delimited: int
    delimited_rate: float | None
    matches: list[SpanMatch]


class AbsenceRecall(BaseModel):
    """The absence layer's score: did the cross-check surface the omitted rights.

    ``recall`` is the fraction of expected whites the system produced; ``precision``
    the fraction of produced whites that were expected, or ``None`` when a side is
    empty. ``missed`` and ``spurious`` name the differences so a failure is
    attributable to a specific checklist item.
    """

    expected: list[str]
    detected: list[str]
    missed: list[str]
    spurious: list[str]
    recall: float | None
    precision: float | None


class Mode2CaseResult(BaseModel):
    """One contract's measured outcome across the segmentation and level layers.

    The counts (not just the rates) are carried so the suite aggregate pools them
    rather than averaging rates, and a big contract cannot dominate the headline.
    ``recall_problematic`` and ``false_tranquility_rate`` share the denominator
    ``problematic_total`` -- the red/orange clauses that were correctly delimited.
    ``precision_problematic`` and ``abstention_rate`` are published together.
    ``outcome_as_expected`` is whether this contract reached the outcome its case
    declared. The ``_e2e`` numbers share the same event definitions as their
    conditioned counterparts but are computed over every problematic reference
    clause, delimited or not -- ``problematic_total_e2e`` is what a tenant's own
    contract actually contained, not what the segmentation layer let through. A
    problematic clause the segmentation layer never delimited is counted in
    ``not_reported_problematic`` rather than folded into either the detected or the
    false-tranquility count, because it was never shown to the user as anything.
    ``displayed_citations`` is how many citations this contract put in front of a
    reader -- the denominator the guardrail's ``violations`` are a count out of.
    All ``_e2e``, ``outcome_*`` and ``displayed_citations`` fields default so an
    artifact written before this schema still reads back.
    """

    id: str
    outcome: str
    outcome_as_expected: bool | None = None
    displayed_citations: int = 0
    segmentation: SegmentationLayer
    clause_predictions: list[ClausePrediction]
    confusion: dict[str, dict[str, int]]
    matched_clauses: int
    problematic_total: int
    problematic_detected: int
    recall_problematic: float | None
    false_tranquility_events: int
    false_tranquility_rate: float | None
    flagged_problematic: int
    flagged_true_positive: int
    precision_problematic: float | None
    abstention_clauses: int
    abstention_rate: float | None
    problematic_total_e2e: int = 0
    problematic_detected_e2e: int = 0
    recall_problematic_e2e: float | None = None
    false_tranquility_events_e2e: int = 0
    false_tranquility_rate_e2e: float | None = None
    not_reported_problematic: int = 0
    absences: AbsenceRecall
    violations: list[GuardrailViolation]


class Mode2SuiteMetrics(BaseModel):
    """The run aggregate: the two headline numbers, their diagnostics, the whites.

    ``recall_problematic`` and ``false_tranquility_rate`` are pooled over every
    correctly delimited red/orange clause in the run. ``precision_problematic`` is
    published beside ``abstention_rate`` by construction. ``confusion`` is the
    pooled per-level matrix. ``segmentation_delimited_rate`` is the layer the level
    numbers are conditioned on, and the absence recall/precision score the whites.
    ``outcome_match_rate`` is the Mode 2 counterpart of Mode 1's own metric: the
    share of cases that reached the outcome their case declared. ``recall_problematic_e2e``
    and ``false_tranquility_rate_e2e`` are pooled over every problematic reference
    clause in the run, delimited or not -- the number a tenant actually experiences
    -- and are published beside, never instead of, the conditioned pair, because the
    conditioned pair is what tells a segmentation miss apart from a classification
    one. ``not_reported_problematic`` is the count folded out of both ``_e2e``
    numbers: a problematic clause the segmentation layer never delimited, so it was
    never shown to the user as reassuring or as anything else. ``displayed_citations``
    is how many citations the run put in front of a reader, so the guardrail's verdict
    is published with the denominator it was taken over rather than as a bare pass.
    All ``_e2e``, ``outcome_match_rate`` and ``displayed_citations`` fields default so
    an artifact written before this schema still reads back.
    """

    cases: int
    outcomes: dict[str, int]
    outcome_match_rate: float | None = None
    displayed_citations: int = 0
    reference_clauses: int
    delimited_clauses: int
    segmentation_delimited_rate: float | None
    matched_clauses: int
    confusion: dict[str, dict[str, int]]
    problematic_total: int
    problematic_detected: int
    recall_problematic: float | None
    false_tranquility_events: int
    false_tranquility_rate: float | None
    flagged_problematic: int
    flagged_true_positive: int
    precision_problematic: float | None
    abstention_clauses: int
    abstention_rate: float | None
    problematic_total_e2e: int = 0
    problematic_detected_e2e: int = 0
    recall_problematic_e2e: float | None = None
    false_tranquility_events_e2e: int = 0
    false_tranquility_rate_e2e: float | None = None
    not_reported_problematic: int = 0
    absence_expected: int
    absence_detected: int
    absence_recall: float | None
    absence_predicted: int
    absence_precision: float | None


CASES = "cases"
OUTCOME_MATCH_RATE = "outcome_match_rate"
RECALL_PROBLEMATIC = "recall_problematic"
FALSE_TRANQUILITY_RATE = "false_tranquility_rate"
RECALL_PROBLEMATIC_E2E = "recall_problematic_e2e"
FALSE_TRANQUILITY_RATE_E2E = "false_tranquility_rate_e2e"
PRECISION_PROBLEMATIC = "precision_problematic"
ABSTENTION_RATE = "abstention_rate"
SEGMENTATION_DELIMITED_RATE = "segmentation_delimited_rate"
ABSENCE_RECALL = "absence_recall"
ABSENCE_PRECISION = "absence_precision"

MODE2_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    CASES: MetricDirection.NEUTRAL,
    OUTCOME_MATCH_RATE: MetricDirection.HIGHER_IS_BETTER,
    RECALL_PROBLEMATIC: MetricDirection.HIGHER_IS_BETTER,
    FALSE_TRANQUILITY_RATE: MetricDirection.LOWER_IS_BETTER,
    RECALL_PROBLEMATIC_E2E: MetricDirection.HIGHER_IS_BETTER,
    FALSE_TRANQUILITY_RATE_E2E: MetricDirection.LOWER_IS_BETTER,
    PRECISION_PROBLEMATIC: MetricDirection.HIGHER_IS_BETTER,
    ABSTENTION_RATE: MetricDirection.NEUTRAL,
    SEGMENTATION_DELIMITED_RATE: MetricDirection.HIGHER_IS_BETTER,
    ABSENCE_RECALL: MetricDirection.HIGHER_IS_BETTER,
    ABSENCE_PRECISION: MetricDirection.HIGHER_IS_BETTER,
}


def scalar_metrics_mode2(metrics: Mode2SuiteMetrics) -> dict[str, float | None]:
    """Project a Mode 2 run's aggregate onto the flat scalars a band or a delta reads.

    Carries both the conditioned-on-segmentation pair and its end-to-end
    counterpart, so a band is measured over the number a tenant experiences and not
    only over the diagnostic.
    """
    return {
        CASES: float(metrics.cases),
        OUTCOME_MATCH_RATE: metrics.outcome_match_rate,
        RECALL_PROBLEMATIC: metrics.recall_problematic,
        FALSE_TRANQUILITY_RATE: metrics.false_tranquility_rate,
        RECALL_PROBLEMATIC_E2E: metrics.recall_problematic_e2e,
        FALSE_TRANQUILITY_RATE_E2E: metrics.false_tranquility_rate_e2e,
        PRECISION_PROBLEMATIC: metrics.precision_problematic,
        ABSTENTION_RATE: metrics.abstention_rate,
        SEGMENTATION_DELIMITED_RATE: metrics.segmentation_delimited_rate,
        ABSENCE_RECALL: metrics.absence_recall,
        ABSENCE_PRECISION: metrics.absence_precision,
    }


def predicted_class(finding: ClauseFinding) -> str:
    """Project a clause finding onto its predicted class for the confusion matrix.

    An evaluated finding reports its risk level; the coverage escapes (informative,
    inconclusive, out of scope) each become their own class, so abstention is never
    folded into a level and cannot be mistaken for a verdict.
    """
    if finding.coverage is CoverageStatus.EVALUADA:
        if finding.level is None:
            raise ValueError(f"evaluated finding '{finding.clause_id}' carries no level")
        return finding.level.value
    return finding.coverage.value


def build_mode2_case_result(
    case: Mode2EvalCase,
    analysis: ContractAnalysis,
    violations: list[GuardrailViolation] | None = None,
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Mode2CaseResult:
    """Measure one analysis against its reference clauses, levels and absences.

    Matches the risk map's clause spans to the reference boundaries, computes the
    level metrics over the correctly delimited clauses alone, and scores the absence
    whites by set recall and precision. A rejected analysis carries an empty risk
    map, so every reference clause is an undelimited segmentation miss -- the honest
    reading of a contract the system refused to analyze. ``violations`` are the
    guardrail failures already found for this case, carried onto the result.
    """
    findings = analysis.risk_map.clause_findings if analysis.risk_map else []
    class_by_finding = {finding.clause_id: predicted_class(finding) for finding in findings}
    matches = match_spans(_reference_spans(case), _predicted_spans(findings), threshold)
    predictions = _clause_predictions(case, matches, class_by_finding)
    matched = [prediction for prediction in predictions if prediction.matched]
    outcome = _outcome_label(analysis)
    return Mode2CaseResult(
        id=case.id,
        outcome=outcome,
        outcome_as_expected=case.expected_outcome == outcome,
        displayed_citations=_displayed_citations(analysis),
        segmentation=_segmentation_layer(case, matches, threshold),
        clause_predictions=predictions,
        confusion=_confusion(matched),
        matched_clauses=len(matched),
        problematic_total=_problematic_total(matched),
        problematic_detected=_problematic_detected(matched),
        recall_problematic=_ratio(_problematic_detected(matched), _problematic_total(matched)),
        false_tranquility_events=_false_tranquility(matched),
        false_tranquility_rate=_ratio(_false_tranquility(matched), _problematic_total(matched)),
        flagged_problematic=_flagged(matched),
        flagged_true_positive=_flagged_true_positive(matched),
        precision_problematic=_ratio(_flagged_true_positive(matched), _flagged(matched)),
        abstention_clauses=_abstained(matched),
        abstention_rate=_ratio(_abstained(matched), len(matched)),
        problematic_total_e2e=_problematic_total_e2e(predictions),
        problematic_detected_e2e=_problematic_detected_e2e(predictions),
        recall_problematic_e2e=_ratio(
            _problematic_detected_e2e(predictions), _problematic_total_e2e(predictions)
        ),
        false_tranquility_events_e2e=_false_tranquility_e2e(predictions),
        false_tranquility_rate_e2e=_ratio(
            _false_tranquility_e2e(predictions), _problematic_total_e2e(predictions)
        ),
        not_reported_problematic=_not_reported_problematic(predictions),
        absences=_absence_recall(case, analysis),
        violations=violations or [],
    )


def aggregate_mode2(results: Sequence[Mode2CaseResult]) -> Mode2SuiteMetrics:
    """Fold per-case results into the run aggregate, pooling counts not rates."""
    reference_clauses = sum(result.segmentation.reference_clauses for result in results)
    delimited = sum(result.segmentation.delimited for result in results)
    matched = sum(result.matched_clauses for result in results)
    problematic_total = sum(result.problematic_total for result in results)
    problematic_detected = sum(result.problematic_detected for result in results)
    false_tranquility = sum(result.false_tranquility_events for result in results)
    flagged = sum(result.flagged_problematic for result in results)
    flagged_true = sum(result.flagged_true_positive for result in results)
    abstained = sum(result.abstention_clauses for result in results)
    problematic_total_e2e = sum(result.problematic_total_e2e for result in results)
    problematic_detected_e2e = sum(result.problematic_detected_e2e for result in results)
    false_tranquility_e2e = sum(result.false_tranquility_events_e2e for result in results)
    not_reported = sum(result.not_reported_problematic for result in results)
    absence_expected = sum(len(result.absences.expected) for result in results)
    absence_detected = sum(len(result.absences.detected) for result in results)
    absence_predicted = sum(
        len(result.absences.detected) + len(result.absences.spurious) for result in results
    )
    return Mode2SuiteMetrics(
        cases=len(results),
        outcomes=_count_outcomes(results),
        outcome_match_rate=_outcome_match_rate(results),
        displayed_citations=sum(result.displayed_citations for result in results),
        reference_clauses=reference_clauses,
        delimited_clauses=delimited,
        segmentation_delimited_rate=_ratio(delimited, reference_clauses),
        matched_clauses=matched,
        confusion=_pool_confusion(results),
        problematic_total=problematic_total,
        problematic_detected=problematic_detected,
        recall_problematic=_ratio(problematic_detected, problematic_total),
        false_tranquility_events=false_tranquility,
        false_tranquility_rate=_ratio(false_tranquility, problematic_total),
        flagged_problematic=flagged,
        flagged_true_positive=flagged_true,
        precision_problematic=_ratio(flagged_true, flagged),
        abstention_clauses=abstained,
        abstention_rate=_ratio(abstained, matched),
        problematic_total_e2e=problematic_total_e2e,
        problematic_detected_e2e=problematic_detected_e2e,
        recall_problematic_e2e=_ratio(problematic_detected_e2e, problematic_total_e2e),
        false_tranquility_events_e2e=false_tranquility_e2e,
        false_tranquility_rate_e2e=_ratio(false_tranquility_e2e, problematic_total_e2e),
        not_reported_problematic=not_reported,
        absence_expected=absence_expected,
        absence_detected=absence_detected,
        absence_recall=_ratio(absence_detected, absence_expected),
        absence_predicted=absence_predicted,
        absence_precision=_ratio(absence_detected, absence_predicted),
    )


def _displayed_citations(analysis: ContractAnalysis) -> int:
    """How many citations the risk map shows: one per cited finding and white."""
    if analysis.risk_map is None:
        return 0
    shown = [*analysis.risk_map.clause_findings, *analysis.risk_map.absence_findings]
    return sum(1 for finding in shown if finding.citation is not None)


def _reference_spans(case: Mode2EvalCase) -> list[LabeledSpan]:
    """The reference clause boundaries as labeled spans."""
    return [
        LabeledSpan(id=clause.clause_id, start=clause.start, end=clause.end)
        for clause in case.clauses
    ]


def _predicted_spans(findings: Sequence[ClauseFinding]) -> list[LabeledSpan]:
    """The risk map's anchored clause spans as labeled spans."""
    return [
        LabeledSpan(id=finding.clause_id, start=finding.start, end=finding.end)
        for finding in findings
    ]


def _clause_predictions(
    case: Mode2EvalCase,
    matches: Sequence[SpanMatch],
    class_by_finding: dict[str, str],
) -> list[ClausePrediction]:
    """Pair each reference clause with the class its matched finding assigned it."""
    predictions = []
    for clause, match in zip(case.clauses, matches, strict=True):
        predicted = (
            class_by_finding.get(match.predicted_id)
            if match.matched and match.predicted_id
            else None
        )
        predictions.append(
            ClausePrediction(
                clause_id=clause.clause_id,
                expected_level=clause.expected_level.value,
                predicted_class=predicted,
                iou=match.iou,
                matched=match.matched,
            )
        )
    return predictions


def _segmentation_layer(
    case: Mode2EvalCase, matches: Sequence[SpanMatch], threshold: float
) -> SegmentationLayer:
    """Assemble the segmentation layer's score from the span matches."""
    delimited = sum(1 for match in matches if match.matched)
    return SegmentationLayer(
        threshold=threshold,
        reference_clauses=len(case.clauses),
        delimited=delimited,
        delimited_rate=_ratio(delimited, len(case.clauses)),
        matches=list(matches),
    )


def _confusion(matched: Sequence[ClausePrediction]) -> dict[str, dict[str, int]]:
    """The per-level confusion matrix over correctly delimited clauses."""
    matrix = {gold: {predicted: 0 for predicted in PREDICTED_CLASSES} for gold in GOLD_LEVELS}
    for prediction in matched:
        if prediction.predicted_class is None:
            raise ValueError(f"matched clause '{prediction.clause_id}' has no predicted class")
        matrix[prediction.expected_level][prediction.predicted_class] += 1
    return matrix


def _problematic(matched: Sequence[ClausePrediction]) -> list[ClausePrediction]:
    """The correctly delimited clauses whose reference level is red or orange."""
    return [p for p in matched if p.expected_level in PROBLEMATIC_CLASSES]


def _problematic_total(matched: Sequence[ClausePrediction]) -> int:
    """How many correctly delimited clauses are really problematic."""
    return len(_problematic(matched))


def _problematic_detected(matched: Sequence[ClausePrediction]) -> int:
    """Problematic clauses the system also placed at a problematic level."""
    return sum(1 for p in _problematic(matched) if p.predicted_class in PROBLEMATIC_CLASSES)


def _false_tranquility(matched: Sequence[ClausePrediction]) -> int:
    """Problematic clauses the system passed off as reassuring (green or silence)."""
    return sum(1 for p in _problematic(matched) if p.predicted_class in REASSURING_CLASSES)


def _flagged(matched: Sequence[ClausePrediction]) -> int:
    """Correctly delimited clauses the system flagged as problematic."""
    return sum(1 for p in matched if p.predicted_class in PROBLEMATIC_CLASSES)


def _flagged_true_positive(matched: Sequence[ClausePrediction]) -> int:
    """Flagged clauses that were really problematic."""
    return sum(
        1
        for p in matched
        if p.predicted_class in PROBLEMATIC_CLASSES and p.expected_level in PROBLEMATIC_CLASSES
    )


def _abstained(matched: Sequence[ClausePrediction]) -> int:
    """Correctly delimited clauses the system placed in an abstention class."""
    return sum(1 for p in matched if p.predicted_class in ABSTENTION_CLASSES)


def _problematic_all(predictions: Sequence[ClausePrediction]) -> list[ClausePrediction]:
    """Every reference clause whose reference level is red or orange, delimited or not."""
    return [p for p in predictions if p.expected_level in PROBLEMATIC_CLASSES]


def _problematic_total_e2e(predictions: Sequence[ClausePrediction]) -> int:
    """How many reference clauses are really problematic, over the whole contract."""
    return len(_problematic_all(predictions))


def _problematic_detected_e2e(predictions: Sequence[ClausePrediction]) -> int:
    """Problematic clauses the system delimited and also placed at a problematic level."""
    return sum(
        1
        for p in _problematic_all(predictions)
        if p.matched and p.predicted_class in PROBLEMATIC_CLASSES
    )


def _false_tranquility_e2e(predictions: Sequence[ClausePrediction]) -> int:
    """Problematic clauses the system delimited and passed off as reassuring.

    A problematic clause the segmentation layer never delimited is not counted
    here -- it was never shown to the user as reassuring or as anything else, so
    it belongs in :func:`_not_reported_problematic` instead.
    """
    return sum(
        1
        for p in _problematic_all(predictions)
        if p.matched and p.predicted_class in REASSURING_CLASSES
    )


def _not_reported_problematic(predictions: Sequence[ClausePrediction]) -> int:
    """Problematic clauses the segmentation layer never delimited at all."""
    return sum(1 for p in _problematic_all(predictions) if not p.matched)


def _absence_recall(case: Mode2EvalCase, analysis: ContractAnalysis) -> AbsenceRecall:
    """Score the produced absence whites against the deliberately omitted rights."""
    expected = {absence.item_id for absence in case.expected_absences}
    produced = {
        finding.item_id
        for finding in (analysis.risk_map.absence_findings if analysis.risk_map else [])
    }
    detected = expected & produced
    return AbsenceRecall(
        expected=sorted(expected),
        detected=sorted(detected),
        missed=sorted(expected - produced),
        spurious=sorted(produced - expected),
        recall=_ratio(len(detected), len(expected)),
        precision=_ratio(len(detected), len(produced)),
    )


def _pool_confusion(results: Iterable[Mode2CaseResult]) -> dict[str, dict[str, int]]:
    """Sum the per-case confusion matrices into the run matrix."""
    matrix = {gold: {predicted: 0 for predicted in PREDICTED_CLASSES} for gold in GOLD_LEVELS}
    for result in results:
        for gold, row in result.confusion.items():
            for predicted, count in row.items():
                matrix[gold][predicted] += count
    return matrix


def _count_outcomes(results: Iterable[Mode2CaseResult]) -> dict[str, int]:
    """Count how many cases ended in each outcome label."""
    counts: dict[str, int] = {}
    for result in results:
        counts[result.outcome] = counts.get(result.outcome, 0) + 1
    return counts


def _outcome_match_rate(results: Iterable[Mode2CaseResult]) -> float | None:
    """The share of cases that reached the outcome their case declared."""
    matches = [
        result.outcome_as_expected for result in results if result.outcome_as_expected is not None
    ]
    return (sum(matches) / len(matches)) if matches else None


def _outcome_label(analysis: ContractAnalysis) -> str:
    """The analysis outcome, naming the rejection reason when it stopped at a gate."""
    if analysis.rejection is not None:
        return f"{analysis.outcome.value}:{analysis.rejection.reason.value}"
    return analysis.outcome.value


def _ratio(numerator: int, denominator: int) -> float | None:
    """A fraction, or ``None`` when the denominator is zero so it is never faked."""
    if denominator == 0:
        return None
    return numerator / denominator
