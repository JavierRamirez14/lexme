"""Mode 2 metrics: the false-tranquility rate, recall, precision and the whites.

These build a reference case and a canned risk map so the asymmetric-error numbers
are exercised without a pipeline. The spans are chosen so a finding at a clause's
own range matches it exactly, isolating classification from segmentation.
"""

from lexme.eval.mode2.metrics import (
    aggregate_mode2,
    build_mode2_case_result,
    predicted_class,
)
from lexme.mode2.models import RejectionReason
from lexme.mode2.risk import CoverageStatus, RiskLevel
from tests.eval.mode2.conftest import (
    absence,
    analysis_of,
    case_of,
    clause_truth,
    covered,
    evaluated,
    rejected,
)

RED = RiskLevel.ILEGAL
ORANGE = RiskLevel.PEOR_QUE_DEFAULT
GREEN = RiskLevel.CORRECTO


def _spans(count: int) -> list[tuple[int, int]]:
    """``count`` adjacent 100-char spans, so each reference clause has a slot."""
    return [(i * 100, i * 100 + 100) for i in range(count)]


def test_predicted_class_reads_the_level_of_an_evaluated_finding() -> None:
    assert predicted_class(evaluated("c1", RED)) == "ilegal"


def test_predicted_class_reads_the_coverage_of_an_abstained_finding() -> None:
    assert predicted_class(covered("c1", CoverageStatus.NO_CONCLUYENTE)) == "no_concluyente"
    assert predicted_class(covered("c1", CoverageStatus.INFORMATIVA)) == "informativa"


def test_a_detected_red_clause_counts_toward_recall_not_false_tranquility() -> None:
    case = case_of(clause_truth("c1", RED, start=0, end=100))
    analysis = analysis_of(evaluated("c1", RED, start=0, end=100))

    result = build_mode2_case_result(case, analysis)

    assert result.problematic_total == 1
    assert result.problematic_detected == 1
    assert result.recall_problematic == 1.0
    assert result.false_tranquility_events == 0
    assert result.false_tranquility_rate == 0.0


def test_a_red_clause_called_correct_is_a_false_tranquility_event() -> None:
    case = case_of(clause_truth("c1", RED, start=0, end=100))
    analysis = analysis_of(evaluated("c1", GREEN, start=0, end=100))

    result = build_mode2_case_result(case, analysis)

    assert result.recall_problematic == 0.0
    assert result.false_tranquility_events == 1
    assert result.false_tranquility_rate == 1.0


def test_a_red_clause_left_silent_as_informative_is_a_false_tranquility_event() -> None:
    case = case_of(clause_truth("c1", ORANGE, start=0, end=100))
    analysis = analysis_of(covered("c1", CoverageStatus.INFORMATIVA, start=0, end=100))

    result = build_mode2_case_result(case, analysis)

    assert result.false_tranquility_events == 1
    assert result.false_tranquility_rate == 1.0


def test_a_red_clause_called_inconclusive_misses_recall_but_is_not_false_tranquility() -> None:
    case = case_of(clause_truth("c1", RED, start=0, end=100))
    analysis = analysis_of(covered("c1", CoverageStatus.NO_CONCLUYENTE, start=0, end=100))

    result = build_mode2_case_result(case, analysis)

    assert result.recall_problematic == 0.0
    assert result.false_tranquility_events == 0
    assert result.false_tranquility_rate == 0.0
    assert result.abstention_clauses == 1


def test_precision_is_published_with_the_abstention_rate() -> None:
    spans = _spans(3)
    case = case_of(
        clause_truth("c1", RED, start=spans[0][0], end=spans[0][1]),
        clause_truth("c2", GREEN, start=spans[1][0], end=spans[1][1]),
        clause_truth("c3", GREEN, start=spans[2][0], end=spans[2][1]),
    )
    analysis = analysis_of(
        evaluated("c1", RED, start=spans[0][0], end=spans[0][1]),
        evaluated("c2", RED, start=spans[1][0], end=spans[1][1]),
        covered("c3", CoverageStatus.NO_CONCLUYENTE, start=spans[2][0], end=spans[2][1]),
    )

    result = build_mode2_case_result(case, analysis)

    assert result.flagged_problematic == 2
    assert result.flagged_true_positive == 1
    assert result.precision_problematic == 0.5
    assert result.abstention_clauses == 1
    assert result.abstention_rate == 1 / 3


def test_a_dropped_clause_lowers_segmentation_and_is_excluded_from_level_metrics() -> None:
    spans = _spans(2)
    case = case_of(
        clause_truth("c1", RED, start=spans[0][0], end=spans[0][1]),
        clause_truth("c2", ORANGE, start=spans[1][0], end=spans[1][1]),
    )
    # Only the first clause is delimited; the second finding overlaps neither well.
    analysis = analysis_of(evaluated("c1", RED, start=spans[0][0], end=spans[0][1]))

    result = build_mode2_case_result(case, analysis)

    assert result.segmentation.delimited == 1
    assert result.segmentation.reference_clauses == 2
    assert result.problematic_total == 1  # c2 is a segmentation miss, not a level miss
    assert result.problematic_detected == 1
    unmatched = [p for p in result.clause_predictions if not p.matched]
    assert [p.clause_id for p in unmatched] == ["c2"]


def test_the_confusion_matrix_records_gold_against_predicted_over_matched_clauses() -> None:
    spans = _spans(2)
    case = case_of(
        clause_truth("c1", RED, start=spans[0][0], end=spans[0][1]),
        clause_truth("c2", GREEN, start=spans[1][0], end=spans[1][1]),
    )
    analysis = analysis_of(
        evaluated("c1", GREEN, start=spans[0][0], end=spans[0][1]),
        evaluated("c2", GREEN, start=spans[1][0], end=spans[1][1]),
    )

    result = build_mode2_case_result(case, analysis)

    assert result.confusion["ilegal"]["correcto"] == 1
    assert result.confusion["correcto"]["correcto"] == 1


def test_absence_recall_and_precision_score_the_omitted_rights() -> None:
    case = case_of(clause_truth("c1", GREEN), absences=("CHK-02", "CHK-05"))
    analysis = analysis_of(evaluated("c1", GREEN), absences=(absence("CHK-02"), absence("CHK-99")))

    result = build_mode2_case_result(case, analysis)

    assert result.absences.detected == ["CHK-02"]
    assert result.absences.missed == ["CHK-05"]
    assert result.absences.spurious == ["CHK-99"]
    assert result.absences.recall == 0.5
    assert result.absences.precision == 0.5


def test_the_displayed_citations_count_covers_findings_and_whites() -> None:
    case = case_of(clause_truth("c1", RED), absences=("CHK-02",))
    analysis = analysis_of(
        evaluated("c1", RED, citation=("a9", "cinco años")),
        evaluated("c2", GREEN),
        absences=(absence("CHK-02", citation=("a36", "una mensualidad")),),
    )

    result = build_mode2_case_result(case, analysis)

    assert result.displayed_citations == 2
    assert aggregate_mode2([result, result]).displayed_citations == 4


def test_a_rejected_contract_shows_no_citations() -> None:
    case = case_of(clause_truth("c1", RED, start=0, end=100))

    result = build_mode2_case_result(case, rejected(RejectionReason.OUT_OF_SCOPE_USE))

    assert result.displayed_citations == 0


def test_a_rejected_contract_delimits_nothing_and_records_its_reason() -> None:
    case = case_of(clause_truth("c1", RED, start=0, end=100))
    analysis = rejected(RejectionReason.OUT_OF_SCOPE_USE)

    result = build_mode2_case_result(case, analysis)

    assert result.outcome == "fuera_de_ambito:uso_fuera_de_ambito"
    assert result.segmentation.delimited == 0
    assert result.problematic_total == 0  # nothing was delimited to judge
    assert all(not p.matched for p in result.clause_predictions)


def test_a_case_that_reaches_its_declared_outcome_matches() -> None:
    case = case_of(clause_truth("c1", RED, start=0, end=100), expected_outcome="analizado")
    analysis = analysis_of(evaluated("c1", RED, start=0, end=100))

    result = build_mode2_case_result(case, analysis)

    assert result.outcome == "analizado"
    assert result.outcome_as_expected is True


def test_a_rejected_case_not_declared_as_rejected_misses_the_outcome() -> None:
    case = case_of(clause_truth("c1", RED, start=0, end=100), expected_outcome="analizado")
    analysis = rejected(RejectionReason.OUT_OF_SCOPE_USE)

    result = build_mode2_case_result(case, analysis)

    assert result.outcome == "fuera_de_ambito:uso_fuera_de_ambito"
    assert result.outcome_as_expected is False


def test_a_case_declared_as_rejected_that_is_rejected_matches() -> None:
    case = case_of(
        clause_truth("c1", RED, start=0, end=100),
        expected_outcome="fuera_de_ambito:uso_fuera_de_ambito",
    )
    analysis = rejected(RejectionReason.OUT_OF_SCOPE_USE)

    result = build_mode2_case_result(case, analysis)

    assert result.outcome_as_expected is True


def test_outcome_match_rate_pools_across_cases() -> None:
    case_a = case_of(clause_truth("c1", RED, start=0, end=100), case_id="a")
    analysis_a = analysis_of(evaluated("c1", RED, start=0, end=100))
    case_b = case_of(
        clause_truth("c1", RED, start=0, end=100), case_id="b", expected_outcome="analizado"
    )
    analysis_b = rejected(RejectionReason.OUT_OF_SCOPE_USE)

    results = [
        build_mode2_case_result(case_a, analysis_a),
        build_mode2_case_result(case_b, analysis_b),
    ]
    metrics = aggregate_mode2(results)

    assert metrics.outcome_match_rate == 0.5


def test_a_problematic_clause_never_delimited_is_not_reported_end_to_end() -> None:
    spans = _spans(2)
    case = case_of(
        clause_truth("c1", RED, start=spans[0][0], end=spans[0][1]),
        clause_truth("c2", ORANGE, start=spans[1][0], end=spans[1][1]),
    )
    # c1 is rejected outright: its analysis carries no risk map, so nothing is delimited.
    analysis = rejected(RejectionReason.OUT_OF_SCOPE_USE)

    result = build_mode2_case_result(case, analysis)

    assert result.problematic_total == 0  # the conditioned number sees nothing
    assert result.recall_problematic is None
    assert result.problematic_total_e2e == 2
    assert result.problematic_detected_e2e == 0
    assert result.recall_problematic_e2e == 0.0
    assert result.not_reported_problematic == 2
    assert result.false_tranquility_events_e2e == 0


def test_an_end_to_end_detection_counts_toward_recall_not_the_conditioned_denominator() -> None:
    spans = _spans(2)
    case = case_of(
        clause_truth("c1", RED, start=spans[0][0], end=spans[0][1]),
        clause_truth("c2", ORANGE, start=spans[1][0], end=spans[1][1]),
    )
    # Only c1 is delimited and correctly flagged; c2 never surfaces at all.
    analysis = analysis_of(evaluated("c1", RED, start=spans[0][0], end=spans[0][1]))

    result = build_mode2_case_result(case, analysis)

    assert result.problematic_total == 1  # conditioned: only c1 was delimited
    assert result.recall_problematic == 1.0
    assert result.problematic_total_e2e == 2  # end to end: c2 counts too
    assert result.problematic_detected_e2e == 1
    assert result.recall_problematic_e2e == 0.5
    assert result.not_reported_problematic == 1


def test_a_delimited_problematic_clause_called_reassuring_is_false_tranquility_end_to_end() -> (
    None
):
    case = case_of(clause_truth("c1", RED, start=0, end=100))
    analysis = analysis_of(evaluated("c1", GREEN, start=0, end=100))

    result = build_mode2_case_result(case, analysis)

    assert result.false_tranquility_events_e2e == 1
    assert result.false_tranquility_rate_e2e == 1.0
    assert result.not_reported_problematic == 0


def test_the_aggregate_pools_e2e_counts_across_cases() -> None:
    case_a = case_of(clause_truth("c1", RED, start=0, end=100), case_id="a")
    analysis_a = rejected(RejectionReason.OUT_OF_SCOPE_USE)
    case_b = case_of(clause_truth("c1", ORANGE, start=0, end=100), case_id="b")
    analysis_b = analysis_of(evaluated("c1", ORANGE, start=0, end=100))

    results = [
        build_mode2_case_result(case_a, analysis_a),
        build_mode2_case_result(case_b, analysis_b),
    ]
    metrics = aggregate_mode2(results)

    assert metrics.problematic_total_e2e == 2
    assert metrics.problematic_detected_e2e == 1
    assert metrics.recall_problematic_e2e == 0.5
    assert metrics.not_reported_problematic == 1


def test_the_aggregate_pools_counts_across_cases() -> None:
    case_a = case_of(clause_truth("c1", RED, start=0, end=100), case_id="a")
    analysis_a = analysis_of(evaluated("c1", GREEN, start=0, end=100))
    case_b = case_of(clause_truth("c1", ORANGE, start=0, end=100), case_id="b")
    analysis_b = analysis_of(evaluated("c1", ORANGE, start=0, end=100))

    results = [
        build_mode2_case_result(case_a, analysis_a),
        build_mode2_case_result(case_b, analysis_b),
    ]
    metrics = aggregate_mode2(results)

    assert metrics.cases == 2
    assert metrics.problematic_total == 2
    assert metrics.problematic_detected == 1
    assert metrics.recall_problematic == 0.5
    assert metrics.false_tranquility_events == 1
    assert metrics.false_tranquility_rate == 0.5
