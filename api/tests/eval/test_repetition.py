"""Tests for the repetition bands: the median, the observed range and the movement."""

import pytest

from lexme.eval.repetition import (
    MetricDirection,
    Movement,
    Span,
    classify_movement,
    observe,
    summarize_repetitions,
)


def _summary(**runs: list[float | None]):
    """A summary over runs given as ``metric=[value per repetition]``."""
    repetitions = max(len(values) for values in runs.values())
    return summarize_repetitions(
        [{metric: values[index] for metric, values in runs.items()} for index in range(repetitions)]
    )


def test_a_band_publishes_the_median_and_the_observed_range() -> None:
    summary = _summary(mean_recall=[0.92, 0.87, 0.90])

    band = summary.band("mean_recall")

    assert band is not None
    assert band.median == 0.90
    assert (band.low, band.high) == (0.87, 0.92)


def test_a_band_keeps_the_value_of_every_repetition_in_run_order() -> None:
    summary = _summary(mean_recall=[0.92, 0.87, 0.90])

    band = summary.band("mean_recall")

    assert band is not None
    assert band.values == [0.92, 0.87, 0.90]


def test_a_metric_no_repetition_measured_gets_no_band() -> None:
    summary = _summary(expected_abstention_recall=[None, None])

    assert summary.band("expected_abstention_recall") is None


def test_a_metric_only_some_repetitions_measured_bands_over_those() -> None:
    summary = _summary(mean_recall=[None, 0.5, 0.7])

    band = summary.band("mean_recall")

    assert band is not None
    assert band.values == [None, 0.5, 0.7]
    assert (band.low, band.high) == (0.5, 0.7)


def test_the_summary_counts_the_repetitions_it_folded() -> None:
    summary = _summary(mean_recall=[0.9, 0.9, 0.9])

    assert summary.repetitions == 3
    assert summary.measures_noise is True


def test_a_single_repetition_measures_no_noise() -> None:
    summary = _summary(mean_recall=[0.9])

    assert summary.measures_noise is False


def test_summarizing_no_run_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_repetitions([])


def test_a_drop_beyond_both_bands_is_a_regression() -> None:
    movement = classify_movement(
        observe(0.92, Span(low=0.90, high=0.94)),
        observe(0.80, Span(low=0.78, high=0.82)),
        MetricDirection.HIGHER_IS_BETTER,
    )

    assert movement is Movement.REGRESSION


def test_a_drop_inside_the_measured_band_is_variance_not_a_regression() -> None:
    movement = classify_movement(
        observe(0.92, Span(low=0.85, high=0.95)),
        observe(0.87, Span(low=0.83, high=0.92)),
        MetricDirection.HIGHER_IS_BETTER,
    )

    assert movement is Movement.VARIANCE


def test_a_rise_beyond_both_bands_is_an_improvement() -> None:
    movement = classify_movement(
        observe(0.80, Span(low=0.78, high=0.82)),
        observe(0.92, Span(low=0.90, high=0.94)),
        MetricDirection.HIGHER_IS_BETTER,
    )

    assert movement is Movement.IMPROVEMENT


def test_on_a_metric_where_less_is_better_a_rise_is_the_regression() -> None:
    movement = classify_movement(
        observe(0.01, Span(low=0.00, high=0.02)),
        observe(0.20, Span(low=0.18, high=0.22)),
        MetricDirection.LOWER_IS_BETTER,
    )

    assert movement is Movement.REGRESSION


def test_a_metric_with_no_better_direction_only_reports_that_it_moved() -> None:
    movement = classify_movement(observe(0.52, None), observe(0.20, None), MetricDirection.NEUTRAL)

    assert movement is Movement.SHIFT


def test_without_a_band_an_unmoved_metric_is_still_variance() -> None:
    movement = classify_movement(
        observe(0.92, None), observe(0.92, None), MetricDirection.HIGHER_IS_BETTER
    )

    assert movement is Movement.VARIANCE


def test_a_metric_one_side_never_measured_is_not_classified() -> None:
    movement = classify_movement(
        observe(None, None), observe(0.92, None), MetricDirection.HIGHER_IS_BETTER
    )

    assert movement is None


def test_a_move_outside_both_bands_but_inside_the_drift_allowance_is_named_drift() -> None:
    base = observe(0.95, Span(low=0.94, high=0.96))
    run = observe(0.90, Span(low=0.89, high=0.91))

    movement = classify_movement(base, run, MetricDirection.HIGHER_IS_BETTER, allowance=0.05)

    assert movement is Movement.DRIFT


def test_a_move_beyond_the_drift_allowance_is_still_a_regression() -> None:
    base = observe(0.95, Span(low=0.94, high=0.96))
    run = observe(0.60, Span(low=0.59, high=0.61))

    movement = classify_movement(base, run, MetricDirection.HIGHER_IS_BETTER, allowance=0.05)

    assert movement is Movement.REGRESSION


def test_a_move_the_bands_already_cover_stays_variance_under_an_allowance() -> None:
    base = observe(0.90, Span(low=0.87, high=0.92))
    run = observe(0.89, Span(low=0.88, high=0.91))

    movement = classify_movement(base, run, MetricDirection.HIGHER_IS_BETTER, allowance=0.05)

    assert movement is Movement.VARIANCE


def test_a_zero_width_band_still_absorbs_a_move_the_allowance_covers() -> None:
    base = observe(0.9474, Span(low=0.9474, high=0.9474))
    run = observe(0.8684, Span(low=0.8684, high=0.8684))

    movement = classify_movement(base, run, MetricDirection.HIGHER_IS_BETTER, allowance=0.08)

    assert movement is Movement.DRIFT


def test_without_an_allowance_the_classification_is_unchanged() -> None:
    base = observe(0.95, Span(low=0.94, high=0.96))
    run = observe(0.90, Span(low=0.89, high=0.91))

    assert classify_movement(base, run, MetricDirection.HIGHER_IS_BETTER) is Movement.REGRESSION


def test_a_neutral_metric_drifting_is_named_drift_rather_than_a_shift() -> None:
    base = observe(0.52, Span(low=0.52, high=0.52))
    run = observe(0.38, Span(low=0.38, high=0.38))

    movement = classify_movement(base, run, MetricDirection.NEUTRAL, allowance=0.15)

    assert movement is Movement.DRIFT
