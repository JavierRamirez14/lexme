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
