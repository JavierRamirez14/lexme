"""Compare a run against a baseline, classifying each metric's move against the noise.

A run's numbers mean nothing in isolation; a regression is a move away from a
known-good baseline. But two runs of the same configuration do not return the same
numbers either, so a raw delta reads a regression and the noise of two consecutive
runs exactly alike. This reports the per-metric delta together with what the move
amounts to once both runs' repetition bands are taken into account -- a regression,
an improvement, or variance -- and, first of all, whether the two runs even share a
configuration fingerprint, because a delta across configurations is not a regression,
it is a different experiment.
"""

from collections.abc import Mapping

from pydantic import BaseModel

from lexme.eval.artifact import RunArtifact
from lexme.eval.metrics import METRIC_DIRECTIONS, scalar_metrics
from lexme.eval.mode2.artifact import Mode2RunArtifact
from lexme.eval.mode2.metrics import MODE2_METRIC_DIRECTIONS, scalar_metrics_mode2
from lexme.eval.repetition import (
    MetricDirection,
    MetricObservation,
    Movement,
    RepetitionSummary,
    Span,
    classify_movement,
    observe,
)


class MetricDelta(BaseModel):
    """One metric's baseline value, run value, difference and what the move means.

    ``base`` or ``run`` is ``None`` when a side did not measure the metric (an
    empty recall, say); ``delta`` is ``None`` unless both sides are present, and so
    is ``movement``. The published value is the median across repetitions whenever
    the side ran more than one, and ``base_span``/``run_span`` are the range it was
    observed across -- absent for a run that measured no band.
    """

    metric: str
    base: float | None
    run: float | None
    delta: float | None
    movement: Movement | None = None
    base_span: Span | None = None
    run_span: Span | None = None


class Comparison(BaseModel):
    """The result of comparing a run against a baseline.

    ``fingerprint_changed`` is the first thing to read: when true, the deltas
    describe two different configurations and are not regressions. ``noise_measured``
    is the second: when false, at least one side ran a single repetition, so no band
    bounds the noise and every non-zero move is reported as if it were a result.
    """

    fingerprint_changed: bool
    base_fingerprint: str
    run_fingerprint: str
    noise_measured: bool
    deltas: list[MetricDelta]


def compare(base: RunArtifact, run: RunArtifact) -> Comparison:
    """Compare ``run`` against ``base``, returning the per-metric deltas.

    Covers the scalar quality metrics and every citation verdict count. The
    fingerprint comparison frames the rest: identical fingerprints make the deltas
    a regression signal, different ones make them an experiment comparison.
    """
    return _compare_scalars(
        base_fingerprint=base.fingerprint.fingerprint,
        run_fingerprint=run.fingerprint.fingerprint,
        base_values=scalar_metrics(base.metrics),
        run_values=scalar_metrics(run.metrics),
        base_repetitions=base.repetitions,
        run_repetitions=run.repetitions,
        directions=METRIC_DIRECTIONS,
    )


def compare_mode2(base: Mode2RunArtifact, run: Mode2RunArtifact) -> Comparison:
    """Compare a Mode 2 ``run`` against ``base``, returning the per-metric deltas.

    Covers both the conditioned-on-segmentation numbers and their end-to-end
    counterparts, so a schema-change comparison shows the same denominator shift the
    fingerprint flags. As with Mode 1, the fingerprint comparison frames the rest.
    """
    return _compare_scalars(
        base_fingerprint=base.fingerprint.fingerprint,
        run_fingerprint=run.fingerprint.fingerprint,
        base_values=scalar_metrics_mode2(base.metrics),
        run_values=scalar_metrics_mode2(run.metrics),
        base_repetitions=base.repetitions,
        run_repetitions=run.repetitions,
        directions=MODE2_METRIC_DIRECTIONS,
    )


def _compare_scalars(
    *,
    base_fingerprint: str,
    run_fingerprint: str,
    base_values: Mapping[str, float | None],
    run_values: Mapping[str, float | None],
    base_repetitions: RepetitionSummary | None,
    run_repetitions: RepetitionSummary | None,
    directions: Mapping[str, MetricDirection],
) -> Comparison:
    """Build the comparison over two runs' scalar projections and their bands."""
    metrics = list(base_values) + [metric for metric in run_values if metric not in base_values]
    return Comparison(
        fingerprint_changed=base_fingerprint != run_fingerprint,
        base_fingerprint=base_fingerprint,
        run_fingerprint=run_fingerprint,
        noise_measured=_measures_noise(base_repetitions) and _measures_noise(run_repetitions),
        deltas=[
            _delta(
                metric,
                _observation(metric, base_values, base_repetitions),
                _observation(metric, run_values, run_repetitions),
                directions.get(metric, MetricDirection.NEUTRAL),
            )
            for metric in metrics
        ],
    )


def _observation(
    metric: str, values: Mapping[str, float | None], repetitions: RepetitionSummary | None
) -> MetricObservation:
    """One side's reading of a metric: its band's median and span, else the bare value."""
    band = repetitions.band(metric) if repetitions is not None else None
    if band is None:
        return observe(values.get(metric), None)
    return observe(band.median, band.span if repetitions.measures_noise else None)


def _delta(
    metric: str,
    base: MetricObservation,
    run: MetricObservation,
    direction: MetricDirection,
) -> MetricDelta:
    """Build a delta, leaving it ``None`` unless both sides measured the metric."""
    delta = run.value - base.value if base.value is not None and run.value is not None else None
    return MetricDelta(
        metric=metric,
        base=base.value,
        run=run.value,
        delta=delta,
        movement=classify_movement(base, run, direction),
        base_span=base.span,
        run_span=run.span,
    )


def _measures_noise(repetitions: RepetitionSummary | None) -> bool:
    """Whether a side ran enough repetitions for its bands to bound the noise."""
    return repetitions is not None and repetitions.measures_noise
