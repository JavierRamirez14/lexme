"""Repetition bands: the noise a metric shows when nothing changes.

Two runs of the same suite under the same fingerprint do not return the same
numbers -- the model is sampled, not evaluated -- so a single value is not a
quantity a later run can be compared against. Running the suite N times turns each
metric into a band: the median it centres on and the range it was observed across.
A movement that stays inside the band is variance, not a result, and this is where
that judgement is made, once, for both modes.
"""

from collections.abc import Mapping, Sequence
from enum import StrEnum
from statistics import median

from pydantic import BaseModel

MIN_REPETITIONS_MEASURING_NOISE = 2


class MetricDirection(StrEnum):
    """Which way a metric has to move to be good news.

    ``NEUTRAL`` is for the metrics that describe a run rather than grade it -- the
    case count, the disambiguation rate, the citation verdict counts -- where a move
    is a fact to explain, not a regression or an improvement.
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    NEUTRAL = "neutral"


class Movement(StrEnum):
    """What a metric's move between two runs amounts to.

    ``VARIANCE`` is the move the measured noise already accounts for; it is not a
    result and must never be reported as one. ``DRIFT`` is the move one session's
    repetitions could not see but the archive of same-fingerprint runs has: outside
    both bands, inside the declared between-session allowance, and no more a result
    than ``VARIANCE`` is. ``SHIFT`` is a move outside both on a metric with no
    better direction.
    """

    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    VARIANCE = "variance"
    DRIFT = "drift"
    SHIFT = "shift"


class Span(BaseModel):
    """The lowest and highest value a metric was observed at across repetitions."""

    low: float
    high: float


class MetricBand(BaseModel):
    """One metric across the repetitions of a single run.

    ``values`` holds every repetition's value in run order, ``None`` where a
    repetition did not measure the metric at all; ``median`` and the span are taken
    over the repetitions that did.
    """

    metric: str
    values: list[float | None]
    median: float
    low: float
    high: float

    @property
    def span(self) -> Span:
        """The observed range as the span a comparison classifies against."""
        return Span(low=self.low, high=self.high)


class RepetitionSummary(BaseModel):
    """Every metric's band over the repetitions one run performed."""

    repetitions: int
    bands: list[MetricBand]

    @property
    def measures_noise(self) -> bool:
        """Whether enough repetitions ran for the bands to describe noise at all."""
        return self.repetitions >= MIN_REPETITIONS_MEASURING_NOISE

    def band(self, metric: str) -> MetricBand | None:
        """The band for ``metric``, or ``None`` when no repetition measured it."""
        return next((band for band in self.bands if band.metric == metric), None)


class MetricObservation(BaseModel):
    """One side's reading of a metric: the value it publishes and the band around it."""

    value: float | None
    span: Span | None


def summarize_repetitions(runs: Sequence[Mapping[str, float | None]]) -> RepetitionSummary:
    """Fold each repetition's metrics into one band per metric.

    ``runs`` is one mapping of metric name to value per repetition, in run order.
    A metric no repetition measured gets no band rather than a fabricated zero.
    Raises :class:`ValueError` when handed no repetition at all.
    """
    if not runs:
        raise ValueError("a repetition summary needs at least one run")
    return RepetitionSummary(
        repetitions=len(runs),
        bands=[band for metric in _metric_names(runs) if (band := _band(metric, runs)) is not None],
    )


def observe(value: float | None, span: Span | None) -> MetricObservation:
    """Read one side of a comparison: its published value and the band it moves in."""
    return MetricObservation(value=value, span=span)


def classify_movement(
    base: MetricObservation,
    run: MetricObservation,
    direction: MetricDirection,
    allowance: float = 0.0,
) -> Movement | None:
    """Say what the move from ``base`` to ``run`` is, against the noise both measured.

    Overlapping bands mean the two runs are indistinguishable at the noise they
    showed, so the move is ``VARIANCE``. A side without a band is read as a
    zero-width one at its own value, which is what a single repetition honestly
    measured: everything but an exact tie then falls outside. ``allowance`` is the
    between-session drift declared for the metric; a gap the two bands leave open
    but the allowance covers is ``DRIFT`` rather than a result, because two runs
    are never the same session. Returns ``None`` when either side never measured
    the metric.
    """
    if base.value is None or run.value is None:
        return None
    if _overlap(_span_of(base), _span_of(run)):
        return Movement.VARIANCE
    if _overlap(_span_of(base), _span_of(run), allowance):
        return Movement.DRIFT
    if direction is MetricDirection.NEUTRAL:
        return Movement.SHIFT
    improved = (
        run.value > base.value
        if direction is MetricDirection.HIGHER_IS_BETTER
        else run.value < base.value
    )
    return Movement.IMPROVEMENT if improved else Movement.REGRESSION


def _metric_names(runs: Sequence[Mapping[str, float | None]]) -> list[str]:
    """Every metric name any repetition reported, first-seen order."""
    names: dict[str, None] = {}
    for run in runs:
        for metric in run:
            names.setdefault(metric, None)
    return list(names)


def _band(metric: str, runs: Sequence[Mapping[str, float | None]]) -> MetricBand | None:
    """The band for one metric, or ``None`` when no repetition measured it."""
    values = [run.get(metric) for run in runs]
    measured = [value for value in values if value is not None]
    if not measured:
        return None
    return MetricBand(
        metric=metric,
        values=values,
        median=median(measured),
        low=min(measured),
        high=max(measured),
    )


def _span_of(observation: MetricObservation) -> Span:
    """The observation's band, or the zero-width span a single value amounts to."""
    if observation.span is not None:
        return observation.span
    assert observation.value is not None  # guarded by the caller
    return Span(low=observation.value, high=observation.value)


def _overlap(first: Span, second: Span, allowance: float = 0.0) -> bool:
    """Whether two observed ranges share a value once each is widened by ``allowance``."""
    return first.low - allowance <= second.high and second.low - allowance <= first.high
