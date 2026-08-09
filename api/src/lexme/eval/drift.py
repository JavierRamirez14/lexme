"""Between-session drift: the noise a metric shows that one session cannot see.

A run's repetition band is measured inside a single session, so the repetitions
share whatever the provider is serving that hour. That makes the band a floor on
the noise rather than a ceiling: a metric whose three repetitions land on the same
value has a zero-width band and still moves between one session and the next, and a
comparison reading only those bands calls the move a regression.

This measures the part the band cannot. The archive of runs already versioned holds
groups of runs that share a configuration fingerprint but not a session; what those
runs disagree by is drift, because nothing in the configuration changed between
them. It is recorded per metric as a declared allowance a comparison widens the
bands by, next to the widest within-session span the archive ever measured, so the
record itself carries the evidence for how far the two diverge.

Drift is measured as the widest *gap between two runs' observed ranges*, never
between their medians. A median is one point drawn from a range the run already
showed, so a median-to-median distance carries the within-session noise a second
time -- and it would be handed to a rule that then adds both bands back. Measuring
the same quantity the comparison classifies keeps the allowance and its use in the
same units, and stops the noise being counted twice.
"""

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from lexme.eval.repetition import Span

DRIFT_FILENAME_TEMPLATE = "drift-{suite}.json"
MIN_SESSIONS_MEASURING_DRIFT = 2


class DriftError(ValueError):
    """Raised when a drift record is malformed or cannot be measured."""


class RunObservation(BaseModel):
    """One archived run as drift reads it: what it published, under which fingerprint.

    ``values`` is the run's scalar projection, ``None`` where the run did not
    measure a metric. ``spans`` carries the within-session range each metric moved
    in, absent for a metric whose run ran a single repetition.
    """

    run: str
    fingerprint: str
    values: dict[str, float | None]
    spans: dict[str, Span] = {}


class DriftBand(BaseModel):
    """One metric's drift across sessions, beside the spread one session sees.

    ``between_sessions`` is the allowance a comparison widens by: the widest gap
    between two runs' ranges that any one fingerprint group showed. ``fingerprint``
    and ``runs`` are that group, so the figure and its provenance can never come
    from different evidence.

    ``within_session`` is the only figure read from the whole archive instead: the
    widest repetition span any run measured for the metric, ``None`` when no run
    banded it. What one session sees is a property of the metric, and the group that
    drifted widest need not be one that banded it at all -- it is here to be read
    against the allowance, not as that group's own spread.
    """

    metric: str
    between_sessions: float
    within_session: float | None
    fingerprint: str
    runs: list[str]


class DriftRecord(BaseModel):
    """The declared per-metric drift a suite's archive measures.

    ``runs`` names every artifact the record was derived from, so a published
    allowance is traceable to the runs behind it rather than to whatever files
    happened to be on disk.
    """

    suite: str
    measured_at: datetime
    runs: list[str]
    bands: list[DriftBand]

    def band(self, metric: str) -> DriftBand | None:
        """The band for ``metric``, or ``None`` when no group measured drift on it."""
        return next((band for band in self.bands if band.metric == metric), None)

    def allowance(self, metric: str) -> float:
        """How far ``metric`` may move between sessions before the move is a result.

        A metric the archive never measured drift on is allowed none: the record
        says nothing about it, and inventing an allowance would hide real moves.
        """
        band = self.band(metric)
        return band.between_sessions if band is not None else 0.0

    def write(self, path: Path) -> None:
        """Serialize the record to ``path`` as indented JSON, creating parents."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def measure_drift(
    suite: str, observations: Sequence[RunObservation], measured_at: datetime
) -> DriftRecord:
    """Measure each metric's between-session drift over an archive of runs.

    Runs are grouped by fingerprint and only groups of at least
    :data:`MIN_SESSIONS_MEASURING_DRIFT` runs contribute: two runs of different
    configurations differ for reasons that are not drift. Raises
    :class:`DriftError` when handed no run at all.
    """
    if not observations:
        raise DriftError("drift needs at least one run to measure")
    groups = _group_by_fingerprint(observations)
    metrics = _metric_names(observations)
    return DriftRecord(
        suite=suite,
        measured_at=measured_at,
        runs=[observation.run for observation in observations],
        bands=[
            band for metric in metrics if (band := _band(metric, groups, observations)) is not None
        ],
    )


def load_drift(path: Path, suite: str | None = None) -> DriftRecord | None:
    """Read the drift record at ``path``, or ``None`` when the file is absent.

    An absent record means drift has not been measured for this suite and a
    comparison classifies against the repetition bands alone. Raises
    :class:`DriftError` when the file is malformed, or when ``suite`` is given and
    the record was measured over another suite's runs -- one suite's drift says
    nothing about another's metrics.
    """
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DriftError(f"drift record {path} is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise DriftError(f"drift record {path} must be a JSON object")
    try:
        record = DriftRecord.model_validate(raw)
    except ValidationError as error:
        raise DriftError(f"drift record {path} is malformed: {error}") from error
    if suite is not None and record.suite != suite:
        raise DriftError(
            f"drift record {path} was measured over '{record.suite}' runs, not '{suite}'"
        )
    return record


def _group_by_fingerprint(
    observations: Sequence[RunObservation],
) -> list[list[RunObservation]]:
    """The runs that share a fingerprint, keeping only groups that span sessions."""
    groups: dict[str, list[RunObservation]] = {}
    for observation in observations:
        groups.setdefault(observation.fingerprint, []).append(observation)
    return [group for group in groups.values() if len(group) >= MIN_SESSIONS_MEASURING_DRIFT]


def _metric_names(observations: Sequence[RunObservation]) -> list[str]:
    """Every metric name any run reported, first-seen order."""
    names: dict[str, None] = {}
    for observation in observations:
        for metric in observation.values:
            names.setdefault(metric, None)
    return list(names)


def _band(
    metric: str,
    groups: Sequence[Sequence[RunObservation]],
    archive: Sequence[RunObservation],
) -> DriftBand | None:
    """The widest drift ``metric`` showed in any group, or ``None`` when none measured it.

    A group contributes only when at least two of its runs measured the metric:
    a single reading has no gap to read, and pairing it with a run that measured
    nothing would report a drift of zero the archive never observed. The
    within-session figure is read from the whole ``archive`` rather than from the
    group that drifted widest, because what one session sees is a property of the
    metric and the widest-drifting group need not be one that banded it at all.
    """
    measured = [(drift, group) for group in groups if (drift := _drift(metric, group)) is not None]
    if not measured:
        return None
    widest, group = max(measured, key=lambda entry: entry[0])
    reading = [observation for observation in group if observation.values.get(metric) is not None]
    return DriftBand(
        metric=metric,
        between_sessions=widest,
        within_session=_widest_span(metric, archive),
        fingerprint=group[0].fingerprint,
        runs=[observation.run for observation in reading],
    )


def _drift(metric: str, group: Sequence[RunObservation]) -> float | None:
    """The widest gap between two runs' ranges for ``metric``, ``None`` if under two read it.

    A run that measured no band is read at the zero-width range its single value
    amounts to, which is what one repetition honestly observed.
    """
    ranges = [
        observation.spans.get(metric) or Span(low=value, high=value)
        for observation in group
        if (value := observation.values.get(metric)) is not None
    ]
    if len(ranges) < MIN_SESSIONS_MEASURING_DRIFT:
        return None
    return max(_gap(first, second) for first in ranges for second in ranges)


def _gap(first: Span, second: Span) -> float:
    """How far apart two ranges lie, zero when they overlap at all."""
    return max(0.0, first.low - second.high, second.low - first.high)


def _widest_span(metric: str, observations: Sequence[RunObservation]) -> float | None:
    """The widest within-session range any of those runs measured for ``metric``."""
    widths = [
        span.high - span.low
        for observation in observations
        if (span := observation.spans.get(metric)) is not None
    ]
    return max(widths) if widths else None
