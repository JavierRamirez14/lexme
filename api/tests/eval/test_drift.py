"""Tests for the between-session drift a run archive measures."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lexme.eval.drift import (
    DriftError,
    RunObservation,
    load_drift,
    measure_drift,
)
from lexme.eval.repetition import Span

MEASURED_AT = datetime(2026, 8, 9, tzinfo=UTC)


def _observation(
    run: str,
    fingerprint: str,
    values: dict[str, float | None],
    spans: dict[str, Span] | None = None,
) -> RunObservation:
    """One archived run's published values and the spans its repetitions measured."""
    return RunObservation(run=run, fingerprint=fingerprint, values=values, spans=spans or {})


def test_the_spread_of_one_fingerprints_runs_is_the_between_session_drift() -> None:
    record = measure_drift(
        "modo1",
        [
            _observation("a.json", "fp", {"mean_recall": 0.92}),
            _observation("b.json", "fp", {"mean_recall": 0.87}),
        ],
        MEASURED_AT,
    )

    assert record.allowance("mean_recall") == pytest.approx(0.05)


def test_runs_of_different_fingerprints_are_never_spread_against_each_other() -> None:
    record = measure_drift(
        "modo1",
        [
            _observation("a.json", "fp-a", {"mean_recall": 0.92}),
            _observation("b.json", "fp-a", {"mean_recall": 0.90}),
            _observation("c.json", "fp-b", {"mean_recall": 0.50}),
        ],
        MEASURED_AT,
    )

    assert record.allowance("mean_recall") == pytest.approx(0.02)


def test_the_widest_group_sets_the_allowance() -> None:
    record = measure_drift(
        "modo1",
        [
            _observation("a.json", "fp-a", {"mean_recall": 0.92}),
            _observation("b.json", "fp-a", {"mean_recall": 0.90}),
            _observation("c.json", "fp-b", {"mean_recall": 0.80}),
            _observation("d.json", "fp-b", {"mean_recall": 0.60}),
        ],
        MEASURED_AT,
    )

    assert record.allowance("mean_recall") == pytest.approx(0.20)


def test_a_fingerprint_only_one_run_measured_contributes_no_drift() -> None:
    record = measure_drift(
        "modo1",
        [
            _observation("a.json", "fp-a", {"mean_recall": 0.92}),
            _observation("b.json", "fp-b", {"mean_recall": 0.10}),
        ],
        MEASURED_AT,
    )

    assert record.bands == []
    assert record.allowance("mean_recall") == 0.0


def test_a_metric_no_group_repeated_gets_no_band() -> None:
    record = measure_drift(
        "modo1",
        [
            _observation("a.json", "fp", {"mean_recall": 0.92, "judge.completeness": 0.8}),
            _observation("b.json", "fp", {"mean_recall": 0.87, "judge.completeness": None}),
        ],
        MEASURED_AT,
    )

    assert [band.metric for band in record.bands] == ["mean_recall"]


def test_the_band_records_the_widest_within_session_span_beside_the_drift() -> None:
    record = measure_drift(
        "modo1",
        [
            _observation(
                "a.json", "fp", {"mean_recall": 0.92}, {"mean_recall": Span(low=0.92, high=0.92)}
            ),
            _observation(
                "b.json", "fp", {"mean_recall": 0.87}, {"mean_recall": Span(low=0.85, high=0.89)}
            ),
        ],
        MEASURED_AT,
    )
    band = record.band("mean_recall")

    assert band is not None
    assert band.within_session == pytest.approx(0.04)
    assert band.between_sessions == pytest.approx(0.03)


def test_a_metric_no_run_banded_records_no_within_session_span() -> None:
    record = measure_drift(
        "modo1",
        [
            _observation("a.json", "fp", {"mean_recall": 0.92}),
            _observation("b.json", "fp", {"mean_recall": 0.87}),
        ],
        MEASURED_AT,
    )
    band = record.band("mean_recall")

    assert band is not None
    assert band.within_session is None


def test_the_band_names_only_the_runs_the_widest_drift_came_from() -> None:
    record = measure_drift(
        "modo1",
        [
            _observation("a.json", "fp-a", {"mean_recall": 0.92}),
            _observation("b.json", "fp-a", {"mean_recall": 0.90}),
            _observation("c.json", "fp-b", {"mean_recall": 0.80}),
            _observation("d.json", "fp-b", {"mean_recall": 0.60}),
        ],
        MEASURED_AT,
    )
    band = record.band("mean_recall")

    assert band is not None
    assert band.fingerprint == "fp-b"
    assert band.runs == ["c.json", "d.json"]


def test_the_record_names_every_run_it_was_derived_from() -> None:
    record = measure_drift(
        "modo1",
        [
            _observation("a.json", "fp", {"mean_recall": 0.92}),
            _observation("b.json", "fp", {"mean_recall": 0.87}),
        ],
        MEASURED_AT,
    )

    assert record.runs == ["a.json", "b.json"]
    assert record.suite == "modo1"
    assert record.measured_at == MEASURED_AT


def test_measuring_drift_over_no_run_at_all_is_refused() -> None:
    with pytest.raises(DriftError):
        measure_drift("modo1", [], MEASURED_AT)


def test_an_unmeasured_metric_is_allowed_no_drift() -> None:
    record = measure_drift(
        "modo1",
        [
            _observation("a.json", "fp", {"mean_recall": 0.92}),
            _observation("b.json", "fp", {"mean_recall": 0.87}),
        ],
        MEASURED_AT,
    )

    assert record.allowance("outcome_match_rate") == 0.0


def test_a_written_record_reads_back(tmp_path: Path) -> None:
    record = measure_drift(
        "modo1",
        [
            _observation("a.json", "fp", {"mean_recall": 0.92}),
            _observation("b.json", "fp", {"mean_recall": 0.87}),
        ],
        MEASURED_AT,
    )
    path = tmp_path / "drift-modo1.json"
    record.write(path)

    assert load_drift(path) == record


def test_an_absent_record_reads_as_no_drift_measured(tmp_path: Path) -> None:
    assert load_drift(tmp_path / "missing.json") is None


def test_a_malformed_record_is_refused_rather_than_read_as_zero(tmp_path: Path) -> None:
    path = tmp_path / "drift-modo1.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(DriftError):
        load_drift(path)


def test_a_record_of_another_suite_cannot_be_read_as_this_ones(tmp_path: Path) -> None:
    record = measure_drift(
        "modo2",
        [
            _observation("a.json", "fp", {"recall_problematic": 1.0}),
            _observation("b.json", "fp", {"recall_problematic": 0.9}),
        ],
        MEASURED_AT,
    )
    path = tmp_path / "drift.json"
    record.write(path)

    with pytest.raises(DriftError):
        load_drift(path, suite="modo1")


def test_a_record_read_under_its_own_suite_is_accepted(tmp_path: Path) -> None:
    record = measure_drift(
        "modo2",
        [
            _observation("a.json", "fp", {"recall_problematic": 1.0}),
            _observation("b.json", "fp", {"recall_problematic": 0.9}),
        ],
        MEASURED_AT,
    )
    path = tmp_path / "drift.json"
    record.write(path)

    assert load_drift(path, suite="modo2") == record


def test_a_record_missing_its_bands_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "drift.json"
    path.write_text(json.dumps({"suite": "modo1"}), encoding="utf-8")

    with pytest.raises(DriftError):
        load_drift(path)


def test_a_run_that_never_measured_the_metric_is_not_named_as_evidence() -> None:
    record = measure_drift(
        "modo1",
        [
            _observation("a.json", "fp", {"judge.completeness": 0.80}),
            _observation("b.json", "fp", {"judge.completeness": 0.75}),
            _observation("c.json", "fp", {"judge.completeness": None}),
        ],
        MEASURED_AT,
    )
    band = record.band("judge.completeness")

    assert band is not None
    assert band.runs == ["a.json", "b.json"]


def test_drift_is_the_gap_between_the_ranges_not_between_the_medians() -> None:
    record = measure_drift(
        "modo1",
        [
            _observation(
                "a.json", "fp", {"mean_recall": 0.95}, {"mean_recall": Span(low=0.89, high=0.95)}
            ),
            _observation(
                "b.json", "fp", {"mean_recall": 0.87}, {"mean_recall": Span(low=0.87, high=0.87)}
            ),
        ],
        MEASURED_AT,
    )

    assert record.allowance("mean_recall") == pytest.approx(0.02)


def test_two_runs_whose_ranges_overlap_drifted_by_nothing() -> None:
    record = measure_drift(
        "modo1",
        [
            _observation(
                "a.json", "fp", {"mean_recall": 0.95}, {"mean_recall": Span(low=0.85, high=0.95)}
            ),
            _observation(
                "b.json", "fp", {"mean_recall": 0.87}, {"mean_recall": Span(low=0.86, high=0.88)}
            ),
        ],
        MEASURED_AT,
    )

    assert record.allowance("mean_recall") == 0.0
