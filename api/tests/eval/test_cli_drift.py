"""``eval drift build``: it measures the archive's between-session spread, and
``eval compare`` reads the record it writes."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from lexme.eval.artifact import build_artifact
from lexme.eval.cli import main
from lexme.eval.drift import load_drift
from lexme.eval.fingerprint import ConfigFingerprint
from lexme.eval.metrics import SuiteMetrics, scalar_metrics
from lexme.eval.mode2.artifact import build_mode2_artifact
from lexme.eval.mode2.metrics import Mode2SuiteMetrics, scalar_metrics_mode2
from lexme.eval.repetition import summarize_repetitions

NOW = datetime(2026, 8, 9, tzinfo=UTC)


def _fingerprint(value: str) -> ConfigFingerprint:
    """A fingerprint whose combined hash is ``value``."""
    return ConfigFingerprint(
        models={},
        vertical="vivienda",
        prompts_hash="prompts",
        vertical_config_hash="cfg",
        corpus_hash="corpus",
        dataset_hash="dataset",
        fingerprint=value,
    )


def _mode1_metrics(mean_recall: float) -> SuiteMetrics:
    """Mode 1 metrics carrying ``mean_recall`` and nothing else worth moving."""
    return SuiteMetrics(
        cases=1,
        outcomes={"respuesta": 1},
        disambiguation={"directo": 1, "reanudado": 0, "sin_respuesta": 0},
        disambiguation_rate=1.0,
        mean_recall=mean_recall,
        mean_first_pass_recall=0.8,
        mean_recall_delta=0.1,
        retrieval_layer_recall={"fused": 0.9},
        outcome_match_rate=1.0,
        abstention_rate=0.0,
        expected_abstention_recall=None,
        mean_agentic_delta=0.0,
        judge=None,
        citation_verdicts={},
    )


def _write_mode1_run(path: Path, fingerprint: str, recalls: list[float]) -> None:
    """Write a Mode 1 artifact whose repetitions observed ``recalls``."""
    artifact = build_artifact(
        "modo1", NOW, _fingerprint(fingerprint), [], _mode1_metrics(recalls[0]), []
    )
    artifact.model_copy(
        update={
            "repetitions": summarize_repetitions(
                [scalar_metrics(_mode1_metrics(recall)) for recall in recalls]
            )
        }
    ).write(path)


def _mode2_metrics(recall: float) -> Mode2SuiteMetrics:
    """Mode 2 metrics carrying ``recall`` as the end-to-end problematic recall."""
    return Mode2SuiteMetrics(
        cases=1,
        outcomes={"analizado": 1},
        outcome_match_rate=1.0,
        reference_clauses=1,
        delimited_clauses=1,
        segmentation_delimited_rate=1.0,
        matched_clauses=1,
        confusion={},
        problematic_total=1,
        problematic_detected=1,
        recall_problematic=1.0,
        false_tranquility_events=0,
        false_tranquility_rate=0.0,
        flagged_problematic=1,
        flagged_true_positive=1,
        precision_problematic=1.0,
        abstention_clauses=0,
        abstention_rate=0.0,
        problematic_total_e2e=1,
        problematic_detected_e2e=1,
        recall_problematic_e2e=recall,
        false_tranquility_events_e2e=0,
        false_tranquility_rate_e2e=0.0,
        not_reported_problematic=0,
        absence_expected=0,
        absence_detected=0,
        absence_recall=None,
        absence_predicted=0,
        absence_precision=None,
    )


def _write_mode2_run(path: Path, fingerprint: str, recalls: list[float]) -> None:
    """Write a Mode 2 artifact whose repetitions observed ``recalls``."""
    artifact = build_mode2_artifact(
        "modo2", NOW, _fingerprint(fingerprint), [], _mode2_metrics(recalls[0]), []
    )
    artifact.model_copy(
        update={
            "repetitions": summarize_repetitions(
                [scalar_metrics_mode2(_mode2_metrics(recall)) for recall in recalls]
            )
        }
    ).write(path)


def test_building_drift_measures_the_spread_of_same_fingerprint_runs(tmp_path: Path) -> None:
    _write_mode1_run(tmp_path / "modo1-20260805T000000Z.json", "fp", [0.95, 0.95, 0.95])
    _write_mode1_run(tmp_path / "modo1-20260809T000000Z.json", "fp", [0.87, 0.87, 0.87])

    exit_code = main(["drift", "build", "--mode", "modo1", "--runs", str(tmp_path)], now=NOW)

    record = load_drift(tmp_path / "drift-modo1.json")
    assert exit_code == 0
    assert record is not None
    assert record.allowance("mean_recall") == pytest.approx(0.08)


def test_the_built_record_carries_the_within_session_span_beside_the_drift(
    tmp_path: Path,
) -> None:
    _write_mode1_run(tmp_path / "modo1-20260805T000000Z.json", "fp", [0.95, 0.95, 0.95])
    _write_mode1_run(tmp_path / "modo1-20260809T000000Z.json", "fp", [0.87, 0.87, 0.87])

    main(["drift", "build", "--mode", "modo1", "--runs", str(tmp_path)], now=NOW)

    record = load_drift(tmp_path / "drift-modo1.json")
    assert record is not None
    band = record.band("mean_recall")
    assert band is not None
    assert band.within_session == pytest.approx(0.0)
    assert band.runs == [
        "modo1-20260805T000000Z.json",
        "modo1-20260809T000000Z.json",
    ]


def test_a_review_sheet_beside_the_runs_is_not_read_as_a_run(tmp_path: Path) -> None:
    _write_mode1_run(tmp_path / "modo1-20260805T000000Z.json", "fp", [0.95])
    _write_mode1_run(tmp_path / "modo1-20260809T000000Z.json", "fp", [0.87])
    (tmp_path / "modo1-20260809T000000Z-judge-review.json").write_text("{}", encoding="utf-8")

    exit_code = main(["drift", "build", "--mode", "modo1", "--runs", str(tmp_path)], now=NOW)

    record = load_drift(tmp_path / "drift-modo1.json")
    assert exit_code == 0
    assert record is not None
    assert record.runs == [
        "modo1-20260805T000000Z.json",
        "modo1-20260809T000000Z.json",
    ]


def test_building_drift_over_the_mode2_archive_reads_the_mode2_metrics(tmp_path: Path) -> None:
    _write_mode2_run(tmp_path / "modo2-20260728T000000Z.json", "fp", [1.0])
    _write_mode2_run(tmp_path / "modo2-20260808T000000Z.json", "fp", [0.8])

    exit_code = main(["drift", "build", "--mode", "modo2", "--runs", str(tmp_path)], now=NOW)

    record = load_drift(tmp_path / "drift-modo2.json")
    assert exit_code == 0
    assert record is not None
    assert record.suite == "modo2"
    assert record.allowance("recall_problematic_e2e") == pytest.approx(0.2)


def test_building_drift_over_an_empty_archive_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one run"):
        main(["drift", "build", "--mode", "modo1", "--runs", str(tmp_path)], now=NOW)


def test_comparing_reads_the_drift_record_lying_beside_the_runs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    base = tmp_path / "modo1-20260805T000000Z.json"
    run = tmp_path / "modo1-20260809T000000Z.json"
    _write_mode1_run(base, "fp", [0.95, 0.95, 0.95])
    _write_mode1_run(run, "fp", [0.87, 0.87, 0.87])
    main(["drift", "build", "--mode", "modo1", "--runs", str(tmp_path)], now=NOW)

    with caplog.at_level("INFO"):
        exit_code = main(["compare", "--base", str(base), "--run", str(run)])

    assert exit_code == 0
    assert any("mean_recall" in line and "drift" in line for line in _messages(caplog))


def test_comparing_without_a_record_beside_the_runs_warns_the_band_is_a_floor(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    base = tmp_path / "modo1-20260805T000000Z.json"
    run = tmp_path / "modo1-20260809T000000Z.json"
    _write_mode1_run(base, "fp", [0.95, 0.95, 0.95])
    _write_mode1_run(run, "fp", [0.87, 0.87, 0.87])

    with caplog.at_level("WARNING"):
        main(["compare", "--base", str(base), "--run", str(run)])

    assert any("within-session" in message for message in _messages(caplog))


def test_comparing_with_no_drift_refuses_the_record_beside_the_runs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    base = tmp_path / "modo1-20260805T000000Z.json"
    run = tmp_path / "modo1-20260809T000000Z.json"
    _write_mode1_run(base, "fp", [0.95, 0.95, 0.95])
    _write_mode1_run(run, "fp", [0.87, 0.87, 0.87])
    main(["drift", "build", "--mode", "modo1", "--runs", str(tmp_path)], now=NOW)

    with caplog.at_level("INFO"):
        main(["compare", "--base", str(base), "--run", str(run), "--no-drift"])

    assert any("regression" in line and "mean_recall" in line for line in _messages(caplog))


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every log line the run emitted, formatted."""
    return [record.getMessage() for record in caplog.records]


def test_a_named_drift_record_that_is_not_there_is_refused(tmp_path: Path) -> None:
    base = tmp_path / "modo1-20260805T000000Z.json"
    run = tmp_path / "modo1-20260809T000000Z.json"
    _write_mode1_run(base, "fp", [0.95, 0.95, 0.95])
    _write_mode1_run(run, "fp", [0.87, 0.87, 0.87])

    with pytest.raises(ValueError, match="no drift record"):
        main(
            [
                "compare",
                "--base",
                str(base),
                "--run",
                str(run),
                "--drift",
                str(tmp_path / "missing.json"),
            ]
        )
