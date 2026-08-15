"""``eval compare``: it dispatches on which suite each artifact belongs to."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lexme.config import Settings
from lexme.eval import cli
from lexme.eval.artifact import build_artifact
from lexme.eval.cli import main
from lexme.eval.fingerprint import ConfigFingerprint
from lexme.eval.metrics import SuiteMetrics
from lexme.eval.mode2.artifact import build_mode2_artifact
from lexme.eval.mode2.metrics import Mode2SuiteMetrics

NOW = datetime(2026, 7, 25, tzinfo=UTC)


def _fingerprint(value: str) -> ConfigFingerprint:
    return ConfigFingerprint(
        models={},
        vertical="vivienda",
        prompts_hash="prompts",
        vertical_config_hash="cfg",
        corpus_hash="corpus",
        dataset_hash="dataset",
        fingerprint=value,
    )


def _write_mode1_artifact(path: Path) -> None:
    metrics = SuiteMetrics(
        cases=1,
        outcomes={"respuesta": 1},
        disambiguation={"directo": 1, "reanudado": 0, "sin_respuesta": 0},
        disambiguation_rate=1.0,
        mean_recall=0.9,
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
    build_artifact("modo1", NOW, _fingerprint("fp"), [], metrics, []).write(path)


def _write_mode2_artifact(path: Path) -> None:
    metrics = Mode2SuiteMetrics(
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
        recall_problematic_e2e=1.0,
        false_tranquility_events_e2e=0,
        false_tranquility_rate_e2e=0.0,
        not_reported_problematic=0,
        absence_expected=0,
        absence_detected=0,
        absence_recall=None,
        absence_predicted=0,
        absence_precision=None,
    )
    build_mode2_artifact("modo2", NOW, _fingerprint("fp"), [], metrics, []).write(path)


def test_comparing_two_mode2_artifacts_reports_the_mode2_metrics(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    base = tmp_path / "base.json"
    run = tmp_path / "run.json"
    _write_mode2_artifact(base)
    _write_mode2_artifact(run)

    with caplog.at_level("INFO"):
        exit_code = main(["compare", "--base", str(base), "--run", str(run)])

    assert exit_code == 0
    assert any("recall_problematic_e2e" in record.message for record in caplog.records)


def test_comparing_two_unrepeated_runs_warns_that_no_noise_band_was_measured(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    base = tmp_path / "base.json"
    run = tmp_path / "run.json"
    _write_mode1_artifact(base)
    _write_mode1_artifact(run)

    with caplog.at_level("WARNING"):
        exit_code = main(["compare", "--base", str(base), "--run", str(run)])

    assert exit_code == 0
    assert any("single repetition" in record.message for record in caplog.records)


def test_comparing_a_mode1_and_a_mode2_artifact_is_refused(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    run = tmp_path / "run.json"
    _write_mode1_artifact(base)
    _write_mode2_artifact(run)

    with pytest.raises(ValueError, match="Mode 1.*Mode 2"):
        main(["compare", "--base", str(base), "--run", str(run)])


def test_a_run_named_by_its_bare_file_name_is_read_from_the_runs_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The archive is listed by file name, so a comparison is typed that way too."""
    _write_mode1_artifact(tmp_path / "base.json")
    _write_mode1_artifact(tmp_path / "run.json")
    settings = Settings(database_url="postgresql://unused", eval_runs_dir=str(tmp_path))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    with caplog.at_level("INFO"):
        exit_code = main(["compare", "--base", "base.json", "--run", "run.json"])

    assert exit_code == 0
    assert any("mean_recall" in record.message for record in caplog.records)


def test_an_old_mode2_artifact_missing_e2e_fields_still_compares(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    _write_mode2_artifact(run_path)
    old_path = tmp_path / "old.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    del payload["metrics"]["recall_problematic_e2e"]
    old_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["compare", "--base", str(old_path), "--run", str(run_path)])

    assert exit_code == 0
