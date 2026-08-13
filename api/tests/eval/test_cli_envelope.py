"""``eval envelope``: it pools every run of one fingerprint into a single band."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from lexme.eval.artifact import build_artifact
from lexme.eval.cli import main
from lexme.eval.fingerprint import ConfigFingerprint
from lexme.eval.metrics import SuiteMetrics, scalar_metrics
from lexme.eval.repetition import summarize_repetitions

NOW = datetime(2026, 8, 13, tzinfo=UTC)


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


def _metrics(mean_recall: float) -> SuiteMetrics:
    """Mode 1 metrics carrying ``mean_recall``."""
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


def _write_run(path: Path, fingerprint: str, recalls: list[float]) -> None:
    """Write a Mode 1 artifact whose repetitions observed ``recalls``."""
    artifact = build_artifact("modo1", NOW, _fingerprint(fingerprint), [], _metrics(recalls[0]), [])
    artifact.model_copy(
        update={
            "repetitions": summarize_repetitions(
                [scalar_metrics(_metrics(recall)) for recall in recalls]
            )
        }
    ).write(path)


def test_the_envelope_bands_every_repetition_of_every_run_at_one_fingerprint(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_run(tmp_path / "modo1-20260809T000000Z.json", "fp", [0.87, 0.87, 0.87])
    _write_run(tmp_path / "modo1-20260810T000000Z.json", "fp", [0.92, 0.84, 0.89])

    with caplog.at_level("INFO"):
        exit_code = main(["envelope", "--mode", "modo1", "--runs", str(tmp_path)])

    assert exit_code == 0
    assert any("6 repetitions" in message for message in _messages(caplog))
    assert any("0.84" in message and "0.92" in message for message in _messages(caplog))


def test_the_envelope_covers_the_newest_fingerprint_by_default(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_run(tmp_path / "modo1-20260701T000000Z.json", "old", [0.10, 0.10, 0.10])
    _write_run(tmp_path / "modo1-20260810T000000Z.json", "fp", [0.92, 0.84, 0.89])

    with caplog.at_level("INFO"):
        main(["envelope", "--mode", "modo1", "--runs", str(tmp_path)])

    assert any("fingerprint 'fp'" in message for message in _messages(caplog))
    assert not any("0.10" in message for message in _messages(caplog))


def test_a_named_fingerprint_is_the_one_pooled(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_run(tmp_path / "modo1-20260701T000000Z.json", "old", [0.10, 0.12, 0.11])
    _write_run(tmp_path / "modo1-20260810T000000Z.json", "fp", [0.92, 0.84, 0.89])

    with caplog.at_level("INFO"):
        main(["envelope", "--mode", "modo1", "--runs", str(tmp_path), "--fingerprint", "old"])

    assert any("0.12" in message for message in _messages(caplog))


def test_an_unknown_fingerprint_is_refused(tmp_path: Path) -> None:
    _write_run(tmp_path / "modo1-20260810T000000Z.json", "fp", [0.92])

    with pytest.raises(ValueError, match="no archived run"):
        main(["envelope", "--mode", "modo1", "--runs", str(tmp_path), "--fingerprint", "nope"])


def test_the_envelope_names_the_runs_it_pooled(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_run(tmp_path / "modo1-20260809T000000Z.json", "fp", [0.87])
    _write_run(tmp_path / "modo1-20260810T000000Z.json", "fp", [0.92])

    with caplog.at_level("INFO"):
        main(["envelope", "--mode", "modo1", "--runs", str(tmp_path)])

    messages = " ".join(_messages(caplog))
    assert "modo1-20260809T000000Z.json" in messages
    assert "modo1-20260810T000000Z.json" in messages


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every log line the run emitted, formatted."""
    return [record.getMessage() for record in caplog.records]
