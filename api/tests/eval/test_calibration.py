"""Tests for the single-pass judge calibration record and its agreement number."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lexme.eval.artifact import build_artifact, read_artifact
from lexme.eval.calibration import (
    CalibrationError,
    CalibrationItem,
    build_calibration,
    compute_agreement,
    load_calibration,
)
from lexme.eval.fingerprint import ConfigFingerprint
from lexme.eval.metrics import aggregate

REVIEWED_AT = datetime(2026, 7, 25, tzinfo=UTC)


def _item(judge: bool, human: bool) -> CalibrationItem:
    """One reviewed key-point ruling with the given judge and human labels."""
    return CalibrationItem(
        case_id="c", kind="key_point", ref="a9", judge_label=judge, human_label=human
    )


def test_agreement_is_the_fraction_of_matching_labels() -> None:
    items = [_item(True, True), _item(True, True), _item(False, True), _item(False, False)]

    assert compute_agreement(items) == 0.75


def test_agreement_is_none_for_an_empty_sample() -> None:
    assert compute_agreement([]) is None


def test_build_derives_agreement_and_sample_size() -> None:
    calibration = build_calibration("gemini-3.5-flash", "human", REVIEWED_AT, [_item(True, True)])

    assert calibration.agreement == 1.0
    assert calibration.sample_size == 1


def test_a_missing_calibration_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_calibration(tmp_path / "absent.json") is None


def test_a_calibration_file_loads_and_recomputes_agreement(tmp_path: Path) -> None:
    path = tmp_path / "judge-calibration.json"
    path.write_text(
        json.dumps(
            {
                "judge_model": "gemini-3.5-flash",
                "reviewed_by": "human",
                "reviewed_at": "2026-07-25T00:00:00+00:00",
                "agreement": 0.0,
                "items": [
                    {
                        "case_id": "c",
                        "kind": "claim",
                        "ref": "x",
                        "judge_label": True,
                        "human_label": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    calibration = load_calibration(path)

    assert calibration is not None
    assert calibration.agreement == 1.0
    assert calibration.sample_size == 1


def test_an_item_with_an_unknown_kind_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "judge-calibration.json"
    path.write_text(
        json.dumps(
            {
                "judge_model": "m",
                "reviewed_by": "h",
                "reviewed_at": "2026-07-25T00:00:00+00:00",
                "items": [
                    {
                        "case_id": "c",
                        "kind": "vibes",
                        "ref": "x",
                        "judge_label": True,
                        "human_label": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CalibrationError, match="kind"):
        load_calibration(path)


def test_the_artifact_publishes_the_calibration_and_round_trips(tmp_path: Path) -> None:
    calibration = build_calibration("gemini-3.5-flash", "human", REVIEWED_AT, [_item(True, True)])
    fingerprint = ConfigFingerprint(
        models={},
        vertical="vivienda",
        prompts_hash="p",
        vertical_config_hash="c",
        corpus_hash="k",
        dataset_hash="d",
        fingerprint="fp",
    )
    artifact = build_artifact(
        "modo1", REVIEWED_AT, fingerprint, [], aggregate([]), [], calibration
    )

    out = tmp_path / "run.json"
    artifact.write(out)
    restored = read_artifact(out)

    assert restored.judge_calibration is not None
    assert restored.judge_calibration.agreement == 1.0
    assert restored == artifact
