"""Tests for the single-pass judge calibration record and its agreement number."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lexme.eval.artifact import build_artifact, read_artifact
from lexme.eval.calibration import (
    REVIEWER_HUMAN,
    REVIEWER_MODEL,
    CalibrationError,
    CalibrationItem,
    build_calibration,
    compute_agreement,
    load_calibration,
)
from lexme.eval.fingerprint import ConfigFingerprint
from lexme.eval.metrics import aggregate

REVIEWED_AT = datetime(2026, 7, 25, tzinfo=UTC)


def _item(judge: bool, reviewer: bool) -> CalibrationItem:
    """One reviewed key-point ruling with the given judge and reviewer labels."""
    return CalibrationItem(
        case_id="c", kind="key_point", ref="a9", judge_label=judge, reviewer_label=reviewer
    )


def test_agreement_is_the_fraction_of_matching_labels() -> None:
    items = [_item(True, True), _item(True, True), _item(False, True), _item(False, False)]

    assert compute_agreement(items) == 0.75


def test_agreement_is_none_for_an_empty_sample() -> None:
    assert compute_agreement([]) is None


def test_build_derives_agreement_and_sample_size() -> None:
    calibration = build_calibration(
        "gemini-3.5-flash", "human", REVIEWER_HUMAN, REVIEWED_AT, [_item(True, True)]
    )

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
                "reviewed_by": "Javier Ramírez",
                "reviewer_kind": "human",
                "reviewed_at": "2026-07-25T00:00:00+00:00",
                "agreement": 0.0,
                "items": [
                    {
                        "case_id": "c",
                        "kind": "claim",
                        "ref": "x",
                        "judge_label": True,
                        "reviewer_label": True,
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
                "reviewer_kind": "human",
                "reviewed_at": "2026-07-25T00:00:00+00:00",
                "items": [
                    {
                        "case_id": "c",
                        "kind": "vibes",
                        "ref": "x",
                        "judge_label": True,
                        "reviewer_label": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CalibrationError, match="kind"):
        load_calibration(path)


def _record(reviewer_kind: object) -> dict:
    """A calibration record body with the given ``reviewer_kind``."""
    return {
        "judge_model": "m",
        "reviewed_by": "someone",
        "reviewer_kind": reviewer_kind,
        "reviewed_at": "2026-07-25T00:00:00+00:00",
        "items": [
            {
                "case_id": "c",
                "kind": "claim",
                "ref": "x",
                "judge_label": True,
                "reviewer_label": True,
            }
        ],
    }


def test_a_calibration_declares_whether_its_reviewer_was_human_or_a_model(tmp_path: Path) -> None:
    path = tmp_path / "judge-calibration.json"
    path.write_text(json.dumps(_record(REVIEWER_MODEL)), encoding="utf-8")

    calibration = load_calibration(path)

    assert calibration is not None
    assert calibration.reviewer_kind == REVIEWER_MODEL


def test_a_calibration_without_a_declared_reviewer_kind_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "judge-calibration.json"
    body = _record(REVIEWER_HUMAN)
    del body["reviewer_kind"]
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(CalibrationError, match="reviewer_kind"):
        load_calibration(path)


def test_a_reviewer_kind_outside_the_two_declared_ones_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "judge-calibration.json"
    path.write_text(json.dumps(_record("committee")), encoding="utf-8")

    with pytest.raises(CalibrationError, match="reviewer_kind"):
        load_calibration(path)


def test_the_artifact_publishes_the_calibration_and_round_trips(tmp_path: Path) -> None:
    calibration = build_calibration(
        "gemini-3.5-flash", "human", REVIEWER_HUMAN, REVIEWED_AT, [_item(True, True)]
    )
    fingerprint = ConfigFingerprint(
        models={},
        vertical="vivienda",
        prompts_hash="p",
        vertical_config_hash="c",
        corpus_hash="k",
        dataset_hash="d",
        fingerprint="fp",
    )
    artifact = build_artifact("modo1", REVIEWED_AT, fingerprint, [], aggregate([]), [], calibration)

    out = tmp_path / "run.json"
    artifact.write(out)
    restored = read_artifact(out)

    assert restored.judge_calibration is not None
    assert restored.judge_calibration.agreement == 1.0
    assert restored == artifact
