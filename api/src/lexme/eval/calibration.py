"""Single-pass judge calibration: the human-judge agreement the score is read with.

A synthetic judge without calibration gives numbers no one can trust. The pattern
is synthetic plus one human calibration pass: a sample of the judge's own per-item
rulings is reviewed by a human with the article in front of them, and the fraction
they agree on is published beside every judged metric ("completeness 0.87, judge at
94% human agreement"). The review is expensive and done once, so it lives as a
versioned record the harness reads and stamps onto each run, rather than being
recomputed per run. The agreement is always derived from the reviewed items here,
never trusted from the file, so a stale number cannot creep in.
"""

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

CALIBRATION_FILENAME = "judge-calibration.json"

KIND_KEY_POINT = "key_point"
KIND_CLAIM = "claim"
_KINDS = (KIND_KEY_POINT, KIND_CLAIM)


class CalibrationError(ValueError):
    """Raised when a calibration record is malformed."""


class CalibrationItem(BaseModel):
    """One reviewed judge ruling: what the judge said and what the human said.

    ``kind`` is the ruling reviewed (a key-point coverage or a claim's support),
    ``ref`` identifies it (a block id or the claim text) and the two labels are the
    booleans compared for agreement.
    """

    case_id: str
    kind: str
    ref: str
    judge_label: bool
    human_label: bool


class JudgeCalibration(BaseModel):
    """The published result of the one-time calibration pass.

    ``agreement`` is the fraction of reviewed items where judge and human agreed,
    ``None`` when the sample is empty (calibration not yet done). ``sample_size`` is
    the number of reviewed items, so a high agreement over three items cannot pass
    for a calibrated judge.
    """

    judge_model: str
    reviewed_by: str
    reviewed_at: datetime
    sample_size: int
    agreement: float | None
    items: list[CalibrationItem]


def compute_agreement(items: list[CalibrationItem]) -> float | None:
    """Return the fraction of items where judge and human agreed, or ``None`` if empty."""
    if not items:
        return None
    agreed = sum(1 for item in items if item.judge_label == item.human_label)
    return agreed / len(items)


def build_calibration(
    judge_model: str, reviewed_by: str, reviewed_at: datetime, items: list[CalibrationItem]
) -> JudgeCalibration:
    """Assemble a calibration record, deriving agreement and sample size from ``items``."""
    return JudgeCalibration(
        judge_model=judge_model,
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at,
        sample_size=len(items),
        agreement=compute_agreement(items),
        items=items,
    )


def load_calibration(path: Path) -> JudgeCalibration | None:
    """Read the calibration record at ``path``, or ``None`` when the file is absent.

    An absent file means the judge has not been calibrated yet and the run publishes
    no agreement, rather than inventing one. Raises :class:`CalibrationError` when
    the file exists but is malformed.
    """
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CalibrationError(f"calibration {path} is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise CalibrationError(f"calibration {path} must be a JSON object")

    judge_model = raw.get("judge_model")
    reviewed_by = raw.get("reviewed_by")
    reviewed_at = raw.get("reviewed_at")
    if not isinstance(judge_model, str) or not judge_model:
        raise CalibrationError(f"calibration {path} missing a non-empty 'judge_model'")
    if not isinstance(reviewed_by, str) or not reviewed_by:
        raise CalibrationError(f"calibration {path} missing a non-empty 'reviewed_by'")
    if not isinstance(reviewed_at, str):
        raise CalibrationError(f"calibration {path} missing an ISO 'reviewed_at'")

    items = _read_items(raw.get("items", []), path)
    return build_calibration(judge_model, reviewed_by, datetime.fromisoformat(reviewed_at), items)


def _read_items(raw_items: object, path: Path) -> list[CalibrationItem]:
    """Validate the reviewed items array into :class:`CalibrationItem` values."""
    if not isinstance(raw_items, list):
        raise CalibrationError(f"calibration {path} 'items' must be an array")
    items = []
    for entry in raw_items:
        if not isinstance(entry, dict):
            raise CalibrationError(f"calibration {path} has a non-object item: {entry!r}")
        if entry.get("kind") not in _KINDS:
            raise CalibrationError(
                f"calibration {path} item 'kind' must be one of {_KINDS}, got {entry.get('kind')!r}"
            )
        try:
            items.append(CalibrationItem.model_validate(entry))
        except ValueError as error:
            raise CalibrationError(f"calibration {path} has a malformed item: {error}") from error
    return items
