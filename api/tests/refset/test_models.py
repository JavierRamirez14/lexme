"""Tests for the reference-set domain models and their invariants."""

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from lexme.mode1.models import Outcome
from lexme.mode2.risk import RiskLevel
from lexme.refset.models import (
    Candidate,
    CaseKind,
    ClauseGroundTruth,
    KeyPoint,
    Mode1ReferenceCase,
    Mode2ReferenceCase,
    Provenance,
    ReviewStatus,
)

_NOW = datetime(2026, 7, 25, tzinfo=UTC)
_NORM = "BOE-A-1994-26003"
_REF_A9 = f"{_NORM}:a9"
_REF_A10 = f"{_NORM}:a10"


def _provenance() -> Provenance:
    return Provenance(generator="test", generated_at=_NOW)


def test_a_mode1_case_accepts_key_points_anchored_to_gold_blocks() -> None:
    case = Mode1ReferenceCase(
        id="plazo",
        question="¿plazo mínimo?",
        gold_block_refs=[_REF_A9],
        expected_outcome=Outcome.ANSWER,
        key_points=[KeyPoint(claim="cinco años", block_ref=_REF_A9)],
        target_date=date(2024, 1, 1),
    )

    assert case.expected_outcome is Outcome.ANSWER
    assert case.key_points[0].block_ref == _REF_A9


def test_a_key_point_off_the_gold_blocks_is_rejected() -> None:
    with pytest.raises(ValidationError, match="absent from gold blocks"):
        Mode1ReferenceCase(
            id="plazo",
            question="¿plazo mínimo?",
            gold_block_refs=[_REF_A9],
            expected_outcome=Outcome.ANSWER,
            key_points=[KeyPoint(claim="x", block_ref=_REF_A10)],
        )


def test_a_clause_labelled_ausente_is_rejected() -> None:
    with pytest.raises(ValidationError, match="clause level"):
        ClauseGroundTruth(
            clause_id="c1",
            heading="Fianza",
            text="...",
            start=0,
            end=3,
            expected_level=RiskLevel.AUSENTE,
            source="consumo",
        )


def test_a_candidate_id_mirrors_its_payload() -> None:
    case = Mode2ReferenceCase(id="contrato-01", document="...")
    candidate = Candidate(kind=CaseKind.MODE2, provenance=_provenance(), mode2=case)

    assert candidate.id == "contrato-01"
    assert candidate.status is ReviewStatus.PENDING


def test_a_candidate_payload_must_match_its_kind() -> None:
    case = Mode2ReferenceCase(id="contrato-01", document="...")
    with pytest.raises(ValidationError, match="modo1 candidate"):
        Candidate(kind=CaseKind.MODE1, provenance=_provenance(), mode2=case)


def test_a_candidate_round_trips_through_json() -> None:
    case = Mode1ReferenceCase(
        id="plazo", question="¿plazo?", gold_block_refs=[_REF_A9], expected_outcome=Outcome.ANSWER
    )
    candidate = Candidate(kind=CaseKind.MODE1, provenance=_provenance(), mode1=case)

    restored = Candidate.model_validate_json(candidate.model_dump_json())

    assert restored == candidate
