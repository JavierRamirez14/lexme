"""Tests for the candidate store and the human review gate."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from lexme.eval.cases import answer_for_branch, load_cases
from lexme.mode1.models import Outcome
from lexme.refset.models import (
    Candidate,
    CaseKind,
    ClarificationAnswer,
    KeyPoint,
    Mode1ReferenceCase,
    Mode2ReferenceCase,
    Provenance,
    ReviewStatus,
)
from lexme.refset.store import CandidateStore, StoreError

_NOW = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
_NORM = "BOE-A-1994-26003"
_REF_A9 = f"{_NORM}:a9"


def _store(tmp_path: Path) -> CandidateStore:
    return CandidateStore(refset_dir=tmp_path / "refset", eval_dir=tmp_path / "eval")


def _mode1(candidate_id: str = "plazo") -> Candidate:
    case = Mode1ReferenceCase(
        id=candidate_id,
        question="¿plazo mínimo?",
        gold_block_refs=[_REF_A9],
        expected_outcome=Outcome.ANSWER,
        key_points=[KeyPoint(claim="cinco años", block_ref=_REF_A9)],
    )
    provenance = Provenance(generator="query_generator", sources=[_REF_A9], generated_at=_NOW)
    return Candidate(kind=CaseKind.MODE1, provenance=provenance, mode1=case)


def _mode2(candidate_id: str = "contrato-01") -> Candidate:
    case = Mode2ReferenceCase(id=candidate_id, document="CONTRATO\n\nFianza\ntexto\n\n")
    provenance = Provenance(generator="assembler", sources=["b1"], generated_at=_NOW)
    return Candidate(kind=CaseKind.MODE2, provenance=provenance, mode2=case)


def test_a_pending_candidate_is_not_in_the_eval_set(tmp_path: Path) -> None:
    store = _store(tmp_path)
    candidate = _mode1()

    store.write(candidate)

    assert store.list_pending() == [candidate]
    assert not store.materialized_path(candidate).exists()


def test_accepting_materializes_the_case_with_its_provenance(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(_mode1())

    path = store.accept("plazo", reviewed_by="javier", now=_NOW)

    assert path.exists()
    cases = load_cases(path)
    assert cases[0].gold_block_refs == (_REF_A9,)
    assert cases[0].expected_outcome == "respuesta"


def test_a_materialized_case_carries_the_clarification_answer_it_pins(tmp_path: Path) -> None:
    store = _store(tmp_path)
    candidate = _mode1()
    assert candidate.mode1 is not None
    candidate.mode1.clarification_answers = [
        ClarificationAnswer(branch_id="fecha_firma", answer="01/06/2023")
    ]
    store.write(candidate)

    path = store.accept("plazo", reviewed_by="javier", now=_NOW)

    (case,) = load_cases(path)
    assert answer_for_branch(case.clarification_answers, "fecha_firma") == "01/06/2023"


def test_an_accepted_candidate_records_its_reviewer(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(_mode1())

    store.accept("plazo", reviewed_by="javier", now=_NOW, note="ok")

    reviewed = store.get("plazo")
    assert reviewed.status is ReviewStatus.ACCEPTED
    assert reviewed.provenance.reviewed_by == "javier"
    assert reviewed.provenance.reviewed_at == _NOW
    assert store.list_pending() == []


def test_rejecting_materializes_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    candidate = _mode1()
    store.write(candidate)

    store.reject("plazo", reviewed_by="javier", reason="mala redacción", now=_NOW)

    assert store.get("plazo").status is ReviewStatus.REJECTED
    assert not store.materialized_path(candidate).exists()


def test_accepting_an_unknown_candidate_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(StoreError, match="unknown candidate"):
        store.accept("nope", reviewed_by="javier", now=_NOW)


def test_a_candidate_cannot_be_accepted_twice(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(_mode1())
    store.accept("plazo", reviewed_by="javier", now=_NOW)

    with pytest.raises(StoreError, match="not pending"):
        store.accept("plazo", reviewed_by="javier", now=_NOW)


def test_an_accepted_candidate_cannot_be_overwritten_by_regeneration(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(_mode1())
    store.accept("plazo", reviewed_by="javier", now=_NOW)

    with pytest.raises(StoreError, match="already accepted"):
        store.write(_mode1())


def test_a_mode2_contract_materializes_into_the_refset_suite(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(_mode2())

    path = store.accept("contrato-01", reviewed_by="javier", now=_NOW)

    assert path.name == "contrato-01.json"
    assert "modo2" in path.parts


def test_list_pending_filters_by_kind(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write(_mode1("plazo"))
    store.write(_mode2("contrato-01"))

    assert [c.id for c in store.list_pending(CaseKind.MODE1)] == ["plazo"]
    assert [c.id for c in store.list_pending(CaseKind.MODE2)] == ["contrato-01"]
