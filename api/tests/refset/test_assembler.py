"""Tests for the synthetic contract assembler and its constructed ground truth."""

from datetime import UTC, datetime

import pytest

from lexme.checklist import SilenceTone
from lexme.refset.assembler import AssemblyError, assemble_contract
from lexme.refset.clause_bank import BankClause
from lexme.refset.models import CaseKind
from tests.refset.conftest import make_checklist, make_item

_NOW = datetime(2026, 7, 25, tzinfo=UTC)


def _clause(clause_id: str, heading: str, text: str, level: str, chk_ids: list[str]) -> BankClause:
    return BankClause(
        id=clause_id, heading=heading, text=text, expected_level=level, chk_ids=chk_ids, source="s"
    )


def test_each_clause_span_indexes_its_own_text() -> None:
    checklist = make_checklist(make_item("CHK-01"))
    clauses = [
        _clause("b1", "Fianza", "Tres mensualidades de fianza.", "peor_que_default", ["CHK-01"]),
        _clause("b2", "Duración", "Un año prorrogable.", "correcto", ["CHK-01"]),
    ]

    candidate = assemble_contract("contrato-01", clauses, checklist, now=_NOW, header="CONTRATO")

    assert candidate.kind is CaseKind.MODE2
    case = candidate.mode2
    assert case is not None
    for truth in case.clauses:
        assert case.document[truth.start : truth.end] == truth.text
    assert [truth.expected_level.value for truth in case.clauses] == [
        "peor_que_default",
        "correcto",
    ]
    assert case.document.startswith("CONTRATO")


def test_omitted_checklist_items_become_expected_absences() -> None:
    checklist = make_checklist(make_item("CHK-01"), make_item("CHK-02"), make_item("CHK-03"))
    clauses = [_clause("b1", "Fianza", "una mensualidad", "correcto", ["CHK-01"])]

    candidate = assemble_contract("contrato-02", clauses, checklist, now=_NOW)

    case = candidate.mode2
    assert case is not None
    absent_ids = {absence.item_id for absence in case.expected_absences}
    assert absent_ids == {"CHK-02", "CHK-03"}


def test_the_closing_meta_rule_is_never_an_expected_absence() -> None:
    checklist = make_checklist(make_item("CHK-01"), make_item("CHK-20", SilenceTone.NOT_APPLICABLE))
    clauses = [_clause("b1", "Fianza", "una mensualidad", "correcto", ["CHK-01"])]

    candidate = assemble_contract("contrato-03", clauses, checklist, now=_NOW)

    case = candidate.mode2
    assert case is not None
    assert case.expected_absences == []


def test_provenance_records_the_source_clauses() -> None:
    checklist = make_checklist(make_item("CHK-01"))
    clauses = [_clause("b1", "Fianza", "texto", "correcto", ["CHK-01"])]

    candidate = assemble_contract("contrato-04", clauses, checklist, now=_NOW)

    assert candidate.provenance.sources == ["b1"]
    assert candidate.provenance.generator == "assembler"


def test_an_empty_contract_is_rejected() -> None:
    with pytest.raises(AssemblyError, match="at least one clause"):
        assemble_contract("contrato-05", [], make_checklist(make_item("CHK-01")), now=_NOW)
