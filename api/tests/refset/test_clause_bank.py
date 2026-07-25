"""Tests for the clause bank loader and its coverage check against a checklist."""

import json
from pathlib import Path

import pytest

from lexme.checklist import SilenceTone
from lexme.refset.clause_bank import (
    ClauseBankError,
    load_clause_bank,
    validate_bank_coverage,
)
from tests.refset.conftest import make_checklist, make_item

_NORM = "BOE-A-1994-26003"


def _write_bank(path: Path, clauses: list[dict]) -> Path:
    payload = {"vertical": "vivienda", "norm_id": _NORM, "clauses": clauses}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _clause(clause_id: str, level: str, chk_ids: list[str]) -> dict:
    return {
        "id": clause_id,
        "heading": "H",
        "text": "clause text",
        "expected_level": level,
        "chk_ids": chk_ids,
        "source": "consumo",
    }


def test_a_well_formed_bank_loads(tmp_path: Path) -> None:
    path = _write_bank(tmp_path / "bank.json", [_clause("BANK-1", "ilegal", ["CHK-01"])])

    bank = load_clause_bank(path)

    assert bank.by_id("BANK-1").expected_level.value == "ilegal"


def test_a_negotiable_clause_with_checklist_items_is_rejected(tmp_path: Path) -> None:
    path = _write_bank(tmp_path / "bank.json", [_clause("BANK-1", "negociable", ["CHK-01"])])

    with pytest.raises(ClauseBankError, match="off the checklist"):
        load_clause_bank(path)


def test_a_duplicate_clause_id_is_rejected(tmp_path: Path) -> None:
    path = _write_bank(
        tmp_path / "bank.json",
        [_clause("BANK-1", "ilegal", ["CHK-01"]), _clause("BANK-1", "correcto", ["CHK-01"])],
    )

    with pytest.raises(ClauseBankError, match="duplicate clause id"):
        load_clause_bank(path)


def test_full_coverage_reports_valid(tmp_path: Path) -> None:
    checklist = make_checklist(make_item("CHK-01"), make_item("CHK-20", SilenceTone.NOT_APPLICABLE))
    path = _write_bank(
        tmp_path / "bank.json",
        [
            _clause("BANK-illegal", "ilegal", ["CHK-01"]),
            _clause("BANK-worse", "peor_que_default", ["CHK-01"]),
            _clause("BANK-ok", "correcto", ["CHK-01"]),
            _clause("BANK-neg", "negociable", []),
        ],
    )

    report = validate_bank_coverage(load_clause_bank(path), checklist)

    assert report.is_valid


def test_an_uncovered_checklist_item_is_reported(tmp_path: Path) -> None:
    checklist = make_checklist(make_item("CHK-01"), make_item("CHK-02"))
    path = _write_bank(
        tmp_path / "bank.json",
        [
            _clause("BANK-illegal", "ilegal", ["CHK-01"]),
            _clause("BANK-worse", "peor_que_default", ["CHK-01"]),
            _clause("BANK-ok", "correcto", ["CHK-01"]),
            _clause("BANK-neg", "negociable", []),
        ],
    )

    report = validate_bank_coverage(load_clause_bank(path), checklist)

    assert not report.is_valid
    assert any("CHK-02" in finding.message for finding in report.findings)


def test_a_missing_citing_level_is_reported(tmp_path: Path) -> None:
    checklist = make_checklist(make_item("CHK-01"))
    path = _write_bank(
        tmp_path / "bank.json",
        [_clause("BANK-ok", "correcto", ["CHK-01"]), _clause("BANK-neg", "negociable", [])],
    )

    report = validate_bank_coverage(load_clause_bank(path), checklist)

    messages = " ".join(finding.message for finding in report.findings)
    assert "ilegal" in messages
    assert "peor_que_default" in messages


def test_a_missing_negotiable_clause_is_reported(tmp_path: Path) -> None:
    checklist = make_checklist(make_item("CHK-01"))
    path = _write_bank(
        tmp_path / "bank.json",
        [
            _clause("BANK-illegal", "ilegal", ["CHK-01"]),
            _clause("BANK-worse", "peor_que_default", ["CHK-01"]),
            _clause("BANK-ok", "correcto", ["CHK-01"]),
        ],
    )

    report = validate_bank_coverage(load_clause_bank(path), checklist)

    assert any("negotiable" in finding.message for finding in report.findings)


def test_a_clause_citing_an_unknown_item_is_reported(tmp_path: Path) -> None:
    checklist = make_checklist(make_item("CHK-01"))
    path = _write_bank(
        tmp_path / "bank.json",
        [
            _clause("BANK-illegal", "ilegal", ["CHK-99"]),
            _clause("BANK-worse", "peor_que_default", ["CHK-01"]),
            _clause("BANK-ok", "correcto", ["CHK-01"]),
            _clause("BANK-neg", "negociable", []),
        ],
    )

    report = validate_bank_coverage(load_clause_bank(path), checklist)

    assert any("CHK-99" in finding.message for finding in report.findings)
