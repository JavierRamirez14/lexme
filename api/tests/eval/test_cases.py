"""Tests for the eval case loader."""

import json
from pathlib import Path

import pytest

from lexme.eval.cases import CasesError, load_cases


def _write_case(directory: Path, name: str, payload: dict) -> None:
    """Write a case JSON file into ``directory``."""
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")


def test_a_directory_of_cases_loads_ordered_by_id(tmp_path: Path) -> None:
    _write_case(tmp_path, "b.json", {"id": "fianza", "question": "¿cuánta fianza?"})
    _write_case(
        tmp_path,
        "a.json",
        {"id": "plazo", "question": "¿plazo mínimo?", "gold_block_ids": ["a9"]},
    )

    cases = load_cases(tmp_path)

    assert [case.id for case in cases] == ["fianza", "plazo"]
    assert cases[1].gold_block_ids == ("a9",)


def test_a_single_case_file_loads(tmp_path: Path) -> None:
    _write_case(tmp_path, "one.json", {"id": "x", "question": "q", "expected_outcome": "respuesta"})

    cases = load_cases(tmp_path / "one.json")

    assert len(cases) == 1
    assert cases[0].expected_outcome == "respuesta"


def test_a_missing_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CasesError, match="not found"):
        load_cases(tmp_path / "nope")


def test_an_empty_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CasesError, match="no '\\*.json' cases"):
        load_cases(tmp_path)


def test_a_case_without_a_question_is_rejected(tmp_path: Path) -> None:
    _write_case(tmp_path, "bad.json", {"id": "x", "question": "  "})

    with pytest.raises(CasesError, match="question"):
        load_cases(tmp_path)


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    _write_case(tmp_path, "a.json", {"id": "dup", "question": "q1"})
    _write_case(tmp_path, "b.json", {"id": "dup", "question": "q2"})

    with pytest.raises(CasesError, match="duplicate case id"):
        load_cases(tmp_path)
