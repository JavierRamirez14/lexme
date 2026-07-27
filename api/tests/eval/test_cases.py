"""Tests for the eval case loader."""

import json
from datetime import date
from pathlib import Path

import pytest

from lexme.eval.cases import (
    CasesError,
    ClarificationAnswer,
    EvalCase,
    answer_for_branch,
    load_cases,
    reject_unknown_branches,
)


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


def test_a_case_loads_its_key_points_and_target_date(tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "kp.json",
        {
            "id": "fianza",
            "question": "¿cuánta fianza?",
            "gold_block_ids": ["a36"],
            "key_points": [{"claim": "la fianza es una mensualidad", "block_id": "a36"}],
            "target_date": "2020-01-01",
        },
    )

    (case,) = load_cases(tmp_path)

    assert case.target_date == date(2020, 1, 1)
    assert case.key_points[0].claim == "la fianza es una mensualidad"
    assert case.key_points[0].block_id == "a36"


def test_a_case_without_key_points_or_target_date_defaults_them(tmp_path: Path) -> None:
    _write_case(tmp_path, "bare.json", {"id": "x", "question": "q"})

    (case,) = load_cases(tmp_path)

    assert case.key_points == ()
    assert case.target_date is None


def test_a_null_target_date_is_treated_as_absent(tmp_path: Path) -> None:
    _write_case(tmp_path, "n.json", {"id": "x", "question": "q", "target_date": None})

    (case,) = load_cases(tmp_path)

    assert case.target_date is None


def test_a_key_point_off_the_gold_blocks_is_rejected(tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "bad.json",
        {
            "id": "x",
            "question": "q",
            "gold_block_ids": ["a36"],
            "key_points": [{"claim": "c", "block_id": "a99"}],
        },
    )

    with pytest.raises(CasesError, match="key point"):
        load_cases(tmp_path)


def test_a_malformed_target_date_is_rejected(tmp_path: Path) -> None:
    _write_case(tmp_path, "bad.json", {"id": "x", "question": "q", "target_date": "ayer"})

    with pytest.raises(CasesError, match="target_date"):
        load_cases(tmp_path)


def test_a_key_point_missing_its_claim_is_rejected(tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "bad.json",
        {"id": "x", "question": "q", "gold_block_ids": ["a1"], "key_points": [{"block_id": "a1"}]},
    )

    with pytest.raises(CasesError, match="key point"):
        load_cases(tmp_path)


def test_a_case_loads_the_answer_it_pins_for_each_critical_branch(tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "pinned.json",
        {
            "id": "x",
            "question": "q",
            "clarification_answers": [
                {"branch_id": "fecha_firma", "answer": "12/06/2025"},
                {"branch_id": "uso_vivienda", "answer": "Es mi vivienda habitual."},
            ],
        },
    )

    (case,) = load_cases(tmp_path)

    assert answer_for_branch(case.clarification_answers, "fecha_firma") == "12/06/2025"
    assert (
        answer_for_branch(case.clarification_answers, "uso_vivienda") == "Es mi vivienda habitual."
    )


def test_a_case_pins_no_answer_for_a_branch_it_does_not_declare(tmp_path: Path) -> None:
    _write_case(tmp_path, "bare.json", {"id": "x", "question": "q"})

    (case,) = load_cases(tmp_path)

    assert case.clarification_answers == ()
    assert answer_for_branch(case.clarification_answers, "fecha_firma") is None


def test_a_blank_pinned_answer_is_rejected(tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "bad.json",
        {
            "id": "x",
            "question": "q",
            "clarification_answers": [{"branch_id": "fecha_firma", "answer": "  "}],
        },
    )

    with pytest.raises(CasesError, match="clarification answer"):
        load_cases(tmp_path)


def test_an_answer_pinned_to_a_branch_the_vertical_does_not_declare_is_rejected() -> None:
    case = EvalCase(
        id="x",
        question="q",
        clarification_answers=(ClarificationAnswer(branch_id="fecha_firmaa", answer="12/06/2025"),),
    )

    with pytest.raises(CasesError, match="fecha_firmaa"):
        reject_unknown_branches([case], ("fecha_firma", "uso_vivienda"))


def test_answers_pinned_to_declared_branches_pass_the_branch_check() -> None:
    case = EvalCase(
        id="x",
        question="q",
        clarification_answers=(ClarificationAnswer(branch_id="fecha_firma", answer="12/06/2025"),),
    )

    reject_unknown_branches([case], ("fecha_firma", "uso_vivienda"))


def test_two_pinned_answers_for_the_same_branch_are_rejected(tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "bad.json",
        {
            "id": "x",
            "question": "q",
            "clarification_answers": [
                {"branch_id": "fecha_firma", "answer": "12/06/2025"},
                {"branch_id": "fecha_firma", "answer": "01/01/2024"},
            ],
        },
    )

    with pytest.raises(CasesError, match="clarification answer"):
        load_cases(tmp_path)
