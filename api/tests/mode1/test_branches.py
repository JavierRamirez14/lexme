"""Tests for loading a vertical's critical-branch package."""

import json
from pathlib import Path

import pytest

from lexme.mode1.branches import AnswerKind, BranchesError, load_branches

REPO_ROOT = Path(__file__).resolve().parents[3]
VIVIENDA_BRANCHES = REPO_ROOT / "verticales" / "vivienda" / "disambiguation.json"

VALID = {
    "vertical": "vivienda",
    "branches": [
        {
            "id": "fecha_firma",
            "question": "¿Cuándo firmaste el contrato?",
            "answer_kind": "fecha",
            "sets_target_date": True,
            "fact_template": "El contrato se firmó el {answer}.",
            "assumption": "Asumo un contrato firmado hoy.",
        }
    ],
}


def write(tmp_path: Path, payload: object) -> Path:
    """Write ``payload`` as a branches file and return its path."""
    path = tmp_path / "disambiguation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_valid_package_loads_its_branches_in_declared_order(tmp_path: Path) -> None:
    payload = {
        "vertical": "vivienda",
        "branches": [
            VALID["branches"][0],
            {
                "id": "uso_vivienda",
                "question": "¿Es tu vivienda habitual?",
                "answer_kind": "texto",
                "sets_target_date": False,
                "fact_template": "El inquilino indica: {answer}.",
                "assumption": "Asumo vivienda habitual.",
            },
        ],
    }

    branches = load_branches(write(tmp_path, payload))

    assert [branch.id for branch in branches] == ["fecha_firma", "uso_vivienda"]
    assert branches[0].answer_kind is AnswerKind.DATE
    assert branches[0].sets_target_date is True
    assert branches[1].answer_kind is AnswerKind.TEXT


def test_a_missing_package_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(BranchesError):
        load_branches(tmp_path / "absent.json")


def test_a_malformed_package_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "disambiguation.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(BranchesError):
        load_branches(path)


def test_a_branch_without_an_id_is_an_error(tmp_path: Path) -> None:
    payload = {"vertical": "vivienda", "branches": [{"question": "¿Cuándo?"}]}

    with pytest.raises(BranchesError):
        load_branches(write(tmp_path, payload))


def test_an_unknown_answer_kind_is_an_error(tmp_path: Path) -> None:
    branch = {**VALID["branches"][0], "answer_kind": "numero"}

    with pytest.raises(BranchesError):
        load_branches(write(tmp_path, {"vertical": "vivienda", "branches": [branch]}))


def test_a_date_branch_must_be_answerable_with_a_date(tmp_path: Path) -> None:
    branch = {**VALID["branches"][0], "answer_kind": "texto", "sets_target_date": True}

    with pytest.raises(BranchesError):
        load_branches(write(tmp_path, {"vertical": "vivienda", "branches": [branch]}))


def test_a_fact_template_without_the_answer_placeholder_is_an_error(tmp_path: Path) -> None:
    branch = {**VALID["branches"][0], "fact_template": "El contrato ya se firmó."}

    with pytest.raises(BranchesError):
        load_branches(write(tmp_path, {"vertical": "vivienda", "branches": [branch]}))


def test_duplicate_branch_ids_are_an_error(tmp_path: Path) -> None:
    branch = VALID["branches"][0]

    with pytest.raises(BranchesError):
        load_branches(write(tmp_path, {"vertical": "vivienda", "branches": [branch, branch]}))


def test_the_shipped_vivienda_package_loads() -> None:
    branches = load_branches(VIVIENDA_BRANCHES)

    assert branches
    assert all(branch.question and branch.assumption for branch in branches)
    assert any(branch.sets_target_date for branch in branches)


def test_a_branch_renders_the_users_answer_as_a_case_fact() -> None:
    branches = load_branches(VIVIENDA_BRANCHES)
    signing = next(branch for branch in branches if branch.sets_target_date)

    assert "2017-05-04" in signing.state_fact("2017-05-04")
