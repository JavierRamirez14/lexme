"""The Mode 2 case loader: it accepts the materialized set and rejects malformed data."""

import json
from pathlib import Path

import pytest

from lexme.eval.mode2.cases import Mode2CasesError, load_mode2_cases

REFSET_MODO2 = Path(__file__).resolve().parents[4] / "verticales" / "vivienda" / "refset" / "modo2"

VALID = {
    "id": "contract-01",
    "document": "PRIMERA. Una cláusula ilegal de ejemplo con texto suficiente.",
    "clauses": [
        {
            "clause_id": "c1",
            "heading": "PRIMERA",
            "text": "Una cláusula ilegal",
            "start": 9,
            "end": 28,
            "expected_level": "ilegal",
            "chk_ids": ["CHK-01"],
        }
    ],
    "expected_absences": [{"item_id": "CHK-02", "right": "Prórroga tácita"}],
}


def _write(tmp_path: Path, payload: dict, name: str = "case.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_valid_case_loads_its_clauses_and_absences(tmp_path: Path) -> None:
    (case,) = load_mode2_cases(_write(tmp_path, VALID))

    assert case.id == "contract-01"
    assert case.clauses[0].expected_level.value == "ilegal"
    assert case.clauses[0].chk_ids == ("CHK-01",)
    assert case.expected_absences[0].item_id == "CHK-02"


def test_the_materialized_reference_set_loads(tmp_path: Path) -> None:
    if not REFSET_MODO2.is_dir():
        pytest.skip("materialized Mode 2 reference set not present")
    cases = load_mode2_cases(REFSET_MODO2)

    assert cases  # at least one accepted contract
    assert all(case.document for case in cases)
    assert all(case.clauses for case in cases)


def test_an_absent_level_on_a_clause_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(json.dumps(VALID))
    payload["clauses"][0]["expected_level"] = "ausente"

    with pytest.raises(Mode2CasesError, match="not a clause level"):
        load_mode2_cases(_write(tmp_path, payload))


def test_a_span_outside_the_document_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(json.dumps(VALID))
    payload["clauses"][0]["end"] = 10_000

    with pytest.raises(Mode2CasesError, match="outside the document"):
        load_mode2_cases(_write(tmp_path, payload))


def test_a_non_string_heading_is_rejected_rather_than_coerced(tmp_path: Path) -> None:
    payload = json.loads(json.dumps(VALID))
    payload["clauses"][0]["heading"] = 123

    with pytest.raises(Mode2CasesError, match="must be a string when present"):
        load_mode2_cases(_write(tmp_path, payload))


def test_a_missing_document_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(json.dumps(VALID))
    del payload["document"]

    with pytest.raises(Mode2CasesError, match="document"):
        load_mode2_cases(_write(tmp_path, payload))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    _write(tmp_path, VALID, "a.json")
    _write(tmp_path, VALID, "b.json")

    with pytest.raises(Mode2CasesError, match="duplicate case id"):
        load_mode2_cases(tmp_path)


def test_an_empty_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(Mode2CasesError, match="no '\\*.json' cases"):
        load_mode2_cases(tmp_path)
