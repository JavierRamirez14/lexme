"""The checklist loader validates structure and rejects malformed packages."""

import json
from pathlib import Path

import pytest

from lexme.checklist import (
    Checklist,
    ChecklistError,
    RuleCharacter,
    SilenceTone,
    load_checklist,
)
from tests.checklist.conftest import CHECKLIST_PATH

_VALID_ITEM = {
    "id": "CHK-01",
    "right": "Plazo mínimo con prórroga obligatoria.",
    "anchors": ["a9"],
    "character": "imperativo",
    "silence_tone": "ex_lege_con_carga",
    "absence_template": "La ley te garantiza el plazo mínimo aunque el contrato calle.",
    "citation": {"block_id": "a9", "text": "se prorrogará obligatoriamente por plazos anuales"},
}


def _write(tmp_path: Path, payload: object) -> Path:
    """Write ``payload`` as JSON to a temp checklist file and return its path."""
    path = tmp_path / "checklist.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _package(**overrides: object) -> dict:
    """A minimal valid checklist package, with optional top-level overrides."""
    base = {"vertical": "vivienda", "norm_id": "BOE-A-1994-26003", "items": [_VALID_ITEM]}
    return base | overrides


def test_loads_a_well_formed_package_into_typed_items(tmp_path: Path) -> None:
    checklist = load_checklist(_write(tmp_path, _package()))

    assert isinstance(checklist, Checklist)
    (item,) = checklist.items
    assert item.id == "CHK-01"
    assert item.anchors == ("a9",)
    assert item.character is RuleCharacter.IMPERATIVE
    assert item.silence_tone is SilenceTone.EX_LEGE_WITH_BURDEN
    assert item.citation.block_id == "a9"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ChecklistError, match="not found"):
        load_checklist(tmp_path / "absent.json")


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "checklist.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ChecklistError, match="not valid JSON"):
        load_checklist(path)


def test_empty_items_array_raises(tmp_path: Path) -> None:
    with pytest.raises(ChecklistError, match="non-empty 'items'"):
        load_checklist(_write(tmp_path, _package(items=[])))


def test_duplicate_item_id_raises(tmp_path: Path) -> None:
    with pytest.raises(ChecklistError, match="duplicate item id"):
        load_checklist(_write(tmp_path, _package(items=[_VALID_ITEM, _VALID_ITEM])))


def test_unknown_character_raises(tmp_path: Path) -> None:
    item = _VALID_ITEM | {"character": "obligatorio"}
    with pytest.raises(ChecklistError, match="unknown character"):
        load_checklist(_write(tmp_path, _package(items=[item])))


def test_unknown_silence_tone_raises(tmp_path: Path) -> None:
    item = _VALID_ITEM | {"silence_tone": "neutro"}
    with pytest.raises(ChecklistError, match="unknown silence_tone"):
        load_checklist(_write(tmp_path, _package(items=[item])))


def test_citation_block_not_among_anchors_raises(tmp_path: Path) -> None:
    item = _VALID_ITEM | {"citation": {"block_id": "a6", "text": "algo"}}
    with pytest.raises(ChecklistError, match="not among the item's anchors"):
        load_checklist(_write(tmp_path, _package(items=[item])))


def test_missing_required_field_raises(tmp_path: Path) -> None:
    item = {key: value for key, value in _VALID_ITEM.items() if key != "right"}
    with pytest.raises(ChecklistError, match="'right'"):
        load_checklist(_write(tmp_path, _package(items=[item])))


def test_ships_the_twenty_lau_items_fully_declared() -> None:
    checklist = load_checklist(CHECKLIST_PATH)

    assert checklist.vertical == "vivienda"
    assert checklist.norm_id == "BOE-A-1994-26003"
    assert [item.id for item in checklist.items] == [f"CHK-{n:02d}" for n in range(1, 21)]
    for item in checklist.items:
        assert item.right
        assert item.anchors
        assert item.absence_template
        assert item.citation.block_id in item.anchors
        assert isinstance(item.character, RuleCharacter)
        assert isinstance(item.silence_tone, SilenceTone)
