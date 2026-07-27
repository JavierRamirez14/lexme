"""Tests for manifest loading and validation."""

import json
from pathlib import Path

import pytest

from lexme.ingestion.manifest import ManifestError, load_manifest
from tests.ingestion.conftest import VIVIENDA_MANIFEST


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest(tmp_path: Path, *norms: dict) -> Path:
    return _write(tmp_path, {"vertical": "vivienda", "norms": list(norms)})


def test_loads_vertical_and_norm_selections(tmp_path: Path) -> None:
    path = _manifest(
        tmp_path, {"norm_id": "A", "label": "First"}, {"norm_id": "B", "label": "Second"}
    )

    manifest = load_manifest(path)

    assert manifest.vertical == "vivienda"
    assert [norm.norm_id for norm in manifest.norms] == ["A", "B"]
    assert [norm.label for norm in manifest.norms] == ["First", "Second"]


def test_a_norm_without_a_block_list_takes_the_whole_norm(tmp_path: Path) -> None:
    path = _manifest(tmp_path, {"norm_id": "A", "label": "First"})

    (norm,) = load_manifest(path).norms

    assert norm.block_ids is None


def test_a_norm_may_declare_the_blocks_it_contributes(tmp_path: Path) -> None:
    path = _manifest(tmp_path, {"norm_id": "A", "label": "First", "blocks": ["a1", "a2"]})

    (norm,) = load_manifest(path).norms

    assert norm.block_ids == ("a1", "a2")


def test_missing_file_raises() -> None:
    with pytest.raises(ManifestError):
        load_manifest(Path("does-not-exist.json"))


def test_empty_norms_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, {"vertical": "vivienda", "norms": []})
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_norm_entry_without_id_raises(tmp_path: Path) -> None:
    path = _manifest(tmp_path, {"eli": "x", "label": "First"})
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_norm_entry_without_label_raises(tmp_path: Path) -> None:
    path = _manifest(tmp_path, {"norm_id": "A"})
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_an_empty_block_list_raises(tmp_path: Path) -> None:
    path = _manifest(tmp_path, {"norm_id": "A", "label": "First", "blocks": []})
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_the_same_norm_declared_twice_raises(tmp_path: Path) -> None:
    path = _manifest(tmp_path, {"norm_id": "A", "label": "First"}, {"norm_id": "A", "label": "Bis"})
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_the_digest_changes_when_the_declared_blocks_change(tmp_path: Path) -> None:
    narrow = load_manifest(_manifest(tmp_path, {"norm_id": "A", "label": "L", "blocks": ["a1"]}))
    wide = load_manifest(
        _manifest(tmp_path, {"norm_id": "A", "label": "L", "blocks": ["a1", "a2"]})
    )

    assert narrow.norms[0].digest != wide.norms[0].digest


def test_the_digest_changes_when_the_label_changes(tmp_path: Path) -> None:
    before = load_manifest(_manifest(tmp_path, {"norm_id": "A", "label": "Old"}))
    after = load_manifest(_manifest(tmp_path, {"norm_id": "A", "label": "New"}))

    assert before.norms[0].digest != after.norms[0].digest


def test_the_digest_ignores_the_order_the_blocks_are_listed_in(tmp_path: Path) -> None:
    ascending = load_manifest(
        _manifest(tmp_path, {"norm_id": "A", "label": "L", "blocks": ["a1", "a2"]})
    )
    descending = load_manifest(
        _manifest(tmp_path, {"norm_id": "A", "label": "L", "blocks": ["a2", "a1"]})
    )

    assert ascending.norms[0].digest == descending.norms[0].digest


def test_shipped_vivienda_manifest_is_valid() -> None:
    manifest = load_manifest(VIVIENDA_MANIFEST)

    assert manifest.vertical == "vivienda"
    assert "BOE-A-1994-26003" in [norm.norm_id for norm in manifest.norms]


def test_shipped_vivienda_manifest_spans_more_than_one_norm() -> None:
    manifest = load_manifest(VIVIENDA_MANIFEST)

    assert len(manifest.norms) > 1
