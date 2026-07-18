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


def test_loads_vertical_and_norm_ids(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {"vertical": "vivienda", "norms": [{"norm_id": "A"}, {"norm_id": "B"}]},
    )

    manifest = load_manifest(path)

    assert manifest.vertical == "vivienda"
    assert manifest.norm_ids == ("A", "B")


def test_missing_file_raises() -> None:
    with pytest.raises(ManifestError):
        load_manifest(Path("does-not-exist.json"))


def test_empty_norms_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, {"vertical": "vivienda", "norms": []})
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_norm_entry_without_id_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, {"vertical": "vivienda", "norms": [{"eli": "x"}]})
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_shipped_vivienda_manifest_is_valid() -> None:
    manifest = load_manifest(VIVIENDA_MANIFEST)

    assert manifest.vertical == "vivienda"
    assert "BOE-A-1994-26003" in manifest.norm_ids
