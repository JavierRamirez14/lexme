"""Loading and validating a vertical's Mode 2 scope package."""

from datetime import date
from pathlib import Path

import pytest

from lexme.mode2 import TenancyUse
from lexme.mode2.scope import ScopeError, load_scope

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIPPED_SCOPE = REPO_ROOT / "verticales" / "vivienda" / "scope.json"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "scope.json"
    path.write_text(body, encoding="utf-8")
    return path


VALID = (
    '{"current_redaction_effective_from": "2019-03-06", '
    '"excluded_uses": ["temporada"], '
    '"use_evidence_markers": {"temporada": ["temporada"]}}'
)


def test_the_shipped_vivienda_scope_loads() -> None:
    scope = load_scope(SHIPPED_SCOPE)

    assert scope.current_redaction_effective_from == date(2019, 3, 6)
    assert scope.excludes(TenancyUse.SEASONAL)
    assert not scope.excludes(TenancyUse.HABITUAL_DWELLING)


def test_the_shipped_markers_tell_a_declared_season_from_a_short_term() -> None:
    scope = load_scope(SHIPPED_SCOPE)

    assert scope.declares(
        TenancyUse.SEASONAL, "El inmueble se arrienda con finalidad de temporada estival"
    )
    assert scope.declares(TenancyUse.NON_DWELLING, "El local se destina a oficina")
    assert not scope.declares(
        TenancyUse.SEASONAL,
        "duración de once meses, sin derecho a prórroga alguna",
    )


def test_an_annex_of_a_dwelling_is_not_a_declaration_of_a_non_dwelling_use() -> None:
    scope = load_scope(SHIPPED_SCOPE)

    assert not scope.declares(
        TenancyUse.NON_DWELLING,
        "La vivienda se arrienda con plaza de garaje y trastero",
    )


def test_markers_are_read_through_case_and_accents() -> None:
    scope = load_scope(SHIPPED_SCOPE)

    assert scope.declares(TenancyUse.SEASONAL, "ARRENDAMIENTO DE TEMPORADA TURISTICA")


def test_a_use_the_scope_does_not_exclude_declares_nothing() -> None:
    scope = load_scope(SHIPPED_SCOPE)

    assert not scope.declares(TenancyUse.HABITUAL_DWELLING, "vivienda habitual del arrendatario")


def test_a_missing_file_is_a_scope_error(tmp_path: Path) -> None:
    with pytest.raises(ScopeError):
        load_scope(tmp_path / "absent.json")


def test_an_invalid_date_is_a_scope_error(tmp_path: Path) -> None:
    path = _write(tmp_path, VALID.replace('"2019-03-06"', '"nope"'))

    with pytest.raises(ScopeError):
        load_scope(path)


def test_an_unknown_use_is_a_scope_error(tmp_path: Path) -> None:
    path = _write(tmp_path, VALID.replace('["temporada"], ', '["marciano"], '))

    with pytest.raises(ScopeError):
        load_scope(path)


def test_an_excluded_use_with_no_markers_is_a_scope_error(tmp_path: Path) -> None:
    path = _write(tmp_path, VALID.replace('{"temporada": ["temporada"]}', "{}"))

    with pytest.raises(ScopeError):
        load_scope(path)


def test_an_empty_marker_list_is_a_scope_error(tmp_path: Path) -> None:
    path = _write(tmp_path, VALID.replace('["temporada"]}}', "[]}}"))

    with pytest.raises(ScopeError):
        load_scope(path)
