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


def test_the_shipped_vivienda_scope_loads() -> None:
    scope = load_scope(SHIPPED_SCOPE)

    assert scope.current_redaction_effective_from == date(2019, 3, 6)
    assert scope.excludes(TenancyUse.SEASONAL)
    assert not scope.excludes(TenancyUse.HABITUAL_DWELLING)


def test_a_missing_file_is_a_scope_error(tmp_path: Path) -> None:
    with pytest.raises(ScopeError):
        load_scope(tmp_path / "absent.json")


def test_an_invalid_date_is_a_scope_error(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"current_redaction_effective_from": "nope", "excluded_uses": []}')

    with pytest.raises(ScopeError):
        load_scope(path)


def test_an_unknown_use_is_a_scope_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        '{"current_redaction_effective_from": "2019-03-06", "excluded_uses": ["marciano"]}',
    )

    with pytest.raises(ScopeError):
        load_scope(path)
