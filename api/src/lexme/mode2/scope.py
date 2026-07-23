"""The vertical's Mode 2 scope package: the art 4.2 exclusions and the temporal edge.

Which uses fall outside the LAU's housing regime (art 4.2) and from which date the
current redaction applies are vertical knowledge, not engine knowledge, so they
ship as data inside the vertical package. The engine only ever sees the two
questions the gates ask: is this use excluded, and is this signing date before the
current redaction took effect. The gate logic reads this package; it never
hardcodes a date or a use.
"""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from lexme.mode2.models import TenancyUse

SCOPE_FILENAME = "scope.json"


class ScopeError(ValueError):
    """Raised when a vertical's scope package is missing or malformed."""


@dataclass(frozen=True)
class ScopePackage:
    """The scope rules the two code gates apply, loaded from vertical data.

    ``current_redaction_effective_from`` is the day the redaction the analysis
    supports took effect; a contract signed before it belongs to a superseded
    regime. ``excluded_uses`` are the uses art 4.2 leaves outside the housing
    regime.
    """

    current_redaction_effective_from: date
    excluded_uses: frozenset[TenancyUse]

    def excludes(self, use: TenancyUse) -> bool:
        """Whether ``use`` is outside the housing regime under art 4.2."""
        return use in self.excluded_uses

    def is_prior_redaction(self, signing_date: date) -> bool:
        """Whether ``signing_date`` falls under a redaction the analysis cannot cover."""
        return signing_date < self.current_redaction_effective_from


def load_scope(path: Path) -> ScopePackage:
    """Read and validate a vertical's scope package.

    Raises :class:`ScopeError` on a missing file, invalid JSON, a missing or
    malformed date, or an unknown use value.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ScopeError(f"scope package not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ScopeError(f"scope package {path} is not valid JSON: {error}") from error

    if not isinstance(raw, dict):
        raise ScopeError(f"scope package {path} must be a JSON object")

    return ScopePackage(
        current_redaction_effective_from=_read_date(raw, "current_redaction_effective_from", path),
        excluded_uses=_read_excluded_uses(raw, path),
    )


def _read_date(raw: dict, field: str, path: Path) -> date:
    """Read a required ISO date field from the scope package."""
    value = raw.get(field)
    if not isinstance(value, str):
        raise ScopeError(f"scope package {path}: missing string '{field}'")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ScopeError(f"scope package {path}: '{field}' is not an ISO date: {error}") from error


def _read_excluded_uses(raw: dict, path: Path) -> frozenset[TenancyUse]:
    """Read the excluded-use list, rejecting unknown use values."""
    values = raw.get("excluded_uses")
    if not isinstance(values, list):
        raise ScopeError(f"scope package {path}: 'excluded_uses' must be an array")
    try:
        return frozenset(TenancyUse(value) for value in values)
    except ValueError as error:
        raise ScopeError(
            f"scope package {path}: unknown use in 'excluded_uses': {error}"
        ) from error
