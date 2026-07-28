"""The vertical's Mode 2 scope package: the art 4.2 exclusions and the temporal edge.

Which uses fall outside the LAU's housing regime (art 4.2), what language actually
declares one of them, and from which date the current redaction applies are all
vertical knowledge, not engine knowledge, so they ship as data inside the vertical
package. The engine only ever sees the three questions the gates ask: is this use
excluded, does this span of the document declare it, and is this signing date
before the current redaction took effect. The gate logic reads this package; it
never hardcodes a date, a use or a word.
"""

import json
import unicodedata
from collections.abc import Mapping
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
    regime, and ``use_evidence_markers`` are the words each of those uses is
    actually declared with -- held case- and accent-folded, the form
    :meth:`declares` compares in -- so the scope gate can tell a document that says
    "temporada" from one that merely says "once meses".
    """

    current_redaction_effective_from: date
    excluded_uses: frozenset[TenancyUse]
    use_evidence_markers: Mapping[TenancyUse, tuple[str, ...]]

    def excludes(self, use: TenancyUse) -> bool:
        """Whether ``use`` is outside the housing regime under art 4.2."""
        return use in self.excluded_uses

    def declares(self, use: TenancyUse, text: str) -> bool:
        """Whether ``text`` declares ``use``, by carrying one of its markers.

        A lexical test, not a quotation: ``text`` and the markers are compared
        case- and accent-folded, because a heading in capitals declares a seasonal
        let as plainly as a sentence does. A use with no markers -- one the vertical
        does not exclude -- is declared by nothing.
        """
        folded = _fold(text)
        return any(marker in folded for marker in self.use_evidence_markers.get(use, ()))

    def is_prior_redaction(self, signing_date: date) -> bool:
        """Whether ``signing_date`` falls under a redaction the analysis cannot cover."""
        return signing_date < self.current_redaction_effective_from


def load_scope(path: Path) -> ScopePackage:
    """Read and validate a vertical's scope package.

    Raises :class:`ScopeError` on a missing file, invalid JSON, a missing or
    malformed date, an unknown use value, or an excluded use with no evidence
    markers behind it -- a scope package that excludes a use it can never recognize
    in a document would leave the gate permanently open without saying so.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ScopeError(f"scope package not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ScopeError(f"scope package {path} is not valid JSON: {error}") from error

    if not isinstance(raw, dict):
        raise ScopeError(f"scope package {path} must be a JSON object")

    excluded_uses = _read_excluded_uses(raw, path)
    return ScopePackage(
        current_redaction_effective_from=_read_date(raw, "current_redaction_effective_from", path),
        excluded_uses=excluded_uses,
        use_evidence_markers=_read_markers(raw, path, excluded_uses),
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


def _read_markers(
    raw: dict, path: Path, excluded_uses: frozenset[TenancyUse]
) -> Mapping[TenancyUse, tuple[str, ...]]:
    """Read the per-use evidence markers, requiring some for every excluded use.

    Markers are stored folded, the same way :meth:`ScopePackage.declares` folds the
    span it tests, so the vertical can write them with their natural accents.
    """
    entries = raw.get("use_evidence_markers", {})
    if not isinstance(entries, dict):
        raise ScopeError(f"scope package {path}: 'use_evidence_markers' must be an object")

    markers: dict[TenancyUse, tuple[str, ...]] = {}
    for key, values in entries.items():
        try:
            use = TenancyUse(key)
        except ValueError as error:
            raise ScopeError(
                f"scope package {path}: unknown use in 'use_evidence_markers': {error}"
            ) from error
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ScopeError(
                f"scope package {path}: markers for '{key}' must be an array of non-empty strings"
            )
        markers[use] = tuple(_fold(value) for value in values)

    for use in sorted(excluded_uses):
        if not markers.get(use):
            raise ScopeError(
                f"scope package {path}: excluded use '{use.value}' has no evidence markers"
            )
    return markers


def _fold(text: str) -> str:
    """Lower-case ``text`` and strip its accents, for marker comparison only."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))
