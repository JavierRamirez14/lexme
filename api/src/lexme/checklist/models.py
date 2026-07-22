"""Load a vertical's legal-default checklist as machine-readable data.

Which protections a tenancy holds by default, and how to read a clause that
touches each one, is vertical knowledge -- so it ships as data inside the
vertical package next to its manifest, never as branching on "vivienda" in the
engine. Each item carries its anchors into the corpus, the character of the
concrete rule, the tone of the silence a missing clause produces, the static
text of the absence finding, and the literal citation that backs the item.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

_EnumT = TypeVar("_EnumT", bound=StrEnum)

CHECKLIST_FILENAME = "checklist.json"


def checklist_path(verticals_dir: str | Path, vertical: str) -> Path:
    """Return the path to a vertical's checklist package inside the data directory."""
    return Path(verticals_dir) / vertical / CHECKLIST_FILENAME


class ChecklistError(ValueError):
    """Raised when a vertical's checklist package is missing or malformed."""


class RuleCharacter(StrEnum):
    """How binding the concrete rule is, which fixes the level a worsening clause fires.

    Read per rule, not per Title: the LAU's Title II is a one-way floor, so the
    same article can hold an imperative core and a dispositive accessory.
    """

    IMPERATIVE = "imperativo"
    DISPOSITIVE = "dispositivo"
    MIXED = "mixto"


class SilenceTone(StrEnum):
    """The tone of the absence finding when no clause mentions the item.

    ``FAVORABLE`` -- the protection stays intact and silence favours the tenant.
    ``EX_LEGE_INFORMATIVE`` -- the right applies by law; the finding only informs.
    ``EX_LEGE_WITH_BURDEN`` -- the right applies but demands timely action or it
    is lost. ``NOT_APPLICABLE`` -- a closing meta-rule that never produces an
    absence finding.
    """

    FAVORABLE = "favorable"
    EX_LEGE_INFORMATIVE = "ex_lege_informativo"
    EX_LEGE_WITH_BURDEN = "ex_lege_con_carga"
    NOT_APPLICABLE = "no_aplica"


@dataclass(frozen=True)
class ChecklistCitation:
    """The literal quote that backs an item, anchored to one corpus block.

    ``block_id`` must be one of the item's anchors; ``text`` is verbatim law that
    must verify against that block's point-in-time redaction.
    """

    block_id: str
    text: str


@dataclass(frozen=True)
class ChecklistItem:
    """One default legal protection, in plain language plus the data the engine reads.

    ``anchors`` are the corpus blocks the item rests on; ``citation`` quotes one
    of them. ``absence_template`` is the static text of the absence finding shown
    when no clause covers the item.
    """

    id: str
    right: str
    anchors: tuple[str, ...]
    character: RuleCharacter
    silence_tone: SilenceTone
    absence_template: str
    citation: ChecklistCitation


@dataclass(frozen=True)
class Checklist:
    """A vertical's ordered checklist, bound to the norm its anchors reference."""

    vertical: str
    norm_id: str
    items: tuple[ChecklistItem, ...]


def load_checklist(path: Path) -> Checklist:
    """Read and validate a vertical's checklist package into a :class:`Checklist`.

    Raises :class:`ChecklistError` on any structural problem: unreadable file,
    invalid JSON, missing or empty required fields, an unknown character or
    silence tone, a duplicate item id, or a citation whose block is not one of
    the item's anchors.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ChecklistError(f"checklist package not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ChecklistError(f"checklist package {path} is not valid JSON: {error}") from error

    if not isinstance(raw, dict):
        raise ChecklistError(f"checklist package {path} must be a JSON object")

    vertical = _read_text(raw, "vertical", path)
    norm_id = _read_text(raw, "norm_id", path)

    entries = raw.get("items")
    if not isinstance(entries, list) or not entries:
        raise ChecklistError(f"checklist package {path} missing a non-empty 'items' array")

    items = tuple(_read_item(entry, path) for entry in entries)
    _reject_duplicates(items, path)
    return Checklist(vertical=vertical, norm_id=norm_id, items=items)


def _read_item(entry: object, path: Path) -> ChecklistItem:
    """Validate one raw entry into a :class:`ChecklistItem`."""
    if not isinstance(entry, dict):
        raise ChecklistError(f"checklist package {path} has a non-object item: {entry!r}")

    anchors = _read_anchors(entry, path)
    citation = _read_citation(entry, path, anchors)
    return ChecklistItem(
        id=_read_text(entry, "id", path),
        right=_read_text(entry, "right", path),
        anchors=anchors,
        character=_read_enum(entry, "character", RuleCharacter, path),
        silence_tone=_read_enum(entry, "silence_tone", SilenceTone, path),
        absence_template=_read_text(entry, "absence_template", path),
        citation=citation,
    )


def _read_anchors(entry: dict, path: Path) -> tuple[str, ...]:
    """Read a non-empty tuple of non-empty block ids from an item entry."""
    value = entry.get("anchors")
    if not isinstance(value, list) or not value:
        raise ChecklistError(f"checklist package {path}: item missing a non-empty 'anchors' array")
    anchors = []
    for anchor in value:
        if not isinstance(anchor, str) or not anchor.strip():
            raise ChecklistError(f"checklist package {path}: item has a non-string anchor")
        anchors.append(anchor.strip())
    return tuple(anchors)


def _read_citation(entry: dict, path: Path, anchors: tuple[str, ...]) -> ChecklistCitation:
    """Read the item's citation and check its block is one of the item's anchors."""
    value = entry.get("citation")
    if not isinstance(value, dict):
        raise ChecklistError(f"checklist package {path}: item missing a 'citation' object")
    block_id = _read_text(value, "block_id", path)
    text = _read_text(value, "text", path)
    if block_id not in anchors:
        raise ChecklistError(
            f"checklist package {path}: citation block '{block_id}' is not among the "
            f"item's anchors {list(anchors)}"
        )
    return ChecklistCitation(block_id=block_id, text=text)


def _read_text(entry: dict, field: str, path: Path) -> str:
    """Read a required non-empty string field from a mapping."""
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ChecklistError(f"checklist package {path}: missing a non-empty '{field}'")
    return value.strip()


def _read_enum(entry: dict, field: str, enum: type[_EnumT], path: Path) -> _EnumT:
    """Read a required field and coerce it to ``enum``, rejecting unknown values."""
    value = entry.get(field)
    try:
        return enum(value)
    except ValueError as error:
        raise ChecklistError(
            f"checklist package {path}: unknown {field} {value!r}; "
            f"expected one of {[member.value for member in enum]}"
        ) from error


def _reject_duplicates(items: tuple[ChecklistItem, ...], path: Path) -> None:
    """Fail when two items share an id, which would make the checklist ambiguous."""
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise ChecklistError(f"checklist package {path}: duplicate item id '{item.id}'")
        seen.add(item.id)
