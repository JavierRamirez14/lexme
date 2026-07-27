"""Load a vertical's ingestion manifest.

A vertical is a data package outside the engine's source tree. The engine only
ever sees an opaque vertical name and the norms to ingest for it; it never
branches on which vertical it is. A norm may contribute its whole text or only the
blocks the manifest names, because the corpus is meant to hold what a user of the
vertical actually asks about rather than every article of every law it touches.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

_DIGEST_SEPARATOR = "|"
_ALL_BLOCKS = "*"


class ManifestError(ValueError):
    """Raised when a manifest file is missing or malformed."""


@dataclass(frozen=True)
class NormSelection:
    """One norm the vertical ingests: its id, its short name, and how much of it.

    ``label`` is the short name the corpus is presented under, next to the BOE's
    own long title. ``block_ids`` names the blocks to keep, or is ``None`` to keep
    every precepto block of the norm.
    """

    norm_id: str
    label: str
    block_ids: tuple[str, ...] | None

    @property
    def digest(self) -> str:
        """A fingerprint of this selection, so a manifest edit forces a re-ingest.

        Ingestion otherwise skips a norm whose BOE text has not moved, which would
        silently ignore a widened or narrowed block list. Block order does not
        change what is ingested, so it does not change the digest.
        """
        blocks = (
            _ALL_BLOCKS
            if self.block_ids is None
            else _DIGEST_SEPARATOR.join(sorted(self.block_ids))
        )
        payload = _DIGEST_SEPARATOR.join((self.label, blocks))
        return hashlib.md5(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VerticalManifest:
    """A vertical name plus the ordered norm selections to ingest for it."""

    vertical: str
    norms: tuple[NormSelection, ...]


def load_manifest(path: Path) -> VerticalManifest:
    """Read and validate a manifest JSON file into a :class:`VerticalManifest`.

    The file must contain a non-empty ``vertical`` string and a ``norms`` array of
    objects each carrying a ``norm_id``, a ``label`` and, optionally, a non-empty
    ``blocks`` array. Raises :class:`ManifestError` on any structural problem,
    including the same norm declared twice.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestError(f"manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ManifestError(f"manifest {path} is not valid JSON: {error}") from error

    vertical = raw.get("vertical")
    if not isinstance(vertical, str) or not vertical:
        raise ManifestError(f"manifest {path} missing a non-empty 'vertical'")

    entries = raw.get("norms")
    if not isinstance(entries, list) or not entries:
        raise ManifestError(f"manifest {path} missing a non-empty 'norms' array")

    norms = tuple(_read_selection(entry, path) for entry in entries)
    _reject_duplicate_norms(norms, path)
    return VerticalManifest(vertical=vertical, norms=norms)


def _read_selection(entry: object, path: Path) -> NormSelection:
    """Build one :class:`NormSelection` from a manifest entry."""
    if not isinstance(entry, dict):
        raise ManifestError(f"manifest {path} has a non-object norm entry: {entry!r}")
    return NormSelection(
        norm_id=_read_text(entry, "norm_id", path),
        label=_read_text(entry, "label", path),
        block_ids=_read_block_ids(entry, path),
    )


def _read_text(entry: dict, field: str, path: Path) -> str:
    """Read a required non-empty string field from a norm entry."""
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"manifest {path} has a norm entry without '{field}': {entry!r}")
    return value.strip()


def _read_block_ids(entry: dict, path: Path) -> tuple[str, ...] | None:
    """Read the optional ``blocks`` array, or ``None`` when the whole norm is taken.

    An empty array is rejected rather than read as "the whole norm": it is far more
    likely a mistake than a deliberate way to say "everything".
    """
    value = entry.get("blocks")
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ManifestError(f"manifest {path} has an empty or non-array 'blocks': {entry!r}")
    for block_id in value:
        if not isinstance(block_id, str) or not block_id.strip():
            raise ManifestError(f"manifest {path} has a non-string block id: {block_id!r}")
    return tuple(block_id.strip() for block_id in value)


def _reject_duplicate_norms(norms: tuple[NormSelection, ...], path: Path) -> None:
    """Fail when a norm is declared twice, since one selection would silently win."""
    seen: set[str] = set()
    for norm in norms:
        if norm.norm_id in seen:
            raise ManifestError(f"manifest {path} declares norm '{norm.norm_id}' twice")
        seen.add(norm.norm_id)
