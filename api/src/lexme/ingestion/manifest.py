"""Load a vertical's ingestion manifest.

A vertical is a data package outside the engine's source tree. The engine only
ever sees an opaque vertical name and a list of BOE norm IDs to ingest; it never
branches on which vertical it is.
"""

import json
from dataclasses import dataclass
from pathlib import Path


class ManifestError(ValueError):
    """Raised when a manifest file is missing or malformed."""


@dataclass(frozen=True)
class VerticalManifest:
    """A vertical name plus the ordered norm IDs to ingest for it."""

    vertical: str
    norm_ids: tuple[str, ...]


def load_manifest(path: Path) -> VerticalManifest:
    """Read and validate a manifest JSON file into a :class:`VerticalManifest`.

    The file must contain a non-empty ``vertical`` string and a ``norms`` array
    of objects each carrying a ``norm_id``. Raises :class:`ManifestError` on any
    structural problem.
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

    norms = raw.get("norms")
    if not isinstance(norms, list) or not norms:
        raise ManifestError(f"manifest {path} missing a non-empty 'norms' array")

    norm_ids = tuple(_read_norm_id(entry, path) for entry in norms)
    return VerticalManifest(vertical=vertical, norm_ids=norm_ids)


def _read_norm_id(entry: object, path: Path) -> str:
    """Extract a non-empty ``norm_id`` from one manifest entry."""
    if not isinstance(entry, dict):
        raise ManifestError(f"manifest {path} has a non-object norm entry: {entry!r}")
    norm_id = entry.get("norm_id")
    if not isinstance(norm_id, str) or not norm_id:
        raise ManifestError(f"manifest {path} has a norm entry without 'norm_id': {entry!r}")
    return norm_id
