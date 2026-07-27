"""Ingestion orchestration: fetch, parse the declared blocks, freshness-check, embed, store.

Rebuilding the corpus from scratch and refreshing it incrementally are the same
code path; the only difference is how many norms turn out to need work. A norm
contributes either its whole text or only the blocks its manifest entry names, and
a norm is reprocessed when either the BOE text or that selection has moved.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

import psycopg

from lexme.ingestion import repository
from lexme.ingestion.manifest import ManifestError, NormSelection, VerticalManifest
from lexme.ingestion.models import ConsolidatedNorm
from lexme.ingestion.xml_parsing import parse_norm_xml

logger = logging.getLogger(__name__)


class NormFetcher(Protocol):
    """Anything that can return the XML of a norm by id."""

    def fetch_norm(self, norm_id: str) -> str: ...


class Embedder(Protocol):
    """Anything that can embed a batch of texts, preserving order."""

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class IngestResult:
    """Outcome of an ingestion run: which norms were rewritten vs. skipped."""

    updated: tuple[str, ...]
    unchanged: tuple[str, ...]


def run_ingest(
    conn: psycopg.Connection,
    manifest: VerticalManifest,
    fetcher: NormFetcher,
    embedder: Embedder,
) -> IngestResult:
    """Ingest every norm in ``manifest``, skipping those already stored as declared.

    A norm is reprocessed only when its BOE ``updated_at`` or its manifest
    selection differs from the stored one; unchanged norms cost no embedding
    calls. Each norm is committed on its own, so a norm that fails to fetch --
    the BOE does drop a multi-megabyte download now and then -- costs only itself
    and the next run resumes from where this one stopped. Raises
    :class:`ManifestError` when a manifest names a block the norm does not have.
    """
    updated: list[str] = []
    unchanged: list[str] = []
    for selection in manifest.norms:
        norm = _parse_selected(fetcher.fetch_norm(selection.norm_id), selection)
        if _is_current(conn, selection, norm):
            logger.info("norm %s unchanged; skipping", selection.norm_id)
            unchanged.append(selection.norm_id)
            continue

        embeddings_by_block = {
            block.block_id: embedder.embed_many(
                [version.text_content for version in block.versions]
            )
            for block in norm.blocks
        }
        with conn.transaction():
            repository.replace_norm(
                conn,
                manifest.vertical,
                norm,
                embeddings_by_block,
                label=selection.label,
                selection_digest=selection.digest,
            )
        conn.commit()
        logger.info("norm %s ingested (%d blocks)", selection.norm_id, len(norm.blocks))
        updated.append(selection.norm_id)

    return IngestResult(updated=tuple(updated), unchanged=tuple(unchanged))


def _parse_selected(xml_text: str, selection: NormSelection) -> ConsolidatedNorm:
    """Parse the norm down to the blocks its manifest entry declares, order preserved.

    A selection naming no blocks keeps the whole norm. Raises
    :class:`ManifestError` when the manifest names a block the norm does not have,
    because a silently dropped anchor is exactly the drift that leaves a checklist
    or an eval case pointing at nothing.
    """
    norm = parse_norm_xml(xml_text, selection.block_ids)
    if selection.block_ids is None:
        return norm
    missing = set(selection.block_ids) - {block.block_id for block in norm.blocks}
    if missing:
        raise ManifestError(
            f"norm {selection.norm_id} has no blocks {sorted(missing)}; "
            "the manifest names blocks the BOE text does not contain"
        )
    return norm


def _is_current(conn: psycopg.Connection, selection: NormSelection, norm: ConsolidatedNorm) -> bool:
    """Whether the stored norm already matches both the BOE revision and the manifest."""
    stored = repository.get_norm_state(conn, selection.norm_id)
    if stored is None:
        return False
    return (
        stored.updated_at == norm.metadata.updated_at
        and stored.selection_digest == selection.digest
    )
