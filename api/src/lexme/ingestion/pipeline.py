"""Ingestion orchestration: fetch, parse, freshness-check, embed, store.

Rebuilding the corpus from scratch and refreshing it incrementally are the same
code path; the only difference is how many norms turn out to need work.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

import psycopg

from lexme.ingestion import repository
from lexme.ingestion.manifest import VerticalManifest
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
    """Ingest every norm in ``manifest``, skipping those whose text is unchanged.

    A norm is reprocessed only when its BOE ``updated_at`` differs from the stored
    value; unchanged norms cost no embedding calls. Each norm is written in its
    own transaction.
    """
    updated: list[str] = []
    unchanged: list[str] = []
    for norm_id in manifest.norm_ids:
        norm = parse_norm_xml(fetcher.fetch_norm(norm_id))
        stored_updated_at = repository.get_norm_updated_at(conn, norm_id)
        if stored_updated_at == norm.metadata.updated_at:
            logger.info("norm %s unchanged; skipping", norm_id)
            unchanged.append(norm_id)
            continue

        embeddings_by_block = {
            block.block_id: embedder.embed_many(
                [version.text_content for version in block.versions]
            )
            for block in norm.blocks
        }
        with conn.transaction():
            repository.replace_norm(conn, manifest.vertical, norm, embeddings_by_block)
        logger.info("norm %s ingested (%d blocks)", norm_id, len(norm.blocks))
        updated.append(norm_id)

    return IngestResult(updated=tuple(updated), unchanged=tuple(unchanged))
