"""SQL persistence for the corpus: norms, blocks and their versions.

This is the only module that speaks SQL for the corpus. A norm's blocks are
replaced wholesale on every change (delete-and-reinsert, cascading to versions)
because the BOE API always returns a block's entire version history, never a
delta, so there is nothing to diff against.
"""

from dataclasses import dataclass
from datetime import date, datetime

import psycopg
from psycopg.rows import class_row

from lexme.ingestion.models import ConsolidatedNorm


@dataclass(frozen=True)
class VersionInForce:
    """The redaction of a block in force at a target date."""

    amending_norm_id: str
    effective_date: date
    text_content: str
    html_content: str


@dataclass(frozen=True)
class CitationAnchor:
    """Everything needed to cite a block back to its source law."""

    norm_id: str
    norm_label: str
    eli: str
    consolidated_html_url: str
    block_id: str
    title: str


@dataclass(frozen=True)
class IngestedNorm:
    """A norm as the corpus currently holds it, for the freshness panel and the CLI."""

    norm_id: str
    label: str
    title: str
    consolidated_html_url: str
    updated_at: datetime
    blocks: int


@dataclass(frozen=True)
class NormState:
    """What the corpus already stores about a norm, to decide whether to re-ingest.

    ``updated_at`` is the BOE's own revision stamp and ``selection_digest`` the
    fingerprint of the manifest selection the stored blocks were built from; a
    norm is reprocessed when either has moved.
    """

    updated_at: datetime
    selection_digest: str


def get_norm_state(conn: psycopg.Connection, norm_id: str) -> NormState | None:
    """Return the norm's stored revision and selection, or ``None`` if not stored."""
    with conn.cursor(row_factory=class_row(NormState)) as cursor:
        return cursor.execute(
            "SELECT updated_at, selection_digest FROM norms WHERE id = %s", (norm_id,)
        ).fetchone()


def replace_norm(
    conn: psycopg.Connection,
    vertical: str,
    norm: ConsolidatedNorm,
    embeddings_by_block: dict[str, list[list[float]]],
    *,
    label: str = "",
    selection_digest: str = "",
) -> None:
    """Upsert a norm and rebuild all of its blocks and versions.

    ``embeddings_by_block`` maps each block's ``block_id`` to one embedding per
    version, aligned to :attr:`Block.versions` order. ``label`` is the short name
    the norm is shown under and ``selection_digest`` the manifest fingerprint the
    stored blocks were built from.
    """
    _upsert_norm(conn, vertical, norm, label, selection_digest)
    conn.execute("DELETE FROM blocks WHERE norm_id = %s", (norm.metadata.norm_id,))
    for block in norm.blocks:
        block_row_id = conn.execute(
            """
            INSERT INTO blocks (norm_id, block_id, title)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (norm.metadata.norm_id, block.block_id, block.title),
        ).fetchone()[0]
        embeddings = embeddings_by_block[block.block_id]
        for version, embedding in zip(block.versions, embeddings, strict=True):
            conn.execute(
                """
                INSERT INTO versions (
                    block_id, amending_norm_id, publication_date, effective_date,
                    text_content, html_content, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    block_row_id,
                    version.amending_norm_id,
                    version.publication_date,
                    version.effective_date,
                    version.text_content,
                    version.html_content,
                    embedding,
                ),
            )


def get_version_in_force(
    conn: psycopg.Connection, norm_id: str, block_id: str, target_date: date
) -> VersionInForce | None:
    """Return the block's redaction in force at ``target_date``, or ``None``.

    Selects the version with the latest ``effective_date`` not exceeding the
    target date -- the point-in-time query the corpus exists to answer.
    """
    with conn.cursor(row_factory=class_row(VersionInForce)) as cursor:
        return cursor.execute(
            """
            SELECT v.amending_norm_id, v.effective_date, v.text_content, v.html_content
            FROM versions v
            JOIN blocks b ON b.id = v.block_id
            WHERE b.norm_id = %s AND b.block_id = %s AND v.effective_date <= %s
            ORDER BY v.effective_date DESC
            LIMIT 1
            """,
            (norm_id, block_id, target_date),
        ).fetchone()


def get_effective_dates(conn: psycopg.Connection, norm_id: str, block_id: str) -> list[date]:
    """Return the effective dates of every stored redaction of a block, oldest first.

    The whole amendment history, not only the versions in force at some date, so
    callers can tell that a cited redaction has been superseded since.
    """
    rows = conn.execute(
        """
        SELECT v.effective_date
        FROM versions v
        JOIN blocks b ON b.id = v.block_id
        WHERE b.norm_id = %s AND b.block_id = %s
        ORDER BY v.effective_date
        """,
        (norm_id, block_id),
    ).fetchall()
    return [row[0] for row in rows]


def get_citation_anchor(
    conn: psycopg.Connection, norm_id: str, block_id: str
) -> CitationAnchor | None:
    """Return the citation anchor (norm identity and ELI + block id/title), or ``None``."""
    with conn.cursor(row_factory=class_row(CitationAnchor)) as cursor:
        return cursor.execute(
            """
            SELECT n.id AS norm_id, n.label AS norm_label, n.eli, n.consolidated_html_url,
                   b.block_id, b.title
            FROM blocks b
            JOIN norms n ON n.id = b.norm_id
            WHERE b.norm_id = %s AND b.block_id = %s
            """,
            (norm_id, block_id),
        ).fetchone()


def get_corpus_last_updated(conn: psycopg.Connection, vertical: str) -> datetime | None:
    """Return the freshest ``updated_at`` across a vertical's norms, or ``None``."""
    return _scalar(conn, "SELECT max(updated_at) FROM norms WHERE vertical = %s", (vertical,))


def list_ingested_norms(conn: psycopg.Connection, vertical: str) -> list[IngestedNorm]:
    """Return the vertical's norms with their block counts, in ingestion-label order.

    The corpus is no longer one law, so its freshness cannot be read as one date:
    this is what the UI needs to name every norm it answers from.
    """
    with conn.cursor(row_factory=class_row(IngestedNorm)) as cursor:
        return cursor.execute(
            """
            SELECT n.id AS norm_id, n.label, n.title, n.consolidated_html_url, n.updated_at,
                   count(b.id) AS blocks
            FROM norms n
            LEFT JOIN blocks b ON b.norm_id = n.id
            WHERE n.vertical = %s
            GROUP BY n.id, n.label, n.title, n.consolidated_html_url, n.updated_at
            ORDER BY n.label
            """,
            (vertical,),
        ).fetchall()


def get_corpus_digest(conn: psycopg.Connection, vertical: str) -> str | None:
    """Return a content digest of the vertical's ingested corpus, or ``None`` if empty.

    Hashes every stored redaction -- norm id, block id, effective date and text --
    in a fixed order, so two identical corpora hash the same and any change to the
    ingested text or its point-in-time structure changes the digest. This is the
    corpus identity a run is stamped with, computed in SQL so no text leaves the DB.
    """
    return _scalar_str(
        conn,
        """
        SELECT md5(string_agg(row_digest, '|' ORDER BY row_digest))
        FROM (
            SELECT
                n.id || ':' || b.block_id || ':' || v.effective_date || ':' || md5(v.text_content)
                    AS row_digest
            FROM norms n
            JOIN blocks b ON b.norm_id = n.id
            JOIN versions v ON v.block_id = b.id
            WHERE n.vertical = %s
        ) rows
        """,
        (vertical,),
    )


def _scalar(conn: psycopg.Connection, query: str, params: tuple[object, ...]) -> datetime | None:
    """Return the first column of the single result row, or ``None`` if empty."""
    row = conn.execute(query, params).fetchone()
    return row[0] if row is not None else None


def _scalar_str(conn: psycopg.Connection, query: str, params: tuple[object, ...]) -> str | None:
    """Return the first column of the single result row as text, or ``None`` if absent."""
    row = conn.execute(query, params).fetchone()
    return row[0] if row is not None and row[0] is not None else None


def _upsert_norm(
    conn: psycopg.Connection,
    vertical: str,
    norm: ConsolidatedNorm,
    label: str,
    selection_digest: str,
) -> None:
    """Insert or update the norm's metadata row."""
    metadata = norm.metadata
    conn.execute(
        """
        INSERT INTO norms (
            id, vertical, label, eli, title, consolidated_html_url, updated_at, selection_digest
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            vertical = EXCLUDED.vertical,
            label = EXCLUDED.label,
            eli = EXCLUDED.eli,
            title = EXCLUDED.title,
            consolidated_html_url = EXCLUDED.consolidated_html_url,
            updated_at = EXCLUDED.updated_at,
            selection_digest = EXCLUDED.selection_digest
        """,
        (
            metadata.norm_id,
            vertical,
            label,
            metadata.eli,
            metadata.title,
            metadata.consolidated_html_url,
            metadata.updated_at,
            selection_digest,
        ),
    )
