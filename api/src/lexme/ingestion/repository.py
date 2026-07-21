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

    eli: str
    consolidated_html_url: str
    block_id: str
    title: str


def get_norm_updated_at(conn: psycopg.Connection, norm_id: str) -> datetime | None:
    """Return the stored ``updated_at`` for a norm, or ``None`` if not stored."""
    return _scalar(conn, "SELECT updated_at FROM norms WHERE id = %s", (norm_id,))


def replace_norm(
    conn: psycopg.Connection,
    vertical: str,
    norm: ConsolidatedNorm,
    embeddings_by_block: dict[str, list[list[float]]],
) -> None:
    """Upsert a norm and rebuild all of its blocks and versions.

    ``embeddings_by_block`` maps each block's ``block_id`` to one embedding per
    version, aligned to :attr:`Block.versions` order.
    """
    _upsert_norm(conn, vertical, norm)
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
    """Return the citation anchor (norm ELI + block id/title), or ``None``."""
    with conn.cursor(row_factory=class_row(CitationAnchor)) as cursor:
        return cursor.execute(
            """
            SELECT n.eli, n.consolidated_html_url, b.block_id, b.title
            FROM blocks b
            JOIN norms n ON n.id = b.norm_id
            WHERE b.norm_id = %s AND b.block_id = %s
            """,
            (norm_id, block_id),
        ).fetchone()


def get_corpus_last_updated(conn: psycopg.Connection, vertical: str) -> datetime | None:
    """Return the freshest ``updated_at`` across a vertical's norms, or ``None``."""
    return _scalar(conn, "SELECT max(updated_at) FROM norms WHERE vertical = %s", (vertical,))


def _scalar(conn: psycopg.Connection, query: str, params: tuple[object, ...]) -> datetime | None:
    """Return the first column of the single result row, or ``None`` if empty."""
    row = conn.execute(query, params).fetchone()
    return row[0] if row is not None else None


def _upsert_norm(conn: psycopg.Connection, vertical: str, norm: ConsolidatedNorm) -> None:
    """Insert or update the norm's metadata row."""
    metadata = norm.metadata
    conn.execute(
        """
        INSERT INTO norms (id, vertical, eli, title, consolidated_html_url, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            vertical = EXCLUDED.vertical,
            eli = EXCLUDED.eli,
            title = EXCLUDED.title,
            consolidated_html_url = EXCLUDED.consolidated_html_url,
            updated_at = EXCLUDED.updated_at
        """,
        (
            metadata.norm_id,
            vertical,
            metadata.eli,
            metadata.title,
            metadata.consolidated_html_url,
            metadata.updated_at,
        ),
    )
