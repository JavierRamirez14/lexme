"""Integration tests for the corpus repository against a real Postgres."""

from datetime import date

import psycopg

from lexme.ingestion import repository
from lexme.ingestion.embeddings import EMBEDDING_DIMENSIONS
from lexme.ingestion.models import ConsolidatedNorm
from lexme.ingestion.xml_parsing import parse_norm_xml
from tests.ingestion.conftest import LAU_NORM_ID


def _store(conn: psycopg.Connection, norm: ConsolidatedNorm, vertical: str = "vivienda") -> None:
    """Store a norm with zero-vector embeddings for every version."""
    embeddings_by_block = {
        block.block_id: [[0.0] * EMBEDDING_DIMENSIONS for _ in block.versions]
        for block in norm.blocks
    }
    repository.replace_norm(conn, vertical, norm, embeddings_by_block)
    conn.commit()


def test_point_in_time_returns_the_redaction_in_force(
    db_connection: psycopg.Connection, lau_sample_xml: str
) -> None:
    norm = parse_norm_xml(lau_sample_xml)
    _store(db_connection, norm)

    in_force = repository.get_version_in_force(db_connection, LAU_NORM_ID, "a9", date(2015, 6, 1))

    assert in_force is not None
    assert in_force.effective_date == date(2013, 6, 6)
    assert in_force.amending_norm_id == "BOE-A-2013-5941"


def test_point_in_time_before_first_version_returns_none(
    db_connection: psycopg.Connection, lau_sample_xml: str
) -> None:
    norm = parse_norm_xml(lau_sample_xml)
    _store(db_connection, norm)

    in_force = repository.get_version_in_force(db_connection, LAU_NORM_ID, "a9", date(1990, 1, 1))

    assert in_force is None


def test_point_in_time_after_last_version_returns_latest(
    db_connection: psycopg.Connection, lau_sample_xml: str
) -> None:
    norm = parse_norm_xml(lau_sample_xml)
    _store(db_connection, norm)

    in_force = repository.get_version_in_force(db_connection, LAU_NORM_ID, "a9", date(2024, 1, 1))

    assert in_force is not None
    assert in_force.effective_date == date(2019, 3, 6)


def test_citation_anchor_binds_eli_and_block(
    db_connection: psycopg.Connection, lau_sample_xml: str
) -> None:
    norm = parse_norm_xml(lau_sample_xml)
    _store(db_connection, norm)

    anchor = repository.get_citation_anchor(db_connection, LAU_NORM_ID, "a9")

    assert anchor is not None
    assert anchor.eli == "https://www.boe.es/eli/es/l/1994/11/24/29"
    assert anchor.block_id == "a9"
    assert anchor.title == "Artículo 9"


def test_corpus_last_updated_reports_freshest_norm(
    db_connection: psycopg.Connection, lau_sample_xml: str
) -> None:
    norm = parse_norm_xml(lau_sample_xml)
    _store(db_connection, norm)

    last_updated = repository.get_corpus_last_updated(db_connection, "vivienda")

    assert last_updated == norm.metadata.updated_at


def test_replace_norm_rebuilds_blocks_without_duplicating(
    db_connection: psycopg.Connection, lau_sample_xml: str
) -> None:
    norm = parse_norm_xml(lau_sample_xml)
    _store(db_connection, norm)
    _store(db_connection, norm)

    block_count = db_connection.execute(
        "SELECT count(*) FROM blocks WHERE norm_id = %s", (LAU_NORM_ID,)
    ).fetchone()[0]
    assert block_count == len(norm.blocks)
