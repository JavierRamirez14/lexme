"""Integration tests for the ingestion pipeline against a real Postgres."""

import psycopg

from lexme.ingestion.manifest import VerticalManifest
from lexme.ingestion.pipeline import run_ingest
from tests.ingestion.conftest import LAU_NORM_ID, CountingEmbedder

ORIGINAL_UPDATED_AT = "20260430T073359Z"
LATER_UPDATED_AT = "20260501T090000Z"


class StaticFetcher:
    """Returns a fixed XML payload for any norm id."""

    def __init__(self, xml: str) -> None:
        self._xml = xml

    def fetch_norm(self, norm_id: str) -> str:
        return self._xml


def _manifest() -> VerticalManifest:
    return VerticalManifest(vertical="vivienda", norm_ids=(LAU_NORM_ID,))


def test_first_run_ingests_the_norm(
    db_connection: psycopg.Connection, lau_sample_xml: str, fake_embedder: CountingEmbedder
) -> None:
    result = run_ingest(db_connection, _manifest(), StaticFetcher(lau_sample_xml), fake_embedder)

    assert result.updated == (LAU_NORM_ID,)
    assert result.unchanged == ()
    version_count = db_connection.execute("SELECT count(*) FROM versions").fetchone()[0]
    assert version_count == 8  # a1 + a2 (one each) + a9 (six)


def test_rerun_with_unchanged_norm_skips_without_embedding(
    db_connection: psycopg.Connection, lau_sample_xml: str, fake_embedder: CountingEmbedder
) -> None:
    manifest = _manifest()
    fetcher = StaticFetcher(lau_sample_xml)
    run_ingest(db_connection, manifest, fetcher, fake_embedder)
    embedded_after_first = fake_embedder.embedded_count

    result = run_ingest(db_connection, manifest, fetcher, fake_embedder)

    assert result.updated == ()
    assert result.unchanged == (LAU_NORM_ID,)
    assert fake_embedder.embedded_count == embedded_after_first


def test_rerun_with_changed_updated_at_reingests(
    db_connection: psycopg.Connection, lau_sample_xml: str, fake_embedder: CountingEmbedder
) -> None:
    manifest = _manifest()
    run_ingest(db_connection, manifest, StaticFetcher(lau_sample_xml), fake_embedder)

    changed_xml = lau_sample_xml.replace(ORIGINAL_UPDATED_AT, LATER_UPDATED_AT)
    result = run_ingest(db_connection, manifest, StaticFetcher(changed_xml), fake_embedder)

    assert result.updated == (LAU_NORM_ID,)
    stored = db_connection.execute(
        "SELECT to_char(updated_at AT TIME ZONE 'UTC', 'YYYYMMDD') FROM norms WHERE id = %s",
        (LAU_NORM_ID,),
    ).fetchone()[0]
    assert stored == "20260501"
