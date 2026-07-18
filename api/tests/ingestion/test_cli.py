"""Acceptance test: the ``ingest`` command builds a queryable LAU corpus.

Drives the real command through the real BoeClient code path with only its HTTP
transport faked, plus a fake embedder, against a real database. This leaves the
small LAU corpus the rest of the suite can rely on.
"""

from datetime import date

import httpx
import psycopg

from lexme.ingestion import repository
from lexme.ingestion.boe_client import BoeClient
from lexme.ingestion.cli import main
from tests.ingestion.conftest import LAU_NORM_ID, VIVIENDA_MANIFEST, CountingEmbedder


def _boe_client_serving(xml: str) -> BoeClient:
    """A real BoeClient whose HTTP transport returns the fixture for the LAU id."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(f"/id/{LAU_NORM_ID}"):
            return httpx.Response(200, text=xml)
        return httpx.Response(404)

    return BoeClient(transport=httpx.MockTransport(handler))


def test_ingest_command_builds_point_in_time_corpus(
    db_connection: psycopg.Connection, lau_sample_xml: str, fake_embedder: CountingEmbedder
) -> None:
    with _boe_client_serving(lau_sample_xml) as fetcher:
        exit_code = main(
            ["--manifest", str(VIVIENDA_MANIFEST)],
            fetcher=fetcher,
            embedder=fake_embedder,
            connection=db_connection,
        )

    assert exit_code == 0

    in_force = repository.get_version_in_force(db_connection, LAU_NORM_ID, "a9", date(2015, 6, 1))
    assert in_force is not None
    assert in_force.effective_date == date(2013, 6, 6)

    anchor = repository.get_citation_anchor(db_connection, LAU_NORM_ID, "a9")
    assert anchor is not None
    assert anchor.eli.endswith("/eli/es/l/1994/11/24/29")

    assert repository.get_corpus_last_updated(db_connection, "vivienda") is not None
