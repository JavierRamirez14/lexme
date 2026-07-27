"""Acceptance test: the ``ingest`` command builds a queryable multi-norm corpus.

Drives the real command through the real BoeClient code path with only its HTTP
transport faked, plus a fake embedder, against a real database. One command builds
every norm the manifest declares, and each norm keeps its own point-in-time history
and its own citation anchor even where two of them share a block id.
"""

import json
from datetime import date
from pathlib import Path

import httpx
import psycopg
import pytest

from lexme.ingestion import repository
from lexme.ingestion.boe_client import MAX_ATTEMPTS, BoeClient
from lexme.ingestion.cli import main
from tests.ingestion.conftest import LAU_NORM_ID, LEC_NORM_ID, CountingEmbedder


def _boe_client_serving(payloads: dict[str, str]) -> BoeClient:
    """A real BoeClient whose HTTP transport returns a fixture per norm id."""

    def handler(request: httpx.Request) -> httpx.Response:
        for norm_id, xml in payloads.items():
            if request.url.path.endswith(f"/id/{norm_id}"):
                return httpx.Response(200, text=xml)
        return httpx.Response(404)

    return BoeClient(transport=httpx.MockTransport(handler))


def _two_norm_manifest(tmp_path: Path) -> Path:
    """A manifest taking the whole LAU fixture and only two blocks of the LEC one."""
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "vertical": "vivienda",
                "norms": [
                    {"norm_id": LAU_NORM_ID, "label": "LAU"},
                    {"norm_id": LEC_NORM_ID, "label": "LEC", "blocks": ["a2", "a22"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_ingest_command_builds_point_in_time_corpus(
    tmp_path: Path,
    db_connection: psycopg.Connection,
    lau_sample_xml: str,
    lec_sample_xml: str,
    fake_embedder: CountingEmbedder,
) -> None:
    payloads = {LAU_NORM_ID: lau_sample_xml, LEC_NORM_ID: lec_sample_xml}
    with _boe_client_serving(payloads) as fetcher:
        exit_code = main(
            ["--manifest", str(_two_norm_manifest(tmp_path))],
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
    assert anchor.norm_label == "LAU"

    assert repository.get_corpus_last_updated(db_connection, "vivienda") is not None


def test_ingest_command_builds_every_declared_norm(
    tmp_path: Path,
    db_connection: psycopg.Connection,
    lau_sample_xml: str,
    lec_sample_xml: str,
    fake_embedder: CountingEmbedder,
) -> None:
    payloads = {LAU_NORM_ID: lau_sample_xml, LEC_NORM_ID: lec_sample_xml}
    with _boe_client_serving(payloads) as fetcher:
        main(
            ["--manifest", str(_two_norm_manifest(tmp_path))],
            fetcher=fetcher,
            embedder=fake_embedder,
            connection=db_connection,
        )

    ingested = repository.list_ingested_norms(db_connection, "vivienda")

    assert [(norm.norm_id, norm.label, norm.blocks) for norm in ingested] == [
        (LAU_NORM_ID, "LAU", 3),
        (LEC_NORM_ID, "LEC", 2),
    ]


def test_a_block_id_two_norms_share_resolves_to_each_norms_own_text(
    tmp_path: Path,
    db_connection: psycopg.Connection,
    lau_sample_xml: str,
    lec_sample_xml: str,
    fake_embedder: CountingEmbedder,
) -> None:
    payloads = {LAU_NORM_ID: lau_sample_xml, LEC_NORM_ID: lec_sample_xml}
    with _boe_client_serving(payloads) as fetcher:
        main(
            ["--manifest", str(_two_norm_manifest(tmp_path))],
            fetcher=fetcher,
            embedder=fake_embedder,
            connection=db_connection,
        )

    today = date(2026, 1, 1)
    lau = repository.get_version_in_force(db_connection, LAU_NORM_ID, "a2", today)
    lec = repository.get_version_in_force(db_connection, LEC_NORM_ID, "a2", today)

    assert lau is not None and "arrendamiento de vivienda" in lau.text_content
    assert lec is not None and "normas procesales" in lec.text_content


def test_each_norm_resolves_its_own_redaction_at_a_past_date(
    tmp_path: Path,
    db_connection: psycopg.Connection,
    lau_sample_xml: str,
    lec_sample_xml: str,
    fake_embedder: CountingEmbedder,
) -> None:
    payloads = {LAU_NORM_ID: lau_sample_xml, LEC_NORM_ID: lec_sample_xml}
    with _boe_client_serving(payloads) as fetcher:
        main(
            ["--manifest", str(_two_norm_manifest(tmp_path))],
            fetcher=fetcher,
            embedder=fake_embedder,
            connection=db_connection,
        )

    past = date(2010, 1, 1)
    lau = repository.get_version_in_force(db_connection, LAU_NORM_ID, "a9", past)
    lec = repository.get_version_in_force(db_connection, LEC_NORM_ID, "a22", past)

    assert lau is not None and lau.effective_date == date(2009, 12, 24)
    assert lec is not None and lec.effective_date == date(2001, 1, 8)


class _InstantBoeClient(BoeClient):
    """A BoeClient that records its backoff instead of sleeping it."""

    def __init__(self, transport: httpx.BaseTransport) -> None:
        super().__init__(transport=transport)
        self.waits: list[float] = []

    def _sleep(self, seconds: float) -> None:
        self.waits.append(seconds)


def test_a_dropped_download_is_retried_before_giving_up(lau_sample_xml: str) -> None:
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        if len(attempts) == 1:
            raise httpx.ReadError("peer closed connection")
        return httpx.Response(200, text=lau_sample_xml)

    with _InstantBoeClient(httpx.MockTransport(handler)) as client:
        xml = client.fetch_norm(LAU_NORM_ID)

    assert len(attempts) == 2
    assert "BOE-A-1994-26003" in xml


def test_the_wait_between_retries_grows(lau_sample_xml: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    client = _InstantBoeClient(httpx.MockTransport(handler))
    with client, pytest.raises(httpx.TransportError):
        client.fetch_norm(LAU_NORM_ID)

    assert client.waits == sorted(client.waits)
    assert len(client.waits) == MAX_ATTEMPTS - 1


def test_a_transport_failure_that_never_recovers_is_raised(lau_sample_xml: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("peer closed connection")

    with (
        _InstantBoeClient(httpx.MockTransport(handler)) as client,
        pytest.raises(httpx.TransportError),
    ):
        client.fetch_norm(LAU_NORM_ID)
