"""Acceptance tests for GET /corpus/status against a live server and a real database."""

import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
import psycopg
import pytest

from lexme.api.dependencies import get_db_connection
from lexme.ingestion.xml_parsing import parse_norm_xml
from lexme.main import app
from tests.conftest import DeterministicEmbedder, seed_norm

LAU_XML = (Path(__file__).resolve().parent / "ingestion" / "fixtures" / "lau_sample.xml").read_text(
    encoding="utf-8"
)


@pytest.fixture
def corpus_status_server(corpus_db: psycopg.Connection, live_server: str):
    """Override the database dependency with the migrated test database."""
    dsn = os.environ["LEXME_TEST_DATABASE_URL"]

    def override_connection():
        with psycopg.connect(dsn) as connection:
            yield connection

    app.dependency_overrides[get_db_connection] = override_connection
    try:
        yield live_server
    finally:
        app.dependency_overrides.clear()


def test_reports_the_freshest_update_across_the_vertical(
    corpus_status_server: str,
    corpus_db: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
) -> None:
    norm = parse_norm_xml(LAU_XML)
    seed_norm(corpus_db, norm, deterministic_embedder)

    response = httpx.get(f"{corpus_status_server}/corpus/status", timeout=10)

    assert response.status_code == 200
    body = response.json()
    assert body["vertical"] == "vivienda"
    reported = datetime.fromisoformat(body["updated_at"])
    assert reported == norm.metadata.updated_at.astimezone(UTC)


def test_reports_no_freshness_when_the_corpus_is_empty(corpus_status_server: str) -> None:
    response = httpx.get(f"{corpus_status_server}/corpus/status", timeout=10)

    assert response.status_code == 200
    assert response.json() == {"vertical": "vivienda", "updated_at": None, "norms": []}


def test_names_every_norm_the_vertical_answers_from(
    corpus_status_server: str,
    corpus_db: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
) -> None:
    seed_norm(corpus_db, parse_norm_xml(LAU_XML), deterministic_embedder, label="LAU")

    response = httpx.get(f"{corpus_status_server}/corpus/status", timeout=10)

    (norm,) = response.json()["norms"]
    assert norm["norm_id"] == "BOE-A-1994-26003"
    assert norm["label"] == "LAU"
    assert norm["blocks"] == 3
    assert norm["consolidated_html_url"].endswith("BOE-A-1994-26003")
