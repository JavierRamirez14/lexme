"""Fixtures for ingestion tests: the sample XML and a clean corpus database.

Database-backed tests require a reachable Postgres with pgvector; point
``LEXME_TEST_DATABASE_URL`` at one to run them. They are skipped otherwise so the
pure-parsing tests can still run anywhere.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector

from lexme.ingestion.embeddings import EMBEDDING_DIMENSIONS

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA_SQL = REPO_ROOT / "db" / "init" / "02-schema.sql"
VIVIENDA_MANIFEST = REPO_ROOT / "verticales" / "vivienda" / "manifest.json"
CORPUS_TABLES = ("versions", "blocks", "norms")
LAU_NORM_ID = "BOE-A-1994-26003"


@pytest.fixture
def lau_sample_xml() -> str:
    """The trimmed LAU XML fixture (articles 1, 2 and 9) as text."""
    return (FIXTURES / "lau_sample.xml").read_text(encoding="utf-8")


class CountingEmbedder:
    """A deterministic embedder that records how many texts it has embedded."""

    def __init__(self) -> None:
        self.embedded_count = 0

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Return one zero vector per text and count the inputs."""
        self.embedded_count += len(texts)
        return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]


@pytest.fixture
def fake_embedder() -> CountingEmbedder:
    """A test double for the TEI embedder, counting embed calls."""
    return CountingEmbedder()


@pytest.fixture
def db_connection() -> Iterator[psycopg.Connection]:
    """Yield a connection to an empty corpus schema, tearing rows down after.

    Applies the schema idempotently, truncates the corpus tables, and registers
    the pgvector adapter so embeddings can be passed as plain lists.
    """
    dsn = os.environ.get("LEXME_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("set LEXME_TEST_DATABASE_URL to run corpus integration tests")

    with psycopg.connect(dsn) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
        conn.execute(f"TRUNCATE {', '.join(CORPUS_TABLES)} RESTART IDENTITY CASCADE")
        conn.commit()
        register_vector(conn)
        yield conn
