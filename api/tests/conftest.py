"""Shared test fixtures: a real uvicorn server on a loopback socket."""

import hashlib
import math
import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
import uvicorn
from pgvector.psycopg import register_vector

os.environ.setdefault("DATABASE_URL", "postgresql://localhost/lexme_test")

from lexme.ingestion import repository
from lexme.ingestion.embeddings import EMBEDDING_DIMENSIONS
from lexme.ingestion.models import ConsolidatedNorm
from lexme.llm import FakeLlmClient
from lexme.main import app

STARTUP_TIMEOUT_SECONDS = 10
SHUTDOWN_TIMEOUT_SECONDS = 5
CONNECT_PROBE_TIMEOUT_SECONDS = 0.25
CONNECT_RETRY_INTERVAL_SECONDS = 0.05

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_INIT_DIR = REPO_ROOT / "db" / "init"
CORPUS_TABLES = ("versions", "blocks", "norms")


def _free_port() -> int:
    """Return an OS-assigned free TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_until_accepting_connections(host: str, port: int) -> None:
    """Block until a TCP connection to host:port succeeds, or raise TimeoutError."""
    stop_at = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < stop_at:
        try:
            with socket.create_connection((host, port), timeout=CONNECT_PROBE_TIMEOUT_SECONDS):
                return
        except OSError:
            time.sleep(CONNECT_RETRY_INTERVAL_SECONDS)
    raise TimeoutError(f"server did not start accepting connections on {host}:{port}")


class DeterministicEmbedder:
    """A network-free embedder whose vectors reflect shared vocabulary.

    Each token is hashed onto a coordinate and its count accumulated, so two texts
    that share words land close in cosine space. Seeding and querying through the
    same embedder makes dense retrieval meaningful and deterministic without TEI.
    """

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Return one normalized bag-of-words vector per input text."""
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        """Hash ``text``'s tokens into a unit-length vector."""
        vector = [0.0] * EMBEDDING_DIMENSIONS
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % EMBEDDING_DIMENSIONS] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


def seed_norm(
    conn: psycopg.Connection,
    norm: ConsolidatedNorm,
    embedder: DeterministicEmbedder,
    vertical: str = "vivienda",
) -> None:
    """Store ``norm`` with one embedding per version, then commit."""
    embeddings_by_block = {
        block.block_id: embedder.embed_many([version.text_content for version in block.versions])
        for block in norm.blocks
    }
    repository.replace_norm(conn, vertical, norm, embeddings_by_block)
    conn.commit()


@pytest.fixture
def deterministic_embedder() -> DeterministicEmbedder:
    """A shared deterministic embedder for seeding and querying the test corpus."""
    return DeterministicEmbedder()


@pytest.fixture
def corpus_db() -> Iterator[psycopg.Connection]:
    """Yield a connection to an empty, fully-migrated corpus schema.

    Applies every ``db/init/*.sql`` migration idempotently, truncates the corpus
    tables and registers the pgvector adapter. Skips when no test database is
    configured so the network-free tests still run anywhere.
    """
    dsn = os.environ.get("LEXME_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("set LEXME_TEST_DATABASE_URL to run corpus integration tests")

    with psycopg.connect(dsn) as conn:
        for sql_file in sorted(DB_INIT_DIR.glob("*.sql")):
            conn.execute(sql_file.read_text(encoding="utf-8"))
        conn.execute(f"TRUNCATE {', '.join(CORPUS_TABLES)} RESTART IDENTITY CASCADE")
        conn.commit()
        register_vector(conn)
        yield conn


@pytest.fixture
def fake_llm() -> FakeLlmClient:
    """The suite's single LLM substitution point: a deterministic, network-free client.

    Every component that needs a model is wired through :class:`LlmClient`, so a
    test programs replies per task here and lets retrieval, verification, the
    checklist cross-check and the code gates run for real.
    """
    return FakeLlmClient()


@pytest.fixture
def live_server() -> Iterator[str]:
    """Run the app in a real uvicorn server on a loopback port for the test.

    Yields the base URL and tears the server down afterwards.
    """
    host = "127.0.0.1"
    port = _free_port()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    _wait_until_accepting_connections(host, port)

    yield f"http://{host}:{port}"

    server.should_exit = True
    thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
