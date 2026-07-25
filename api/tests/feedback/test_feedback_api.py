"""Acceptance tests for POST /feedback against a live server and a real database.

The snapshot is opaque JSON from the endpoint's point of view -- whatever the
client sends, verbatim -- so these tests assert the full round trip: an arbitrary
Mode 1 or Mode 2 snapshot goes in, and the exact same structure reads back out.
"""

import os

import httpx
import psycopg
import pytest

from lexme.api.dependencies import get_db_connection
from lexme.main import app

pytestmark = pytest.mark.usefixtures("corpus_db")


@pytest.fixture
def feedback_server(corpus_db: psycopg.Connection, live_server: str):
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


def test_a_consultation_vote_persists_its_full_snapshot(feedback_server: str) -> None:
    snapshot = {
        "outcome": "respuesta",
        "answer": {
            "fundamento": [{"block_id": "a9", "text": "La duración del arrendamiento..."}],
            "explicacion": "El plazo mínimo se pacta libremente.",
        },
    }

    response = httpx.post(
        f"{feedback_server}/feedback",
        json={"mode": "consulta", "vote": "positivo", "snapshot": snapshot},
        timeout=10,
    )

    assert response.status_code == 201
    feedback_id = response.json()["id"]

    with psycopg.connect(os.environ["LEXME_TEST_DATABASE_URL"]) as conn:
        row = conn.execute(
            "SELECT mode, vote, snapshot FROM feedback WHERE id = %s", (feedback_id,)
        ).fetchone()

    assert row == ("consulta", "positivo", snapshot)


def test_a_contract_vote_persists_its_full_snapshot(feedback_server: str) -> None:
    snapshot = {
        "outcome": "analizado",
        "sheet": {"arrendador": "Juan Pérez", "renta": "800 euros al mes"},
        "risk_map": {"clause_findings": [], "absence_findings": []},
    }

    response = httpx.post(
        f"{feedback_server}/feedback",
        json={"mode": "contrato", "vote": "negativo", "snapshot": snapshot},
        timeout=10,
    )

    assert response.status_code == 201

    with psycopg.connect(os.environ["LEXME_TEST_DATABASE_URL"]) as conn:
        row = conn.execute(
            "SELECT mode, vote, snapshot FROM feedback WHERE id = %s",
            (response.json()["id"],),
        ).fetchone()

    assert row == ("contrato", "negativo", snapshot)


def test_an_empty_snapshot_is_rejected_at_the_boundary(feedback_server: str) -> None:
    response = httpx.post(
        f"{feedback_server}/feedback",
        json={"mode": "consulta", "vote": "positivo", "snapshot": {}},
        timeout=10,
    )

    assert response.status_code == 422


def test_an_unknown_vote_is_rejected_at_the_boundary(feedback_server: str) -> None:
    response = httpx.post(
        f"{feedback_server}/feedback",
        json={"mode": "consulta", "vote": "neutro", "snapshot": {"outcome": "respuesta"}},
        timeout=10,
    )

    assert response.status_code == 422
