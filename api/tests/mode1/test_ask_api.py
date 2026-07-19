"""Acceptance tests over the HTTP surface of Mode 1 with a faked LLM.

The whole path runs for real against a live server -- hybrid retrieval over the
seeded corpus, citation verification and the deterministic gate -- with only the
LLM and the query embedder substituted. They assert the output contract: a cited
answer, a discarded corrupt citation, and a forced abstention.
"""

import os
from collections.abc import Iterator

import httpx
import psycopg
import pytest
from pgvector.psycopg import register_vector

from lexme.api.dependencies import (
    get_db_connection,
    get_embedder,
    get_llm_client,
    get_target_date,
)
from lexme.llm import FakeLlmClient
from lexme.main import app
from lexme.mode1.models import Mode1Synthesis
from lexme.mode1.synthesis import SYNTHESIS_TASK
from lexme.verification import ProposedCitation
from tests.conftest import DeterministicEmbedder
from tests.mode1.conftest import AS_OF

QUESTION = "¿cuál es el plazo mínimo del arrendamiento de vivienda?"
VERBATIM_QUOTE = "La duración del arrendamiento será libremente pactada por las partes"
FABRICATED = "El arrendador podrá desalojar al inquilino en cualquier momento sin preaviso."


@pytest.fixture
def ask_server(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    fake_llm: FakeLlmClient,
    live_server: str,
) -> Iterator[tuple[str, FakeLlmClient]]:
    """Override the endpoint's collaborators and yield the base URL and fake LLM."""
    dsn = os.environ["LEXME_TEST_DATABASE_URL"]

    def override_connection() -> Iterator[psycopg.Connection]:
        with psycopg.connect(dsn) as connection:
            register_vector(connection)
            yield connection

    app.dependency_overrides[get_db_connection] = override_connection
    app.dependency_overrides[get_embedder] = lambda: deterministic_embedder
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_target_date] = lambda: AS_OF
    try:
        yield live_server, fake_llm
    finally:
        app.dependency_overrides.clear()


def test_a_grounded_question_returns_a_three_layer_cited_answer(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    fake_llm.queue(
        SYNTHESIS_TASK,
        Mode1Synthesis(
            fundamento=[ProposedCitation(block_id="a9", text=VERBATIM_QUOTE)],
            explicacion="El plazo mínimo se pacta libremente, con una duración protegida.",
            accion=["Revisa la fecha de tu contrato"],
        ),
    )

    response = httpx.post(f"{base_url}/ask", json={"question": QUESTION}, timeout=10)

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "respuesta"
    citation = body["answer"]["fundamento"][0]
    assert citation["block_id"] == "a9"
    assert citation["verdict"] == "verificada_directa"
    assert citation["anchor"]["effective_date"] == "2019-03-06"
    assert body["retrieval"]["dense"] and body["retrieval"]["lexical"]


def test_a_corrupt_citation_is_discarded_and_never_shown(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    fake_llm.queue(
        SYNTHESIS_TASK,
        Mode1Synthesis(
            fundamento=[
                ProposedCitation(block_id="a9", text=VERBATIM_QUOTE),
                ProposedCitation(block_id="a9", text=FABRICATED),
            ],
            explicacion="El plazo mínimo se pacta libremente.",
            accion=[],
        ),
    )

    response = httpx.post(f"{base_url}/ask", json={"question": QUESTION}, timeout=10)

    body = response.json()
    assert body["outcome"] == "respuesta"
    texts = [citation["text"] for citation in body["answer"]["fundamento"]]
    assert all(FABRICATED not in text for text in texts)
    assert body["citation_verdicts"]["descartada"] == 1


def test_no_verifiable_citation_yields_an_abstention_contract(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    fake_llm.queue(
        SYNTHESIS_TASK,
        Mode1Synthesis(
            fundamento=[ProposedCitation(block_id="a9", text=FABRICATED)],
            explicacion="...",
            accion=[],
        ),
    )

    response = httpx.post(f"{base_url}/ask", json={"question": QUESTION}, timeout=10)

    body = response.json()
    assert body["outcome"] == "abstencion"
    assert body["answer"] is None
    assert body["abstention"]["reason"] == "sin_cita_verificable"
    assert body["abstention"]["scope_reminder"]


def test_a_blank_question_is_rejected_at_the_boundary(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, _ = ask_server

    response = httpx.post(f"{base_url}/ask", json={"question": "   "}, timeout=10)

    assert response.status_code == 422
