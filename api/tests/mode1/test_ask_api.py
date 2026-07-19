"""Acceptance tests over the HTTP surface of Mode 1 with a faked LLM.

The whole agentic path runs for real against a live server -- router, planner,
hybrid retrieval, self-critique, verification and the gate -- with only the LLM
and the query embedder substituted. They assert the output contract on ``/ask``
(a cited answer, a discarded corrupt citation, a scope rejection) and that
``/ask/stream`` narrates the run as ordered SSE steps ending in a result.
"""

import json
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
from lexme.mode1.graph.critique import CRITIQUE_TASK
from lexme.mode1.graph.planner import PLANNING_TASK
from lexme.mode1.graph.router import ROUTER_TASK
from lexme.mode1.models import SubQueryVerdict
from lexme.mode1.synthesis import SYNTHESIS_TASK
from tests.conftest import DeterministicEmbedder
from tests.mode1.conftest import (
    AS_OF,
    critique_of,
    in_scope,
    out_of_scope,
    plan_of,
    program_single_sufficient,
    synthesis_of,
)

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


def test_a_grounded_question_returns_a_cited_answer_with_agentic_trace(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_single_sufficient(
        fake_llm,
        query_text=QUESTION,
        citation=("a9", VERBATIM_QUOTE),
        accion=("Revisa la fecha de tu contrato",),
    )

    response = httpx.post(f"{base_url}/ask", json={"question": QUESTION}, timeout=10)

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "respuesta"
    citation = body["answer"]["fundamento"][0]
    assert citation["block_id"] == "a9"
    assert citation["verdict"] == "verificada_directa"
    assert body["agentic"]["subqueries"][0]["verdict"] == "suficiente"
    assert body["agentic"]["subqueries"][0]["retrieval"]["dense"]


def test_an_out_of_scope_question_returns_a_router_rejection(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    fake_llm.queue(ROUTER_TASK, out_of_scope("Eso es materia laboral, no alquiler de vivienda."))

    response = httpx.post(
        f"{base_url}/ask", json={"question": "¿me pueden despedir estando de baja?"}, timeout=10
    )

    body = response.json()
    assert body["outcome"] == "rechazo_router"
    assert body["answer"] is None
    assert body["agentic"] is None
    assert body["rejection"]["scope_reminder"]


def test_a_corrupt_citation_is_discarded_and_never_shown(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    fake_llm.queue(ROUTER_TASK, in_scope())
    fake_llm.queue(PLANNING_TASK, plan_of((QUESTION, "Plazo mínimo.", True)))
    fake_llm.queue(CRITIQUE_TASK, critique_of(("sq1", SubQueryVerdict.SUFFICIENT, "")))
    fake_llm.queue(SYNTHESIS_TASK, synthesis_of(("a9", VERBATIM_QUOTE), ("a9", FABRICATED)))

    response = httpx.post(f"{base_url}/ask", json={"question": QUESTION}, timeout=10)

    body = response.json()
    assert body["outcome"] == "respuesta"
    texts = [citation["text"] for citation in body["answer"]["fundamento"]]
    assert all(FABRICATED not in text for text in texts)
    assert body["citation_verdicts"]["descartada"] == 1


def test_a_peripheral_gap_returns_a_partial_answer_over_the_api(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    fake_llm.queue(ROUTER_TASK, in_scope())
    fake_llm.queue(
        PLANNING_TASK,
        plan_of(
            (QUESTION, "Plazo mínimo.", True),
            ("subvenciones ayudas alquiler joven", "Ayudas al alquiler.", False),
        ),
    )
    fake_llm.queue(
        CRITIQUE_TASK,
        *[
            critique_of(
                ("sq1", SubQueryVerdict.SUFFICIENT, ""),
                ("sq2", SubQueryVerdict.INSUFFICIENT, "ayudas alquiler"),
            )
            for _ in range(3)
        ],
    )
    fake_llm.queue(SYNTHESIS_TASK, synthesis_of(("a9", VERBATIM_QUOTE)))

    response = httpx.post(f"{base_url}/ask", json={"question": QUESTION}, timeout=10)

    body = response.json()
    assert body["outcome"] == "respuesta_parcial"
    assert body["answer"]["huecos_declarados"] == ["Ayudas al alquiler."]


def test_no_verifiable_citation_returns_an_abstention_over_the_api(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    fake_llm.queue(ROUTER_TASK, in_scope())
    fake_llm.queue(PLANNING_TASK, plan_of((QUESTION, "Plazo mínimo.", True)))
    fake_llm.queue(CRITIQUE_TASK, critique_of(("sq1", SubQueryVerdict.SUFFICIENT, "")))
    fake_llm.queue(SYNTHESIS_TASK, synthesis_of(("a9", FABRICATED)))

    response = httpx.post(f"{base_url}/ask", json={"question": QUESTION}, timeout=10)

    body = response.json()
    assert body["outcome"] == "abstencion"
    assert body["answer"] is None
    assert body["abstention"]["reason"] == "sin_cita_verificable"
    assert body["abstention"]["scope_reminder"]


def test_the_stream_narrates_the_run_as_ordered_steps_then_a_result(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_single_sufficient(fake_llm, query_text=QUESTION, citation=("a9", VERBATIM_QUOTE))

    events = _read_sse(base_url, QUESTION)

    step_names = [data["step"] for name, data in events if name == "step"]
    assert "planificando" in step_names
    assert "recuperando" in step_names
    assert "sintetizando" in step_names
    result_events = [data for name, data in events if name == "result"]
    assert len(result_events) == 1
    assert result_events[0]["outcome"] == "respuesta"


def test_a_blank_question_is_rejected_at_the_boundary(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, _ = ask_server

    response = httpx.post(f"{base_url}/ask", json={"question": "   "}, timeout=10)

    assert response.status_code == 422


def _read_sse(base_url: str, question: str) -> list[tuple[str, dict]]:
    """POST to the stream endpoint and collect the ``(event, data)`` pairs in order."""
    events: list[tuple[str, dict]] = []
    with httpx.stream(
        "POST", f"{base_url}/ask/stream", json={"question": question}, timeout=10
    ) as response:
        event_name = "message"
        for line in response.iter_lines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                events.append((event_name, json.loads(line[len("data:") :].strip())))
    return events
