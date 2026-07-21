"""Acceptance tests for the pause-and-resume cycle, over the HTTP surface.

The point of these is that the pause survives the request: ``/ask`` comes back
with a question and a thread id, the connection closes, and a later ``/ask/resume``
picks the run up inside the graph exactly where it stopped. They run against the
shipped vivienda branch package, so what counts as a regime-changing question is
the real data, not a fixture.
"""

import httpx
import pytest

from lexme.llm import FakeLlmClient
from lexme.mode1 import CriticalBranch
from lexme.mode1.graph.critique import CRITIQUE_TASK
from lexme.mode1.graph.planner import PLANNING_TASK
from lexme.mode1.graph.router import ROUTER_TASK
from lexme.mode1.models import QueryType, SubQueryVerdict
from lexme.mode1.synthesis import SYNTHESIS_TASK
from tests.mode1.conftest import (
    AS_OF,
    critique_of,
    in_scope,
    plan_of,
    program_single_sufficient,
    read_resume_sse,
    read_sse,
    synthesis_of,
)

QUESTION = "me quieren echar del piso, ¿cuánto dura mi contrato?"
QUERY_TEXT = "duración plazo mínimo arrendamiento vivienda"
THREE_YEAR_QUOTE = "Si ésta fuera inferior a tres años"
FIVE_YEAR_QUOTE = "Si esta fuera inferior a cinco años"
SIGNING_BRANCH = "fecha_firma"


@pytest.fixture
def server_branches(
    vivienda_branches: tuple[CriticalBranch, ...],
) -> tuple[CriticalBranch, ...]:
    """Serve the real vivienda branch package, so these runs can pause."""
    return vivienda_branches


def program_situational_run(fake_llm: FakeLlmClient, *, citation: tuple[str, str]) -> None:
    """Queue a situational run that leaves the signing-date branch unresolved."""
    program_single_sufficient(
        fake_llm,
        query_text=QUERY_TEXT,
        citation=citation,
        query_type=QueryType.SITUATIONAL,
        unresolved=(SIGNING_BRANCH,),
    )


def ask(base_url: str, question: str = QUESTION) -> dict:
    """POST a question to the blocking endpoint and return the decoded body."""
    response = httpx.post(f"{base_url}/ask", json={"question": question}, timeout=10)
    assert response.status_code == 200
    return response.json()


def reply(base_url: str, thread_id: str, answer: str) -> httpx.Response:
    """POST a reply to the paused run's thread."""
    return httpx.post(
        f"{base_url}/ask/resume",
        json={"thread_id": thread_id, "answer": answer},
        timeout=10,
    )


def test_a_critical_branch_pauses_the_run_with_one_question(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_situational_run(fake_llm, citation=("a9", THREE_YEAR_QUOTE))

    body = ask(base_url)

    assert body["outcome"] == "desambiguacion"
    assert body["answer"] is None
    assert body["clarification"]["branch_id"] == SIGNING_BRANCH
    assert body["clarification"]["answer_kind"] == "fecha"
    assert body["thread_id"]


def test_the_paused_run_asks_before_retrieving_anything(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_situational_run(fake_llm, citation=("a9", THREE_YEAR_QUOTE))

    ask(base_url)

    assert [call.task for call in fake_llm.calls] == [ROUTER_TASK, PLANNING_TASK]


def test_answering_resumes_the_run_and_answers_for_that_case(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_situational_run(fake_llm, citation=("a9", THREE_YEAR_QUOTE))
    paused = ask(base_url)

    response = reply(base_url, paused["thread_id"], "2017-05-04")

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "respuesta"
    assert body["answer"]["fecha_objetivo"] == "2017-05-04"
    assert body["answer"]["fundamento"][0]["anchor"]["effective_date"] == "2013-06-06"


def test_the_users_answer_reaches_synthesis_as_a_case_fact(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_situational_run(fake_llm, citation=("a9", THREE_YEAR_QUOTE))
    paused = ask(base_url)

    reply(base_url, paused["thread_id"], "2017-05-04")

    synthesis_prompt = next(
        call.messages[-1].content for call in fake_llm.calls if call.task == SYNTHESIS_TASK
    )
    assert "2017-05-04" in synthesis_prompt


def test_a_blank_answer_proceeds_on_an_explicit_assumption(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_situational_run(fake_llm, citation=("a9", FIVE_YEAR_QUOTE))
    paused = ask(base_url)

    body = reply(base_url, paused["thread_id"], "").json()

    assert body["outcome"] == "respuesta"
    assert body["answer"]["fecha_objetivo"] == AS_OF.isoformat()
    assert any("Asumo" in assumption for assumption in body["answer"]["asunciones"])


def test_an_unreadable_answer_proceeds_on_an_explicit_assumption(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_situational_run(fake_llm, citation=("a9", FIVE_YEAR_QUOTE))
    paused = ask(base_url)

    body = reply(base_url, paused["thread_id"], "ni idea, hace años").json()

    assert body["outcome"] == "respuesta"
    assert body["answer"]["fecha_objetivo"] == AS_OF.isoformat()
    assert any("Asumo" in assumption for assumption in body["answer"]["asunciones"])


def test_the_run_never_asks_a_second_time(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    fake_llm.queue(ROUTER_TASK, in_scope(QueryType.SITUATIONAL))
    fake_llm.queue(
        PLANNING_TASK,
        plan_of(
            (QUERY_TEXT, "Responder la pregunta.", True),
            unresolved=("fecha_firma", "uso_vivienda"),
        ),
    )
    fake_llm.queue(CRITIQUE_TASK, critique_of(("sq1", SubQueryVerdict.SUFFICIENT, "")))
    fake_llm.queue(SYNTHESIS_TASK, synthesis_of(("a9", THREE_YEAR_QUOTE)))
    paused = ask(base_url)

    body = reply(base_url, paused["thread_id"], "2017-05-04").json()

    assert body["outcome"] == "respuesta"


def test_an_informational_question_is_answered_without_asking(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_single_sufficient(
        fake_llm,
        query_text=QUERY_TEXT,
        citation=("a9", FIVE_YEAR_QUOTE),
        query_type=QueryType.INFORMATIONAL,
        unresolved=(SIGNING_BRANCH,),
    )

    body = ask(base_url, "¿qué dice la ley sobre la duración mínima?")

    assert body["outcome"] == "respuesta"


def test_replying_to_an_unknown_run_is_a_not_found(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, _ = ask_server

    response = reply(base_url, "no-such-thread", "2017-05-04")

    assert response.status_code == 404


def test_replying_twice_to_the_same_run_is_a_not_found(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_situational_run(fake_llm, citation=("a9", THREE_YEAR_QUOTE))
    paused = ask(base_url)
    reply(base_url, paused["thread_id"], "2017-05-04")

    response = reply(base_url, paused["thread_id"], "2019-06-01")

    assert response.status_code == 404


def test_a_blank_thread_id_is_rejected_at_the_boundary(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, _ = ask_server

    response = reply(base_url, "  ", "2017-05-04")

    assert response.status_code == 422


def test_the_stream_ends_on_the_question_and_the_resumed_stream_on_the_answer(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_situational_run(fake_llm, citation=("a9", THREE_YEAR_QUOTE))

    paused_events = read_sse(base_url, QUESTION)

    paused = next(data for name, data in paused_events if name == "result")
    assert paused["outcome"] == "desambiguacion"
    assert paused["agentic"]["subqueries"]

    resumed_events = read_resume_sse(base_url, paused["thread_id"], "2017-05-04")

    steps = [data["step"] for name, data in resumed_events if name == "step"]
    assert "preguntando" in steps
    assert "recuperando" in steps
    resumed = next(data for name, data in resumed_events if name == "result")
    assert resumed["outcome"] == "respuesta"
    assert resumed["answer"]["fecha_objetivo"] == "2017-05-04"


def test_a_branch_left_unasked_becomes_an_explicit_assumption(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    fake_llm.queue(ROUTER_TASK, in_scope(QueryType.SITUATIONAL))
    fake_llm.queue(
        PLANNING_TASK,
        plan_of(
            (QUERY_TEXT, "Responder la pregunta.", True),
            unresolved=("fecha_firma", "uso_vivienda"),
        ),
    )
    fake_llm.queue(CRITIQUE_TASK, critique_of(("sq1", SubQueryVerdict.SUFFICIENT, "")))
    fake_llm.queue(SYNTHESIS_TASK, synthesis_of(("a9", THREE_YEAR_QUOTE)))
    paused = ask(base_url)

    body = reply(base_url, paused["thread_id"], "2017-05-04").json()

    assert any("vivienda habitual" in assumption for assumption in body["answer"]["asunciones"])
    assert not any("firmado hoy" in assumption for assumption in body["answer"]["asunciones"])


def test_an_informational_question_states_its_open_branches_as_assumptions(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_single_sufficient(
        fake_llm,
        query_text=QUERY_TEXT,
        citation=("a9", FIVE_YEAR_QUOTE),
        query_type=QueryType.INFORMATIONAL,
        unresolved=("uso_vivienda",),
    )

    body = ask(base_url, "¿qué dice la ley sobre la duración mínima?")

    assert body["outcome"] == "respuesta"
    assert any("vivienda habitual" in assumption for assumption in body["answer"]["asunciones"])
