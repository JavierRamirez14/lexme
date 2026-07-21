"""Acceptance tests for answering at a past date, over the HTTP surface.

Article 9 of the seeded LAU has six redactions, so "what applied in 2017" and
"what applies now" are genuinely different texts: in 2017 a contract shorter than
*three* years was extended to three, and since March 2019 the floor is *five*.
These tests drive the whole path over HTTP -- the planner reads the date out of
the question, the SQL resolves the redaction, the verifier matches against that
same redaction -- and check both that the right text comes back and that the
notices warning it is not today's law are there. The clock is fixed at
:data:`AS_OF`, so nothing here depends on when it runs.
"""

import httpx

from lexme.llm import FakeLlmClient
from tests.mode1.conftest import AS_OF, program_single_sufficient

QUESTION_2017 = "firmé el contrato en 2017, ¿qué duración mínima me aplicaba?"
QUESTION_NOW = "¿cuál es la duración mínima del arrendamiento de vivienda?"

THREE_YEAR_QUOTE = "Si ésta fuera inferior a tres años"
FIVE_YEAR_QUOTE = "Si esta fuera inferior a cinco años"


def ask(base_url: str, question: str) -> dict:
    """POST a question to the blocking endpoint and return the decoded body."""
    response = httpx.post(f"{base_url}/ask", json={"question": question}, timeout=10)
    assert response.status_code == 200
    return response.json()


def notice_codes(body: dict) -> list[str]:
    """The codes of the answer's in-force notices, in order."""
    return [notice["code"] for notice in body["answer"]["avisos_vigencia"]]


def test_a_question_anchored_in_the_past_is_answered_with_that_dates_redaction(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_single_sufficient(
        fake_llm,
        query_text="duración plazo mínimo arrendamiento vivienda",
        citation=("a9", THREE_YEAR_QUOTE),
        target_date_reference="2017",
    )

    body = ask(base_url, QUESTION_2017)

    assert body["outcome"] == "respuesta"
    assert body["answer"]["fecha_objetivo"] == "2017-01-01"
    citation = body["answer"]["fundamento"][0]
    assert citation["verdict"] == "verificada_directa"
    assert citation["anchor"]["effective_date"] == "2013-06-06"
    assert THREE_YEAR_QUOTE in citation["text"]


def test_the_same_question_without_a_date_is_answered_with_todays_redaction(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_single_sufficient(
        fake_llm,
        query_text="duración plazo mínimo arrendamiento vivienda",
        citation=("a9", FIVE_YEAR_QUOTE),
    )

    body = ask(base_url, QUESTION_NOW)

    assert body["answer"]["fecha_objetivo"] == AS_OF.isoformat()
    assert body["answer"]["fundamento"][0]["anchor"]["effective_date"] == "2019-03-06"


def test_a_past_answer_warns_that_it_is_not_todays_law(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_single_sufficient(
        fake_llm,
        query_text="duración plazo mínimo arrendamiento vivienda",
        citation=("a9", THREE_YEAR_QUOTE),
        target_date_reference="2017-05-04",
    )

    body = ask(base_url, QUESTION_2017)

    assert "fecha_objetivo_pasada" in notice_codes(body)
    assert "redaccion_superada" in notice_codes(body)


def test_the_superseded_notice_names_the_article_and_the_amendment_date(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_single_sufficient(
        fake_llm,
        query_text="duración plazo mínimo arrendamiento vivienda",
        citation=("a9", THREE_YEAR_QUOTE),
        target_date_reference="2017-05-04",
    )

    body = ask(base_url, QUESTION_2017)

    superseded = next(
        notice
        for notice in body["answer"]["avisos_vigencia"]
        if notice["code"] == "redaccion_superada"
    )
    assert superseded["block_id"] == "a9"
    assert "Artículo 9" in superseded["message"]
    assert "19 de diciembre de 2018" in superseded["message"]


def test_an_answer_at_today_is_never_warned_about_as_historical(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_single_sufficient(
        fake_llm,
        query_text="duración plazo mínimo arrendamiento vivienda",
        citation=("a9", FIVE_YEAR_QUOTE),
    )

    body = ask(base_url, QUESTION_NOW)

    assert "fecha_objetivo_pasada" not in notice_codes(body)
    assert "redaccion_superada" not in notice_codes(body)


def test_a_redaction_amended_within_the_last_year_is_flagged_as_recent(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_single_sufficient(
        fake_llm,
        query_text="duración plazo mínimo arrendamiento vivienda",
        citation=("a9", FIVE_YEAR_QUOTE),
    )

    body = ask(base_url, QUESTION_NOW)

    assert notice_codes(body) == ["modificacion_reciente"]


def test_a_future_date_falls_back_to_today_rather_than_promising_a_redaction(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_single_sufficient(
        fake_llm,
        query_text="duración plazo mínimo arrendamiento vivienda",
        citation=("a9", FIVE_YEAR_QUOTE),
        target_date_reference="2030-01-01",
    )

    body = ask(base_url, "¿qué duración mínima me aplicará en 2030?")

    assert body["answer"]["fecha_objetivo"] == AS_OF.isoformat()
    assert "fecha_objetivo_pasada" not in notice_codes(body)


def test_a_bare_year_that_straddles_a_reform_is_flagged_as_ambiguous(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_single_sufficient(
        fake_llm,
        query_text="duración plazo mínimo arrendamiento vivienda",
        citation=("a9", FIVE_YEAR_QUOTE),
        target_date_reference="2019",
    )

    body = ask(base_url, "firmé en 2019, ¿qué duración mínima me aplicaba?")

    assert body["answer"]["fecha_objetivo"] == "2019-01-01"
    ambiguous = next(
        notice
        for notice in body["answer"]["avisos_vigencia"]
        if notice["code"] == "anio_objetivo_ambiguo"
    )
    assert "24 de enero de 2019" in ambiguous["message"]


def test_a_year_with_no_reform_inside_it_is_not_flagged_as_ambiguous(
    ask_server: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = ask_server
    program_single_sufficient(
        fake_llm,
        query_text="duración plazo mínimo arrendamiento vivienda",
        citation=("a9", THREE_YEAR_QUOTE),
        target_date_reference="2017",
    )

    body = ask(base_url, QUESTION_2017)

    assert "anio_objetivo_ambiguo" not in notice_codes(body)
