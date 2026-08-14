"""Tests for the citation guardrail: displayed citations must re-verify against the corpus."""

from datetime import date

from lexme.eval.guardrail import (
    REASON_DISCARDED,
    REASON_NO_EVIDENCE,
    REASON_NOT_LITERAL,
    REASON_UNRESOLVED,
    check_case_citations,
    read_verification_date,
)
from lexme.mode1 import AskResponse
from lexme.verification import CitationVerdict
from tests.eval.conftest import (
    AS_OF,
    NORM_ID,
    DatedCorpus,
    InMemoryCorpus,
    answer_response,
    clarification_response,
)

BLOCK_TEXT = "La duración del arrendamiento será libremente pactada por las partes."
QUOTE = "La duración del arrendamiento será libremente pactada"
FABRICATED = "El arrendador podrá desalojar al inquilino sin preaviso."

# Art. 22 LEC as a block with two redactions: the newer one moves an internal
# remission and adds a paragraph, so a literal quote of the older redaction is absent
# from it and the two dates cannot both hold the same quote.
LEC = "BOE-A-2000-323"
SIGNED_ON = date(2023, 1, 15)
RUN_DATE = date(2026, 8, 13)
LEC_22_2015 = (
    "Los procesos de desahucio de finca urbana por falta de pago terminarán si el "
    "arrendatario paga al actor o pone a su disposición en el tribunal o notarialmente "
    "las cantidades reclamadas, conforme a lo dispuesto en el apartado 3 del artículo 440."
)
LEC_22_2025 = (
    "Los procesos de desahucio de finca urbana por falta de pago terminarán si el "
    "arrendatario paga al actor o pone a su disposición en el tribunal o notarialmente "
    "las cantidades reclamadas, conforme a lo dispuesto en el apartado 4 del artículo 439. "
    "En tal caso, las costas se impondrán al arrendatario salvo que el pago se hubiera "
    "producido antes del requerimiento."
)
QUOTE_2015 = "las cantidades reclamadas, conforme a lo dispuesto en el apartado 3 del artículo 440"
QUOTE_2025 = "las costas se impondrán al arrendatario salvo que el pago se hubiera producido"


def _lec_corpus() -> DatedCorpus:
    """A corpus holding both redactions of art. 22 LEC, each from its effective date."""
    return DatedCorpus(
        {(LEC, "a22"): [(date(2015, 10, 1), LEC_22_2015), (date(2025, 4, 3), LEC_22_2025)]}
    )


def _enervacion_response(quote: str) -> AskResponse:
    """The `mh-02` answer as displayed: given for the signing date, citing art. 22 LEC."""
    return answer_response(
        ("a22", quote), evidence=((LEC, "a22"),), norm_id=LEC, fecha_objetivo=SIGNED_ON
    )


def _corpus() -> InMemoryCorpus:
    """A corpus resolving block a9 to its real LAU text."""
    return InMemoryCorpus({(NORM_ID, "a9"): BLOCK_TEXT})


def test_a_literal_displayed_citation_passes() -> None:
    response = answer_response(("a9", QUOTE), evidence=((NORM_ID, "a9"),))

    violations = check_case_citations("plazo", response, _corpus(), AS_OF)

    assert violations == []


def test_a_fabricated_displayed_citation_is_a_hard_failure() -> None:
    response = answer_response(("a9", FABRICATED), evidence=((NORM_ID, "a9"),))

    violations = check_case_citations("plazo", response, _corpus(), AS_OF)

    assert len(violations) == 1
    assert violations[0].case_id == "plazo"
    assert violations[0].block_ref == "BOE-A-1994-26003:a9"
    assert violations[0].reason == REASON_NOT_LITERAL


def test_a_citation_without_matching_evidence_cannot_re_verify() -> None:
    response = answer_response(("a9", QUOTE), evidence=())

    violations = check_case_citations("plazo", response, _corpus(), AS_OF)

    assert [violation.reason for violation in violations] == [REASON_NO_EVIDENCE]


def test_a_citation_whose_block_is_absent_from_the_corpus_fails() -> None:
    response = answer_response(("a9", QUOTE), evidence=((NORM_ID, "a9"),))

    violations = check_case_citations("plazo", response, InMemoryCorpus({}), AS_OF)

    assert [violation.reason for violation in violations] == [REASON_UNRESOLVED]


def test_a_displayed_citation_carrying_a_discarded_verdict_is_a_hard_failure() -> None:
    response = answer_response(
        ("a9", QUOTE), evidence=((NORM_ID, "a9"),), verdict=CitationVerdict.DISCARDED
    )

    violations = check_case_citations("plazo", response, _corpus(), AS_OF)

    assert [violation.reason for violation in violations] == [REASON_DISCARDED]


def test_a_response_with_no_answer_has_no_citations_to_check() -> None:
    response = answer_response(evidence=((NORM_ID, "a9"),))
    response.answer = None

    assert check_case_citations("plazo", response, _corpus(), AS_OF) == []


def test_an_answer_given_for_a_past_date_re_verifies_against_the_law_of_then() -> None:
    response = _enervacion_response(QUOTE_2015)

    violations = check_case_citations("mh-02", response, _lec_corpus(), RUN_DATE)

    assert violations == []


def test_a_quote_literal_only_in_another_redaction_than_the_declared_one_still_fails() -> None:
    response = _enervacion_response(QUOTE_2025)

    violations = check_case_citations("mh-02", response, _lec_corpus(), RUN_DATE)

    assert [violation.reason for violation in violations] == [REASON_NOT_LITERAL]


def test_a_violation_records_the_displayed_quote_and_the_date_it_was_resolved_at() -> None:
    response = _enervacion_response(QUOTE_2025)

    violation = check_case_citations("mh-02", response, _lec_corpus(), RUN_DATE)[0]

    assert violation.quote == QUOTE_2025
    assert violation.verified_at == SIGNED_ON


def test_the_declared_date_is_read_from_the_answer_and_not_from_the_case() -> None:
    response = _enervacion_response(QUOTE_2015)

    assert read_verification_date(response, RUN_DATE) == SIGNED_ON


def test_a_response_that_reached_no_answer_declares_no_date_and_falls_back_to_the_case() -> None:
    assert read_verification_date(clarification_response(), RUN_DATE) == RUN_DATE
