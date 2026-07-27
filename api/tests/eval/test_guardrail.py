"""Tests for the citation guardrail: displayed citations must re-verify against the corpus."""

from lexme.eval.guardrail import (
    REASON_DISCARDED,
    REASON_NO_EVIDENCE,
    REASON_NOT_LITERAL,
    REASON_UNRESOLVED,
    check_case_citations,
)
from lexme.verification import CitationVerdict
from tests.eval.conftest import AS_OF, NORM_ID, InMemoryCorpus, answer_response

BLOCK_TEXT = "La duración del arrendamiento será libremente pactada por las partes."
QUOTE = "La duración del arrendamiento será libremente pactada"
FABRICATED = "El arrendador podrá desalojar al inquilino sin preaviso."


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
