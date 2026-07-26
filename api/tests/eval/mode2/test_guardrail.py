"""The Mode 2 citation guardrail: every finding's citation must re-verify literally."""

from datetime import date

from lexme.eval.guardrail import REASON_NOT_LITERAL, REASON_UNRESOLVED
from lexme.eval.mode2.guardrail import check_mode2_citations
from lexme.mode2.models import RejectionReason
from lexme.mode2.risk import RiskLevel
from lexme.verification import ResolvedBlock
from tests.eval.mode2.conftest import (
    NORM_ID,
    absence,
    analysis_of,
    anchor,
    evaluated,
    rejected,
)

TARGET_DATE = date(2024, 6, 1)
QUOTE = "se prorrogará obligatoriamente por plazos anuales"
BLOCK_TEXT = "El plazo del arrendamiento " + QUOTE + " hasta cinco años."


class OneBlockCorpus:
    """A corpus that resolves a single ``block_id`` to a fixed text, else ``None``."""

    def __init__(self, block_id: str, text: str) -> None:
        self._block_id = block_id
        self._text = text

    def resolve_block(self, norm_id: str, block_id: str, target_date: date) -> ResolvedBlock | None:
        if block_id != self._block_id:
            return None
        return ResolvedBlock(text=self._text, anchor=anchor(block_id))


def _corpus() -> OneBlockCorpus:
    return OneBlockCorpus("a9", BLOCK_TEXT)


def test_a_literal_clause_citation_passes() -> None:
    analysis = analysis_of(evaluated("c1", RiskLevel.ILEGAL, citation=("a9", QUOTE)))

    violations = check_mode2_citations("case", analysis, _corpus(), NORM_ID, TARGET_DATE)

    assert violations == []


def test_a_non_literal_clause_citation_is_a_hard_failure() -> None:
    analysis = analysis_of(evaluated("c1", RiskLevel.ILEGAL, citation=("a9", "texto inventado")))

    violations = check_mode2_citations("case", analysis, _corpus(), NORM_ID, TARGET_DATE)

    assert [v.reason for v in violations] == [REASON_NOT_LITERAL]
    assert violations[0].block_id == "a9"


def test_a_citation_that_no_longer_resolves_is_a_hard_failure() -> None:
    analysis = analysis_of(evaluated("c1", RiskLevel.ILEGAL, citation=("a404", QUOTE)))

    violations = check_mode2_citations("case", analysis, _corpus(), NORM_ID, TARGET_DATE)

    assert [v.reason for v in violations] == [REASON_UNRESOLVED]


def test_the_guardrail_also_covers_absence_whites() -> None:
    analysis = analysis_of(
        evaluated("c1", RiskLevel.CORRECTO, citation=("a9", QUOTE)),
        absences=(absence("CHK-02", citation=("a9", "no aparece literalmente aquí")),),
    )

    violations = check_mode2_citations("case", analysis, _corpus(), NORM_ID, TARGET_DATE)

    assert [v.reason for v in violations] == [REASON_NOT_LITERAL]


def test_a_finding_without_a_citation_is_not_checked() -> None:
    analysis = analysis_of(evaluated("c1", RiskLevel.NEGOCIABLE))

    violations = check_mode2_citations("case", analysis, _corpus(), NORM_ID, TARGET_DATE)

    assert violations == []


def test_a_rejected_analysis_has_no_citations_to_check() -> None:
    analysis = rejected(RejectionReason.NOT_EXTRACTABLE)

    violations = check_mode2_citations("case", analysis, _corpus(), NORM_ID, TARGET_DATE)

    assert violations == []
