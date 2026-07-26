"""Builders for the Mode 2 eval tests: reference cases and canned risk maps.

The metrics and guardrail tests need no pipeline: a reference case and a hand-built
:class:`ContractAnalysis` exercise them deterministically. The spans are chosen so a
finding placed at a clause's own range matches it exactly, letting a test isolate a
classification error from a segmentation error.
"""

from datetime import date

from lexme.eval.mode2.cases import AbsenceTruth, ClauseTruth, Mode2EvalCase
from lexme.mode1.models import VerifiedCitation
from lexme.mode2.models import ContractAnalysis, Mode2Outcome, Rejection, RejectionReason
from lexme.mode2.risk import (
    AbsenceFinding,
    ClauseFinding,
    CoverageStatus,
    RiskLevel,
    RiskMap,
)
from lexme.verification import CitationVerdict, VerifiedAnchor

NORM_ID = "BOE-A-1994-26003"


def clause_truth(
    clause_id: str,
    level: RiskLevel,
    *,
    start: int = 0,
    end: int = 100,
    chk_ids: tuple[str, ...] = (),
) -> ClauseTruth:
    """A reference clause with a known level and span."""
    return ClauseTruth(
        clause_id=clause_id,
        heading=clause_id,
        text="x",
        start=start,
        end=end,
        expected_level=level,
        chk_ids=chk_ids,
    )


def case_of(
    *clauses: ClauseTruth,
    absences: tuple[str, ...] = (),
    document: str = "d",
    case_id: str = "case",
) -> Mode2EvalCase:
    """A reference case from clauses and the item ids it deliberately omits."""
    return Mode2EvalCase(
        id=case_id,
        document=document,
        clauses=clauses,
        expected_absences=tuple(AbsenceTruth(item_id=item, right=item) for item in absences),
    )


def anchor(block_id: str) -> VerifiedAnchor:
    """A filled-in anchor for a finding's citation."""
    return VerifiedAnchor(
        eli="https://www.boe.es/eli/es/l/1994/11/24/29",
        consolidated_html_url="https://www.boe.es/buscar/act.php?id=" + NORM_ID,
        block_id=block_id,
        title="Artículo",
        effective_date=date(2019, 3, 6),
    )


def evaluated(
    clause_id: str,
    level: RiskLevel,
    *,
    start: int = 0,
    end: int = 100,
    citation: tuple[str, str] | None = None,
) -> ClauseFinding:
    """An evaluated clause finding at ``level``, optionally citing ``(block_id, text)``."""
    return ClauseFinding(
        clause_id=clause_id,
        heading=clause_id,
        snippet="x",
        start=start,
        end=end,
        coverage=CoverageStatus.EVALUADA,
        level=level,
        citation=_citation(citation),
    )


def covered(
    clause_id: str,
    coverage: CoverageStatus,
    *,
    start: int = 0,
    end: int = 100,
) -> ClauseFinding:
    """A finding placed at a non-evaluated coverage status (informative, abstained)."""
    return ClauseFinding(
        clause_id=clause_id,
        heading=clause_id,
        snippet="x",
        start=start,
        end=end,
        coverage=coverage,
    )


def absence(item_id: str, *, citation: tuple[str, str] | None = None) -> AbsenceFinding:
    """An absence white for ``item_id``, optionally citing ``(block_id, text)``."""
    return AbsenceFinding(
        item_id=item_id,
        right=item_id,
        silence_tone="ex_lege_informativo",
        explanation="La ley te lo reconoce.",
        citation=_citation(citation),
    )


def analysis_of(
    *findings: ClauseFinding,
    absences: tuple[AbsenceFinding, ...] = (),
) -> ContractAnalysis:
    """An analyzed contract carrying a risk map of the given findings and whites."""
    return ContractAnalysis(
        outcome=Mode2Outcome.ANALYZED,
        risk_map=RiskMap(clause_findings=list(findings), absence_findings=list(absences)),
    )


def rejected(reason: RejectionReason) -> ContractAnalysis:
    """A rejected analysis that carries no risk map."""
    outcome = (
        Mode2Outcome.OUT_OF_SCOPE
        if reason in (RejectionReason.OUT_OF_SCOPE_USE, RejectionReason.PRIOR_REDACTION)
        else Mode2Outcome.NOT_ANALYZABLE
    )
    return ContractAnalysis(outcome=outcome, rejection=Rejection(reason=reason, message="stopped"))


def _citation(citation: tuple[str, str] | None) -> VerifiedCitation | None:
    """Build a verified citation from ``(block_id, text)`` or return ``None``."""
    if citation is None:
        return None
    block_id, text = citation
    return VerifiedCitation(
        block_id=block_id,
        text=text,
        verdict=CitationVerdict.VERIFIED_DIRECT,
        anchor=anchor(block_id),
    )
