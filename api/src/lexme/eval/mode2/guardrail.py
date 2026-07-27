"""The citation guardrail, extended to the Mode 2 risk map's findings.

The same invariant Mode 1 holds -- every citation the system shows must re-resolve
against the corpus and quote it literally -- applies to the norm behind each risk
finding. A clause placed at illegal, worse-than-default or conforming carries the
article that grounds that level, and an absence white carries the law that grants
the omitted right; both are shown to the tenant, so both are re-verified here at the
run's point-in-time date. A citation that no longer resolves, or whose quote is not
literally present, is a hard failure, exactly as in Mode 1.
"""

from datetime import date

from lexme.blocks import BlockRef
from lexme.eval.guardrail import (
    REASON_DISCARDED,
    REASON_EMPTY,
    REASON_NOT_LITERAL,
    REASON_UNQUALIFIED,
    REASON_UNRESOLVED,
    GuardrailViolation,
)
from lexme.mode1.models import VerifiedCitation
from lexme.mode2.models import ContractAnalysis
from lexme.verification import CitationVerdict, CorpusReader
from lexme.verification.normalization import (
    contains_segments_in_order,
    normalize,
    split_into_segments,
)


def check_mode2_citations(
    case_id: str,
    analysis: ContractAnalysis,
    corpus: CorpusReader,
    target_date: date,
) -> list[GuardrailViolation]:
    """Re-verify every citation the risk map displays, returning its violations.

    Walks the clause findings and the absence whites, re-resolving each citation's
    own norm-qualified block against ``corpus`` at ``target_date`` and checking the
    shown quote is literally present. Returns an empty list when the analysis
    produced no risk map or every citation re-verifies.
    """
    if analysis.risk_map is None:
        return []
    violations = []
    for finding in analysis.risk_map.clause_findings:
        _collect(violations, case_id, finding.citation, corpus, target_date)
    for absence in analysis.risk_map.absence_findings:
        _collect(violations, case_id, absence.citation, corpus, target_date)
    return violations


def _collect(
    violations: list[GuardrailViolation],
    case_id: str,
    citation: VerifiedCitation | None,
    corpus: CorpusReader,
    target_date: date,
) -> None:
    """Append a violation for ``citation`` when it fails re-verification."""
    if citation is None:
        return
    reason = _check_one(citation, corpus, target_date)
    if reason is not None:
        violations.append(
            GuardrailViolation(case_id=case_id, block_ref=citation.block_ref, reason=reason)
        )


def _check_one(
    citation: VerifiedCitation,
    corpus: CorpusReader,
    target_date: date,
) -> str | None:
    """Check one displayed citation, returning a failure reason or ``None`` if it holds."""
    if citation.verdict is CitationVerdict.DISCARDED:
        return REASON_DISCARDED
    cited = BlockRef.parse(citation.block_ref)
    if cited is None:
        return REASON_UNQUALIFIED
    resolved = corpus.resolve_block(cited.norm_id, cited.block_id, target_date)
    if resolved is None:
        return REASON_UNRESOLVED
    segments, _ = split_into_segments(citation.text)
    if not segments:
        return REASON_EMPTY
    if not contains_segments_in_order(normalize(resolved.text), segments):
        return REASON_NOT_LITERAL
    return None
