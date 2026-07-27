"""The citation guardrail: every displayed citation must re-verify against the corpus.

Existence and literality of shown citations are an invariant, not a metric: a 98%
is not a good score, it is a bug. Two checks enforce it. First, the verifier's own
telemetry: a citation still carrying a ``descartada`` verdict must never be shown.
Second, independent re-verification: each displayed citation is re-resolved from the
corpus at the run's point-in-time date and its literality re-checked against the
quote as shown, so a defect in the runtime path that let a non-literal quote through
cannot hide behind its own passing verdict. Any violation is a hard failure.
"""

from datetime import date

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from lexme.blocks import BlockRef
from lexme.mode1 import AskResponse, VerifiedCitation
from lexme.verification import CitationVerdict, CorpusReader
from lexme.verification.normalization import (
    contains_segments_in_order,
    normalize,
    split_into_segments,
)

REASON_DISCARDED = "displayed citation still carries a 'descartada' verdict from the verifier"
REASON_UNQUALIFIED = "displayed citation does not name a 'norm:block' reference"
REASON_NO_EVIDENCE = "cited block is not among the run's evidence, so it cannot be re-verified"
REASON_UNRESOLVED = "cited block did not resolve against the corpus at the run's date"
REASON_EMPTY = "displayed citation has no quotable text"
REASON_NOT_LITERAL = "displayed quote is not literally present in the corpus block"


class GuardrailViolation(BaseModel):
    """One displayed citation that failed re-verification, with why and where.

    ``case_id`` names the failing case so a run can point at it; ``block_ref`` and
    ``reason`` say which citation broke the literality invariant and how. The
    reference also reads its pre-expansion name so older run artifacts still load.
    """

    model_config = ConfigDict(populate_by_name=True)

    case_id: str
    block_ref: str = Field(validation_alias=AliasChoices("block_ref", "block_id"))
    reason: str


def check_case_citations(
    case_id: str,
    response: AskResponse,
    corpus: CorpusReader,
    target_date: date,
) -> list[GuardrailViolation]:
    """Re-verify every displayed citation of one case, returning its violations.

    Reads the norm each citation belongs to from its own reference, checks that
    reference was really among the run's retrieval evidence, re-resolves it through
    ``corpus`` at ``target_date`` and checks the shown quote is literally present.
    Returns an empty list when the case displayed no answer or every citation
    re-verifies.
    """
    if response.answer is None:
        return []
    evidence = _evidence_refs(response)
    violations = []
    for citation in response.answer.fundamento:
        reason = _check_one(citation, evidence, corpus, target_date)
        if reason is not None:
            violations.append(
                GuardrailViolation(case_id=case_id, block_ref=citation.block_ref, reason=reason)
            )
    return violations


def _check_one(
    citation: VerifiedCitation,
    evidence: set[BlockRef],
    corpus: CorpusReader,
    target_date: date,
) -> str | None:
    """Check one displayed citation, returning a failure reason or ``None`` if it holds.

    Reads the verifier's per-citation verdict first, then re-verifies literality
    against the corpus independently of that verdict.
    """
    if citation.verdict is CitationVerdict.DISCARDED:
        return REASON_DISCARDED
    cited = BlockRef.parse(citation.block_ref)
    if cited is None:
        return REASON_UNQUALIFIED
    if cited not in evidence:
        return REASON_NO_EVIDENCE
    return _reverify(cited, citation.text, corpus, target_date)


def _reverify(
    cited: BlockRef,
    quote: str,
    corpus: CorpusReader,
    target_date: date,
) -> str | None:
    """Re-verify one displayed quote, returning a failure reason or ``None`` if it holds."""
    resolved = corpus.resolve_block(cited.norm_id, cited.block_id, target_date)
    if resolved is None:
        return REASON_UNRESOLVED
    segments, _ = split_into_segments(quote)
    if not segments:
        return REASON_EMPTY
    if not contains_segments_in_order(normalize(resolved.text), segments):
        return REASON_NOT_LITERAL
    return None


def _evidence_refs(response: AskResponse) -> set[BlockRef]:
    """The norm-qualified blocks the run surfaced as evidence, from its agentic trace."""
    if response.agentic is None:
        return set()
    return {
        BlockRef(norm_id=block.norm_id, block_id=block.block_id)
        for subquery in response.agentic.subqueries
        for block in subquery.evidence
    }
