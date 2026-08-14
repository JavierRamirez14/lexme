"""The citation guardrail: every displayed citation must re-verify against the corpus.

Existence and literality of shown citations are an invariant, not a metric: a 98%
is not a good score, it is a bug. Two checks enforce it. First, the verifier's own
telemetry: a citation still carrying a ``descartada`` verdict must never be shown.
Second, independent re-verification: each displayed citation is re-resolved from the
corpus at the date the answer itself was given for and its literality re-checked
against the quote as shown, so a defect in the runtime path that let a non-literal
quote through cannot hide behind its own passing verdict. Any violation is a hard
failure.

The date is the answer's, not the run's, and the distinction is the whole check: an
answer given for 2023 quotes the redaction in force in 2023, and re-resolving it at
today's date compares it against a law that did not exist when it was written. Both
directions matter -- a quote literal only in some *other* redaction than the one the
answer declares is still a hard failure.
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
    ``reason`` say which citation broke the literality invariant and how. ``quote``
    is the text as displayed and ``verified_at`` the date the block was re-resolved
    at: without those two a hard failure can only be diagnosed by running the case
    again. Both are empty in an artifact that did not record them, as is the
    reference read under its pre-expansion name, so older run artifacts still load.
    """

    model_config = ConfigDict(populate_by_name=True)

    case_id: str
    block_ref: str = Field(validation_alias=AliasChoices("block_ref", "block_id"))
    reason: str
    quote: str = ""
    verified_at: date | None = None

    @classmethod
    def from_citation(
        cls, case_id: str, citation: VerifiedCitation, reason: str, verified_at: date
    ) -> "GuardrailViolation":
        """Record a displayed citation's failure, keeping the quote and the date checked."""
        return cls(
            case_id=case_id,
            block_ref=citation.block_ref,
            reason=reason,
            quote=citation.text,
            verified_at=verified_at,
        )


def read_verification_date(response: AskResponse, case_date: date) -> date:
    """The date a response's displayed citations must be re-verified at.

    An answer states the date its corpus was resolved at, and that -- not the date
    the case was launched at -- is the law its quotes were taken from; the two part
    company whenever a disambiguation branch moves the clock back. A response that
    reached no answer declares no date, and falls back to the case's.
    """
    if response.answer is None:
        return case_date
    return response.answer.fecha_objetivo


def check_case_citations(
    case_id: str,
    response: AskResponse,
    corpus: CorpusReader,
    case_date: date,
) -> list[GuardrailViolation]:
    """Re-verify every displayed citation of one case, returning its violations.

    Reads the norm each citation belongs to from its own reference, checks that
    reference was really among the run's retrieval evidence, re-resolves it through
    ``corpus`` at the date the answer declares -- ``case_date`` only backs that up --
    and checks the shown quote is literally present. Returns an empty list when the
    case displayed no answer or every citation re-verifies.
    """
    if response.answer is None:
        return []
    target_date = read_verification_date(response, case_date)
    evidence = _evidence_refs(response)
    violations = []
    for citation in response.answer.fundamento:
        reason = _check_one(citation, evidence, corpus, target_date)
        if reason is not None:
            violations.append(
                GuardrailViolation.from_citation(case_id, citation, reason, target_date)
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
