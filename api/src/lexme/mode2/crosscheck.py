"""The checklist × contract cross-check: pure code, zero model, the absence whites.

The clause findings say which checklist items each clause touches. A right that no
clause addresses is one the contract stays silent about, and code -- not the
model -- turns that silence into an :class:`AbsenceFinding`: the item's static
absence text (which already carries any consequence and deadline), its plain
right, and the law that grants it. A clause only *addresses* a right when it was
actually judged against it -- evaluated, or found relevant but inconclusive; a
clause the mapping tied to a right but that turned out merely informative does not
reflect the right, so it cannot suppress the white. The reflected and worsened
cases live in the clause findings themselves; the closing meta-rule (a right that
is never omitted, only invoked) is excluded.
"""

from collections.abc import Sequence
from datetime import date

from lexme.checklist import Checklist, ChecklistItem, SilenceTone
from lexme.mode1.models import VerifiedCitation
from lexme.mode2.risk import AbsenceFinding, ClauseFinding, CoverageStatus
from lexme.verification import CitationVerdict, CorpusReader, VerifiedAnchor

_ADDRESSING_COVERAGE: frozenset[CoverageStatus] = frozenset(
    {CoverageStatus.EVALUADA, CoverageStatus.NO_CONCLUYENTE}
)


def cross_check(
    checklist: Checklist,
    clause_findings: Sequence[ClauseFinding],
    corpus: CorpusReader,
    target_date: date,
) -> list[AbsenceFinding]:
    """Produce one absence finding per checklist right no clause addresses.

    Walks the checklist in order; an item a judged clause addressed (evaluated or
    inconclusive) is either reflected in that clause's own finding or inherits its
    inconclusiveness, so it yields no white. A right only mapped to informative or
    out-of-scope clauses is still silent and does produce one. The closing
    meta-rule, never an omissible right, is skipped, and each white's citation
    anchor is hydrated from the corpus for display.
    """
    addressed = {
        chk_id
        for finding in clause_findings
        if finding.coverage in _ADDRESSING_COVERAGE
        for chk_id in finding.chk_ids
    }
    findings: list[AbsenceFinding] = []
    for item in checklist.items:
        if item.silence_tone is SilenceTone.NOT_APPLICABLE:
            continue
        if item.id in addressed:
            continue
        findings.append(_absence_finding(item, checklist.norm_id, corpus, target_date))
    return findings


def _absence_finding(
    item: ChecklistItem,
    norm_id: str,
    corpus: CorpusReader,
    target_date: date,
) -> AbsenceFinding:
    """Build the white for one omitted item from its static template and citation."""
    return AbsenceFinding(
        item_id=item.id,
        right=item.right,
        silence_tone=item.silence_tone.value,
        explanation=item.absence_template,
        citation=_hydrate_citation(item, norm_id, corpus, target_date),
    )


def _hydrate_citation(
    item: ChecklistItem,
    norm_id: str,
    corpus: CorpusReader,
    target_date: date,
) -> VerifiedCitation | None:
    """Hydrate the item's static citation with its corpus anchor for display.

    The quote's literalness is guaranteed at build time, so this only resolves the
    anchor (ELI, URL, title) to show it; an unresolvable block yields ``None``
    rather than a citation with no anchor.
    """
    resolved = corpus.resolve_block(norm_id, item.citation.block_id, target_date)
    if resolved is None:
        return None
    anchor: VerifiedAnchor = resolved.anchor
    return VerifiedCitation(
        block_id=item.citation.block_id,
        text=item.citation.text,
        verdict=CitationVerdict.VERIFIED_DIRECT,
        anchor=anchor,
    )
