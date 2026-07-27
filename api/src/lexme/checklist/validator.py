"""Validate a checklist against the ingested corpus, so it cannot drift silently.

Two independent checks run per item at a target date: every anchor block must
resolve against the corpus, and the item's citation must verify *directly* --
the exact quote present in the exact block -- through the shared runtime citation
verifier. Reusing :func:`verify_citations` holds the checklist to the same bar as
a model-authored citation at runtime; a snap repair or a re-anchor is treated as
a failure here, because a curated datum whose text no longer matches the law is
precisely the drift this check exists to catch.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from lexme.blocks import BlockRef
from lexme.checklist.models import Checklist, ChecklistItem
from lexme.verification import (
    CitationVerdict,
    CorpusReader,
    ProposedCitation,
    verify_citations,
)


@dataclass(frozen=True)
class ChecklistFinding:
    """One reason an item failed validation: which item, which block, and why."""

    item_id: str
    block_id: str
    message: str


@dataclass(frozen=True)
class ChecklistReport:
    """The outcome of validating a checklist: every finding, in item order.

    ``is_valid`` is true only when nothing failed; the CLI turns that into an
    exit code and CI turns the exit code into a red build.
    """

    findings: tuple[ChecklistFinding, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether the checklist validated cleanly against the corpus."""
        return not self.findings


def validate_checklist(
    checklist: Checklist,
    corpus: CorpusReader,
    target_date: date,
) -> ChecklistReport:
    """Validate every item's anchors and citation against ``corpus`` at ``target_date``.

    Returns a :class:`ChecklistReport` collecting one finding per unresolved
    anchor and one per citation that does not verify directly, so a run reports
    every problem at once rather than only the first.
    """
    findings: list[ChecklistFinding] = []
    for item in checklist.items:
        findings.extend(_validate_item(item, checklist.norm_id, corpus, target_date))
    return ChecklistReport(findings=tuple(findings))


def _validate_item(
    item: ChecklistItem,
    norm_id: str,
    corpus: CorpusReader,
    target_date: date,
) -> list[ChecklistFinding]:
    """Collect the anchor-resolution and citation-verification findings for one item."""
    evidence = [BlockRef(norm_id=norm_id, block_id=anchor) for anchor in item.anchors]
    findings = list(_unresolved_anchors(item, evidence, corpus, target_date))
    findings.extend(_citation_findings(item, norm_id, evidence, corpus, target_date))
    return findings


def _unresolved_anchors(
    item: ChecklistItem,
    evidence: Sequence[BlockRef],
    corpus: CorpusReader,
    target_date: date,
) -> list[ChecklistFinding]:
    """A finding for each anchor block that does not resolve against the corpus."""
    findings = []
    for block in evidence:
        if corpus.resolve_block(block.norm_id, block.block_id, target_date) is None:
            findings.append(
                ChecklistFinding(
                    item_id=item.id,
                    block_id=block.block_id,
                    message=(
                        f"anchor {block.block_id} does not resolve in {block.norm_id} "
                        f"at {target_date.isoformat()}"
                    ),
                )
            )
    return findings


def _citation_findings(
    item: ChecklistItem,
    norm_id: str,
    evidence: Sequence[BlockRef],
    corpus: CorpusReader,
    target_date: date,
) -> list[ChecklistFinding]:
    """A finding when the item's citation does not verify directly against its block."""
    ref = BlockRef(norm_id=norm_id, block_id=item.citation.block_id)
    citation = ProposedCitation(block_ref=str(ref), text=item.citation.text)
    (result,) = verify_citations([citation], evidence, target_date, corpus)
    if result.verdict is CitationVerdict.VERIFIED_DIRECT:
        return []
    return [
        ChecklistFinding(
            item_id=item.id,
            block_id=item.citation.block_id,
            message=f"citation does not verify directly (verdict={result.verdict.value})",
        )
    ]
