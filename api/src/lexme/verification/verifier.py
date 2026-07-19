"""The runtime citation verifier, shared by both modes.

No citation reaches the user without being re-verified against the corpus. Each
citation ends at exactly one :class:`CitationVerdict`; the pipeline is:

    id in this run's evidence?  -- no -->  discarded
    strict match in the block's point-in-time redaction  -->  verificada_directa
    else snap in that block  -->  reparada_snap (text replaced by the real span)
    else snap in exactly one other evidence block  -->  reparada_anclaje
    else  -->  descartada

The verifier never calls the model and never invents an anchor; anchors are
hydrated by code through the :class:`CorpusReader` port.
"""

import logging
from collections import Counter
from collections.abc import Sequence
from datetime import date

from lexme.verification.models import (
    CitationResult,
    CitationVerdict,
    CorpusReader,
    EvidenceBlock,
    ProposedCitation,
    ResolvedBlock,
)
from lexme.verification.normalization import (
    ELLIPSIS,
    contains_segments_in_order,
    normalize,
    segments_meet_minimum,
    split_into_segments,
)
from lexme.verification.snap import SnapResult, snap_citation

logger = logging.getLogger(__name__)


def verify_citations(
    citations: Sequence[ProposedCitation],
    evidence: Sequence[EvidenceBlock],
    target_date: date,
    corpus: CorpusReader,
) -> list[CitationResult]:
    """Verify every citation against the corpus at ``target_date``.

    ``evidence`` are the blocks retrieved in this run; a citation may only anchor
    to one of them. Returns one :class:`CitationResult` per citation, in order,
    and logs the per-verdict counts as the run's citation telemetry.
    """
    resolver = _BlockResolver(corpus, target_date)
    evidence_by_id: dict[str, EvidenceBlock] = {}
    for block in evidence:
        evidence_by_id.setdefault(block.block_id, block)

    results = [
        _verify_one(citation, evidence, evidence_by_id, resolver) for citation in citations
    ]
    _log_summary(results)
    return results


def summarize(results: Sequence[CitationResult]) -> dict[str, int]:
    """Return the per-verdict counts for ``results``, keyed by telemetry value."""
    counts = Counter(result.verdict.value for result in results)
    return {verdict.value: counts.get(verdict.value, 0) for verdict in CitationVerdict}


def _verify_one(
    citation: ProposedCitation,
    evidence: Sequence[EvidenceBlock],
    evidence_by_id: dict[str, EvidenceBlock],
    resolver: "_BlockResolver",
) -> CitationResult:
    """Run the full pipeline for one citation and return its single verdict."""
    cited = evidence_by_id.get(citation.block_id)
    if cited is None:
        return _discarded(citation.block_id)

    resolved = resolver.resolve(cited)
    if resolved is None:
        return _discarded(citation.block_id)

    segments, has_ellipsis = split_into_segments(citation.text)
    if not segments or not segments_meet_minimum(segments, has_ellipsis):
        return _discarded(citation.block_id)

    block_text = normalize(resolved.text)
    if contains_segments_in_order(block_text, segments):
        return CitationResult(
            verdict=CitationVerdict.VERIFIED_DIRECT,
            block_id=citation.block_id,
            text=_render(segments),
            anchor=resolved.anchor,
        )

    snap = snap_citation(segments, block_text)
    if snap is not None:
        return _repaired(CitationVerdict.REPAIRED_SNAP, citation.block_id, resolved, snap)

    return _reanchor(segments, citation.block_id, evidence, resolver)


def _reanchor(
    segments: list[str],
    cited_block_id: str,
    evidence: Sequence[EvidenceBlock],
    resolver: "_BlockResolver",
) -> CitationResult:
    """Try to snap the citation into exactly one other evidence block.

    A single unambiguous match re-anchors the citation; zero or several matches
    are treated as what they are -- a lack of guarantee -- and discarded.
    """
    matches: list[tuple[ResolvedBlock, EvidenceBlock, SnapResult]] = []
    for block in evidence:
        if block.block_id == cited_block_id:
            continue
        resolved = resolver.resolve(block)
        if resolved is None:
            continue
        snap = snap_citation(segments, normalize(resolved.text))
        if snap is not None:
            matches.append((resolved, block, snap))

    if len(matches) != 1:
        return _discarded(cited_block_id)
    resolved, block, snap = matches[0]
    return _repaired(CitationVerdict.REPAIRED_ANCHOR, block.block_id, resolved, snap)


def _repaired(
    verdict: CitationVerdict,
    block_id: str,
    resolved: ResolvedBlock,
    snap: SnapResult,
) -> CitationResult:
    """Build a repaired result carrying the real corpus text and snap similarity."""
    return CitationResult(
        verdict=verdict,
        block_id=block_id,
        text=snap.text,
        anchor=resolved.anchor,
        snap_similarity=snap.similarity,
    )


def _discarded(block_id: str) -> CitationResult:
    """Build a discarded result: no anchor, nothing to show."""
    return CitationResult(verdict=CitationVerdict.DISCARDED, block_id=block_id, text="")


def _render(segments: list[str]) -> str:
    """Rejoin verified segments with the canonical ellipsis for display."""
    return f" {ELLIPSIS} ".join(segments)


def _log_summary(results: Sequence[CitationResult]) -> None:
    """Emit the run's per-verdict citation counts to the structured log."""
    logger.info("citation verification verdicts: %s", summarize(results))


class _BlockResolver:
    """Resolve evidence blocks through the corpus port, caching by block."""

    def __init__(self, corpus: CorpusReader, target_date: date) -> None:
        self._corpus = corpus
        self._target_date = target_date
        self._cache: dict[tuple[str, str], ResolvedBlock | None] = {}

    def resolve(self, block: EvidenceBlock) -> ResolvedBlock | None:
        """Return the block's point-in-time redaction and anchor, memoized."""
        key = (block.norm_id, block.block_id)
        if key not in self._cache:
            self._cache[key] = self._corpus.resolve_block(
                block.norm_id, block.block_id, self._target_date
            )
        return self._cache[key]
