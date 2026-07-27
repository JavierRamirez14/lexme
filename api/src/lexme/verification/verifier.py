"""The runtime citation verifier, shared by both modes.

No citation reaches the user without being re-verified against the corpus. Each
citation ends at exactly one :class:`CitationVerdict`; the pipeline is:

    reference parses as ``norm:block``?  -- no -->  discarded
    reference in this run's evidence?    -- no -->  discarded
    strict match in the block's point-in-time redaction  -->  verificada_directa
    else snap in that block  -->  reparada_snap (text replaced by the real span)
    else snap in exactly one other evidence block  -->  reparada_anclaje
    else  -->  descartada

References are norm qualified end to end: with several norms in the corpus a bare
block id would let a citation of one law's article 9 be checked against another's.
The verifier never calls the model and never invents an anchor; anchors are
hydrated by code through the :class:`CorpusReader` port.
"""

import logging
from collections import Counter
from collections.abc import Sequence
from datetime import date

from lexme.blocks import BlockRef
from lexme.verification.models import (
    CitationResult,
    CitationVerdict,
    CorpusReader,
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
    evidence: Sequence[BlockRef],
    target_date: date,
    corpus: CorpusReader,
) -> list[CitationResult]:
    """Verify every citation against the corpus at ``target_date``.

    ``evidence`` are the blocks retrieved in this run; a citation may only anchor
    to one of them. Returns one :class:`CitationResult` per citation, in order,
    and logs the per-verdict counts as the run's citation telemetry.
    """
    resolver = _BlockResolver(corpus, target_date)
    allowed = set(evidence)

    results = [_verify_one(citation, evidence, allowed, resolver) for citation in citations]
    _log_summary(results)
    return results


def summarize(results: Sequence[CitationResult]) -> dict[str, int]:
    """Return the per-verdict counts for ``results``, keyed by telemetry value."""
    counts = Counter(result.verdict.value for result in results)
    return {verdict.value: counts.get(verdict.value, 0) for verdict in CitationVerdict}


def _verify_one(
    citation: ProposedCitation,
    evidence: Sequence[BlockRef],
    allowed: set[BlockRef],
    resolver: "_BlockResolver",
) -> CitationResult:
    """Run the full pipeline for one citation and return its single verdict."""
    cited = BlockRef.parse(citation.block_ref)
    if cited is None or cited not in allowed:
        return _discarded(citation.block_ref)

    resolved = resolver.resolve(cited)
    if resolved is None:
        return _discarded(citation.block_ref)

    segments, has_ellipsis = split_into_segments(citation.text)
    if not segments or not segments_meet_minimum(segments, has_ellipsis):
        return _discarded(citation.block_ref)

    block_text = normalize(resolved.text)
    if contains_segments_in_order(block_text, segments):
        return CitationResult(
            verdict=CitationVerdict.VERIFIED_DIRECT,
            block_ref=str(cited),
            text=_render(segments),
            anchor=resolved.anchor,
        )

    snap = snap_citation(segments, block_text)
    if snap is not None:
        return _repaired(CitationVerdict.REPAIRED_SNAP, cited, resolved, snap)

    return _reanchor(segments, cited, evidence, resolver)


def _reanchor(
    segments: list[str],
    cited: BlockRef,
    evidence: Sequence[BlockRef],
    resolver: "_BlockResolver",
) -> CitationResult:
    """Try to snap the citation into exactly one other evidence block.

    A single unambiguous match re-anchors the citation; zero or several matches
    are treated as what they are -- a lack of guarantee -- and discarded.
    """
    matches: list[tuple[ResolvedBlock, BlockRef, SnapResult]] = []
    for block in evidence:
        if block == cited:
            continue
        resolved = resolver.resolve(block)
        if resolved is None:
            continue
        snap = snap_citation(segments, normalize(resolved.text))
        if snap is not None:
            matches.append((resolved, block, snap))

    if len(matches) != 1:
        return _discarded(str(cited))
    resolved, block, snap = matches[0]
    return _repaired(CitationVerdict.REPAIRED_ANCHOR, block, resolved, snap)


def _repaired(
    verdict: CitationVerdict,
    block: BlockRef,
    resolved: ResolvedBlock,
    snap: SnapResult,
) -> CitationResult:
    """Build a repaired result carrying the real corpus text and snap similarity."""
    return CitationResult(
        verdict=verdict,
        block_ref=str(block),
        text=snap.text,
        anchor=resolved.anchor,
        snap_similarity=snap.similarity,
    )


def _discarded(block_ref: str) -> CitationResult:
    """Build a discarded result: no anchor, nothing to show."""
    return CitationResult(verdict=CitationVerdict.DISCARDED, block_ref=block_ref, text="")


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
        self._cache: dict[BlockRef, ResolvedBlock | None] = {}

    def resolve(self, block: BlockRef) -> ResolvedBlock | None:
        """Return the block's point-in-time redaction and anchor, memoized."""
        if block not in self._cache:
            self._cache[block] = self._corpus.resolve_block(
                block.norm_id, block.block_id, self._target_date
            )
        return self._cache[block]
