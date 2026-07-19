"""Deterministic snap repair: align a citation to the real block text.

Snap is the only repair. The LLM never re-writes its own citation: renegotiating
with the model that just fabricated invites a better-crafted fabrication and puts
a non-deterministic call inside the guardrail that presumes determinism. Snap
aligns each segment to the block by sequence matching (difflib, no model) and, on
acceptance, replaces the quote with the real corpus span -- literal by
construction. If the official text does not back it, the citation dies.

The acceptance thresholds are constants under regression test, not sacred numbers.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher

from lexme.verification.normalization import ELLIPSIS

SNAP_SIMILARITY_THRESHOLD = 0.90
SNAP_LENGTH_TOLERANCE = 0.20


@dataclass(frozen=True)
class SnapResult:
    """A successful snap of a whole citation onto one block.

    ``text`` is the real block span (segments rejoined with ``[…]``);
    ``similarity`` is the weakest per-segment similarity.
    """

    text: str
    similarity: float


@dataclass(frozen=True)
class _SegmentSnap:
    """One segment aligned to a span of the block."""

    start: int
    end: int
    text: str
    similarity: float


def snap_citation(segments: list[str], block_text: str) -> SnapResult | None:
    """Snap every segment onto ``block_text`` in order, or return ``None``.

    ``block_text`` and ``segments`` are already normalized. Every segment must
    align above threshold and within the length tolerance, and the spans must run
    left to right without overlap. On success the real spans are rejoined with the
    canonical ellipsis so an elision stays marked.
    """
    snaps: list[_SegmentSnap] = []
    cursor = 0
    for segment in segments:
        snap = _snap_segment(segment, block_text, cursor)
        if snap is None:
            return None
        snaps.append(snap)
        cursor = snap.end
    text = f" {ELLIPSIS} ".join(snap.text for snap in snaps)
    return SnapResult(text=text, similarity=min(snap.similarity for snap in snaps))


def _snap_segment(segment: str, block_text: str, offset: int) -> _SegmentSnap | None:
    """Align ``segment`` to the best span of ``block_text`` at or after ``offset``."""
    region = block_text[offset:]
    matcher = SequenceMatcher(None, region, segment, autojunk=False)
    matches = [match for match in matcher.get_matching_blocks() if match.size > 0]
    if not matches:
        return None
    # Span the block from the first to the last aligned run. A far trailing run
    # can inflate the span; that inflation trips the length tolerance below and
    # discards the citation -- the safe direction, never a wrong span shown.
    start = offset + matches[0].a
    end = offset + matches[-1].a + matches[-1].size
    span = block_text[start:end]
    similarity = SequenceMatcher(None, span, segment, autojunk=False).ratio()
    if similarity < SNAP_SIMILARITY_THRESHOLD:
        return None
    if not _within_length_tolerance(len(span), len(segment)):
        return None
    return _SegmentSnap(start=start, end=end, text=span, similarity=similarity)


def _within_length_tolerance(span_length: int, segment_length: int) -> bool:
    """Return whether ``span_length`` is within the tolerance band of ``segment_length``."""
    if segment_length == 0:
        return False
    return abs(span_length - segment_length) <= SNAP_LENGTH_TOLERANCE * segment_length
