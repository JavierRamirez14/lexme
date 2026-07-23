"""Code-side anchoring of proposed clauses to literal spans of the document.

The model proposes clause snippets; it does not get to assert they are in the
document. Each snippet is normalized and aligned to a real span of the normalized
document -- by strict containment first, then by a deterministic difflib snap that
only accepts a near-exact match, reusing the same thresholds as citation snap. A
snippet that will not anchor breaks the whole segmentation: a lease we cannot
faithfully quote back is not one we analyze, so a broken anchor becomes an honest
``no_analizable`` rather than text the document does not contain.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from lexme.verification.normalization import normalize
from lexme.verification.snap import SNAP_LENGTH_TOLERANCE, SNAP_SIMILARITY_THRESHOLD

MIN_CLAUSE_CHARS = 20


@dataclass(frozen=True)
class AnchoredSpan:
    """A clause anchored to a literal span of the normalized document.

    ``text`` is the real document span (equal to the snippet when it matched
    strictly, or the near-exact span it snapped to); ``start`` and ``end`` are its
    half-open range in the normalized document.
    """

    start: int
    end: int
    text: str


def anchor_clauses(document: str, snippets: Sequence[str]) -> list[AnchoredSpan] | None:
    """Anchor every snippet to a literal span of ``document``, or return ``None``.

    Anchoring is all-or-nothing: one snippet that will not align to a real span of
    the document fails the whole segmentation, so the caller can reject the
    document instead of showing a clause it cannot back. Snippets are anchored
    independently, each to its best literal span anywhere in the document.
    """
    normalized_document = normalize(document)
    spans: list[AnchoredSpan] = []
    for snippet in snippets:
        span = _anchor_one(normalized_document, snippet)
        if span is None:
            return None
        spans.append(span)
    return spans


def _anchor_one(document: str, snippet: str) -> AnchoredSpan | None:
    """Anchor one snippet: exact containment, else a near-exact snap."""
    needle = normalize(snippet)
    if len(needle) < MIN_CLAUSE_CHARS:
        return None
    index = document.find(needle)
    if index != -1:
        return AnchoredSpan(start=index, end=index + len(needle), text=needle)
    return _snap(document, needle)


def _snap(document: str, needle: str) -> AnchoredSpan | None:
    """Align ``needle`` to the best-matching span of ``document`` above threshold.

    The span runs from the first to the last aligned run; an inflated span trips
    the length tolerance and fails -- the safe direction, never a wrong span shown.
    """
    matcher = SequenceMatcher(None, document, needle, autojunk=False)
    matches = [match for match in matcher.get_matching_blocks() if match.size > 0]
    if not matches:
        return None
    start = matches[0].a
    end = matches[-1].a + matches[-1].size
    span = document[start:end]
    similarity = SequenceMatcher(None, span, needle, autojunk=False).ratio()
    if similarity < SNAP_SIMILARITY_THRESHOLD:
        return None
    if abs(len(span) - len(needle)) > SNAP_LENGTH_TOLERANCE * len(needle):
        return None
    return AnchoredSpan(start=start, end=end, text=span)
