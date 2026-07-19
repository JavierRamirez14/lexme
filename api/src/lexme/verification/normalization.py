"""Conservative typographic normalization and ellipsis handling.

Normalization removes only typographic noise -- Unicode form, whitespace runs,
quote and dash variants, hard spaces. It never touches accents or punctuation:
in a legal corpus, ``"articulo"`` is not ``"artículo"`` and that drift is exactly
what verification must catch. The citation and the block text are normalized the
same way, so an equal-after-normalization comparison compares like with like.
"""

import re
import unicodedata

MIN_SEGMENT_CHARS = 15
ELLIPSIS = "[…]"

_QUOTE_TRANSLATION = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
    }
)
_DASH_TRANSLATION = str.maketrans(
    {
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "‐": "-",
        "‑": "-",
    }
)
_WHITESPACE_RUN = re.compile(r"\s+")
_ELLIPSIS_RUN = re.compile(r"[(\[]?\s*\.{3,}\s*[)\]]?")


def normalize(text: str) -> str:
    """Return ``text`` with typographic noise removed but content intact.

    Applies Unicode NFC, unifies quote and dash variants, collapses every
    whitespace run (including the newlines of embedded BOE HTML and hard spaces)
    to a single space, and strips the ends.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_QUOTE_TRANSLATION)
    text = text.translate(_DASH_TRANSLATION)
    return _WHITESPACE_RUN.sub(" ", text).strip()


def split_into_segments(quote: str) -> tuple[list[str], bool]:
    """Normalize ``quote`` and split it on ellipsis markers.

    Collapses every ellipsis variant (``...``, ``…``, ``(...)``) to the canonical
    ``[…]`` before splitting. Returns the non-empty normalized segments and
    whether the quote contained an ellipsis.
    """
    canonical = _canonicalize_ellipsis(normalize(quote))
    has_ellipsis = ELLIPSIS in canonical
    segments = [segment.strip() for segment in canonical.split(ELLIPSIS)]
    return [segment for segment in segments if segment], has_ellipsis


def segments_meet_minimum(segments: list[str], has_ellipsis: bool) -> bool:
    """Return whether every ellipsis segment is at least ``MIN_SEGMENT_CHARS`` long.

    The minimum only applies once a quote is split by an ellipsis; it stops a
    trivial fragment from matching against any article. A quote without ellipsis
    has no minimum.
    """
    if not has_ellipsis:
        return True
    return all(len(segment) >= MIN_SEGMENT_CHARS for segment in segments)


def contains_segments_in_order(block_text: str, segments: list[str]) -> bool:
    """Return whether ``block_text`` contains every segment, in order, without overlap.

    ``block_text`` and ``segments`` are already normalized. A greedy
    left-to-right substring search enforces both ordering and non-overlap: each
    segment is sought only past the end of the previous match.
    """
    cursor = 0
    for segment in segments:
        index = block_text.find(segment, cursor)
        if index == -1:
            return False
        cursor = index + len(segment)
    return True


def _canonicalize_ellipsis(text: str) -> str:
    """Collapse every ellipsis variant in normalized ``text`` to ``[…]``."""
    text = text.replace("…", "...")
    return _ELLIPSIS_RUN.sub(f" {ELLIPSIS} ", text)
