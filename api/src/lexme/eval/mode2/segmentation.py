"""Segmentation as its own eval layer: match predicted spans to reference spans.

The risk map anchors every finding to a character span of the document; if the
segmentation fuses or splits clauses, the verdicts land on the wrong text and the
level metrics measure segmentation error instead of classification. So the layer is
scored on its own, before any level is read: each reference clause is paired with
the predicted span it overlaps most, and counts as correctly delimited only when
that overlap reaches the IoU threshold. The level metrics are then computed over the
well-delimited clauses alone, so each number measures one thing.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

DEFAULT_IOU_THRESHOLD = 0.8


@dataclass(frozen=True)
class LabeledSpan:
    """A half-open character range ``[start, end)`` carrying its owner's id."""

    id: str
    start: int
    end: int


class SpanMatch(BaseModel):
    """One reference clause paired with its best-overlapping predicted span.

    ``predicted_id`` is the predicted clause whose span overlaps this reference
    clause most, or ``None`` when no predicted span overlaps it at all. ``iou`` is
    that best character IoU and ``matched`` whether it reached the threshold, so a
    reference clause is "correctly delimited" exactly when ``matched`` is true.
    """

    reference_id: str
    predicted_id: str | None
    iou: float
    matched: bool


def character_iou(a: LabeledSpan, b: LabeledSpan) -> float:
    """The intersection-over-union of two half-open character ranges.

    Returns 0.0 when the ranges do not overlap or either is empty, and 1.0 when
    they coincide exactly.
    """
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    if intersection == 0:
        return 0.0
    union = (a.end - a.start) + (b.end - b.start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def match_spans(
    reference: Sequence[LabeledSpan],
    predicted: Sequence[LabeledSpan],
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[SpanMatch]:
    """Pair each reference span with the predicted span it overlaps most.

    Returns one :class:`SpanMatch` per reference span, in reference order, marking
    it matched when its best character IoU reaches ``threshold``. A reference span
    no prediction overlaps yields a match with ``predicted_id`` ``None`` and IoU 0,
    so a dropped clause is visible rather than silently absent.
    """
    matches = []
    for ref in reference:
        best_id, best_iou = _best_overlap(ref, predicted)
        matches.append(
            SpanMatch(
                reference_id=ref.id,
                predicted_id=best_id,
                iou=best_iou,
                matched=best_iou >= threshold,
            )
        )
    return matches


def _best_overlap(ref: LabeledSpan, predicted: Sequence[LabeledSpan]) -> tuple[str | None, float]:
    """The predicted span id with the highest IoU against ``ref``, and that IoU."""
    best_id: str | None = None
    best_iou = 0.0
    for candidate in predicted:
        iou = character_iou(ref, candidate)
        if iou > best_iou:
            best_id, best_iou = candidate.id, iou
    return best_id, best_iou
