"""The segmentation layer: character IoU and reference-to-prediction matching."""

from lexme.eval.mode2.segmentation import LabeledSpan, character_iou, match_spans


def _span(span_id: str, start: int, end: int) -> LabeledSpan:
    return LabeledSpan(id=span_id, start=start, end=end)


def test_identical_spans_have_iou_one() -> None:
    assert character_iou(_span("a", 10, 20), _span("b", 10, 20)) == 1.0


def test_disjoint_spans_have_iou_zero() -> None:
    assert character_iou(_span("a", 0, 10), _span("b", 10, 20)) == 0.0


def test_partial_overlap_is_intersection_over_union() -> None:
    # overlap [10,20) = 10; union [0,25) = 25
    assert character_iou(_span("a", 0, 20), _span("b", 10, 25)) == 10 / 25


def test_a_reference_span_matches_its_best_overlapping_prediction() -> None:
    reference = [_span("r1", 0, 100)]
    predicted = [_span("p1", 0, 40), _span("p2", 0, 95)]

    matches = match_spans(reference, predicted)

    assert matches[0].predicted_id == "p2"
    assert matches[0].matched is True
    assert matches[0].iou == 95 / 100


def test_a_reference_span_below_the_threshold_is_not_matched() -> None:
    reference = [_span("r1", 0, 100)]
    predicted = [_span("p1", 0, 70)]

    matches = match_spans(reference, predicted)

    assert matches[0].predicted_id == "p1"
    assert matches[0].iou == 70 / 100
    assert matches[0].matched is False


def test_a_reference_span_no_prediction_overlaps_is_an_unmatched_drop() -> None:
    matches = match_spans([_span("r1", 0, 50)], [_span("p1", 60, 90)])

    assert matches[0].predicted_id is None
    assert matches[0].iou == 0.0
    assert matches[0].matched is False


def test_the_threshold_is_configurable() -> None:
    reference = [_span("r1", 0, 100)]
    predicted = [_span("p1", 0, 70)]

    assert match_spans(reference, predicted, threshold=0.7)[0].matched is True


def test_one_match_is_returned_per_reference_span_in_order() -> None:
    reference = [_span("r1", 0, 10), _span("r2", 10, 20), _span("r3", 20, 30)]
    predicted = [_span("p", 0, 10)]

    matches = match_spans(reference, predicted)

    assert [match.reference_id for match in matches] == ["r1", "r2", "r3"]
