"""Unit tests for conservative normalization and ellipsis segmentation."""

from lexme.verification.normalization import (
    MIN_SEGMENT_CHARS,
    contains_segments_in_order,
    normalize,
    segments_meet_minimum,
    split_into_segments,
)


def test_normalize_collapses_whitespace_and_newlines() -> None:
    assert normalize("el   arrendador\n\t podrá") == "el arrendador podrá"


def test_normalize_unifies_typographic_quotes_and_dashes() -> None:
    assert normalize("«renta» —anual") == '"renta" -anual'


def test_normalize_applies_unicode_nfc() -> None:
    decomposed = "artículo"
    assert normalize(decomposed) == "artículo"


def test_normalize_keeps_accents_and_punctuation() -> None:
    assert normalize("artículo") != normalize("articulo")
    assert normalize("renta, fianza.") == "renta, fianza."


def test_split_without_ellipsis_is_a_single_segment() -> None:
    segments, has_ellipsis = split_into_segments("la renta anual")
    assert segments == ["la renta anual"]
    assert has_ellipsis is False


def test_split_canonicalizes_every_ellipsis_variant() -> None:
    for marker in ("[…]", "...", "…", "(...)", "[...]"):
        segments, has_ellipsis = split_into_segments(f"la renta {marker} será la pactada")
        assert has_ellipsis is True
        assert segments == ["la renta", "será la pactada"]


def test_segments_meet_minimum_only_gates_ellipsis_quotes() -> None:
    assert segments_meet_minimum(["short"], has_ellipsis=False) is True
    assert segments_meet_minimum(["short", "also short"], has_ellipsis=True) is False


def test_segments_meet_minimum_passes_long_enough_segments() -> None:
    long_segment = "x" * MIN_SEGMENT_CHARS
    assert segments_meet_minimum([long_segment, long_segment], has_ellipsis=True) is True


def test_contains_segments_in_order_accepts_ordered_non_overlapping() -> None:
    block = "la renta será la que libremente estipulen las partes"
    assert contains_segments_in_order(block, ["la renta", "estipulen las partes"]) is True


def test_contains_segments_rejects_reordered_segments() -> None:
    block = "la renta será la que libremente estipulen las partes"
    assert contains_segments_in_order(block, ["estipulen las partes", "la renta"]) is False


def test_contains_segments_rejects_overlapping_segments() -> None:
    block = "la renta será la que libremente estipulen las partes"
    assert contains_segments_in_order(block, ["la renta será", "será la que"]) is False


def test_contains_segments_rejects_absent_segment() -> None:
    block = "la renta será la que libremente estipulen las partes"
    assert contains_segments_in_order(block, ["la fianza"]) is False
