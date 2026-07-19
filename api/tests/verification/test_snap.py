"""Unit tests for deterministic snap repair and its regression constants."""

from lexme.verification.snap import (
    SNAP_LENGTH_TOLERANCE,
    SNAP_SIMILARITY_THRESHOLD,
    _within_length_tolerance,
    snap_citation,
)

BLOCK = "la renta sera la que libremente estipulen las partes en el contrato"


def test_regression_constants_are_pinned() -> None:
    assert SNAP_SIMILARITY_THRESHOLD == 0.90
    assert SNAP_LENGTH_TOLERANCE == 0.20


def test_snap_repairs_a_single_changed_word_to_the_real_text() -> None:
    quote = "la renta sera la que libremente estipulan las partes"
    result = snap_citation([quote], BLOCK)
    assert result is not None
    assert result.text == "la renta sera la que libremente estipulen las partes"
    assert result.similarity >= SNAP_SIMILARITY_THRESHOLD


def test_snap_rejects_text_unlike_the_block() -> None:
    assert snap_citation(["el plazo minimo de duracion sera de cinco anos"], BLOCK) is None


def test_snap_rejects_a_scattered_subsequence_fabrication() -> None:
    assert snap_citation(["renta partes contrato"], BLOCK) is None


def test_length_tolerance_band_tracks_the_pinned_constant() -> None:
    assert _within_length_tolerance(120, 100) is True
    assert _within_length_tolerance(80, 100) is True
    assert _within_length_tolerance(121, 100) is False
    assert _within_length_tolerance(79, 100) is False
    assert _within_length_tolerance(1, 0) is False


def test_snap_requires_every_ellipsis_segment_to_align() -> None:
    good = "la renta sera la que libremente estipulan"
    bad = "clausula completamente inventada por el modelo hoy"
    assert snap_citation([good, bad], BLOCK) is None


def test_snap_rejoins_aligned_segments_with_the_canonical_ellipsis() -> None:
    result = snap_citation(
        ["la renta sera la que", "estipulen las partes en el"], BLOCK
    )
    assert result is not None
    assert result.text == "la renta sera la que […] estipulen las partes en el"
