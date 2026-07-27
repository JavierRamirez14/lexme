"""Tests for the norm-qualified block reference."""

from lexme.blocks import BlockRef


def test_renders_as_norm_and_block_joined_by_a_colon() -> None:
    assert str(BlockRef(norm_id="BOE-A-1994-26003", block_id="a9")) == "BOE-A-1994-26003:a9"


def test_parses_its_own_rendering() -> None:
    ref = BlockRef(norm_id="BOE-A-1889-4763", block_id="art1554")

    assert BlockRef.parse(str(ref)) == ref


def test_parses_a_norm_id_containing_hyphens() -> None:
    ref = BlockRef.parse("BOE-A-2000-323:a22")

    assert ref == BlockRef(norm_id="BOE-A-2000-323", block_id="a22")


def test_parses_ignoring_surrounding_whitespace() -> None:
    assert BlockRef.parse("  BOE-A-1994-26003:a9 ") == BlockRef(
        norm_id="BOE-A-1994-26003", block_id="a9"
    )


def test_an_unqualified_block_id_does_not_parse() -> None:
    assert BlockRef.parse("a9") is None


def test_a_reference_missing_either_side_does_not_parse() -> None:
    assert BlockRef.parse(":a9") is None
    assert BlockRef.parse("BOE-A-1994-26003:") is None


def test_the_first_colon_separates_norm_from_block() -> None:
    assert BlockRef.parse("BOE-A-1994-26003:a9:extra") == BlockRef(
        norm_id="BOE-A-1994-26003", block_id="a9:extra"
    )


def test_two_norms_sharing_a_block_id_are_different_references() -> None:
    lau = BlockRef(norm_id="BOE-A-1994-26003", block_id="a9")
    vivienda = BlockRef(norm_id="BOE-A-2023-12203", block_id="a9")

    assert lau != vivienda
    assert len({lau, vivienda}) == 2
