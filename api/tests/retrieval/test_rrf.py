"""Unit tests for the reciprocal rank fusion primitive."""

from lexme.retrieval.rrf import RRF_K, reciprocal_rank_fusion


def test_a_key_ranked_by_both_lists_beats_one_ranked_by_a_single_list() -> None:
    dense = ["a", "b", "c"]
    lexical = ["b", "d", "e"]

    fused = reciprocal_rank_fusion([dense, lexical])

    assert fused[0][0] == "b"


def test_score_sums_the_reciprocal_ranks_across_lists() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "a"]]))

    expected = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 2)
    assert fused["a"] == expected
    assert fused["b"] == expected


def test_ties_are_broken_by_first_appearance_for_determinism() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])

    assert [key for key, _ in fused] == ["a", "b"]


def test_a_smaller_k_widens_the_gap_between_top_and_lower_ranks() -> None:
    ranking = [["top", "second"]]

    small_k = dict(reciprocal_rank_fusion(ranking, k=1))
    large_k = dict(reciprocal_rank_fusion(ranking, k=1000))

    assert small_k["top"] - small_k["second"] > large_k["top"] - large_k["second"]


def test_empty_rankings_fuse_to_nothing() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


def test_tuple_keys_are_supported_for_block_references() -> None:
    dense = [("BOE-A-1", "a9"), ("BOE-A-1", "a2")]
    lexical = [("BOE-A-1", "a9")]

    fused = reciprocal_rank_fusion([dense, lexical])

    assert fused[0][0] == ("BOE-A-1", "a9")
