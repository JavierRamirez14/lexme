"""Tests for retrieval recall by layer and per sub-query, and the agentic delta."""

from lexme.eval.cases import EvalCase
from lexme.eval.metrics import build_case_result
from tests.eval.conftest import NORM_ID, block_ref, traced_response


def _keys(*block_ids: str) -> tuple[tuple[str, str], ...]:
    """Turn block ids into ``(norm_id, block_id)`` ranking keys."""
    return tuple((NORM_ID, block_id) for block_id in block_ids)


def _gold(*block_ids: str) -> tuple[str, ...]:
    """Turn block ids into the norm-qualified references a case declares as gold."""
    return tuple(block_ref(block_id) for block_id in block_ids)


def test_recall_is_attributed_per_retrieval_layer() -> None:
    case = EvalCase(id="c", question="q", gold_block_refs=_gold("a9", "a36"))
    response = traced_response(
        dense=_keys("a9", "x"),
        lexical=_keys("a36", "y"),
        fused=_keys("a9", "a36"),
        evidence=_keys("a9", "a36"),
    )

    result = build_case_result(case, response, [])

    by_layer = {layer.layer: layer.recall for layer in result.retrieval.layers}
    assert by_layer["dense"] == 0.5
    assert by_layer["lexical"] == 0.5
    assert by_layer["fused"] == 1.0
    assert by_layer["evidence"] == 1.0


def test_recall_is_attributed_per_sub_query() -> None:
    case = EvalCase(id="c", question="q", gold_block_refs=_gold("a9", "a36"))
    response = traced_response(
        dense=_keys("a9"),
        lexical=_keys("a36"),
        fused=_keys("a9", "a36"),
        evidence=_keys("a9", "a36"),
    )

    result = build_case_result(case, response, [])

    (sub,) = result.retrieval.subqueries
    assert sub.id == "sq1"
    assert sub.recovered_gold == list(_gold("a9", "a36"))
    assert sub.recall == 1.0


def test_the_agentic_delta_is_a_first_pass_to_final_recall_pair() -> None:
    case = EvalCase(id="c", question="q", gold_block_refs=_gold("a9", "a36"))
    response = traced_response(
        dense=_keys("a9", "a36"),
        lexical=_keys("a9", "a36"),
        fused=_keys("a9", "a36"),
        evidence=_keys("a9", "a36"),
        first_pass_evidence=("a9",),
    )

    result = build_case_result(case, response, [])

    assert result.first_pass_recall == 0.5
    assert result.recall == 1.0
    assert result.recall_delta == 0.5


def test_a_case_without_gold_has_no_layer_recall() -> None:
    case = EvalCase(id="c", question="q")
    response = traced_response(
        dense=_keys("a9"), lexical=_keys("a9"), fused=_keys("a9"), evidence=_keys("a9")
    )

    result = build_case_result(case, response, [])

    assert all(layer.recall is None for layer in result.retrieval.layers)
    assert result.first_pass_recall is None
    assert result.recall_delta is None
