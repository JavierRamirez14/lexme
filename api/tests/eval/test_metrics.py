"""Tests for the eval quality metrics: recall over accumulated evidence, and aggregates."""

from lexme.eval.cases import EvalCase
from lexme.eval.metrics import aggregate, build_case_result
from tests.eval.conftest import NORM_ID, answer_response


def test_recall_counts_gold_blocks_in_the_accumulated_evidence() -> None:
    case = EvalCase(id="c", question="q", gold_block_ids=("a9", "a36"))
    response = answer_response(("a9", "quote"), evidence=((NORM_ID, "a9"),))

    result = build_case_result(case, response, [])

    assert result.recall == 0.5
    assert result.retrieved_block_ids == ["a9"]


def test_a_case_without_gold_blocks_has_no_recall() -> None:
    case = EvalCase(id="c", question="q")
    response = answer_response(("a9", "quote"), evidence=((NORM_ID, "a9"),))

    result = build_case_result(case, response, [])

    assert result.recall is None


def test_the_aggregate_averages_only_scored_recalls() -> None:
    scored = build_case_result(
        EvalCase(id="scored", question="q", gold_block_ids=("a9",)),
        answer_response(("a9", "quote"), evidence=((NORM_ID, "a9"),)),
        [],
    )
    unscored = build_case_result(
        EvalCase(id="unscored", question="q"),
        answer_response(("a9", "quote"), evidence=((NORM_ID, "a9"),)),
        [],
    )

    metrics = aggregate([scored, unscored])

    assert metrics.cases == 2
    assert metrics.mean_recall == 1.0
    assert metrics.outcomes == {"respuesta": 2}
    assert metrics.mean_agentic_delta == 1.0


def test_the_aggregate_sums_citation_verdicts() -> None:
    result = build_case_result(
        EvalCase(id="c", question="q"),
        answer_response(("a9", "quote"), evidence=((NORM_ID, "a9"),)),
        [],
    )

    metrics = aggregate([result, result])

    assert metrics.citation_verdicts["verificada_directa"] == 2
