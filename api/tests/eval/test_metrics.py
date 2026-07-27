"""Tests for the eval quality metrics: recall over accumulated evidence, and aggregates."""

from lexme.eval.cases import EvalCase, KeyPoint
from lexme.eval.judge import ClaimAssessment, JudgeVerdict, KeyPointCoverage
from lexme.eval.metrics import aggregate, build_case_result
from lexme.mode1 import Outcome
from tests.eval.conftest import NORM_ID, answer_response

REF_A9 = f"{NORM_ID}:a9"
REF_A36 = f"{NORM_ID}:a36"


def _abstention() -> object:
    """An abstention response with an agentic trace but no answer."""
    response = answer_response(("a9", "quote"), evidence=((NORM_ID, "a9"),))
    return response.model_copy(update={"outcome": Outcome.ABSTENTION, "answer": None})


def test_recall_counts_gold_blocks_in_the_accumulated_evidence() -> None:
    case = EvalCase(id="c", question="q", gold_block_refs=(REF_A9, REF_A36))
    response = answer_response(("a9", "quote"), evidence=((NORM_ID, "a9"),))

    result = build_case_result(case, response, [])

    assert result.recall == 0.5
    assert result.retrieved_block_refs == [REF_A9]


def test_a_case_without_gold_blocks_has_no_recall() -> None:
    case = EvalCase(id="c", question="q")
    response = answer_response(("a9", "quote"), evidence=((NORM_ID, "a9"),))

    result = build_case_result(case, response, [])

    assert result.recall is None


def test_the_aggregate_averages_only_scored_recalls() -> None:
    scored = build_case_result(
        EvalCase(id="scored", question="q", gold_block_refs=(REF_A9,)),
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


def test_abstention_is_reported_as_a_rate_and_an_expected_recall() -> None:
    answered = build_case_result(
        EvalCase(id="ans", question="q", expected_outcome="respuesta"),
        answer_response(("a9", "quote"), evidence=((NORM_ID, "a9"),)),
        [],
    )
    abstained = build_case_result(
        EvalCase(id="abs", question="q", expected_outcome="abstencion"),
        _abstention(),
        [],
    )

    metrics = aggregate([answered, abstained])

    assert metrics.abstention_rate == 0.5
    assert metrics.expected_abstention_recall == 1.0


def test_expected_abstention_recall_is_none_when_no_case_should_abstain() -> None:
    answered = build_case_result(
        EvalCase(id="ans", question="q", expected_outcome="respuesta"),
        answer_response(("a9", "quote"), evidence=((NORM_ID, "a9"),)),
        [],
    )

    assert aggregate([answered]).expected_abstention_recall is None


def test_the_judge_aggregate_pools_unsupported_claims_across_cases() -> None:
    case = EvalCase(
        id="c",
        question="q",
        gold_block_refs=(REF_A9,),
        key_points=(KeyPoint(claim="k", block_ref=REF_A9),),
    )
    verdict = JudgeVerdict(
        key_points=[KeyPointCoverage(block_ref=REF_A9, covered=True)],
        claims=[
            ClaimAssessment(claim="a", supported=True, supporting_block_ref=REF_A9),
            ClaimAssessment(claim="b", supported=False),
        ],
        clarity=5,
    )
    result = build_case_result(
        case, answer_response(("a9", "quote"), evidence=((NORM_ID, "a9"),)), [], verdict
    )

    metrics = aggregate([result, result])

    assert metrics.judge is not None
    assert metrics.judge.judged_cases == 2
    assert metrics.judge.mean_completeness == 1.0
    assert metrics.judge.unsupported_claim_rate == 0.5
    assert metrics.judge.mean_clarity == 5.0


def test_the_judge_aggregate_is_none_when_no_case_was_judged() -> None:
    result = build_case_result(
        EvalCase(id="c", question="q"),
        answer_response(("a9", "quote"), evidence=((NORM_ID, "a9"),)),
        [],
    )

    assert aggregate([result]).judge is None
