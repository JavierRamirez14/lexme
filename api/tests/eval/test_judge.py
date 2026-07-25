"""Tests for the with-reference judge, its derived metrics and its config invariant."""

import pytest

from lexme.eval.cases import EvalCase, KeyPoint
from lexme.eval.judge import (
    JUDGE_TASK,
    ClaimAssessment,
    JudgeConfigError,
    JudgeVerdict,
    KeyPointCoverage,
    LlmJudge,
    assert_judge_distinct_from_generator,
    build_judge_metrics,
)
from lexme.llm import FakeLlmClient, TaskModel, TaskRegistry
from lexme.mode1 import SYNTHESIS_TASK, Outcome
from tests.eval.conftest import AS_OF, NORM_ID, InMemoryCorpus, answer_response

ARTICLE_TEXT = "La fianza será de una mensualidad de renta en el arrendamiento de viviendas."


def _case() -> EvalCase:
    """A case with one key point tied to its single gold block."""
    return EvalCase(
        id="fianza",
        question="¿cuánta fianza?",
        gold_block_ids=("a36",),
        key_points=(KeyPoint(claim="la fianza es una mensualidad", block_id="a36"),),
    )


def _verdict() -> JudgeVerdict:
    """A judge verdict covering the key point with one supported and one unsupported claim."""
    return JudgeVerdict(
        key_points=[KeyPointCoverage(block_id="a36", covered=True, evidence="una mensualidad")],
        claims=[
            ClaimAssessment(
                claim="la fianza es una mensualidad", supported=True, supporting_block_id="a36"
            ),
            ClaimAssessment(claim="se devuelve en 24h", supported=False),
        ],
        clarity=4,
    )


def test_the_judge_grades_an_answered_case_against_its_reference() -> None:
    llm = FakeLlmClient({JUDGE_TASK: [_verdict()]})
    corpus = InMemoryCorpus({(NORM_ID, "a36"): ARTICLE_TEXT})
    response = answer_response(("a36", "una mensualidad"), evidence=((NORM_ID, "a36"),))

    verdict = LlmJudge(llm=llm, corpus=corpus).judge(_case(), response, AS_OF)

    assert verdict is not None
    call = llm.calls[0]
    assert call.task == JUDGE_TASK
    assert call.response_model is JudgeVerdict
    prompt = call.messages[1].content
    assert ARTICLE_TEXT in prompt
    assert "la fianza es una mensualidad" in prompt


def test_the_judge_skips_a_case_with_no_key_points() -> None:
    llm = FakeLlmClient()
    corpus = InMemoryCorpus({})
    response = answer_response(("a36", "una mensualidad"), evidence=((NORM_ID, "a36"),))
    case = EvalCase(id="x", question="q", gold_block_ids=("a36",))

    assert LlmJudge(llm=llm, corpus=corpus).judge(case, response, AS_OF) is None
    assert llm.calls == []


def test_the_judge_skips_a_case_that_did_not_answer() -> None:
    llm = FakeLlmClient()
    corpus = InMemoryCorpus({})
    response = answer_response(evidence=((NORM_ID, "a36"),))
    response = response.model_copy(update={"outcome": Outcome.ABSTENTION, "answer": None})

    assert LlmJudge(llm=llm, corpus=corpus).judge(_case(), response, AS_OF) is None
    assert llm.calls == []


def test_judge_metrics_are_completeness_hallucination_and_clarity() -> None:
    metrics = build_judge_metrics(_verdict())

    assert metrics.completeness == 1.0
    assert metrics.unsupported_claim_rate == 0.5
    assert metrics.clarity == 4
    assert metrics.claims_total == 2
    assert metrics.claims_unsupported == 1


def _registry(judge_model: str, judge_temp: float = 0.0) -> TaskRegistry:
    """A registry with a generator and a judge, for the distinctness invariant."""
    return TaskRegistry(
        _by_task={
            SYNTHESIS_TASK: TaskModel(SYNTHESIS_TASK, "openrouter", "nemotron", 0.0),
            JUDGE_TASK: TaskModel(JUDGE_TASK, "gemini", judge_model, judge_temp),
        }
    )


def test_a_distinct_pinned_temp0_judge_passes_the_invariant() -> None:
    assert_judge_distinct_from_generator(_registry("gemini-3.5-flash"))


def test_a_judge_sharing_the_generator_model_is_rejected() -> None:
    registry = TaskRegistry(
        _by_task={
            SYNTHESIS_TASK: TaskModel(SYNTHESIS_TASK, "openrouter", "nemotron", 0.0),
            JUDGE_TASK: TaskModel(JUDGE_TASK, "openrouter", "nemotron", 0.0),
        }
    )
    with pytest.raises(JudgeConfigError, match="different family"):
        assert_judge_distinct_from_generator(registry)


def test_a_judge_above_temperature_zero_is_rejected() -> None:
    with pytest.raises(JudgeConfigError, match="temperature"):
        assert_judge_distinct_from_generator(_registry("gemini-3.5-flash", judge_temp=0.7))
