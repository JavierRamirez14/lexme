"""Tests for the with-reference judge, its derived metrics and its config invariant."""

from datetime import date

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
from lexme.llm.protocol import LlmError
from lexme.mode1 import SYNTHESIS_TASK, Outcome
from tests.eval.conftest import AS_OF, NORM_ID, DatedCorpus, InMemoryCorpus, answer_response

REF_A36 = f"{NORM_ID}:a36"

ARTICLE_TEXT = "La fianza será de una mensualidad de renta en el arrendamiento de viviendas."

# Art. 22 LEC in its two redactions, as in the guardrail's tests: the newer one
# moves an internal remission and adds a paragraph about costs.
LEC = "BOE-A-2000-323"
REF_A22 = f"{LEC}:a22"
SIGNED_ON = date(2023, 1, 15)
RUN_DATE = date(2026, 8, 14)
LEC_22_2015 = "…conforme a lo dispuesto en el apartado 3 del artículo 440."
LEC_22_2025 = (
    "…conforme a lo dispuesto en el apartado 4 del artículo 439. En tal caso, las "
    "costas se impondrán al arrendatario."
)


def _case() -> EvalCase:
    """A case with one key point tied to its single gold block."""
    return EvalCase(
        id="fianza",
        question="¿cuánta fianza?",
        gold_block_refs=(REF_A36,),
        key_points=(KeyPoint(claim="la fianza es una mensualidad", block_ref=REF_A36),),
    )


def _verdict() -> JudgeVerdict:
    """A judge verdict covering the key point with one supported and one unsupported claim."""
    return JudgeVerdict(
        key_points=[KeyPointCoverage(block_ref=REF_A36, covered=True, evidence="una mensualidad")],
        claims=[
            ClaimAssessment(
                claim="la fianza es una mensualidad",
                supported=True,
                supporting_block_ref=REF_A36,
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


def _ruling_on(block_ref: str) -> JudgeVerdict:
    """A verdict whose two rulings both name ``block_ref`` as the model wrote it."""
    return JudgeVerdict(
        key_points=[KeyPointCoverage(block_ref=block_ref, covered=True, evidence="e")],
        claims=[ClaimAssessment(claim="c", supported=True, supporting_block_ref=block_ref)],
        clarity=4,
    )


def _enervacion_case() -> EvalCase:
    """The `mh-02` shape: one key point on art. 22 LEC, no pinned date of its own."""
    return EvalCase(
        id="mh-02",
        question="¿puedo parar el desahucio pagando?",
        gold_block_refs=(REF_A22,),
        key_points=(KeyPoint(claim="el pago enerva el desahucio", block_ref=REF_A22),),
    )


def _judged_prompt_for(declared: date) -> str:
    """The prompt the judge is sent for an answer declaring ``declared``, run at ``RUN_DATE``."""
    llm = FakeLlmClient({JUDGE_TASK: [_ruling_on(REF_A22)]})
    corpus = DatedCorpus(
        {(LEC, "a22"): [(date(2015, 10, 1), LEC_22_2015), (date(2025, 4, 3), LEC_22_2025)]}
    )
    response = answer_response(
        ("a22", "el arrendatario paga"),
        evidence=((LEC, "a22"),),
        norm_id=LEC,
        fecha_objetivo=declared,
    )

    LlmJudge(llm=llm, corpus=corpus).judge(_enervacion_case(), response, RUN_DATE)

    return llm.calls[0].messages[1].content


def test_an_answer_given_for_a_past_date_is_graded_against_the_law_of_then() -> None:
    prompt = _judged_prompt_for(SIGNED_ON)

    assert LEC_22_2015 in prompt
    assert LEC_22_2025 not in prompt


def test_an_answer_given_for_today_is_still_graded_against_the_law_of_today() -> None:
    prompt = _judged_prompt_for(RUN_DATE)

    assert LEC_22_2025 in prompt
    assert LEC_22_2015 not in prompt


def test_an_unusable_verdict_leaves_the_case_unjudged_instead_of_ending_the_run() -> None:
    llm = FakeLlmClient({JUDGE_TASK: ["no soy JSON, soy una tabla markdown"]})
    corpus = InMemoryCorpus({(NORM_ID, "a36"): ARTICLE_TEXT})
    response = answer_response(("a36", "una mensualidad"), evidence=((NORM_ID, "a36"),))

    assert LlmJudge(llm=llm, corpus=corpus).judge(_case(), response, AS_OF) is None


def test_a_provider_failure_is_not_swallowed_as_an_unjudged_case() -> None:
    llm = FakeLlmClient()
    corpus = InMemoryCorpus({(NORM_ID, "a36"): ARTICLE_TEXT})
    response = answer_response(("a36", "una mensualidad"), evidence=((NORM_ID, "a36"),))

    with pytest.raises(LlmError):
        LlmJudge(llm=llm, corpus=corpus).judge(_case(), response, AS_OF)


def test_the_judge_skips_a_case_with_no_key_points() -> None:
    llm = FakeLlmClient()
    corpus = InMemoryCorpus({})
    response = answer_response(("a36", "una mensualidad"), evidence=((NORM_ID, "a36"),))
    case = EvalCase(id="x", question="q", gold_block_refs=(REF_A36,))

    assert LlmJudge(llm=llm, corpus=corpus).judge(case, response, AS_OF) is None
    assert llm.calls == []


def test_the_judge_skips_a_case_that_did_not_answer() -> None:
    llm = FakeLlmClient()
    corpus = InMemoryCorpus({})
    response = answer_response(evidence=((NORM_ID, "a36"),))
    response = response.model_copy(update={"outcome": Outcome.ABSTENTION, "answer": None})

    assert LlmJudge(llm=llm, corpus=corpus).judge(_case(), response, AS_OF) is None
    assert llm.calls == []


def _judged(verdict: JudgeVerdict) -> JudgeVerdict | None:
    """Run ``verdict`` back through the judge, as the model had returned it."""
    llm = FakeLlmClient({JUDGE_TASK: [verdict]})
    corpus = InMemoryCorpus({(NORM_ID, "a36"): ARTICLE_TEXT})
    response = answer_response(("a36", "una mensualidad"), evidence=((NORM_ID, "a36"),))
    return LlmJudge(llm=llm, corpus=corpus).judge(_case(), response, AS_OF)


def test_a_reference_the_judge_wrapped_in_brackets_is_read_as_the_block_it_names() -> None:
    verdict = _judged(_ruling_on(f"[{REF_A36}]"))

    assert verdict is not None
    assert verdict.key_points[0].block_ref == REF_A36
    assert verdict.claims[0].supporting_block_ref == REF_A36


def test_a_reference_the_judge_echoed_with_its_claim_is_read_as_the_block_it_names() -> None:
    verdict = _judged(_ruling_on(f"[{REF_A36}] la fianza es una mensualidad"))

    assert verdict is not None
    assert verdict.key_points[0].block_ref == REF_A36


def test_a_bare_block_id_is_qualified_with_the_norm_the_case_names() -> None:
    verdict = _judged(_ruling_on("a36"))

    assert verdict is not None
    assert verdict.key_points[0].block_ref == REF_A36
    assert verdict.claims[0].supporting_block_ref == REF_A36


def test_a_reference_is_read_as_the_longest_block_it_names_not_a_prefix_of_it() -> None:
    long_ref = f"{NORM_ID}:a90"
    case = EvalCase(
        id="prefix",
        question="q",
        gold_block_refs=(f"{NORM_ID}:a9", long_ref),
        key_points=(
            KeyPoint(claim="nueve", block_ref=f"{NORM_ID}:a9"),
            KeyPoint(claim="noventa", block_ref=long_ref),
        ),
    )
    llm = FakeLlmClient({JUDGE_TASK: [_ruling_on(f"[{long_ref}]")]})
    corpus = InMemoryCorpus({(NORM_ID, "a36"): ARTICLE_TEXT})
    response = answer_response(("a36", "una mensualidad"), evidence=((NORM_ID, "a36"),))

    verdict = LlmJudge(llm=llm, corpus=corpus).judge(case, response, AS_OF)

    assert verdict is not None
    assert verdict.key_points[0].block_ref == long_ref
    assert verdict.claims[0].supporting_block_ref == long_ref


def test_a_bracketed_bare_block_id_is_qualified_like_a_naked_one() -> None:
    verdict = _judged(_ruling_on("[a36]"))

    assert verdict is not None
    assert verdict.key_points[0].block_ref == REF_A36


def test_a_bare_block_id_two_norms_share_is_left_unresolved() -> None:
    other_norm = "BOE-A-2023-12203"
    case = EvalCase(
        id="ambiguous",
        question="q",
        gold_block_refs=(REF_A36, f"{other_norm}:a36"),
        key_points=(
            KeyPoint(claim="uno", block_ref=REF_A36),
            KeyPoint(claim="otro", block_ref=f"{other_norm}:a36"),
        ),
    )
    llm = FakeLlmClient({JUDGE_TASK: [_ruling_on("a36")]})
    corpus = InMemoryCorpus({(NORM_ID, "a36"): ARTICLE_TEXT})
    response = answer_response(("a36", "una mensualidad"), evidence=((NORM_ID, "a36"),))

    verdict = LlmJudge(llm=llm, corpus=corpus).judge(case, response, AS_OF)

    assert verdict is not None
    assert verdict.key_points[0].block_ref == "a36"
    assert verdict.claims[0].supporting_block_ref is None


def test_a_reference_to_a_block_the_case_never_named_is_not_invented_away() -> None:
    verdict = _judged(_ruling_on(f"{NORM_ID}:a99"))

    assert verdict is not None
    assert verdict.key_points[0].block_ref == f"{NORM_ID}:a99"
    assert verdict.claims[0].supporting_block_ref is None


def test_an_unsupported_claim_keeps_no_backing_reference() -> None:
    verdict = _judged(
        JudgeVerdict(
            key_points=[KeyPointCoverage(block_ref=REF_A36, covered=True, evidence="e")],
            claims=[ClaimAssessment(claim="c", supported=False)],
            clarity=4,
        )
    )

    assert verdict is not None
    assert verdict.claims[0].supporting_block_ref is None


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
