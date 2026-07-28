"""Tests for turning a run's judge rulings into a human-reviewable calibration sample."""

from datetime import UTC, datetime

import pytest

from lexme.eval.artifact import RunArtifact, build_artifact
from lexme.eval.calibration import KIND_CLAIM, KIND_KEY_POINT, REVIEWER_HUMAN
from lexme.eval.cases import EvalCase, KeyPoint
from lexme.eval.fingerprint import ConfigFingerprint, TaskFingerprint
from lexme.eval.judge import JUDGE_TASK, ClaimAssessment, JudgeVerdict, KeyPointCoverage
from lexme.eval.metrics import aggregate, build_case_result
from lexme.eval.review import (
    ReviewError,
    build_sample,
    calibration_from_review,
    collect_rulings,
    parse_disagreements,
    render_sheet,
    with_article_texts,
)
from tests.eval.conftest import NORM_ID, answer_response

NOW = datetime(2026, 7, 27, tzinfo=UTC)
REVIEWED_AT = datetime(2026, 7, 28, tzinfo=UTC)
JUDGE_MODEL = "openai/gpt-oss-20b:free"
REF = f"{NORM_ID}:a36"
KEY_POINT_CLAIM = "la fianza es de una mensualidad"
ANSWER_CLAIM = "el arrendador debe exigir una mensualidad de fianza"
QUOTE = "una mensualidad"
ARTICLE_TEXT = "A la celebración del contrato será obligatoria la exigencia de una mensualidad."


def _fingerprint() -> ConfigFingerprint:
    """A fingerprint pinning the judge model the sample is reviewed for."""
    return ConfigFingerprint(
        models={
            JUDGE_TASK: TaskFingerprint(provider="openrouter", model=JUDGE_MODEL, temperature=0.0)
        },
        vertical="vivienda",
        prompts_hash="prompts",
        vertical_config_hash="cfg",
        corpus_hash="corpus",
        dataset_hash="dataset",
        fingerprint="fp-under-test",
    )


def _case(case_id: str = "fianza") -> EvalCase:
    """A case with one key point tied to its single gold block."""
    return EvalCase(
        id=case_id,
        question="¿cuánta fianza me pueden pedir?",
        gold_block_refs=(REF,),
        key_points=(KeyPoint(claim=KEY_POINT_CLAIM, block_ref=REF),),
    )


def _verdict(*, covered: bool = True, supported: bool = True) -> JudgeVerdict:
    """A verdict with one key-point ruling and one claim ruling."""
    return JudgeVerdict(
        key_points=[KeyPointCoverage(block_ref=REF, covered=covered, evidence=QUOTE)],
        claims=[
            ClaimAssessment(
                claim=ANSWER_CLAIM,
                supported=supported,
                supporting_block_ref=REF if supported else None,
            )
        ],
        clarity=4,
    )


def _artifact(*judged: tuple[EvalCase, JudgeVerdict | None]) -> RunArtifact:
    """A run artifact whose cases carry the given verdicts."""
    response = answer_response(("a36", QUOTE), evidence=((NORM_ID, "a36"),))
    results = [build_case_result(case, response, [], verdict) for case, verdict in judged]
    return build_artifact("modo1", NOW, _fingerprint(), results, aggregate(results), [])


def _judged_cases(count: int) -> tuple[RunArtifact, list[EvalCase]]:
    """A run of ``count`` judged cases and the case set behind it."""
    cases = [_case(f"case-{index:02d}") for index in range(count)]
    return _artifact(*[(case, _verdict()) for case in cases]), cases


def test_the_artifact_keeps_every_ruling_the_judge_made() -> None:
    artifact = _artifact((_case(), _verdict()))

    assert artifact.cases[0].judge_verdict == _verdict()


def test_every_key_point_and_claim_ruling_becomes_one_numbered_item() -> None:
    artifact, cases = _judged_cases(3)

    rulings = collect_rulings(artifact, cases)

    assert [ruling.number for ruling in rulings] == [1, 2, 3, 4, 5, 6]
    assert [ruling.kind for ruling in rulings] == [KIND_KEY_POINT, KIND_CLAIM] * 3


def test_a_key_point_ruling_carries_the_reference_claim_and_the_judges_quote() -> None:
    artifact = _artifact((_case(), _verdict()))

    ruling = collect_rulings(artifact, [_case()])[0]

    assert ruling.ref == f"{REF}#1"
    assert ruling.reference == KEY_POINT_CLAIM
    assert ruling.quoted == QUOTE
    assert ruling.judge_label is True
    assert ruling.article_ref == REF


def _two_point_case() -> EvalCase:
    """A case whose two key points hang off the same gold block."""
    return EvalCase(
        id="fianza",
        question="¿cuánta fianza me pueden pedir?",
        gold_block_refs=(REF,),
        key_points=(
            KeyPoint(claim="la fianza es de una mensualidad", block_ref=REF),
            KeyPoint(claim="la garantía adicional no excede dos mensualidades", block_ref=REF),
        ),
    )


def _two_point_verdict() -> JudgeVerdict:
    """A verdict ruling on both key points of :func:`_two_point_case`, in order."""
    return JudgeVerdict(
        key_points=[
            KeyPointCoverage(block_ref=REF, covered=True, evidence="una mensualidad"),
            KeyPointCoverage(block_ref=REF, covered=False),
        ],
        claims=[],
        clarity=4,
    )


def test_key_points_sharing_a_block_each_carry_their_own_reference() -> None:
    artifact = _artifact((_two_point_case(), _two_point_verdict()))

    rulings = collect_rulings(artifact, [_two_point_case()])

    assert [ruling.reference for ruling in rulings] == [
        "la fianza es de una mensualidad",
        "la garantía adicional no excede dos mensualidades",
    ]


def test_key_points_sharing_a_block_are_told_apart_in_the_calibration_record() -> None:
    artifact = _artifact((_two_point_case(), _two_point_verdict()))

    refs = [ruling.ref for ruling in collect_rulings(artifact, [_two_point_case()])]

    assert len(set(refs)) == 2
    assert all(REF in ref for ref in refs)


def test_a_verdict_that_rules_on_a_different_number_of_points_is_not_aligned_by_position() -> None:
    verdict = JudgeVerdict(
        key_points=[KeyPointCoverage(block_ref=REF, covered=True, evidence="e")],
        claims=[],
        clarity=4,
    )
    artifact = _artifact((_two_point_case(), verdict))

    ruling = collect_rulings(artifact, [_two_point_case()])[0]

    assert ruling.reference == ""
    assert ruling.ref == REF


def test_a_claim_ruling_is_identified_by_the_claim_and_names_its_backing_article() -> None:
    artifact = _artifact((_case(), _verdict()))

    ruling = collect_rulings(artifact, [_case()])[1]

    assert ruling.ref == ANSWER_CLAIM
    assert ruling.article_ref == REF
    assert ruling.judge_label is True


def test_an_unsupported_claim_names_no_backing_article() -> None:
    artifact = _artifact((_case(), _verdict(supported=False)))

    ruling = collect_rulings(artifact, [_case()])[1]

    assert ruling.judge_label is False
    assert ruling.article_ref is None


def test_an_unjudged_case_contributes_no_rulings() -> None:
    artifact = _artifact((_case(), None))

    assert collect_rulings(artifact, [_case()]) == []


def test_the_sample_declares_the_population_it_was_drawn_from() -> None:
    artifact, cases = _judged_cases(10)

    sample = build_sample(artifact, cases, size=6, seed=20)

    assert sample.population == 20
    assert sample.size == 6
    assert sample.judge_model == JUDGE_MODEL


def test_the_same_seed_draws_the_same_sample() -> None:
    artifact, cases = _judged_cases(10)

    first = build_sample(artifact, cases, size=6, seed=20)
    second = build_sample(artifact, cases, size=6, seed=20)

    assert [ruling.number for ruling in first.rulings] == [
        ruling.number for ruling in second.rulings
    ]


def test_another_seed_draws_another_sample() -> None:
    artifact, cases = _judged_cases(10)

    first = build_sample(artifact, cases, size=6, seed=20)
    other = build_sample(artifact, cases, size=6, seed=21)

    assert [ruling.number for ruling in first.rulings] != [
        ruling.number for ruling in other.rulings
    ]


def test_the_sample_reads_in_run_order() -> None:
    artifact, cases = _judged_cases(10)

    sample = build_sample(artifact, cases, size=6, seed=20)

    numbers = [ruling.number for ruling in sample.rulings]
    assert numbers == sorted(numbers)


def test_a_sample_wider_than_the_population_takes_every_ruling() -> None:
    artifact, cases = _judged_cases(2)

    sample = build_sample(artifact, cases, size=50, seed=20)

    assert sample.size == 4
    assert sample.population == 4


def test_a_run_with_no_judged_case_cannot_be_sampled() -> None:
    artifact = _artifact((_case(), None))

    with pytest.raises(ReviewError, match="no judge rulings"):
        build_sample(artifact, [_case()], size=10, seed=20)


def test_article_texts_are_attached_per_case_and_article() -> None:
    artifact, cases = _judged_cases(1)
    sample = build_sample(artifact, cases, size=2, seed=20)

    resolved = with_article_texts(sample, {("case-00", REF): ARTICLE_TEXT})

    assert [ruling.article_text for ruling in resolved.rulings] == [ARTICLE_TEXT, ARTICLE_TEXT]


def test_an_article_left_unresolved_stays_empty() -> None:
    artifact, cases = _judged_cases(1)
    sample = build_sample(artifact, cases, size=2, seed=20)

    resolved = with_article_texts(sample, {})

    assert [ruling.article_text for ruling in resolved.rulings] == ["", ""]


def test_the_sheet_shows_each_ruling_numbered_with_the_label_to_confirm() -> None:
    artifact, cases = _judged_cases(1)
    sample = with_article_texts(
        build_sample(artifact, cases, size=2, seed=20), {("case-00", REF): ARTICLE_TEXT}
    )

    sheet = render_sheet(sample)

    assert "#1" in sheet
    assert KEY_POINT_CLAIM in sheet
    assert ANSWER_CLAIM in sheet
    assert ARTICLE_TEXT in sheet
    assert JUDGE_MODEL in sheet


def test_a_reviewed_sample_agrees_except_where_the_human_disagreed() -> None:
    artifact, cases = _judged_cases(5)
    sample = build_sample(artifact, cases, size=10, seed=20)

    calibration = calibration_from_review(sample, (2, 4), "Reviewer", REVIEWER_HUMAN, REVIEWED_AT)

    assert calibration.sample_size == 10
    assert calibration.agreement == 0.8
    assert calibration.judge_model == JUDGE_MODEL
    assert calibration.reviewed_by == "Reviewer"


def test_a_disagreed_ruling_records_the_opposite_human_label() -> None:
    artifact, cases = _judged_cases(1)
    sample = build_sample(artifact, cases, size=2, seed=20)

    calibration = calibration_from_review(sample, (1,), "Reviewer", REVIEWER_HUMAN, REVIEWED_AT)

    assert calibration.items[0].judge_label is True
    assert calibration.items[0].reviewer_label is False
    assert calibration.items[1].reviewer_label is True


def test_a_disagreement_outside_the_sample_is_rejected() -> None:
    artifact, cases = _judged_cases(1)
    sample = build_sample(artifact, cases, size=2, seed=20)

    with pytest.raises(ReviewError, match="99"):
        calibration_from_review(sample, (99,), "Reviewer", REVIEWER_HUMAN, REVIEWED_AT)


def test_disagreements_are_read_as_a_comma_separated_list() -> None:
    assert parse_disagreements(" 3, 7 ,12") == (3, 7, 12)


def test_no_disagreement_means_the_human_confirmed_every_ruling() -> None:
    assert parse_disagreements("") == ()
    assert parse_disagreements("none") == ()


def test_a_disagreement_list_that_is_not_numbers_is_rejected() -> None:
    with pytest.raises(ReviewError, match="siete"):
        parse_disagreements("3, siete")
