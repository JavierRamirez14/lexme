"""Tests driving the harness through the CLI with injected seams, no network.

These are the allowed suite/harness intersection: the harness produces its
artifact, and the citation guardrail turns a corrupt displayed citation into a hard
failure that names the case and fails the run. The real Mode 1 system is stubbed
here; the end-to-end run behind the fake LLM lives in the integration test.
"""

import json
from datetime import UTC, date, datetime
from pathlib import Path

from lexme.eval.artifact import RunArtifact, read_artifact
from lexme.eval.calibration import (
    REVIEWER_MODEL,
    CalibrationItem,
    JudgeCalibration,
    build_calibration,
)
from lexme.eval.cases import ClarificationAnswer, EvalCase
from lexme.eval.cli import _parse_args, main
from lexme.eval.fingerprint import ConfigFingerprint, TaskFingerprint, build_fingerprint
from lexme.eval.judge import JUDGE_TASK
from lexme.eval.metrics import Disambiguation
from lexme.eval.runner import CaseRun, run_suite
from lexme.llm import load_task_registry
from lexme.mode1 import Outcome
from tests.eval.conftest import (
    AS_OF,
    NORM_ID,
    InMemoryCorpus,
    StubRunner,
    answer_response,
    clarification_response,
)

QUESTION = "¿cuál es el plazo mínimo del arrendamiento?"
GOLD_A9 = f"{NORM_ID}:a9"
BLOCK_TEXT = "La duración del arrendamiento será libremente pactada por las partes."
QUOTE = "La duración del arrendamiento será libremente pactada"
FABRICATED = "El arrendador podrá desalojar al inquilino sin preaviso."
NOW = datetime(2026, 7, 25, 9, 30, tzinfo=UTC)


def _fingerprint() -> ConfigFingerprint:
    """A fixed fingerprint standing in for the real configuration hash."""
    return ConfigFingerprint(
        models={},
        vertical="vivienda",
        prompts_hash="prompts",
        vertical_config_hash="cfg",
        corpus_hash="corpus",
        dataset_hash="dataset",
        fingerprint="fp-under-test",
    )


def _cases_dir(tmp_path: Path) -> Path:
    """A one-case directory the CLI loads, keyed to the seeded question."""
    directory = tmp_path / "cases"
    directory.mkdir()
    (directory / "plazo.json").write_text(
        json.dumps({"id": "plazo", "question": QUESTION, "gold_block_refs": [GOLD_A9]}),
        encoding="utf-8",
    )
    return directory


def _run(tmp_path: Path, runner: StubRunner, corpus: InMemoryCorpus) -> tuple[int, Path]:
    """Invoke ``eval run`` with injected seams and return its exit code and artifact path."""
    out = tmp_path / "run.json"
    code = main(
        ["run", "--vertical", "vivienda", "--cases", str(_cases_dir(tmp_path)), "--out", str(out)],
        runner=runner,
        corpus=corpus,
        fingerprint=_fingerprint(),
        today=AS_OF,
        now=NOW,
    )
    return code, out


def test_eval_run_emits_an_artifact_with_the_full_fingerprint_and_passes(tmp_path: Path) -> None:
    runner = StubRunner(answer_response(("a9", QUOTE), evidence=((NORM_ID, "a9"),)))
    corpus = InMemoryCorpus({(NORM_ID, "a9"): BLOCK_TEXT})

    code, out = _run(tmp_path, runner, corpus)

    assert code == 0
    artifact = read_artifact(out)
    assert artifact.passed is True
    assert artifact.fingerprint.fingerprint == "fp-under-test"
    assert artifact.metrics.cases == 1
    assert artifact.metrics.mean_recall == 1.0
    assert runner.questions == [QUESTION]


def test_a_case_is_answered_and_verified_at_its_own_point_in_time_date(tmp_path: Path) -> None:
    pinned = date(2015, 6, 1)
    directory = tmp_path / "cases"
    directory.mkdir()
    (directory / "pit.json").write_text(
        json.dumps(
            {
                "id": "pit",
                "question": QUESTION,
                "gold_block_refs": [GOLD_A9],
                "target_date": pinned.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    runner = StubRunner(answer_response(("a9", QUOTE), evidence=((NORM_ID, "a9"),)))
    corpus = InMemoryCorpus({(NORM_ID, "a9"): BLOCK_TEXT})
    out = tmp_path / "run.json"

    code = main(
        ["run", "--vertical", "vivienda", "--cases", str(directory), "--out", str(out)],
        runner=runner,
        corpus=corpus,
        fingerprint=_fingerprint(),
        today=AS_OF,
        now=NOW,
    )

    assert code == 0
    assert runner.dates == [pinned]


def test_the_artifact_separates_direct_resumed_and_unanswered_cases(tmp_path: Path) -> None:
    directory = tmp_path / "cases"
    directory.mkdir()
    for name in ("a", "b", "c"):
        (directory / f"{name}.json").write_text(
            json.dumps({"id": name, "question": QUESTION, "gold_block_refs": [GOLD_A9]}),
            encoding="utf-8",
        )
    answered = answer_response(("a9", QUOTE), evidence=((NORM_ID, "a9"),))
    runner = StubRunner(
        CaseRun(response=answered, disambiguation=Disambiguation.DIRECT),
        CaseRun(response=answered, disambiguation=Disambiguation.RESUMED),
        CaseRun(response=clarification_response(), disambiguation=Disambiguation.UNANSWERED),
    )
    out = tmp_path / "run.json"

    code = main(
        ["run", "--vertical", "vivienda", "--cases", str(directory), "--out", str(out)],
        runner=runner,
        corpus=InMemoryCorpus({(NORM_ID, "a9"): BLOCK_TEXT}),
        fingerprint=_fingerprint(),
        today=AS_OF,
        now=NOW,
    )

    assert code == 0
    metrics = read_artifact(out).metrics
    assert metrics.disambiguation == {"directo": 1, "reanudado": 1, "sin_respuesta": 1}
    assert metrics.disambiguation_rate == 2 / 3


def test_a_case_paused_without_a_pinned_answer_is_recorded_as_a_disambiguation(
    tmp_path: Path,
) -> None:
    runner = StubRunner(
        CaseRun(response=clarification_response(), disambiguation=Disambiguation.UNANSWERED)
    )

    code, out = _run(tmp_path, runner, InMemoryCorpus({}))

    assert code == 0
    (case,) = read_artifact(out).cases
    assert case.outcome == Outcome.CLARIFICATION.value
    assert case.disambiguation == Disambiguation.UNANSWERED.value
    assert case.outcome_as_expected is None


def test_the_runner_receives_the_answer_the_case_pins(tmp_path: Path) -> None:
    directory = tmp_path / "cases"
    directory.mkdir()
    (directory / "pinned.json").write_text(
        json.dumps(
            {
                "id": "pinned",
                "question": QUESTION,
                "clarification_answers": [{"branch_id": "fecha_firma", "answer": "12/06/2025"}],
            }
        ),
        encoding="utf-8",
    )
    runner = StubRunner(answer_response(("a9", QUOTE), evidence=((NORM_ID, "a9"),)))

    main(
        ["run", "--vertical", "vivienda", "--cases", str(directory), "--out", str(tmp_path / "r")],
        runner=runner,
        corpus=InMemoryCorpus({(NORM_ID, "a9"): BLOCK_TEXT}),
        fingerprint=_fingerprint(),
        today=AS_OF,
        now=NOW,
    )

    assert runner.pinned_answers == [(ClarificationAnswer("fecha_firma", "12/06/2025"),)]


def _repeat(
    tmp_path: Path, runner: StubRunner, corpus: InMemoryCorpus, times: int
) -> tuple[int, Path]:
    """Invoke ``eval run --repeat`` with injected seams and return its code and artifact."""
    out = tmp_path / "run.json"
    code = main(
        [
            "run",
            "--vertical",
            "vivienda",
            "--cases",
            str(_cases_dir(tmp_path)),
            "--out",
            str(out),
            "--repeat",
            str(times),
        ],
        runner=runner,
        corpus=corpus,
        fingerprint=_fingerprint(),
        today=AS_OF,
        now=NOW,
    )
    return code, out


def test_repeating_the_suite_bands_each_metric_over_the_repetitions(tmp_path: Path) -> None:
    recalled = answer_response(evidence=((NORM_ID, "a9"),))
    missed = answer_response(evidence=())
    runner = StubRunner(recalled, missed, recalled)

    code, out = _repeat(tmp_path, runner, InMemoryCorpus({(NORM_ID, "a9"): BLOCK_TEXT}), times=3)

    assert code == 0
    repetitions = read_artifact(out).repetitions
    assert repetitions is not None
    assert repetitions.repetitions == 3
    band = repetitions.band("mean_recall")
    assert band is not None
    assert band.values == [1.0, 0.0, 1.0]
    assert (band.median, band.low, band.high) == (1.0, 0.0, 1.0)


def test_a_repeated_run_keeps_the_first_repetition_as_its_per_case_detail(
    tmp_path: Path,
) -> None:
    recalled = answer_response(evidence=((NORM_ID, "a9"),))
    runner = StubRunner(recalled, answer_response(evidence=()))

    code, out = _repeat(tmp_path, runner, InMemoryCorpus({(NORM_ID, "a9"): BLOCK_TEXT}), times=2)

    assert code == 0
    artifact = read_artifact(out)
    assert [case.recall for case in artifact.cases] == [1.0]
    assert artifact.metrics.mean_recall == 1.0


def test_a_single_repetition_still_runs_and_records_that_it_measured_no_noise(
    tmp_path: Path,
) -> None:
    runner = StubRunner(answer_response(("a9", QUOTE), evidence=((NORM_ID, "a9"),)))

    code, out = _run(tmp_path, runner, InMemoryCorpus({(NORM_ID, "a9"): BLOCK_TEXT}))

    assert code == 0
    repetitions = read_artifact(out).repetitions
    assert repetitions is not None
    assert repetitions.repetitions == 1
    assert repetitions.measures_noise is False


def test_a_corrupt_citation_in_a_later_repetition_still_hard_fails_the_run(
    tmp_path: Path,
) -> None:
    clean = answer_response(("a9", QUOTE), evidence=((NORM_ID, "a9"),))
    corrupt = answer_response(("a9", FABRICATED), evidence=((NORM_ID, "a9"),))
    runner = StubRunner(clean, corrupt)

    code, out = _repeat(tmp_path, runner, InMemoryCorpus({(NORM_ID, "a9"): BLOCK_TEXT}), times=2)

    assert code == 1
    assert read_artifact(out).passed is False


def test_a_corrupt_displayed_citation_hard_fails_the_run_and_flags_the_case(
    tmp_path: Path,
) -> None:
    runner = StubRunner(answer_response(("a9", FABRICATED), evidence=((NORM_ID, "a9"),)))
    corpus = InMemoryCorpus({(NORM_ID, "a9"): BLOCK_TEXT})

    code, out = _run(tmp_path, runner, corpus)

    assert code == 1
    artifact = read_artifact(out)
    assert artifact.passed is False
    assert [failure.case_id for failure in artifact.hard_failures] == ["plazo"]
    assert artifact.hard_failures[0].block_ref == GOLD_A9


def test_a_corrupt_citation_in_a_resumed_answer_still_hard_fails_the_run(tmp_path: Path) -> None:
    corrupt = answer_response(("a9", FABRICATED), evidence=((NORM_ID, "a9"),))
    runner = StubRunner(CaseRun(response=corrupt, disambiguation=Disambiguation.RESUMED))
    corpus = InMemoryCorpus({(NORM_ID, "a9"): BLOCK_TEXT})

    code, out = _run(tmp_path, runner, corpus)

    assert code == 1
    artifact = read_artifact(out)
    assert artifact.passed is False
    assert artifact.cases[0].disambiguation == Disambiguation.RESUMED.value
    assert [failure.case_id for failure in artifact.hard_failures] == ["plazo"]


def _judged_fingerprint(judge_model: str) -> ConfigFingerprint:
    """A fingerprint pinning ``judge_model`` to the judge task, as a live run does."""
    return _fingerprint().model_copy(
        update={
            "models": {
                JUDGE_TASK: TaskFingerprint(
                    provider="openrouter", model=judge_model, temperature=0.0
                )
            }
        }
    )


def _calibration_of(judge_model: str) -> JudgeCalibration:
    """A calibration record grading ``judge_model``, agreement 1.0 over one ruling."""
    return build_calibration(
        judge_model,
        "reviewer",
        REVIEWER_MODEL,
        NOW,
        [
            CalibrationItem(
                case_id="plazo",
                kind="key_point",
                ref=GOLD_A9,
                judge_label=True,
                reviewer_label=True,
            )
        ],
    )


def _suite_with(fingerprint: ConfigFingerprint, calibration: JudgeCalibration) -> RunArtifact:
    """Run a one-case suite under ``fingerprint`` carrying ``calibration``."""
    return run_suite(
        "modo1",
        [EvalCase(id="plazo", question=QUESTION, gold_block_refs=(GOLD_A9,))],
        StubRunner(answer_response(("a9", QUOTE), evidence=((NORM_ID, "a9"),))),
        InMemoryCorpus({(NORM_ID, "a9"): BLOCK_TEXT}),
        AS_OF,
        fingerprint,
        NOW,
        calibration=calibration,
    )


def test_a_run_publishes_the_calibration_of_the_judge_it_actually_used() -> None:
    calibration = _calibration_of("qwen/qwen3-235b-a22b-2507")

    artifact = _suite_with(_judged_fingerprint("qwen/qwen3-235b-a22b-2507"), calibration)

    assert artifact.judge_calibration == calibration


def test_a_run_that_swapped_the_judge_publishes_no_agreement_from_the_old_one() -> None:
    artifact = _suite_with(
        _judged_fingerprint("qwen/qwen3-235b-a22b-2507"),
        _calibration_of("openai/gpt-oss-20b:free"),
    )

    assert artifact.judge_calibration is None


def test_the_harness_runs_judged_unless_asked_not_to() -> None:
    judged = _parse_args(["run", "--vertical", "vivienda"])
    unjudged = _parse_args(["run", "--vertical", "vivienda", "--no-judge"])

    assert judged.no_judge is False
    assert unjudged.no_judge is True


def test_an_unjudged_run_pins_no_judge_and_so_publishes_no_agreement() -> None:
    artifact = _suite_with(_fingerprint(), _calibration_of("deepseek/deepseek-v3.2"))

    assert JUDGE_TASK not in artifact.fingerprint.models
    assert artifact.judge_calibration is None


def test_dropping_the_judge_task_leaves_its_provider_out_of_the_fingerprint() -> None:
    registry = load_task_registry()

    judged = build_fingerprint(registry, "vivienda", Path("."), None, "p", "d")
    unjudged = build_fingerprint(
        registry.without(JUDGE_TASK), "vivienda", Path("."), None, "p", "d"
    )

    assert JUDGE_TASK in judged.models
    assert JUDGE_TASK not in unjudged.models
    assert judged.fingerprint != unjudged.fingerprint
