"""Tests driving the harness through the CLI with injected seams, no network.

These are the allowed suite/harness intersection: the harness produces its
artifact, and the citation guardrail turns a corrupt displayed citation into a hard
failure that names the case and fails the run. The real Mode 1 system is stubbed
here; the end-to-end run behind the fake LLM lives in the integration test.
"""

import json
from datetime import UTC, date, datetime
from pathlib import Path

from lexme.eval.artifact import read_artifact
from lexme.eval.cases import ClarificationAnswer
from lexme.eval.cli import main
from lexme.eval.fingerprint import ConfigFingerprint
from lexme.eval.metrics import Disambiguation
from lexme.eval.runner import CaseRun
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
