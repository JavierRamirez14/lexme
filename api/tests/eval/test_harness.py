"""Tests driving the harness through the CLI with injected seams, no network.

These are the allowed suite/harness intersection: the harness produces its
artifact, and the citation guardrail turns a corrupt displayed citation into a hard
failure that names the case and fails the run. The real Mode 1 system is stubbed
here; the end-to-end run behind the fake LLM lives in the integration test.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from lexme.eval.artifact import read_artifact
from lexme.eval.cli import main
from lexme.eval.fingerprint import ConfigFingerprint
from tests.eval.conftest import AS_OF, NORM_ID, InMemoryCorpus, StubRunner, answer_response

QUESTION = "¿cuál es el plazo mínimo del arrendamiento?"
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
        json.dumps({"id": "plazo", "question": QUESTION, "gold_block_ids": ["a9"]}),
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
    assert artifact.hard_failures[0].block_id == "a9"
