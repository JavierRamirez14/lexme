"""Tests for ``eval calibrate``: drawing the review sample and recording its labels."""

import json
from datetime import UTC, datetime
from pathlib import Path

from lexme.eval.artifact import build_artifact, read_artifact
from lexme.eval.calibration import load_calibration
from lexme.eval.cases import load_cases
from lexme.eval.cli import main
from lexme.eval.fingerprint import ConfigFingerprint, TaskFingerprint
from lexme.eval.judge import JUDGE_TASK, ClaimAssessment, JudgeVerdict, KeyPointCoverage
from lexme.eval.metrics import aggregate, build_case_result
from lexme.eval.review import read_sample
from tests.eval.conftest import NORM_ID, InMemoryCorpus, answer_response

NOW = datetime(2026, 7, 27, tzinfo=UTC)
JUDGE_MODEL = "openai/gpt-oss-20b:free"
REF = f"{NORM_ID}:a36"
QUESTION = "¿cuánta fianza me pueden pedir?"
KEY_POINT_CLAIM = "la fianza es de una mensualidad"
ANSWER_CLAIM = "el arrendador debe exigir una mensualidad de fianza"
ARTICLE_TEXT = "A la celebración del contrato será obligatoria la exigencia de una mensualidad."


def _cases_dir(tmp_path: Path, count: int) -> Path:
    """A directory of judgeable cases, each with one key point on the same block."""
    directory = tmp_path / "cases"
    directory.mkdir()
    for index in range(count):
        (directory / f"case-{index:02d}.json").write_text(
            json.dumps(
                {
                    "id": f"case-{index:02d}",
                    "question": QUESTION,
                    "gold_block_refs": [REF],
                    "key_points": [{"claim": KEY_POINT_CLAIM, "block_ref": REF}],
                }
            ),
            encoding="utf-8",
        )
    return directory


def _verdict() -> JudgeVerdict:
    """A verdict with one key-point ruling and one claim ruling."""
    return JudgeVerdict(
        key_points=[KeyPointCoverage(block_ref=REF, covered=True, evidence="una mensualidad")],
        claims=[ClaimAssessment(claim=ANSWER_CLAIM, supported=True, supporting_block_ref=REF)],
        clarity=4,
    )


def _run_artifact(tmp_path: Path, count: int) -> tuple[Path, Path]:
    """Write a run artifact whose cases carry verdicts; return it and its case set."""
    directory = _cases_dir(tmp_path, count)
    response = answer_response(("a36", "una mensualidad"), evidence=((NORM_ID, "a36"),))
    results = [build_case_result(case, response, [], _verdict()) for case in load_cases(directory)]
    fingerprint = ConfigFingerprint(
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
    artifact = build_artifact("modo1", NOW, fingerprint, results, aggregate(results), [])
    path = tmp_path / "modo1-run.json"
    artifact.write(path)
    return path, directory


def _export(tmp_path: Path, count: int = 5, size: int = 6) -> tuple[int, Path, Path]:
    """Draw a sample from a written run and return the exit code and both outputs."""
    run, directory = _run_artifact(tmp_path, count)
    sample = tmp_path / "sample.json"
    sheet = tmp_path / "sheet.md"
    code = main(
        [
            "calibrate",
            "export",
            "--vertical",
            "vivienda",
            "--run",
            str(run),
            "--cases",
            str(directory),
            "--size",
            str(size),
            "--seed",
            "20",
            "--out",
            str(sample),
            "--sheet",
            str(sheet),
        ],
        corpus=InMemoryCorpus({(NORM_ID, "a36"): ARTICLE_TEXT}),
    )
    return code, sample, sheet


def test_export_draws_a_seeded_sample_of_the_runs_rulings(tmp_path: Path) -> None:
    code, sample_path, _ = _export(tmp_path)

    assert code == 0
    sample = read_sample(sample_path)
    assert sample.population == 10
    assert sample.size == 6
    assert sample.seed == 20
    assert sample.judge_model == JUDGE_MODEL


def test_the_exported_sheet_quotes_the_article_each_ruling_hangs_on(tmp_path: Path) -> None:
    code, _, sheet_path = _export(tmp_path)

    assert code == 0
    sheet = sheet_path.read_text(encoding="utf-8")
    assert ARTICLE_TEXT in sheet
    assert KEY_POINT_CLAIM in sheet


def test_build_records_the_reviewed_sample_as_the_calibration_the_run_reads(
    tmp_path: Path,
) -> None:
    _, sample_path, _ = _export(tmp_path, count=5, size=10)
    out = tmp_path / "judge-calibration.json"

    code = main(
        [
            "calibrate",
            "build",
            "--vertical",
            "vivienda",
            "--sample",
            str(sample_path),
            "--reviewed-by",
            "Reviewer",
            "--disagree",
            "2, 4",
            "--out",
            str(out),
        ],
        now=NOW,
    )

    assert code == 0
    calibration = load_calibration(out)
    assert calibration is not None
    assert calibration.sample_size == 10
    assert calibration.agreement == 0.8
    assert calibration.judge_model == JUDGE_MODEL
    assert calibration.reviewed_by == "Reviewer"
    assert calibration.reviewed_at == NOW


def test_build_stamps_the_agreement_onto_the_run_the_sample_came_from(tmp_path: Path) -> None:
    _, sample_path, _ = _export(tmp_path, count=5, size=10)
    run = tmp_path / "modo1-run.json"

    code = main(
        [
            "calibrate",
            "build",
            "--vertical",
            "vivienda",
            "--sample",
            str(sample_path),
            "--reviewed-by",
            "Reviewer",
            "--disagree",
            "2",
            "--out",
            str(tmp_path / "judge-calibration.json"),
            "--stamp",
            str(run),
        ],
        now=NOW,
    )

    assert code == 0
    stamped = read_artifact(run)
    assert stamped.judge_calibration is not None
    assert stamped.judge_calibration.agreement == 0.9
    assert stamped.judge_calibration.sample_size == 10


def test_a_review_with_no_disagreement_is_recorded_as_full_agreement(tmp_path: Path) -> None:
    _, sample_path, _ = _export(tmp_path, count=5, size=10)
    out = tmp_path / "judge-calibration.json"

    code = main(
        [
            "calibrate",
            "build",
            "--vertical",
            "vivienda",
            "--sample",
            str(sample_path),
            "--reviewed-by",
            "Reviewer",
            "--disagree",
            "none",
            "--out",
            str(out),
        ],
        now=NOW,
    )

    assert code == 0
    calibration = load_calibration(out)
    assert calibration is not None
    assert calibration.agreement == 1.0
