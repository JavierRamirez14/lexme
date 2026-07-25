"""``eval`` command: run the Mode 1 regression harness or compare two runs.

``eval run`` drives a case set end-to-end through the real system -- the same
image the API serves -- stamps the run with its configuration fingerprint, applies
the citation guardrail and writes a versionable artifact, exiting non-zero if any
citation broke the literality invariant. ``eval compare`` reads two artifacts and
reports the per-metric delta. The harness uses a real model; tests inject the
runner, corpus and fingerprint to exercise it without the network.
"""

import argparse
import contextlib
import logging
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import psycopg
from langgraph.checkpoint.memory import InMemorySaver
from pgvector.psycopg import register_vector

from lexme.config import Settings, get_settings
from lexme.eval.artifact import RunArtifact, read_artifact
from lexme.eval.calibration import CALIBRATION_FILENAME, JudgeCalibration, load_calibration
from lexme.eval.cases import EvalCase, load_cases
from lexme.eval.compare import Comparison, compare
from lexme.eval.fingerprint import (
    ConfigFingerprint,
    build_fingerprint,
    compute_dataset_digest,
    compute_prompts_digest,
)
from lexme.eval.judge import Judge, LlmJudge, assert_judge_distinct_from_generator
from lexme.eval.metrics import JudgeAggregate
from lexme.eval.runner import CaseRunner, Mode1CaseRunner, run_suite
from lexme.ingestion.embeddings import TeiEmbedder
from lexme.ingestion.repository import get_corpus_digest
from lexme.llm import build_llm_client, load_task_registry
from lexme.mode1 import Mode1Deps, PsycopgVersionHistory, load_branches
from lexme.verification import CorpusReader, PsycopgCorpusReader

logger = logging.getLogger(__name__)

DEFAULT_SUITE = "modo1"
DISAMBIGUATION_FILENAME = "disambiguation.json"


@dataclass(frozen=True)
class _Harness:
    """The real collaborators an ``eval run`` needs, built once from settings."""

    runner: CaseRunner
    corpus: CorpusReader
    fingerprint: ConfigFingerprint
    judge: Judge


def main(
    argv: list[str] | None = None,
    *,
    runner: CaseRunner | None = None,
    corpus: CorpusReader | None = None,
    fingerprint: ConfigFingerprint | None = None,
    today: date | None = None,
    now: datetime | None = None,
) -> int:
    """Run the ``eval`` CLI and return its exit code.

    ``runner``, ``corpus`` and ``fingerprint`` default to the real system built
    from settings; tests inject all three to run the harness without a model or the
    network. ``today`` fixes the point-in-time clock and ``now`` the artifact
    timestamp, so a test run is deterministic.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    if args.command == "compare":
        return _do_compare(args)
    return _do_run(
        args, runner=runner, corpus=corpus, fingerprint=fingerprint, today=today, now=now
    )


def _do_run(
    args: argparse.Namespace,
    *,
    runner: CaseRunner | None,
    corpus: CorpusReader | None,
    fingerprint: ConfigFingerprint | None,
    today: date | None,
    now: datetime | None,
) -> int:
    """Run a suite end-to-end, write its artifact and return 0 unless it hard-failed."""
    settings = get_settings()
    target_date = today or date.today()
    created_at = now or datetime.now(UTC)
    cases = load_cases(_cases_path(args, settings))
    calibration = load_calibration(_calibration_path(args, settings))

    with ExitStack() as stack:
        judge: Judge | None = None
        if runner is None:
            harness = _build_real_harness(stack, settings, args.vertical, cases)
            runner, corpus, fingerprint = harness.runner, harness.corpus, harness.fingerprint
            judge = harness.judge
        if corpus is None or fingerprint is None:
            raise ValueError("corpus and fingerprint must be provided alongside an injected runner")
        artifact = run_suite(
            args.suite,
            cases,
            runner,
            corpus,
            target_date,
            fingerprint,
            created_at,
            judge,
            calibration,
        )

    out_path = _out_path(args, settings, created_at)
    artifact.write(out_path)
    _report_run(artifact, out_path)
    return 0 if artifact.passed else 1


def _do_compare(args: argparse.Namespace) -> int:
    """Compare a run against a baseline artifact and report the deltas."""
    comparison = compare(read_artifact(args.base), read_artifact(args.run))
    _report_comparison(comparison)
    return 0


def _build_real_harness(
    stack: ExitStack,
    settings: Settings,
    vertical: str,
    cases: Sequence[EvalCase],
) -> _Harness:
    """Wire the real Mode 1 system, corpus, judge and fingerprint for a live eval run."""
    connection = _open_connection(stack, settings.database_url)
    embedder = stack.enter_context(contextlib.closing(TeiEmbedder(settings.tei_url)))
    corpus = PsycopgCorpusReader(connection)
    vertical_dir = Path(settings.verticals_dir) / vertical
    registry = load_task_registry()
    assert_judge_distinct_from_generator(registry)
    llm = build_llm_client(settings)
    deps = Mode1Deps(
        connection=connection,
        embedder=embedder,
        corpus=corpus,
        history=PsycopgVersionHistory(connection),
        llm=llm,
        branches=load_branches(vertical_dir / DISAMBIGUATION_FILENAME),
    )
    runner = Mode1CaseRunner(
        deps=deps,
        checkpointer=InMemorySaver(),
        vertical=vertical,
    )
    fingerprint = build_fingerprint(
        registry,
        vertical,
        vertical_dir,
        get_corpus_digest(connection, vertical),
        compute_prompts_digest(),
        compute_dataset_digest(cases),
    )
    return _Harness(
        runner=runner,
        corpus=corpus,
        fingerprint=fingerprint,
        judge=LlmJudge(llm=llm, corpus=corpus),
    )


def _open_connection(stack: ExitStack, database_url: str) -> psycopg.Connection:
    """Open a database connection with the pgvector adapter registered."""
    connection = stack.enter_context(psycopg.connect(database_url))
    register_vector(connection)
    return connection


def _cases_path(args: argparse.Namespace, settings: Settings) -> Path:
    """The case set to run: the ``--cases`` path, or the vertical's suite directory."""
    if args.cases is not None:
        return args.cases
    return Path(settings.verticals_dir) / args.vertical / "eval" / args.suite


def _calibration_path(args: argparse.Namespace, settings: Settings) -> Path:
    """The vertical's judge-calibration record, read to publish the agreement number."""
    return Path(settings.verticals_dir) / args.vertical / "eval" / CALIBRATION_FILENAME


def _out_path(args: argparse.Namespace, settings: Settings, created_at: datetime) -> Path:
    """The artifact destination: ``--out``, or a timestamped file in the runs directory."""
    if args.out is not None:
        return args.out
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return Path(settings.eval_runs_dir) / f"{args.suite}-{stamp}.json"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``run`` and ``compare`` subcommands and their options."""
    parser = argparse.ArgumentParser(description="Run the Lexme evaluation harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a case set end-to-end and emit its artifact")
    run.add_argument("--vertical", required=True, help="the vertical to evaluate (e.g. vivienda)")
    run.add_argument("--suite", default=DEFAULT_SUITE, help="the case suite to run")
    run.add_argument("--cases", type=Path, help="path to a case file or directory of cases")
    run.add_argument("--out", type=Path, help="artifact destination path")

    comparison = sub.add_parser("compare", help="compare a run against a baseline artifact")
    comparison.add_argument("--base", type=Path, required=True, help="baseline artifact path")
    comparison.add_argument("--run", type=Path, required=True, help="run artifact path")

    return parser.parse_args(argv)


def _report_run(artifact: RunArtifact, out_path: Path) -> None:
    """Log the run's fingerprint, headline metrics and any hard failures."""
    metrics = artifact.metrics
    logger.info("eval '%s' fingerprint %s", artifact.suite, artifact.fingerprint.fingerprint)
    logger.info(
        "cases=%d outcomes=%s outcome_match_rate=%s",
        metrics.cases,
        metrics.outcomes,
        metrics.outcome_match_rate,
    )
    logger.info(
        "retrieval recall first_pass=%s -> final=%s (delta=%s); by layer %s",
        metrics.mean_first_pass_recall,
        metrics.mean_recall,
        metrics.mean_recall_delta,
        metrics.retrieval_layer_recall,
    )
    logger.info(
        "abstention_rate=%s expected_abstention_recall=%s",
        metrics.abstention_rate,
        metrics.expected_abstention_recall,
    )
    _report_judge(metrics.judge, artifact.judge_calibration)
    logger.info("artifact written to %s", out_path)
    _report_guardrail(artifact)


def _report_judge(judge: JudgeAggregate | None, calibration: JudgeCalibration | None) -> None:
    """Log the judged end-to-end numbers next to the human-judge agreement they carry."""
    if judge is None:
        logger.info("judge: no answered case carried reference key points; not judged")
        return
    if calibration is None or calibration.agreement is None:
        agreement = "not calibrated"
    else:
        agreement = f"{calibration.agreement:.2f} over {calibration.sample_size} reviewed"
    logger.info(
        "judge (%d cases): completeness=%s unsupported_claim_rate=%s mean_clarity=%s; "
        "human agreement %s",
        judge.judged_cases,
        judge.mean_completeness,
        judge.unsupported_claim_rate,
        judge.mean_clarity,
        agreement,
    )


def _report_guardrail(artifact: RunArtifact) -> None:
    """Log the citation guardrail's verdict and every hard failure it found."""
    if artifact.passed:
        logger.info("citation guardrail passed: every displayed citation re-verified")
        return
    for failure in artifact.hard_failures:
        logger.error(
            "guardrail failure in case '%s' [%s]: %s",
            failure.case_id,
            failure.block_id,
            failure.reason,
        )
    logger.error("citation guardrail FAILED: %d hard failure(s)", len(artifact.hard_failures))


def _report_comparison(comparison: Comparison) -> None:
    """Log whether the fingerprint changed and each metric's delta."""
    if comparison.fingerprint_changed:
        logger.warning(
            "fingerprint changed (%s -> %s): deltas are an experiment, not a regression",
            comparison.base_fingerprint,
            comparison.run_fingerprint,
        )
    else:
        logger.info(
            "fingerprint unchanged (%s): deltas are a regression signal",
            comparison.run_fingerprint,
        )
    for delta in comparison.deltas:
        logger.info("%s: base=%s run=%s delta=%s", delta.metric, delta.base, delta.run, delta.delta)


if __name__ == "__main__":
    sys.exit(main())
