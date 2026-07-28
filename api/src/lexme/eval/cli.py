"""``eval`` command: run the Mode 1 regression harness, compare runs or calibrate the judge.

``eval run`` drives a case set end-to-end through the real system -- the same
image the API serves -- stamps the run with its configuration fingerprint, applies
the citation guardrail and writes a versionable artifact, exiting non-zero if any
citation broke the literality invariant. ``eval compare`` reads two artifacts and
reports the per-metric delta. ``eval calibrate`` draws the judge's own rulings out
of a run for a reviewer to confirm and records the labels the agreement is derived
from. The harness uses a real model; tests inject the runner, corpus and
fingerprint to exercise it without the network.
"""

import argparse
import contextlib
import json
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

from lexme.blocks import BlockRef
from lexme.checklist import checklist_path, load_checklist
from lexme.config import Settings, get_settings
from lexme.eval.artifact import RunArtifact, read_artifact
from lexme.eval.calibration import (
    CALIBRATION_FILENAME,
    REVIEWER_HUMAN,
    REVIEWER_KINDS,
    JudgeCalibration,
    load_calibration,
)
from lexme.eval.cases import EvalCase, load_cases, reject_unknown_branches
from lexme.eval.compare import Comparison, compare, compare_mode2
from lexme.eval.fingerprint import (
    ConfigFingerprint,
    build_fingerprint,
    compute_dataset_digest,
    compute_mode2_dataset_digest,
    compute_mode2_prompts_digest,
    compute_prompts_digest,
)
from lexme.eval.guardrail import GuardrailViolation
from lexme.eval.judge import Judge, LlmJudge, assert_judge_distinct_from_generator
from lexme.eval.metrics import JudgeAggregate
from lexme.eval.mode2 import (
    DEFAULT_IOU_THRESHOLD,
    Mode2EvalCase,
    Mode2RunArtifact,
    PipelineMode2CaseRunner,
    load_mode2_cases,
    read_mode2_artifact,
    run_mode2_suite,
)
from lexme.eval.review import (
    ReviewSample,
    build_sample,
    calibration_from_review,
    parse_disagreements,
    read_sample,
    render_sheet,
    with_article_texts,
)
from lexme.eval.runner import CaseRunner, Mode1CaseRunner, run_suite
from lexme.ingestion.embeddings import TeiEmbedder
from lexme.ingestion.repository import get_corpus_digest
from lexme.llm import build_llm_client, load_task_registry
from lexme.mode1 import Mode1Deps, PsycopgVersionHistory, load_branches
from lexme.mode2 import HybridClauseRetriever
from lexme.mode2.scope import SCOPE_FILENAME, load_scope
from lexme.verification import CorpusReader, PsycopgCorpusReader

logger = logging.getLogger(__name__)

MODE1 = "modo1"
MODE2 = "modo2"
DEFAULT_SUITE = MODE1
DISAMBIGUATION_FILENAME = "disambiguation.json"
MODE2_CASES_DIRNAME = "refset"
CALIBRATE_EXPORT = "export"
CALIBRATE_BUILD = "build"
DEFAULT_SAMPLE_SIZE = 50
DEFAULT_SAMPLE_SEED = 0
REVIEW_SUFFIX = "-judge-review"


@dataclass(frozen=True)
class _Harness:
    """The real collaborators an ``eval run`` needs, built once from settings."""

    runner: CaseRunner
    corpus: CorpusReader
    fingerprint: ConfigFingerprint
    judge: Judge


@dataclass(frozen=True)
class _Mode2Harness:
    """The real collaborators a Mode 2 ``eval run`` needs, built once from settings.

    ``norm_id`` is the vertical's governing norm, the id the citation guardrail
    re-resolves every finding's block against.
    """

    runner: PipelineMode2CaseRunner
    corpus: CorpusReader
    fingerprint: ConfigFingerprint
    norm_id: str


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
    if args.command == "calibrate":
        return _do_calibrate(args, corpus=corpus, now=now)
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
    if args.mode == MODE2:
        return _do_run_mode2(args, today=today, now=now)
    return _do_run_mode1(
        args, runner=runner, corpus=corpus, fingerprint=fingerprint, today=today, now=now
    )


def _do_run_mode1(
    args: argparse.Namespace,
    *,
    runner: CaseRunner | None,
    corpus: CorpusReader | None,
    fingerprint: ConfigFingerprint | None,
    today: date | None,
    now: datetime | None,
) -> int:
    """Run the Mode 1 suite end-to-end and write its artifact."""
    settings = get_settings()
    suite = args.suite or DEFAULT_SUITE
    target_date = today or date.today()
    created_at = now or datetime.now(UTC)
    cases = load_cases(_cases_path(args, settings, suite))
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
            suite,
            cases,
            runner,
            corpus,
            target_date,
            fingerprint,
            created_at,
            judge,
            calibration,
        )

    out_path = _out_path(args, settings, suite, created_at)
    artifact.write(out_path)
    _report_run(artifact, out_path)
    return 0 if artifact.passed else 1


def _do_run_mode2(
    args: argparse.Namespace,
    *,
    today: date | None,
    now: datetime | None,
) -> int:
    """Run the Mode 2 contract suite end-to-end and write its artifact.

    Wires the real risk-map pipeline against the ingested corpus and the vertical's
    scope and checklist, drives every synthetic contract through it, re-verifies the
    findings' citations and writes the run artifact. Returns 0 unless a citation
    broke the literality invariant.
    """
    settings = get_settings()
    suite = args.suite or MODE2
    target_date = today or date.today()
    created_at = now or datetime.now(UTC)
    cases = load_mode2_cases(_mode2_cases_path(args, settings))

    with ExitStack() as stack:
        harness = _build_mode2_harness(stack, settings, args.vertical, cases)
        artifact = run_mode2_suite(
            suite,
            cases,
            harness.runner,
            harness.corpus,
            harness.norm_id,
            target_date,
            harness.fingerprint,
            created_at,
        )

    out_path = _out_path(args, settings, suite, created_at)
    artifact.write(out_path)
    _report_mode2_run(artifact, out_path)
    return 0 if artifact.passed else 1


def _do_compare(args: argparse.Namespace) -> int:
    """Compare a run against a baseline artifact and report the deltas.

    Dispatches on which suite each artifact belongs to -- Mode 2's metrics schema
    is not Mode 1's, so each side is read with its own model. Comparing a Mode 1
    artifact against a Mode 2 one is refused rather than silently misread.
    """
    base_is_mode2 = _is_mode2_artifact(args.base)
    run_is_mode2 = _is_mode2_artifact(args.run)
    if base_is_mode2 != run_is_mode2:
        raise ValueError(
            f"cannot compare a Mode 1 artifact against a Mode 2 one: "
            f"base={args.base} (mode2={base_is_mode2}) run={args.run} (mode2={run_is_mode2})"
        )
    if base_is_mode2:
        comparison = compare_mode2(read_mode2_artifact(args.base), read_mode2_artifact(args.run))
    else:
        comparison = compare(read_artifact(args.base), read_artifact(args.run))
    _report_comparison(comparison)
    return 0


def _is_mode2_artifact(path: Path) -> bool:
    """Whether the artifact at ``path`` is a Mode 2 run, read without full validation.

    Peeks at the raw JSON for a field only :class:`Mode2SuiteMetrics` carries, so
    dispatch does not depend on the ``suite`` name a run happened to be given.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return "problematic_total" in raw.get("metrics", {})


def _do_calibrate(
    args: argparse.Namespace, *, corpus: CorpusReader | None, now: datetime | None
) -> int:
    """Draw the judge's rulings for review, or record the labels a review came back with."""
    if args.calibration_command == CALIBRATE_EXPORT:
        return _do_calibrate_export(args, corpus=corpus)
    return _do_calibrate_build(args, now=now)


def _do_calibrate_export(args: argparse.Namespace, *, corpus: CorpusReader | None) -> int:
    """Draw a seeded sample of a run's judge rulings and write it with its review sheet.

    Each ruling is rendered with the article it hangs on, resolved at the case's own
    point-in-time date so the reviewer reads the redaction the case was judged
    under. ``corpus`` defaults to the ingested corpus; tests inject it.
    """
    settings = get_settings()
    run_path = _runs_path(args.run, settings)
    artifact = read_artifact(run_path)
    cases = load_cases(_cases_path(args, settings, DEFAULT_SUITE))
    sample = build_sample(artifact, cases, args.size, args.seed, run=run_path.name)

    with ExitStack() as stack:
        if corpus is None:
            corpus = PsycopgCorpusReader(_open_connection(stack, settings.database_url))
        sample = with_article_texts(
            sample, _resolve_articles(sample, cases, corpus, artifact.created_at.date())
        )

    sample_path = args.out or _review_path(run_path, settings, ".json")
    sheet_path = args.sheet or _review_path(run_path, settings, ".md")
    sample.write(sample_path)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_path.write_text(render_sheet(sample), encoding="utf-8")
    logger.info(
        "drew %d of %d judge rulings (seed %d) from %s",
        sample.size,
        sample.population,
        sample.seed,
        args.run.name,
    )
    logger.info("review sheet written to %s; sample to %s", sheet_path, sample_path)
    return 0


def _do_calibrate_build(args: argparse.Namespace, *, now: datetime | None) -> int:
    """Record a reviewed sample as the calibration record the harness publishes.

    ``--stamp`` also writes the agreement onto a run artifact, so the run whose own
    rulings were reviewed publishes the number rather than waiting for the next one.
    """
    settings = get_settings()
    sample = read_sample(_runs_path(args.sample, settings))
    disagreed = parse_disagreements(args.disagree)
    calibration = calibration_from_review(
        sample, disagreed, args.reviewed_by, args.reviewer_kind, now or datetime.now(UTC)
    )
    out_path = args.out or _calibration_path(args, settings)
    calibration.write(out_path)
    if args.stamp is not None:
        stamped = _runs_path(args.stamp, settings)
        artifact = read_artifact(stamped)
        artifact.model_copy(update={"judge_calibration": calibration}).write(stamped)
        logger.info("stamped the agreement onto %s", stamped)
    logger.info(
        "judge '%s' agrees with the %s reviewer on %s of %d rulings (%d disagreements) -> %s",
        calibration.judge_model,
        calibration.reviewer_kind,
        f"{calibration.agreement:.2f}" if calibration.agreement is not None else "n/a",
        calibration.sample_size,
        len(disagreed),
        out_path,
    )
    return 0


def _resolve_articles(
    sample: ReviewSample,
    cases: Sequence[EvalCase],
    corpus: CorpusReader,
    default_date: date,
) -> dict[tuple[str, str], str]:
    """The in-force text of every article a ruling hangs on, keyed by case and reference."""
    dates = {case.id: case.target_date or default_date for case in cases}
    texts: dict[tuple[str, str], str] = {}
    for ruling in sample.rulings:
        key = (ruling.case_id, ruling.article_ref or "")
        if not ruling.article_ref or key in texts:
            continue
        block = BlockRef.parse(ruling.article_ref)
        if block is None:
            continue
        resolved = corpus.resolve_block(
            block.norm_id, block.block_id, dates.get(ruling.case_id, default_date)
        )
        if resolved is not None:
            texts[key] = resolved.text
    return texts


def _build_real_harness(
    stack: ExitStack,
    settings: Settings,
    vertical: str,
    cases: Sequence[EvalCase],
) -> _Harness:
    """Wire the real Mode 1 system, corpus, judge and fingerprint for a live eval run.

    Cross-checks the cases' pinned clarification answers against the vertical's
    branch package here, the one place both are known, so a case that names a
    branch the vertical does not declare fails the run instead of silently going
    unmeasured behind a pause nothing answers.
    """
    connection = _open_connection(stack, settings.database_url)
    embedder = stack.enter_context(contextlib.closing(TeiEmbedder(settings.tei_url)))
    corpus = PsycopgCorpusReader(connection)
    vertical_dir = Path(settings.verticals_dir) / vertical
    registry = load_task_registry()
    assert_judge_distinct_from_generator(registry)
    llm = build_llm_client(settings)
    branches = load_branches(vertical_dir / DISAMBIGUATION_FILENAME)
    reject_unknown_branches(cases, [branch.id for branch in branches])
    deps = Mode1Deps(
        connection=connection,
        embedder=embedder,
        corpus=corpus,
        history=PsycopgVersionHistory(connection),
        llm=llm,
        branches=branches,
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


def _build_mode2_harness(
    stack: ExitStack,
    settings: Settings,
    vertical: str,
    cases: Sequence[Mode2EvalCase],
) -> "_Mode2Harness":
    """Wire the real Mode 2 pipeline, corpus and fingerprint for a live eval run."""
    connection = _open_connection(stack, settings.database_url)
    embedder = stack.enter_context(contextlib.closing(TeiEmbedder(settings.tei_url)))
    corpus = PsycopgCorpusReader(connection)
    vertical_dir = Path(settings.verticals_dir) / vertical
    checklist = load_checklist(checklist_path(settings.verticals_dir, vertical))
    scope = load_scope(vertical_dir / SCOPE_FILENAME)
    registry = load_task_registry()
    llm = build_llm_client(settings)
    runner = PipelineMode2CaseRunner(
        llm=llm,
        scope=scope,
        checklist=checklist,
        corpus=corpus,
        retriever=HybridClauseRetriever(connection, embedder),
        vertical=vertical,
    )
    fingerprint = build_fingerprint(
        registry,
        vertical,
        vertical_dir,
        get_corpus_digest(connection, vertical),
        compute_mode2_prompts_digest(),
        compute_mode2_dataset_digest(cases),
    )
    return _Mode2Harness(
        runner=runner, corpus=corpus, fingerprint=fingerprint, norm_id=checklist.norm_id
    )


def _open_connection(stack: ExitStack, database_url: str) -> psycopg.Connection:
    """Open a database connection with the pgvector adapter registered."""
    connection = stack.enter_context(psycopg.connect(database_url))
    register_vector(connection)
    return connection


def _cases_path(args: argparse.Namespace, settings: Settings, suite: str) -> Path:
    """The case set to run: the ``--cases`` path, or the vertical's suite directory."""
    if args.cases is not None:
        return args.cases
    return Path(settings.verticals_dir) / args.vertical / "eval" / suite


def _mode2_cases_path(args: argparse.Namespace, settings: Settings) -> Path:
    """The Mode 2 contract set: ``--cases``, or the vertical's ``refset/modo2`` directory."""
    if args.cases is not None:
        return args.cases
    return Path(settings.verticals_dir) / args.vertical / MODE2_CASES_DIRNAME / MODE2


def _calibration_path(args: argparse.Namespace, settings: Settings) -> Path:
    """The vertical's judge-calibration record, read to publish the agreement number."""
    return Path(settings.verticals_dir) / args.vertical / "eval" / CALIBRATION_FILENAME


def _review_path(run_path: Path, settings: Settings, suffix: str) -> Path:
    """The default destination for a run's review sample, beside the run it came from."""
    return Path(settings.eval_runs_dir) / f"{run_path.stem}{REVIEW_SUFFIX}{suffix}"


def _runs_path(path: Path, settings: Settings) -> Path:
    """Read an artifact path, taking a bare file name as one of the runs directory."""
    if path.parent == Path("."):
        return Path(settings.eval_runs_dir) / path
    return path


def _out_path(
    args: argparse.Namespace, settings: Settings, suite: str, created_at: datetime
) -> Path:
    """The artifact destination: ``--out``, or a timestamped file in the runs directory."""
    if args.out is not None:
        return args.out
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return Path(settings.eval_runs_dir) / f"{suite}-{stamp}.json"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``run`` and ``compare`` subcommands and their options."""
    parser = argparse.ArgumentParser(description="Run the Lexme evaluation harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a case set end-to-end and emit its artifact")
    run.add_argument("--vertical", required=True, help="the vertical to evaluate (e.g. vivienda)")
    run.add_argument(
        "--mode",
        choices=(MODE1, MODE2),
        default=MODE1,
        help="which pipeline to evaluate (default: modo1)",
    )
    run.add_argument("--suite", default=None, help="the artifact label (default: the mode name)")
    run.add_argument("--cases", type=Path, help="path to a case file or directory of cases")
    run.add_argument("--out", type=Path, help="artifact destination path")

    comparison = sub.add_parser("compare", help="compare a run against a baseline artifact")
    comparison.add_argument("--base", type=Path, required=True, help="baseline artifact path")
    comparison.add_argument("--run", type=Path, required=True, help="run artifact path")

    _add_calibrate_parser(sub)

    return parser.parse_args(argv)


def _add_calibrate_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``calibrate export`` and ``calibrate build`` and their options."""
    calibrate = sub.add_parser("calibrate", help="draw and record the judge's calibration")
    calibrate_sub = calibrate.add_subparsers(dest="calibration_command", required=True)

    export = calibrate_sub.add_parser(
        CALIBRATE_EXPORT, help="draw a sample of a run's judge rulings for review"
    )
    export.add_argument("--vertical", required=True, help="the vertical the run belongs to")
    export.add_argument(
        "--run",
        type=Path,
        required=True,
        help="run artifact to draw rulings from; a bare name is read from the runs directory",
    )
    export.add_argument("--cases", type=Path, help="path to a case file or directory of cases")
    export.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"rulings to draw (default: {DEFAULT_SAMPLE_SIZE})",
    )
    export.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help=f"seed the draw is reproducible from (default: {DEFAULT_SAMPLE_SEED})",
    )
    export.add_argument("--out", type=Path, help="sample destination path")
    export.add_argument("--sheet", type=Path, help="review sheet destination path")

    build = calibrate_sub.add_parser(
        CALIBRATE_BUILD, help="record a reviewed sample as the vertical's calibration"
    )
    build.add_argument("--vertical", required=True, help="the vertical the calibration is for")
    build.add_argument("--sample", type=Path, required=True, help="the drawn sample reviewed")
    build.add_argument("--reviewed-by", required=True, help="who reviewed the sample")
    build.add_argument(
        "--reviewer-kind",
        required=True,
        choices=REVIEWER_KINDS,
        help="what the reviewer was; the agreement can only be read as strong evidence "
        "about the judge when it is 'human'",
    )
    build.add_argument(
        "--disagree",
        required=True,
        help="comma-separated numbers of the rulings the reviewer disagreed with, or 'none'",
    )
    build.add_argument("--out", type=Path, help="calibration record destination path")
    build.add_argument(
        "--stamp",
        type=Path,
        help="run artifact to publish the agreement on, usually the one reviewed",
    )


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
    logger.info(
        "disambiguation_rate=%s; cases %s",
        metrics.disambiguation_rate,
        metrics.disambiguation,
    )
    _report_judge(metrics.judge, artifact.judge_calibration)
    logger.info("artifact written to %s", out_path)
    _report_guardrail(artifact)


def _report_judge(judge: JudgeAggregate | None, calibration: JudgeCalibration | None) -> None:
    """Log the judged end-to-end numbers next to the reviewer agreement they carry."""
    if judge is None:
        logger.info("judge: no answered case carried reference key points; not judged")
        return
    if calibration is None or calibration.agreement is None:
        agreement = "not calibrated"
    else:
        qualifier = "" if calibration.reviewer_kind == REVIEWER_HUMAN else " (model reviewer)"
        agreement = (
            f"{calibration.agreement:.2f} over {calibration.sample_size} reviewed{qualifier}"
        )
    logger.info(
        "judge (%d cases): completeness=%s unsupported_claim_rate=%s mean_clarity=%s; "
        "reviewer agreement %s",
        judge.judged_cases,
        judge.mean_completeness,
        judge.unsupported_claim_rate,
        judge.mean_clarity,
        agreement,
    )


def _report_guardrail(artifact: RunArtifact) -> None:
    """Log the citation guardrail's verdict and every hard failure it found."""
    _log_guardrail(
        artifact.passed,
        artifact.hard_failures,
        "citation guardrail passed: every displayed citation re-verified",
    )


def _log_guardrail(
    passed: bool, hard_failures: list[GuardrailViolation], passed_message: str
) -> None:
    """Log the guardrail's verdict: a pass line, or every hard failure and a total."""
    if passed:
        logger.info("%s", passed_message)
        return
    for failure in hard_failures:
        logger.error(
            "guardrail failure in case '%s' [%s]: %s",
            failure.case_id,
            failure.block_ref,
            failure.reason,
        )
    logger.error("citation guardrail FAILED: %d hard failure(s)", len(hard_failures))


def _report_mode2_run(artifact: Mode2RunArtifact, out_path: Path) -> None:
    """Log the Mode 2 headline numbers, their diagnostics and any hard failures."""
    metrics = artifact.metrics
    logger.info("eval '%s' fingerprint %s", artifact.suite, artifact.fingerprint.fingerprint)
    logger.info(
        "cases=%d outcomes=%s outcome_match_rate=%s",
        metrics.cases,
        metrics.outcomes,
        metrics.outcome_match_rate,
    )
    logger.info(
        "recall_problematic=%s (%d/%d) | false_tranquility_rate=%s (%d/%d) "
        "[conditioned on segmentation]",
        metrics.recall_problematic,
        metrics.problematic_detected,
        metrics.problematic_total,
        metrics.false_tranquility_rate,
        metrics.false_tranquility_events,
        metrics.problematic_total,
    )
    logger.info(
        "recall_problematic_e2e=%s (%d/%d) | false_tranquility_rate_e2e=%s (%d/%d) | "
        "not_reported_problematic=%d [end to end, every reference clause]",
        metrics.recall_problematic_e2e,
        metrics.problematic_detected_e2e,
        metrics.problematic_total_e2e,
        metrics.false_tranquility_rate_e2e,
        metrics.false_tranquility_events_e2e,
        metrics.problematic_total_e2e,
        metrics.not_reported_problematic,
    )
    logger.info(
        "precision_problematic=%s (%d/%d) | abstention_rate=%s (%d/%d)",
        metrics.precision_problematic,
        metrics.flagged_true_positive,
        metrics.flagged_problematic,
        metrics.abstention_rate,
        metrics.abstention_clauses,
        metrics.matched_clauses,
    )
    logger.info(
        "segmentation delimited_rate=%s (%d/%d, IoU>=%.2f) | level metrics over matched clauses",
        metrics.segmentation_delimited_rate,
        metrics.delimited_clauses,
        metrics.reference_clauses,
        DEFAULT_IOU_THRESHOLD,
    )
    logger.info(
        "absence recall=%s (%d/%d) precision=%s",
        metrics.absence_recall,
        metrics.absence_detected,
        metrics.absence_expected,
        metrics.absence_precision,
    )
    logger.info("confusion (gold -> predicted): %s", metrics.confusion)
    logger.info("artifact written to %s", out_path)
    _report_mode2_guardrail(artifact)


def _report_mode2_guardrail(artifact: Mode2RunArtifact) -> None:
    """Log the Mode 2 citation guardrail's verdict and every hard failure it found."""
    _log_guardrail(
        artifact.passed,
        artifact.hard_failures,
        "citation guardrail passed: every finding citation re-verified",
    )


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
