"""``eval`` command: run the Mode 1 regression harness, compare runs or calibrate the judge.

``eval run`` drives a case set end-to-end through the real system -- the same
image the API serves -- stamps the run with its configuration fingerprint, applies
the citation guardrail and writes a versionable artifact, exiting non-zero if any
citation broke the literality invariant. ``--repeat`` runs the suite more than once
under that one fingerprint, so each metric is published as the band it moved in.
``eval compare`` reads two artifacts and reports the per-metric delta together with
what it amounts to against those bands and against the between-session drift
``eval drift build`` measures over the archive -- a band is the spread one session
saw, and two runs are never the same session. ``eval envelope`` pools the
repetitions of every archived run at one fingerprint into a single band, which is
the range a published figure should carry: how wide a single session's band comes
out is itself unstable. ``--no-judge`` drops the one task on a
paid provider, so a checkout without that credit still measures every other metric.
``eval calibrate`` draws the judge's own rulings out of a run for a reviewer to
confirm and records the labels the agreement is derived from. The harness uses a
real model; tests inject the runner, corpus and fingerprint to exercise it without
the network.
"""

import argparse
import contextlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
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
from lexme.eval.drift import (
    DRIFT_FILENAME_TEMPLATE,
    DriftRecord,
    RunObservation,
    load_drift,
    measure_drift,
)
from lexme.eval.fingerprint import (
    ConfigFingerprint,
    build_fingerprint,
    compute_dataset_digest,
    compute_mode2_dataset_digest,
    compute_mode2_prompts_digest,
    compute_prompts_digest,
)
from lexme.eval.guardrail import GuardrailViolation
from lexme.eval.judge import JUDGE_TASK, Judge, LlmJudge, assert_judge_distinct_from_generator
from lexme.eval.metrics import JudgeAggregate, scalar_metrics
from lexme.eval.mode2 import (
    DEFAULT_IOU_THRESHOLD,
    Mode2EvalCase,
    Mode2RunArtifact,
    PipelineMode2CaseRunner,
    load_mode2_cases,
    read_mode2_artifact,
    run_mode2_suite,
)
from lexme.eval.mode2.metrics import scalar_metrics_mode2
from lexme.eval.repetition import (
    MetricBand,
    RepetitionSummary,
    Span,
    repetition_values,
    summarize_repetitions,
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
from lexme.llm import build_client, load_task_registry
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
NOT_RECORDED = "not recorded"


@dataclass(frozen=True)
class _Harness:
    """The real collaborators an ``eval run`` needs, built once from settings."""

    runner: CaseRunner
    corpus: CorpusReader
    fingerprint: ConfigFingerprint
    judge: Judge | None


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
    if args.command == "drift":
        return _do_drift(args, now=now)
    if args.command == "envelope":
        return _do_envelope(args)
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
            harness = _build_real_harness(
                stack, settings, args.vertical, cases, judged=not args.no_judge
            )
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
            _repetitions(args, settings),
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
            _repetitions(args, settings),
        )

    out_path = _out_path(args, settings, suite, created_at)
    artifact.write(out_path)
    _report_mode2_run(artifact, out_path)
    return 0 if artifact.passed else 1


def _do_compare(args: argparse.Namespace) -> int:
    """Compare a run against a baseline artifact and report the deltas.

    Both artifacts are named the way every other subcommand names one: a bare file
    name is read from the runs directory, so a comparison is typed the way the
    archive is listed rather than as a path into the container.

    Dispatches on which suite each artifact belongs to -- Mode 2's metrics schema
    is not Mode 1's, so each side is read with its own model. Comparing a Mode 1
    artifact against a Mode 2 one is refused rather than silently misread. The
    suite's drift record is read from beside the baseline unless ``--no-drift``
    says to classify against the repetition bands alone.
    """
    settings = get_settings()
    base = _runs_path(args.base, settings)
    run = _runs_path(args.run, settings)
    base_is_mode2 = _is_mode2_artifact(base)
    run_is_mode2 = _is_mode2_artifact(run)
    if base_is_mode2 != run_is_mode2:
        raise ValueError(
            f"cannot compare a Mode 1 artifact against a Mode 2 one: "
            f"base={base} (mode2={base_is_mode2}) run={run} (mode2={run_is_mode2})"
        )
    suite = MODE2 if base_is_mode2 else MODE1
    drift = _drift_for_comparison(args, suite, base)
    if base_is_mode2:
        comparison = compare_mode2(read_mode2_artifact(base), read_mode2_artifact(run), drift)
    else:
        comparison = compare(read_artifact(base), read_artifact(run), drift)
    _report_comparison(comparison, drift)
    return 0


def _drift_for_comparison(args: argparse.Namespace, suite: str, base: Path) -> DriftRecord | None:
    """The drift record a comparison classifies against, or ``None`` when it has none.

    ``--no-drift`` refuses one and ``--drift`` names one, which must exist: a path
    typed by hand and silently missed would classify as if drift had never been
    measured. Otherwise the record is looked for beside the baseline artifact --
    ``base`` resolved, not as it was typed, so naming the baseline by its bare file
    name still finds the record ``eval drift build`` wrote over that archive -- and
    its absence is the honest answer that none has been measured yet.
    """
    if args.no_drift:
        return None
    if args.drift is not None:
        record = load_drift(args.drift, suite=suite)
        if record is None:
            raise ValueError(f"no drift record at {args.drift}")
        return record
    return load_drift(base.parent / DRIFT_FILENAME_TEMPLATE.format(suite=suite), suite=suite)


def _do_drift(args: argparse.Namespace, *, now: datetime | None) -> int:
    """Measure the archive's between-session drift and write the record.

    Reads every run artifact of the suite in the runs directory, groups them by
    fingerprint and records how far each metric moved between runs that share one.
    """
    runs_dir = args.runs or Path(get_settings().eval_runs_dir)
    paths = _archived_runs(runs_dir, args.mode)
    record = measure_drift(
        args.mode,
        [_observe_archived_run(path, args.mode) for path in paths],
        now or datetime.now(UTC),
    )
    out_path = args.out or runs_dir / DRIFT_FILENAME_TEMPLATE.format(suite=args.mode)
    record.write(out_path)
    _report_drift(record, out_path)
    return 0


def _do_envelope(args: argparse.Namespace) -> int:
    """Band every repetition of every archived run at one fingerprint, and report it.

    A single run's band is one session's draw of the noise, and how wide that draw
    comes out is itself unstable. Pooling the repetitions of every run that shares a
    fingerprint gives the range the configuration has actually been observed across,
    which is what a published figure should carry. Raises :class:`ValueError` when
    no archived run bears the fingerprint asked for.
    """
    runs_dir = args.runs or Path(get_settings().eval_runs_dir)
    observed = [
        (path, _observe_archived_run(path, args.mode))
        for path in _archived_runs(runs_dir, args.mode)
    ]
    fingerprint = args.fingerprint or _newest_fingerprint(observed, runs_dir)
    pooled = [(path, run) for path, run in observed if run.fingerprint == fingerprint]
    if not pooled:
        raise ValueError(f"no archived run of '{args.mode}' at fingerprint {fingerprint!r}")
    _report_envelope(fingerprint, pooled, _pooled_repetitions(runs_dir, pooled, args.mode))
    return 0


def _newest_fingerprint(observed: Sequence[tuple[Path, RunObservation]], runs_dir: Path) -> str:
    """The fingerprint of the most recent archived run, by artifact name."""
    if not observed:
        raise ValueError(f"no archived run to read a fingerprint from in {runs_dir}")
    return observed[-1][1].fingerprint


def _pooled_repetitions(
    runs_dir: Path, pooled: Sequence[tuple[Path, RunObservation]], suite: str
) -> RepetitionSummary:
    """Fold every repetition of every pooled run into one band per metric.

    A run that measured no band at all still contributes the one draw it is, so a
    single-repetition run is not silently dropped from the envelope it belongs in.
    """
    draws: list[Mapping[str, float | None]] = []
    for path, _ in pooled:
        repetitions = _artifact_repetitions(path, suite)
        if repetitions is None:
            draws.append(_bare_values(path, suite))
            continue
        draws.extend(repetition_values(repetitions))
    return summarize_repetitions(draws)


def _artifact_repetitions(path: Path, suite: str) -> RepetitionSummary | None:
    """The repetition summary an archived run carries, or ``None`` for an older one."""
    if suite == MODE2:
        return read_mode2_artifact(path).repetitions
    return read_artifact(path).repetitions


def _bare_values(path: Path, suite: str) -> Mapping[str, float | None]:
    """One run's scalar projection, for a run that banded nothing."""
    if suite == MODE2:
        return scalar_metrics_mode2(read_mode2_artifact(path).metrics)
    return scalar_metrics(read_artifact(path).metrics)


def _archived_runs(runs_dir: Path, suite: str) -> list[Path]:
    """Every run artifact of ``suite`` in ``runs_dir``, oldest name first.

    The review sheets a calibration pass leaves beside the runs carry the run's own
    name and would be read as runs; they are excluded by their suffix rather than
    by failing to parse.
    """
    return sorted(
        path for path in runs_dir.glob(f"{suite}-*.json") if REVIEW_SUFFIX not in path.stem
    )


def _observe_archived_run(path: Path, suite: str) -> RunObservation:
    """Read one archived run of ``suite`` as the values and spans drift is measured over."""
    if suite == MODE2:
        mode2 = read_mode2_artifact(path)
        return _run_observation(
            path,
            mode2.fingerprint.fingerprint,
            scalar_metrics_mode2(mode2.metrics),
            mode2.repetitions,
        )
    artifact = read_artifact(path)
    return _run_observation(
        path,
        artifact.fingerprint.fingerprint,
        scalar_metrics(artifact.metrics),
        artifact.repetitions,
    )


def _run_observation(
    path: Path,
    fingerprint: str,
    values: dict[str, float | None],
    repetitions: RepetitionSummary | None,
) -> RunObservation:
    """Fold a run's published values and its repetition bands into one observation.

    A banded metric publishes its median rather than the first repetition's draw,
    the same value a comparison reads, so the spread measured across runs is the
    spread of what the runs actually claim.
    """
    bands = _measured_bands(repetitions)
    return RunObservation(
        run=path.name,
        fingerprint=fingerprint,
        values={**values, **{metric: band.median for metric, band in bands.items()}},
        spans={metric: band.span for metric, band in bands.items()},
    )


def _measured_bands(repetitions: RepetitionSummary | None) -> dict[str, MetricBand]:
    """The run's bands by metric, empty when it ran too few repetitions to band."""
    if repetitions is None or not repetitions.measures_noise:
        return {}
    return {band.metric: band for band in repetitions.bands}


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
    judged: bool = True,
) -> _Harness:
    """Wire the real Mode 1 system, corpus, judge and fingerprint for a live eval run.

    Cross-checks the cases' pinned clarification answers against the vertical's
    branch package here, the one place both are known, so a case that names a
    branch the vertical does not declare fails the run instead of silently going
    unmeasured behind a pause nothing answers.

    An unjudged run drops the judge task from the registry rather than building it
    and ignoring it. The judge is the one task on a paid provider, so dropping it
    leaves that provider unreferenced and the run needs no key for it -- and the
    fingerprint, which pins whatever the registry holds, stops claiming a judge the
    run never consulted.
    """
    connection = _open_connection(stack, settings.database_url)
    embedder = stack.enter_context(contextlib.closing(TeiEmbedder(settings.tei_url)))
    corpus = PsycopgCorpusReader(connection)
    vertical_dir = Path(settings.verticals_dir) / vertical
    registry = load_task_registry()
    if judged:
        assert_judge_distinct_from_generator(registry)
    else:
        registry = registry.without(JUDGE_TASK)
        logger.info("--no-judge: running unjudged, no completeness or claim numbers")
    llm = build_client(registry, settings)
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
        judge=LlmJudge(llm=llm, corpus=corpus) if judged else None,
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
    llm = build_client(registry, settings)
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


def _repetitions(args: argparse.Namespace, settings: Settings) -> int:
    """How many times to run the suite: ``--repeat``, else the configured default.

    The default is configuration rather than a constant because each repetition
    multiplies the run's model calls, and what a provider's quota affords is a
    property of the deployment, not of the harness.
    """
    if args.repeat is not None:
        return args.repeat
    return settings.eval_repetitions


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
    """Parse the ``run``, ``compare``, ``drift`` and ``calibrate`` subcommands."""
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
    run.add_argument(
        "--no-judge",
        action="store_true",
        help="run the suite without the LLM judge, so every non-judged metric is still "
        "measured; the judge is the one task on a paid provider, and this is how a "
        "checkout without that credit runs the harness",
    )
    run.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="times to run the suite under the same fingerprint, so each metric is "
        "published as a band rather than a single value; every repetition costs a "
        "full set of model calls (default: EVAL_REPETITIONS)",
    )

    comparison = sub.add_parser("compare", help="compare a run against a baseline artifact")
    comparison.add_argument("--base", type=Path, required=True, help="baseline artifact path")
    comparison.add_argument("--run", type=Path, required=True, help="run artifact path")
    comparison.add_argument(
        "--drift",
        type=Path,
        default=None,
        help="drift record to classify against (default: the one beside the baseline)",
    )
    comparison.add_argument(
        "--no-drift",
        action="store_true",
        help="classify against the repetition bands alone, which measure the spread "
        "inside one session and not the drift between them",
    )

    _add_drift_parser(sub)
    _add_envelope_parser(sub)
    _add_calibrate_parser(sub)

    return parser.parse_args(argv)


def _add_drift_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``drift build`` and its options."""
    drift = sub.add_parser("drift", help="measure the archive's between-session drift")
    drift_sub = drift.add_subparsers(dest="drift_command", required=True)

    build = drift_sub.add_parser("build", help="measure drift over the archived runs")
    build.add_argument(
        "--mode",
        choices=(MODE1, MODE2),
        default=MODE1,
        help="which suite's archive to measure (default: modo1)",
    )
    build.add_argument(
        "--runs", type=Path, help="directory of run artifacts (default: the runs directory)"
    )
    build.add_argument("--out", type=Path, help="drift record destination path")


def _add_envelope_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``envelope`` and its options."""
    envelope = sub.add_parser(
        "envelope", help="band every repetition of every archived run at one fingerprint"
    )
    envelope.add_argument(
        "--mode",
        choices=(MODE1, MODE2),
        default=MODE1,
        help="which suite's archive to pool (default: modo1)",
    )
    envelope.add_argument(
        "--runs", type=Path, help="directory of run artifacts (default: the runs directory)"
    )
    envelope.add_argument(
        "--fingerprint",
        default=None,
        help="the configuration to pool (default: the newest archived run's)",
    )


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
    _report_repetitions(artifact.repetitions)
    logger.info("artifact written to %s", out_path)
    _report_guardrail(artifact)


def _report_repetitions(repetitions: RepetitionSummary | None) -> None:
    """Log each metric as the median and range its repetitions observed.

    A single repetition is reported as such: it measured no band, and the metrics
    above are one draw rather than a number a later run can be compared against.
    """
    if repetitions is None:
        return
    if not repetitions.measures_noise:
        logger.info(
            "one repetition: the numbers above are a single draw, no noise band was measured"
        )
        return
    logger.info(
        "%d repetitions under the same fingerprint; median [min, max]", repetitions.repetitions
    )
    for band in repetitions.bands:
        logger.info(
            "%s: %s [%s, %s] over %s", band.metric, band.median, band.low, band.high, band.values
        )


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
            "guardrail failure in case '%s' [%s] re-verified at %s: %s -- displayed quote: %s",
            failure.case_id,
            failure.block_ref,
            failure.verified_at.isoformat() if failure.verified_at else NOT_RECORDED,
            failure.reason,
            repr(failure.quote) if failure.quote else NOT_RECORDED,
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
    _report_repetitions(artifact.repetitions)
    logger.info("artifact written to %s", out_path)
    _report_mode2_guardrail(artifact)


def _report_mode2_guardrail(artifact: Mode2RunArtifact) -> None:
    """Log the Mode 2 citation guardrail's verdict, its denominator and any failures."""
    _log_guardrail(
        artifact.passed,
        artifact.hard_failures,
        "citation guardrail passed: all "
        f"{artifact.metrics.displayed_citations} finding citations re-verified",
    )


def _report_comparison(comparison: Comparison, drift: DriftRecord | None) -> None:
    """Log whether the fingerprint changed and each metric's delta and movement."""
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
    if not comparison.noise_measured:
        logger.warning(
            "at least one side ran a single repetition: no noise band was measured, so a "
            "move outside an exact tie is reported as a result whether or not it is one"
        )
    _report_drift_source(drift)
    for delta in comparison.deltas:
        logger.info(
            "%s: base=%s%s run=%s%s delta=%s [%s]",
            delta.metric,
            delta.base,
            _span_label(delta.base_span),
            delta.run,
            _span_label(delta.run_span),
            delta.delta,
            delta.movement.value if delta.movement is not None else "not measured",
        )


def _report_drift_source(drift: DriftRecord | None) -> None:
    """Log which drift record framed the classification, or that none did."""
    if drift is None:
        logger.warning(
            "no drift record: moves are classified against the within-session repetition "
            "bands alone, which are a floor on the noise and not its ceiling"
        )
        return
    logger.info(
        "drift measured over %d archived runs (%s): a move the declared allowance covers "
        "is reported as drift, not as a result",
        len(drift.runs),
        drift.measured_at.date().isoformat(),
    )


def _report_envelope(
    fingerprint: str,
    pooled: Sequence[tuple[Path, RunObservation]],
    envelope: RepetitionSummary,
) -> None:
    """Log the pooled band per metric and the runs whose repetitions built it."""
    logger.info(
        "envelope at fingerprint '%s': %d repetitions over %d run(s)",
        fingerprint,
        envelope.repetitions,
        len(pooled),
    )
    for path, _ in pooled:
        logger.info("  pooled %s", path.name)
    for band in envelope.bands:
        logger.info(
            "%s: %s [%s, %s] over %s", band.metric, band.median, band.low, band.high, band.values
        )


def _report_drift(record: DriftRecord, out_path: Path) -> None:
    """Log each metric's between-session drift beside the spread one session saw."""
    logger.info(
        "drift for '%s' over %d archived runs; between sessions vs within one",
        record.suite,
        len(record.runs),
    )
    for band in record.bands:
        logger.info(
            "%s: between=%s within=%s over %d runs at %s",
            band.metric,
            band.between_sessions,
            "not banded" if band.within_session is None else band.within_session,
            len(band.runs),
            band.fingerprint,
        )
    logger.info("drift record written to %s", out_path)


def _span_label(span: Span | None) -> str:
    """The observed range as a suffix to the value, empty when none was measured."""
    if span is None:
        return ""
    return f" [{span.low}, {span.high}]"


if __name__ == "__main__":
    sys.exit(main())
