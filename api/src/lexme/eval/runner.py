"""The harness that runs a case set end-to-end and assembles its artifact.

The harness drives each case through a :class:`CaseRunner` -- the seam behind
which the real Mode 1 consultation lives -- applies the citation guardrail to the
response, measures the quality metrics and folds everything into one artifact. The
runner is a seam so the harness itself is tested with the real system behind a
fake LLM, and never needs a live model of its own.

The suite can be run more than once under the same fingerprint. Two runs of the
same configuration do not return the same numbers, so the harness measures that
spread itself and publishes each metric as the band it moved in rather than as a
single value a later run could not honestly be compared against.

A case may stop mid-run on the disambiguation gate. The harness answers it only
with the reply the case brings written down: it hands the runner the case's pinned
answers, the runner resumes the paused thread with the one matching the branch
actually asked, and a pause it has no written answer for is recorded as such rather
than guessed at.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, runtime_checkable
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver

from lexme.eval.artifact import RunArtifact, build_artifact
from lexme.eval.calibration import JudgeCalibration, calibration_for_judge
from lexme.eval.cases import ClarificationAnswer, EvalCase, answer_for_branch
from lexme.eval.fingerprint import ConfigFingerprint
from lexme.eval.guardrail import GuardrailViolation, check_case_citations
from lexme.eval.judge import JUDGE_TASK, Judge
from lexme.eval.metrics import (
    CaseResult,
    Disambiguation,
    aggregate,
    build_case_result,
    scalar_metrics,
)
from lexme.eval.repetition import summarize_repetitions
from lexme.mode1 import AskResponse, Mode1Deps, Mode1Run, Outcome
from lexme.verification import CorpusReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseRun:
    """What running one case produced: its final response and how it got there.

    ``response`` is the outcome the metrics, the guardrail and the judge are
    applied to -- the answer a resumed case reached, never the pause it stopped at.
    ``disambiguation`` records whether the case answered directly, was resumed with
    its pinned reply, or ended at a pause it brought no reply for.
    """

    response: AskResponse
    disambiguation: Disambiguation


@runtime_checkable
class CaseRunner(Protocol):
    """Runs one eval question through the system and returns what it produced."""

    def run(
        self,
        question: str,
        target_date: date,
        clarification_answers: Sequence[ClarificationAnswer] = (),
    ) -> CaseRun:
        """Answer ``question`` at ``target_date``, resuming with a pinned reply if asked."""
        ...


@dataclass(frozen=True)
class Mode1CaseRunner:
    """A :class:`CaseRunner` backed by a real Mode 1 consultation per case.

    Each case runs on its own checkpoint thread so no run resumes another's paused
    state. The per-case ``target_date`` fixes the point-in-time clock the corpus is
    resolved at, and a pinned reply on a date branch may move it: what the answer
    ends up declaring, not what the case started at, is the date it is verified
    against.
    """

    deps: Mode1Deps
    checkpointer: BaseCheckpointSaver
    vertical: str

    def run(
        self,
        question: str,
        target_date: date,
        clarification_answers: Sequence[ClarificationAnswer] = (),
    ) -> CaseRun:
        """Start a fresh Mode 1 run for ``question`` at ``target_date``.

        A run that pauses is continued on the same thread with the reply the case
        pins for the branch asked; without such a reply the pause is returned as
        the case's outcome.
        """
        run = Mode1Run(
            thread_id=f"eval-{uuid4()}",
            vertical=self.vertical,
            today=target_date,
            deps=self.deps,
            checkpointer=self.checkpointer,
        )
        response = run.answer(question)
        if response.outcome is not Outcome.CLARIFICATION:
            return CaseRun(response=response, disambiguation=Disambiguation.DIRECT)
        return self._resume(run, response, clarification_answers)

    def _resume(
        self,
        run: Mode1Run,
        paused: AskResponse,
        clarification_answers: Sequence[ClarificationAnswer],
    ) -> CaseRun:
        """Continue a paused run with the case's pinned reply, or leave it paused."""
        branch_id = paused.clarification.branch_id if paused.clarification else None
        pinned = answer_for_branch(clarification_answers, branch_id) if branch_id else None
        if pinned is None:
            logger.info("case paused on branch '%s' with no pinned answer", branch_id)
            return CaseRun(response=paused, disambiguation=Disambiguation.UNANSWERED)
        return CaseRun(response=run.resume(pinned), disambiguation=Disambiguation.RESUMED)


def run_suite(
    suite: str,
    cases: Sequence[EvalCase],
    runner: CaseRunner,
    corpus: CorpusReader,
    default_date: date,
    fingerprint: ConfigFingerprint,
    created_at: datetime,
    judge: Judge | None = None,
    calibration: JudgeCalibration | None = None,
    repetitions: int = 1,
) -> RunArtifact:
    """Run every case ``repetitions`` times and build the run artifact.

    Each case is answered at its own ``target_date`` when it pins one, else at
    ``default_date``; the guardrail re-verifies that case's citations at the date
    the answer itself declares, so a point-in-time case is measured against the law
    as it stood then even when a disambiguation reply moved that date away from the
    one the case was launched at. A case that stops on the disambiguation gate is
    resumed with the reply it pins, and the guardrail and the judge then see that
    resumed answer rather than the pause. When a ``judge`` is given, every answered
    case carrying key points is graded against them, and ``calibration`` is the
    reviewer agreement those scores are published with -- but only when it graded
    the judge model this run's fingerprint pins. A record for any other judge is
    dropped, so swapping the judge leaves the run honestly uncalibrated instead of
    quoting the old grader's number.

    Every repetition runs the same cases under the same fingerprint, so what moves
    between them is the model's own variance and nothing else; the artifact carries
    each metric's band across them, and its ``metrics`` and ``cases`` are the first
    repetition's. ``passed`` is false if any repetition's citation broke the
    literality invariant, and the hard failures of all of them are kept.
    Raises :class:`ValueError` when asked for fewer than one repetition.
    """
    if repetitions < 1:
        raise ValueError(f"a run needs at least one repetition, got {repetitions}")
    passes = [_run_pass(cases, runner, corpus, default_date, judge) for _ in range(repetitions)]
    per_repetition = [aggregate(results) for results, _ in passes]
    hard_failures = [failure for _, failures in passes for failure in failures]
    return build_artifact(
        suite,
        created_at,
        fingerprint,
        passes[0][0],
        per_repetition[0],
        hard_failures,
        calibration_for_judge(calibration, _judge_model(fingerprint)),
        summarize_repetitions([scalar_metrics(metrics) for metrics in per_repetition]),
    )


def _judge_model(fingerprint: ConfigFingerprint) -> str | None:
    """The judge model the run pinned, or ``None`` when it pinned none."""
    pinned = fingerprint.models.get(JUDGE_TASK)
    return pinned.model if pinned is not None else None


def _run_pass(
    cases: Sequence[EvalCase],
    runner: CaseRunner,
    corpus: CorpusReader,
    default_date: date,
    judge: Judge | None,
) -> tuple[list[CaseResult], list[GuardrailViolation]]:
    """Drive every case once, returning its results and the guardrail's failures."""
    results: list[CaseResult] = []
    hard_failures: list[GuardrailViolation] = []
    for case in cases:
        case_date = case.target_date or default_date
        case_run = runner.run(case.question, case_date, case.clarification_answers)
        response = case_run.response
        violations = check_case_citations(case.id, response, corpus, case_date)
        hard_failures.extend(violations)
        verdict = judge.judge(case, response, case_date) if judge is not None else None
        results.append(
            build_case_result(case, response, violations, verdict, case_run.disambiguation)
        )
    return results, hard_failures
