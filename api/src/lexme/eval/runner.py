"""The harness that runs a case set end-to-end and assembles its artifact.

The harness drives each case through a :class:`CaseRunner` -- the seam behind
which the real Mode 1 consultation lives -- applies the citation guardrail to the
response, measures the quality metrics and folds everything into one artifact. The
runner is a seam so the harness itself is tested with the real system behind a
fake LLM, and never needs a live model of its own.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, runtime_checkable
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver

from lexme.eval.artifact import RunArtifact, build_artifact
from lexme.eval.calibration import JudgeCalibration
from lexme.eval.cases import EvalCase
from lexme.eval.fingerprint import ConfigFingerprint
from lexme.eval.guardrail import GuardrailViolation, check_case_citations
from lexme.eval.judge import Judge
from lexme.eval.metrics import aggregate, build_case_result
from lexme.mode1 import AskResponse, Mode1Deps, Mode1Run
from lexme.verification import CorpusReader


@runtime_checkable
class CaseRunner(Protocol):
    """Runs one eval question through the system and returns its response."""

    def run(self, question: str, target_date: date) -> AskResponse:
        """Answer ``question`` situated at ``target_date`` and return the response."""
        ...


@dataclass(frozen=True)
class Mode1CaseRunner:
    """A :class:`CaseRunner` backed by a real Mode 1 consultation per case.

    Each case runs on its own checkpoint thread so no run resumes another's paused
    state. The per-case ``target_date`` fixes the point-in-time clock the corpus is
    resolved at, the same date the guardrail re-verifies against.
    """

    deps: Mode1Deps
    checkpointer: BaseCheckpointSaver
    vertical: str

    def run(self, question: str, target_date: date) -> AskResponse:
        """Start a fresh Mode 1 run for ``question`` at ``target_date``."""
        run = Mode1Run(
            thread_id=f"eval-{uuid4()}",
            vertical=self.vertical,
            today=target_date,
            deps=self.deps,
            checkpointer=self.checkpointer,
        )
        return run.answer(question)


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
) -> RunArtifact:
    """Run every case, apply the guardrail and metrics, and build the run artifact.

    Each case is answered at its own ``target_date`` when it pins one, else at
    ``default_date``; the guardrail re-verifies that case's citations at the same
    date, so a point-in-time case is measured against the law as it stood then.
    When a ``judge`` is given, every answered case carrying key points is graded
    against them, and ``calibration`` is the human-judge agreement those scores are
    published with. The returned artifact's ``passed`` is false if any case's
    citation broke the literality invariant.
    """
    results = []
    hard_failures: list[GuardrailViolation] = []
    for case in cases:
        case_date = case.target_date or default_date
        response = runner.run(case.question, case_date)
        violations = check_case_citations(case.id, response, corpus, case_date)
        hard_failures.extend(violations)
        verdict = judge.judge(case, response, case_date) if judge is not None else None
        results.append(build_case_result(case, response, violations, verdict))
    metrics = aggregate(results)
    return build_artifact(
        suite, created_at, fingerprint, results, metrics, hard_failures, calibration
    )
