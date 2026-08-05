"""The Mode 2 run artifact: the versionable record of one contract-eval run.

Like the Mode 1 artifact, it stamps the run with its configuration fingerprint so a
number is comparable and reproducible, carries the suite metrics and the per-case
results, and lists the guardrail's hard failures. ``passed`` is the guardrail's
verdict alone: a run with any non-literal citation has not passed, whatever the
false-tranquility rate says.
"""

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from lexme.eval.fingerprint import ConfigFingerprint
from lexme.eval.guardrail import GuardrailViolation
from lexme.eval.mode2.metrics import Mode2CaseResult, Mode2SuiteMetrics
from lexme.eval.repetition import RepetitionSummary


class Mode2RunArtifact(BaseModel):
    """One Mode 2 eval run's full, self-describing result.

    ``passed`` is ``False`` whenever ``hard_failures`` is non-empty, independent of
    the metrics; the metrics are the soft numbers, the guardrail is the invariant.
    ``repetitions`` holds the band each metric moved in across the run's repetitions
    and is the number to publish; ``metrics`` and ``cases`` are the first repetition
    alone. It is ``None`` only in an artifact written before the harness repeated
    anything.
    """

    suite: str
    created_at: datetime
    fingerprint: ConfigFingerprint
    metrics: Mode2SuiteMetrics
    cases: list[Mode2CaseResult]
    hard_failures: list[GuardrailViolation]
    passed: bool
    repetitions: RepetitionSummary | None = None

    def write(self, path: Path) -> None:
        """Serialize the artifact to ``path`` as indented JSON, creating parents."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def build_mode2_artifact(
    suite: str,
    created_at: datetime,
    fingerprint: ConfigFingerprint,
    results: list[Mode2CaseResult],
    metrics: Mode2SuiteMetrics,
    hard_failures: list[GuardrailViolation],
    repetitions: RepetitionSummary | None = None,
) -> Mode2RunArtifact:
    """Assemble a :class:`Mode2RunArtifact`, deriving ``passed`` from the guardrail alone."""
    return Mode2RunArtifact(
        suite=suite,
        created_at=created_at,
        fingerprint=fingerprint,
        metrics=metrics,
        cases=results,
        hard_failures=hard_failures,
        passed=not hard_failures,
        repetitions=repetitions,
    )


def read_mode2_artifact(path: Path) -> Mode2RunArtifact:
    """Read a Mode 2 run artifact back from its JSON file at ``path``."""
    return Mode2RunArtifact.model_validate(json.loads(path.read_text(encoding="utf-8")))
