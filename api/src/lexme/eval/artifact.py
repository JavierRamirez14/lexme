"""The run artifact: the versionable record of one eval run.

Every run emits one JSON artifact carrying its configuration fingerprint, its
quality metrics, its per-case results and the guardrail's hard failures. Without
the fingerprint a run's numbers are not comparable or reproducible; with it, the
run history becomes a time series a regression can be read off. ``passed`` is the
guardrail's verdict alone: a run with any hard failure has not passed, whatever
its metrics say.
"""

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from lexme.eval.fingerprint import ConfigFingerprint
from lexme.eval.guardrail import GuardrailViolation
from lexme.eval.metrics import CaseResult, SuiteMetrics


class RunArtifact(BaseModel):
    """One eval run's full, self-describing result.

    ``passed`` is ``False`` whenever ``hard_failures`` is non-empty, independent of
    the metrics; the metrics are the soft numbers, the guardrail is the invariant.
    """

    suite: str
    created_at: datetime
    fingerprint: ConfigFingerprint
    metrics: SuiteMetrics
    cases: list[CaseResult]
    hard_failures: list[GuardrailViolation]
    passed: bool

    def write(self, path: Path) -> None:
        """Serialize the artifact to ``path`` as indented JSON, creating parents."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def build_artifact(
    suite: str,
    created_at: datetime,
    fingerprint: ConfigFingerprint,
    results: list[CaseResult],
    metrics: SuiteMetrics,
    hard_failures: list[GuardrailViolation],
) -> RunArtifact:
    """Assemble a :class:`RunArtifact`, deriving ``passed`` from the guardrail alone."""
    return RunArtifact(
        suite=suite,
        created_at=created_at,
        fingerprint=fingerprint,
        metrics=metrics,
        cases=results,
        hard_failures=hard_failures,
        passed=not hard_failures,
    )


def read_artifact(path: Path) -> RunArtifact:
    """Read a run artifact back from its JSON file at ``path``."""
    return RunArtifact.model_validate(json.loads(path.read_text(encoding="utf-8")))
