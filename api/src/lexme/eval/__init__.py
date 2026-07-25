"""Evaluation boundary: the Mode 1 regression harness and its artifacts.

The harness is development infrastructure, not a showcase: it runs a case set
end-to-end against the real system, stamps each run with the configuration
fingerprint it was produced under, applies the citation guardrail as a hard
invariant and reports quality metrics against a baseline. It uses a real model and
is never run in CI as a test; the test suite only checks that the harness produces
its artifacts and that its guardrail fires -- the one intersection between suite
and harness.
"""

from lexme.eval.artifact import RunArtifact, build_artifact, read_artifact
from lexme.eval.cases import CasesError, EvalCase, load_cases
from lexme.eval.compare import Comparison, MetricDelta, compare
from lexme.eval.fingerprint import (
    ConfigFingerprint,
    TaskFingerprint,
    build_fingerprint,
    compute_dataset_digest,
    compute_prompts_digest,
)
from lexme.eval.guardrail import GuardrailViolation, check_case_citations
from lexme.eval.metrics import CaseResult, SuiteMetrics, aggregate, build_case_result
from lexme.eval.runner import CaseRunner, Mode1CaseRunner, run_suite

__all__ = [
    "CaseResult",
    "CaseRunner",
    "CasesError",
    "Comparison",
    "ConfigFingerprint",
    "EvalCase",
    "GuardrailViolation",
    "MetricDelta",
    "Mode1CaseRunner",
    "RunArtifact",
    "SuiteMetrics",
    "TaskFingerprint",
    "aggregate",
    "build_artifact",
    "build_case_result",
    "build_fingerprint",
    "check_case_citations",
    "compare",
    "compute_dataset_digest",
    "compute_prompts_digest",
    "load_cases",
    "read_artifact",
    "run_suite",
]
