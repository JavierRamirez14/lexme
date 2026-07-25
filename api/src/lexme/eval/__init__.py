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
from lexme.eval.calibration import (
    CalibrationError,
    CalibrationItem,
    JudgeCalibration,
    build_calibration,
    compute_agreement,
    load_calibration,
)
from lexme.eval.cases import CasesError, EvalCase, KeyPoint, load_cases
from lexme.eval.compare import Comparison, MetricDelta, compare
from lexme.eval.fingerprint import (
    ConfigFingerprint,
    TaskFingerprint,
    build_fingerprint,
    compute_dataset_digest,
    compute_prompts_digest,
)
from lexme.eval.guardrail import GuardrailViolation, check_case_citations
from lexme.eval.judge import (
    ClaimAssessment,
    Judge,
    JudgeConfigError,
    JudgeMetrics,
    JudgeVerdict,
    KeyPointCoverage,
    LlmJudge,
    assert_judge_distinct_from_generator,
    build_judge_metrics,
)
from lexme.eval.metrics import (
    CaseResult,
    JudgeAggregate,
    LayerRecall,
    RetrievalRecall,
    SubQueryRecall,
    SuiteMetrics,
    aggregate,
    build_case_result,
)
from lexme.eval.runner import CaseRunner, Mode1CaseRunner, run_suite

__all__ = [
    "CalibrationError",
    "CalibrationItem",
    "CaseResult",
    "CaseRunner",
    "CasesError",
    "ClaimAssessment",
    "Comparison",
    "ConfigFingerprint",
    "EvalCase",
    "GuardrailViolation",
    "Judge",
    "JudgeAggregate",
    "JudgeCalibration",
    "JudgeConfigError",
    "JudgeMetrics",
    "JudgeVerdict",
    "KeyPoint",
    "KeyPointCoverage",
    "LayerRecall",
    "LlmJudge",
    "MetricDelta",
    "Mode1CaseRunner",
    "RetrievalRecall",
    "RunArtifact",
    "SubQueryRecall",
    "SuiteMetrics",
    "TaskFingerprint",
    "aggregate",
    "assert_judge_distinct_from_generator",
    "build_artifact",
    "build_calibration",
    "build_case_result",
    "build_fingerprint",
    "build_judge_metrics",
    "check_case_citations",
    "compare",
    "compute_agreement",
    "compute_dataset_digest",
    "compute_prompts_digest",
    "load_calibration",
    "load_cases",
    "read_artifact",
    "run_suite",
]
