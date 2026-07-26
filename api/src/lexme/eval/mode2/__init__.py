"""The Mode 2 contract-eval harness: the false-tranquility measurement.

A peer to the Mode 1 harness in the parent package. It runs the synthetic reference
contracts end-to-end through the real risk-map pipeline, scores segmentation as its
own layer, and publishes the two numbers the Mode 2 design lives or dies by -- the
recall of problematic clauses and the false-tranquility rate -- next to the
precision, abstention and absence diagnostics. The citation guardrail applies to the
findings' citations exactly as it does to a Mode 1 answer's.
"""

from lexme.eval.mode2.artifact import (
    Mode2RunArtifact,
    build_mode2_artifact,
    read_mode2_artifact,
)
from lexme.eval.mode2.cases import (
    AbsenceTruth,
    ClauseTruth,
    Mode2CasesError,
    Mode2EvalCase,
    load_mode2_cases,
)
from lexme.eval.mode2.guardrail import check_mode2_citations
from lexme.eval.mode2.metrics import (
    AbsenceRecall,
    ClausePrediction,
    Mode2CaseResult,
    Mode2SuiteMetrics,
    SegmentationLayer,
    aggregate_mode2,
    build_mode2_case_result,
    predicted_class,
)
from lexme.eval.mode2.runner import (
    Mode2CaseRunner,
    PipelineMode2CaseRunner,
    run_mode2_suite,
)
from lexme.eval.mode2.segmentation import (
    DEFAULT_IOU_THRESHOLD,
    LabeledSpan,
    SpanMatch,
    character_iou,
    match_spans,
)

__all__ = [
    "DEFAULT_IOU_THRESHOLD",
    "AbsenceRecall",
    "AbsenceTruth",
    "ClausePrediction",
    "ClauseTruth",
    "LabeledSpan",
    "Mode2CaseResult",
    "Mode2CaseRunner",
    "Mode2CasesError",
    "Mode2EvalCase",
    "Mode2RunArtifact",
    "Mode2SuiteMetrics",
    "PipelineMode2CaseRunner",
    "SegmentationLayer",
    "SpanMatch",
    "aggregate_mode2",
    "build_mode2_artifact",
    "build_mode2_case_result",
    "character_iou",
    "check_mode2_citations",
    "load_mode2_cases",
    "match_spans",
    "predicted_class",
    "read_mode2_artifact",
    "run_mode2_suite",
]
