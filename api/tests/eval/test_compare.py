"""Tests for run comparison: per-metric deltas, framed by whether config changed."""

from datetime import UTC, datetime

from lexme.eval.artifact import RunArtifact, build_artifact
from lexme.eval.compare import compare
from lexme.eval.fingerprint import ConfigFingerprint
from lexme.eval.metrics import SuiteMetrics

NOW = datetime(2026, 7, 25, tzinfo=UTC)


def _fingerprint(value: str) -> ConfigFingerprint:
    """A fingerprint whose combined hash is ``value``."""
    return ConfigFingerprint(
        models={},
        vertical="vivienda",
        prompts_hash="prompts",
        vertical_config_hash="cfg",
        corpus_hash="corpus",
        dataset_hash="dataset",
        fingerprint=value,
    )


def _artifact(fingerprint: str, mean_recall: float, agentic_delta: float) -> RunArtifact:
    """A metrics-only artifact carrying the headline numbers under a fingerprint."""
    metrics = SuiteMetrics(
        cases=2,
        outcomes={"respuesta": 2},
        mean_recall=mean_recall,
        mean_first_pass_recall=mean_recall - 0.1,
        mean_recall_delta=0.1,
        retrieval_layer_recall={"fused": mean_recall},
        outcome_match_rate=1.0,
        abstention_rate=0.0,
        expected_abstention_recall=None,
        mean_agentic_delta=agentic_delta,
        judge=None,
        citation_verdicts={"verificada_directa": 3, "descartada": 0},
    )
    return build_artifact("modo1", NOW, _fingerprint(fingerprint), [], metrics, [])


def _delta(comparison, metric: str):
    """The delta entry for ``metric``."""
    return next(entry for entry in comparison.deltas if entry.metric == metric)


def test_identical_fingerprints_make_the_deltas_a_regression_signal() -> None:
    base = _artifact("fp", mean_recall=0.80, agentic_delta=1.0)
    run = _artifact("fp", mean_recall=0.90, agentic_delta=1.5)

    comparison = compare(base, run)

    assert comparison.fingerprint_changed is False
    assert _delta(comparison, "mean_recall").delta == 0.90 - 0.80
    assert _delta(comparison, "mean_agentic_delta").delta == 0.5


def test_a_changed_fingerprint_is_flagged() -> None:
    base = _artifact("fp-a", mean_recall=0.80, agentic_delta=1.0)
    run = _artifact("fp-b", mean_recall=0.90, agentic_delta=1.0)

    comparison = compare(base, run)

    assert comparison.fingerprint_changed is True
    assert comparison.base_fingerprint == "fp-a"
    assert comparison.run_fingerprint == "fp-b"


def test_the_agentic_recall_pair_is_compared() -> None:
    base = _artifact("fp", mean_recall=0.80, agentic_delta=1.0)
    run = _artifact("fp", mean_recall=0.95, agentic_delta=1.0)

    comparison = compare(base, run)

    assert _delta(comparison, "mean_first_pass_recall").delta == 0.95 - 0.80
    assert _delta(comparison, "abstention_rate").delta == 0.0
    assert _delta(comparison, "mean_recall_delta").base == 0.1


def test_verdict_counts_are_compared_per_verdict() -> None:
    base = _artifact("fp", mean_recall=0.80, agentic_delta=1.0)
    run = _artifact("fp", mean_recall=0.80, agentic_delta=1.0)

    comparison = compare(base, run)

    assert _delta(comparison, "citation_verdicts.verificada_directa").delta == 0
