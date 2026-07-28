"""Tests for run comparison: per-metric deltas, framed by whether config changed."""

import json
from datetime import UTC, datetime
from pathlib import Path

from lexme.eval.artifact import RunArtifact, build_artifact, read_artifact
from lexme.eval.compare import compare, compare_mode2
from lexme.eval.fingerprint import ConfigFingerprint
from lexme.eval.metrics import SuiteMetrics
from lexme.eval.mode2.artifact import Mode2RunArtifact, build_mode2_artifact, read_mode2_artifact
from lexme.eval.mode2.metrics import Mode2SuiteMetrics

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


def _artifact(
    fingerprint: str,
    mean_recall: float,
    agentic_delta: float,
    disambiguation_rate: float = 0.0,
) -> RunArtifact:
    """A metrics-only artifact carrying the headline numbers under a fingerprint."""
    metrics = SuiteMetrics(
        cases=2,
        outcomes={"respuesta": 2},
        disambiguation={"directo": 2, "reanudado": 0, "sin_respuesta": 0},
        disambiguation_rate=disambiguation_rate,
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


def test_an_artifact_written_before_the_disambiguation_metric_still_reads_back(
    tmp_path: Path,
) -> None:
    artifact = _artifact("fp", mean_recall=0.8, agentic_delta=1.0)
    payload = artifact.model_dump(mode="json")
    del payload["metrics"]["disambiguation"]
    del payload["metrics"]["disambiguation_rate"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    older = read_artifact(path)

    assert older.metrics.disambiguation_rate is None
    assert _delta(compare(older, artifact), "disambiguation_rate").delta is None


def test_the_disambiguation_rate_is_compared() -> None:
    base = _artifact("fp", mean_recall=0.80, agentic_delta=1.0, disambiguation_rate=0.6)
    run = _artifact("fp", mean_recall=0.80, agentic_delta=1.0, disambiguation_rate=0.2)

    comparison = compare(base, run)

    assert _delta(comparison, "disambiguation_rate").delta == 0.2 - 0.6


def test_verdict_counts_are_compared_per_verdict() -> None:
    base = _artifact("fp", mean_recall=0.80, agentic_delta=1.0)
    run = _artifact("fp", mean_recall=0.80, agentic_delta=1.0)

    comparison = compare(base, run)

    assert _delta(comparison, "citation_verdicts.verificada_directa").delta == 0


def _mode2_metrics(**overrides: object) -> Mode2SuiteMetrics:
    """A Mode 2 suite metrics object with every field filled, for a test to override."""
    base = dict(
        cases=1,
        outcomes={"analizado": 1},
        outcome_match_rate=1.0,
        reference_clauses=2,
        delimited_clauses=2,
        segmentation_delimited_rate=1.0,
        matched_clauses=2,
        confusion={},
        problematic_total=1,
        problematic_detected=1,
        recall_problematic=1.0,
        false_tranquility_events=0,
        false_tranquility_rate=0.0,
        flagged_problematic=1,
        flagged_true_positive=1,
        precision_problematic=1.0,
        abstention_clauses=0,
        abstention_rate=0.0,
        problematic_total_e2e=1,
        problematic_detected_e2e=1,
        recall_problematic_e2e=1.0,
        false_tranquility_events_e2e=0,
        false_tranquility_rate_e2e=0.0,
        not_reported_problematic=0,
        absence_expected=0,
        absence_detected=0,
        absence_recall=None,
        absence_predicted=0,
        absence_precision=None,
    )
    base.update(overrides)
    return Mode2SuiteMetrics(**base)


def _mode2_artifact(fingerprint: str, **overrides: object) -> Mode2RunArtifact:
    """A Mode 2 metrics-only artifact carrying its headline numbers under a fingerprint."""
    return build_mode2_artifact(
        "modo2", NOW, _fingerprint(fingerprint), [], _mode2_metrics(**overrides), []
    )


def test_mode2_identical_fingerprints_make_the_deltas_a_regression_signal() -> None:
    base = _mode2_artifact("fp", recall_problematic_e2e=0.29)
    run = _mode2_artifact("fp", recall_problematic_e2e=0.5)

    comparison = compare_mode2(base, run)

    assert comparison.fingerprint_changed is False
    assert _delta(comparison, "recall_problematic_e2e").delta == 0.5 - 0.29


def test_mode2_a_changed_fingerprint_is_flagged() -> None:
    base = _mode2_artifact("fp-a")
    run = _mode2_artifact("fp-b")

    comparison = compare_mode2(base, run)

    assert comparison.fingerprint_changed is True
    assert comparison.base_fingerprint == "fp-a"
    assert comparison.run_fingerprint == "fp-b"


def test_mode2_publishes_both_the_conditioned_and_the_end_to_end_recall() -> None:
    base = _mode2_artifact("fp", recall_problematic=1.0, recall_problematic_e2e=0.29)
    run = _mode2_artifact("fp", recall_problematic=1.0, recall_problematic_e2e=0.29)

    comparison = compare_mode2(base, run)

    assert _delta(comparison, "recall_problematic").base == 1.0
    assert _delta(comparison, "recall_problematic_e2e").base == 0.29


def test_a_mode2_artifact_written_before_the_e2e_schema_change_still_reads_back(
    tmp_path: Path,
) -> None:
    artifact = _mode2_artifact("fp")
    payload = artifact.model_dump(mode="json")
    del payload["metrics"]["recall_problematic_e2e"]
    del payload["metrics"]["problematic_total_e2e"]
    del payload["metrics"]["problematic_detected_e2e"]
    del payload["metrics"]["false_tranquility_events_e2e"]
    del payload["metrics"]["false_tranquility_rate_e2e"]
    del payload["metrics"]["not_reported_problematic"]
    del payload["metrics"]["outcome_match_rate"]
    for case in payload["cases"]:
        del case["outcome_as_expected"]
    path = tmp_path / "old_modo2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    older = read_mode2_artifact(path)

    assert older.metrics.recall_problematic_e2e is None
    assert older.metrics.outcome_match_rate is None
    delta = _delta(compare_mode2(older, artifact), "recall_problematic_e2e")
    assert delta.base is None
    assert delta.run == 1.0
    assert delta.delta is None
