"""Compare a run against a baseline, reporting the delta on each metric.

A run's numbers mean nothing in isolation; a regression is a move away from a
known-good baseline. This reports the per-metric delta and, first of all, whether
the two runs even share a configuration fingerprint -- because a metric delta
across different configurations is not a regression, it is a different experiment,
and reading it as a regression is the mistake the fingerprint exists to prevent.
"""

from pydantic import BaseModel

from lexme.eval.artifact import RunArtifact

MEAN_RECALL = "mean_recall"
MEAN_FIRST_PASS_RECALL = "mean_first_pass_recall"
MEAN_RECALL_DELTA = "mean_recall_delta"
OUTCOME_MATCH_RATE = "outcome_match_rate"
ABSTENTION_RATE = "abstention_rate"
EXPECTED_ABSTENTION_RECALL = "expected_abstention_recall"
MEAN_AGENTIC_DELTA = "mean_agentic_delta"
JUDGE_COMPLETENESS = "judge.completeness"
JUDGE_UNSUPPORTED_CLAIM_RATE = "judge.unsupported_claim_rate"
JUDGE_MEAN_CLARITY = "judge.mean_clarity"
CASES = "cases"


class MetricDelta(BaseModel):
    """One metric's baseline value, run value and their difference.

    ``base`` or ``run`` is ``None`` when a side did not measure the metric (an
    empty recall, say); ``delta`` is ``None`` unless both sides are present.
    """

    metric: str
    base: float | None
    run: float | None
    delta: float | None


class Comparison(BaseModel):
    """The result of comparing a run against a baseline.

    ``fingerprint_changed`` is the first thing to read: when true, the deltas
    describe two different configurations and are not regressions.
    """

    fingerprint_changed: bool
    base_fingerprint: str
    run_fingerprint: str
    deltas: list[MetricDelta]


def compare(base: RunArtifact, run: RunArtifact) -> Comparison:
    """Compare ``run`` against ``base``, returning the per-metric deltas.

    Covers the scalar quality metrics and every citation verdict count. The
    fingerprint comparison frames the rest: identical fingerprints make the deltas
    a regression signal, different ones make them an experiment comparison.
    """
    deltas = [
        _scalar_delta(CASES, base.metrics.cases, run.metrics.cases),
        _scalar_delta(MEAN_RECALL, base.metrics.mean_recall, run.metrics.mean_recall),
        _scalar_delta(
            MEAN_FIRST_PASS_RECALL,
            base.metrics.mean_first_pass_recall,
            run.metrics.mean_first_pass_recall,
        ),
        _scalar_delta(
            MEAN_RECALL_DELTA, base.metrics.mean_recall_delta, run.metrics.mean_recall_delta
        ),
        _scalar_delta(
            OUTCOME_MATCH_RATE, base.metrics.outcome_match_rate, run.metrics.outcome_match_rate
        ),
        _scalar_delta(ABSTENTION_RATE, base.metrics.abstention_rate, run.metrics.abstention_rate),
        _scalar_delta(
            EXPECTED_ABSTENTION_RECALL,
            base.metrics.expected_abstention_recall,
            run.metrics.expected_abstention_recall,
        ),
        _scalar_delta(
            MEAN_AGENTIC_DELTA, base.metrics.mean_agentic_delta, run.metrics.mean_agentic_delta
        ),
    ]
    deltas.extend(_judge_deltas(base, run))
    deltas.extend(_verdict_deltas(base, run))
    return Comparison(
        fingerprint_changed=base.fingerprint.fingerprint != run.fingerprint.fingerprint,
        base_fingerprint=base.fingerprint.fingerprint,
        run_fingerprint=run.fingerprint.fingerprint,
        deltas=deltas,
    )


def _judge_deltas(base: RunArtifact, run: RunArtifact) -> list[MetricDelta]:
    """Deltas on the judged end-to-end numbers, treating an unjudged run as absent."""
    base_judge = base.metrics.judge
    run_judge = run.metrics.judge
    return [
        _scalar_delta(
            JUDGE_COMPLETENESS,
            base_judge.mean_completeness if base_judge else None,
            run_judge.mean_completeness if run_judge else None,
        ),
        _scalar_delta(
            JUDGE_UNSUPPORTED_CLAIM_RATE,
            base_judge.unsupported_claim_rate if base_judge else None,
            run_judge.unsupported_claim_rate if run_judge else None,
        ),
        _scalar_delta(
            JUDGE_MEAN_CLARITY,
            base_judge.mean_clarity if base_judge else None,
            run_judge.mean_clarity if run_judge else None,
        ),
    ]


def _verdict_deltas(base: RunArtifact, run: RunArtifact) -> list[MetricDelta]:
    """A delta for each citation verdict count seen in either run, in verdict order."""
    verdicts = list(base.metrics.citation_verdicts) + [
        verdict
        for verdict in run.metrics.citation_verdicts
        if verdict not in base.metrics.citation_verdicts
    ]
    return [
        _scalar_delta(
            f"citation_verdicts.{verdict}",
            base.metrics.citation_verdicts.get(verdict),
            run.metrics.citation_verdicts.get(verdict),
        )
        for verdict in verdicts
    ]


def _scalar_delta(metric: str, base: float | None, run: float | None) -> MetricDelta:
    """Build a delta, leaving it ``None`` unless both sides measured the metric."""
    delta = run - base if base is not None and run is not None else None
    return MetricDelta(metric=metric, base=base, run=run, delta=delta)
