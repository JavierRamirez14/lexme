"""The deterministic gate: code, not the model, picks the terminal outcome.

The gate reads the critique verdicts and the surviving citations and applies a
fixed rule -- the model never judges whether there is enough ground to answer.
A critical sub-query left ungrounded forces an abstention even if peripheral
citations held; only a peripheral gap downgrades a full answer to a partial one.
"""

from lexme.mode1.graph.state import Mode1State
from lexme.mode1.graph.steps import Step
from lexme.mode1.models import AbstentionReason, Outcome


def gate_node(state: Mode1State) -> dict:
    """Decide the terminal outcome from the evidence, verdicts and verified citations.

    Abstains when nothing was retrieved, when no citation survived verification, or
    when a critical sub-query stayed ungrounded; downgrades to a partial answer on
    a peripheral gap; otherwise answers.
    """
    if state.total_evidence == 0:
        return _abstain(AbstentionReason.NO_EVIDENCE)
    if not state.verified:
        return _abstain(AbstentionReason.NO_VERIFIABLE_CITATION)
    if _has_ungrounded_critical(state):
        return _abstain(AbstentionReason.INSUFFICIENT_CORE)
    if _has_ungrounded_peripheral(state):
        return {"outcome": Outcome.PARTIAL_ANSWER, "current_step": Step.DECIDING}
    return {"outcome": Outcome.ANSWER, "current_step": Step.DECIDING}


def _has_ungrounded_critical(state: Mode1State) -> bool:
    """Whether any critical sub-query was left insufficient by the critique."""
    return any(sub.subquery.is_critical and not sub.is_sufficient for sub in state.subqueries)


def _has_ungrounded_peripheral(state: Mode1State) -> bool:
    """Whether any peripheral sub-query was left insufficient by the critique."""
    return any(not sub.subquery.is_critical and not sub.is_sufficient for sub in state.subqueries)


def _abstain(reason: AbstentionReason) -> dict:
    """Build the abstention update for a given reason."""
    return {
        "outcome": Outcome.ABSTENTION,
        "abstention_reason": reason,
        "current_step": Step.DECIDING,
    }
