"""The in-force node: code-derived warnings about the distance from today's law.

Runs after the gate, over the citations that actually survived, so a notice never
describes a citation the answer does not make. It reads the block's amendment
history from the corpus and applies the rules in :mod:`lexme.mode1.notices`; the
model is not involved at any point.
"""

from lexme.mode1.graph.deps import Mode1Deps
from lexme.mode1.graph.state import Mode1State
from lexme.mode1.graph.steps import Step
from lexme.mode1.models import Outcome
from lexme.mode1.notices import build_notices


def notices_node(state: Mode1State, *, deps: Mode1Deps) -> dict:
    """Derive the run's in-force notices from its target date and its citations."""
    if state.outcome not in (Outcome.ANSWER, Outcome.PARTIAL_ANSWER):
        return {"notices": [], "current_step": Step.CHECKING_CURRENCY}
    return {
        "notices": build_notices(
            state.cited_blocks,
            target_date=state.anchor,
            today=state.today,
            history=deps.history,
        ),
        "current_step": Step.CHECKING_CURRENCY,
    }
