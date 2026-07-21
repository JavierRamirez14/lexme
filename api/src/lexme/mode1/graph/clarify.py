"""The disambiguation node: at most one question, and no silent guesses.

Stopping to ask costs a round trip and the user's patience, so the graph only
pauses for a branch the vertical declares as regime-changing, only on a question
about the user's own case, and only once per run. The pause is a LangGraph
``interrupt``: the run is checkpointed mid-graph and resumes on the same thread
when the answer arrives, which is what lets the cycle survive between HTTP
requests.

The node runs on every in-scope question, because its second job has nothing to
do with asking: every branch it leaves open -- the ones it chose not to ask
about, and the one it asked about but got no usable answer for -- becomes an
assumption stated out loud in the answer. An open branch is never dropped
silently.
"""

from dataclasses import dataclass
from datetime import date

from langgraph.types import interrupt

from lexme.mode1.branches import CriticalBranch
from lexme.mode1.dates import DatePrecision, parse_target_date
from lexme.mode1.graph.deps import Mode1Deps
from lexme.mode1.graph.state import Mode1State
from lexme.mode1.graph.steps import Step
from lexme.mode1.models import Clarification, QueryType


@dataclass(frozen=True)
class _Resolution:
    """A branch the user actually resolved: the fact stated, and any new anchor."""

    fact: str
    target_date: date | None = None
    precision: DatePrecision | None = None


def branch_to_ask(state: Mode1State, branches: tuple[CriticalBranch, ...]) -> CriticalBranch | None:
    """The one branch worth pausing for, or ``None`` to go straight to retrieval.

    Returns nothing once a question has already been asked (the budget is a single
    round) or when the question is purely informational, which does not depend on
    the user's own case. Otherwise the first branch the planner left unresolved,
    in the vertical's declared priority order.
    """
    if state.clarification_asked or state.query_type is QueryType.INFORMATIONAL:
        return None
    pending = set(state.unresolved_branch_ids)
    return next((branch for branch in branches if branch.id in pending), None)


def clarify_node(state: Mode1State, *, deps: Mode1Deps) -> dict:
    """Ask at most one question, then state an assumption for every branch left open.

    Suspends the graph on the interrupt when there is something worth asking; on
    resume, returns the state update carrying the resolved case fact (and, for the
    signing-date branch, the new target date) plus the assumptions standing in for
    whatever stayed open.
    """
    branch = branch_to_ask(state, deps.branches)
    if branch is None:
        return _situate(state, deps.branches, asked=None, resolution=None)

    answer = interrupt(
        Clarification(
            branch_id=branch.id,
            question=branch.question,
            answer_kind=branch.answer_kind,
        ).model_dump(mode="json")
    )
    return _situate(
        state, deps.branches, asked=branch, resolution=_resolve(state, branch, str(answer or ""))
    )


def _situate(
    state: Mode1State,
    branches: tuple[CriticalBranch, ...],
    *,
    asked: CriticalBranch | None,
    resolution: _Resolution | None,
) -> dict:
    """Fold the reply into the run and assume out loud for every branch still open."""
    resolved_ids = {asked.id} if asked and resolution else set()
    still_open = [
        branch
        for branch in branches
        if branch.id in set(state.unresolved_branch_ids) and branch.id not in resolved_ids
    ]

    update: dict = {
        "clarification_asked": True,
        "assumptions": [*state.assumptions, *(branch.assumption for branch in still_open)],
        "current_step": Step.CLARIFYING if asked else Step.SITUATING,
    }
    if resolution is None:
        return update

    update["case_facts"] = [*state.case_facts, resolution.fact]
    if resolution.target_date is not None and resolution.precision is not None:
        update["target_date"] = resolution.target_date
        update["target_date_precision"] = resolution.precision
    return update


def _resolve(state: Mode1State, branch: CriticalBranch, answer: str) -> _Resolution | None:
    """Read the user's reply, or ``None`` when it does not resolve the branch."""
    reply = answer.strip()
    if not reply:
        return None
    if not branch.sets_target_date:
        return _Resolution(fact=branch.state_fact(reply))

    signed_on = parse_target_date(reply)
    if signed_on is None or signed_on.value > state.today:
        return None
    return _Resolution(
        fact=branch.state_fact(signed_on.value.isoformat()),
        target_date=signed_on.value,
        precision=signed_on.precision,
    )
