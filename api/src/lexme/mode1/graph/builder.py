"""Assembly of the Mode 1 LangGraph: nodes, the loop, and the two short-circuits.

The topology is linear with a single self-critique loop: an out-of-scope router
verdict ends the graph before planning, the clarify node either pauses for one
disambiguating question or falls straight through, and the critique either loops
back to retrieval (reformulated gaps, within the pass budget) or falls through to
synthesis. Each node is bound to the run's collaborators here, so the LangGraph
node signature stays ``(state) -> update``. The graph is compiled with a
checkpointer because a paused run has to survive between HTTP requests.
"""

from functools import partial

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from lexme.mode1.graph.clarify import clarify_node
from lexme.mode1.graph.critique import critique_node
from lexme.mode1.graph.deps import Mode1Deps
from lexme.mode1.graph.gate import gate_node
from lexme.mode1.graph.notices_node import notices_node
from lexme.mode1.graph.planner import planner_node
from lexme.mode1.graph.retrieval import retrieve_node
from lexme.mode1.graph.router import router_node
from lexme.mode1.graph.state import MAX_PASSES, Mode1State
from lexme.mode1.graph.synthesis_node import synthesize_node
from lexme.mode1.models import Outcome


def build_mode1_graph(deps: Mode1Deps, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Build and compile the Mode 1 graph with its nodes bound to ``deps``."""
    graph = StateGraph(Mode1State)
    graph.add_node("route", partial(router_node, deps=deps))
    graph.add_node("plan", partial(planner_node, deps=deps))
    graph.add_node("clarify", partial(clarify_node, deps=deps))
    graph.add_node("retrieve", partial(retrieve_node, deps=deps))
    graph.add_node("critique", partial(critique_node, deps=deps))
    graph.add_node("synthesize", partial(synthesize_node, deps=deps))
    graph.add_node("gate", gate_node)
    graph.add_node("notices", partial(notices_node, deps=deps))

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _after_route, {"plan": "plan", "end": END})
    graph.add_edge("plan", "clarify")
    graph.add_edge("clarify", "retrieve")
    graph.add_edge("retrieve", "critique")
    graph.add_conditional_edges(
        "critique", _after_critique, {"retrieve": "retrieve", "synthesize": "synthesize"}
    )
    graph.add_edge("synthesize", "gate")
    graph.add_edge("gate", "notices")
    graph.add_edge("notices", END)
    return graph.compile(checkpointer=checkpointer)


def _after_route(state: Mode1State) -> str:
    """End on an out-of-scope rejection; otherwise proceed to planning."""
    return "end" if state.outcome is Outcome.ROUTER_REJECTION else "plan"


def _after_critique(state: Mode1State) -> str:
    """Loop back to retrieval while gaps remain and the pass budget holds."""
    if state.all_sufficient or state.pass_number >= MAX_PASSES:
        return "synthesize"
    return "retrieve"
