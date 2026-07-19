"""The Mode 1 agentic graph: a LangGraph state machine over typed collaborators.

The topology is linear with one self-critique loop: ``route -> plan -> retrieve ->
critique -(loop)- retrieve -> synthesize -> gate``. An out-of-scope router verdict
ends the run before planning. LangGraph earns its place here as the issue 07
disambiguation adds an ``interrupt()`` node to this same graph; for now it gives a
typed state channel the SSE stream and the eval harness read straight from.
"""

from lexme.mode1.graph.builder import build_mode1_graph
from lexme.mode1.graph.deps import Mode1Deps
from lexme.mode1.graph.state import MAX_PASSES, MAX_SUBQUERIES, Mode1State

__all__ = [
    "MAX_PASSES",
    "MAX_SUBQUERIES",
    "Mode1Deps",
    "Mode1State",
    "build_mode1_graph",
]
