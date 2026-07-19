"""The retrieval node: hybrid retrieval per sub-query, re-running only the gaps.

On the first pass every sub-query is retrieved; on a loop re-entry only the
sub-queries the critique left insufficient are retrieved again, with the fresh
query text the critique wrote. Each sub-query keeps its own dense, lexical and
fused rankings, so the hybrid retrieval stays measurable per sub-query.
"""

from lexme.mode1.graph.deps import Mode1Deps
from lexme.mode1.graph.state import Mode1State, SubQueryState
from lexme.mode1.graph.steps import Step
from lexme.retrieval import hybrid_retrieve


def retrieve_node(state: Mode1State, *, deps: Mode1Deps) -> dict:
    """Retrieve evidence for every sub-query still lacking it, and count the pass.

    Sufficient sub-queries keep the evidence they already hold; the rest are
    (re-)retrieved with their current query text. Increments the pass counter.
    """
    retrieved = [_retrieve_one(sub, state, deps) for sub in state.subqueries]
    return {
        "subqueries": retrieved,
        "pass_number": state.pass_number + 1,
        "current_step": Step.RETRIEVING,
    }


def _retrieve_one(sub: SubQueryState, state: Mode1State, deps: Mode1Deps) -> SubQueryState:
    """Run hybrid retrieval for one sub-query, unless it is already sufficient."""
    if sub.is_sufficient:
        return sub
    result = hybrid_retrieve(
        deps.connection,
        deps.embedder,
        query=sub.query_text,
        vertical=state.vertical,
        target_date=state.target_date,
    )
    return sub.model_copy(
        update={
            "evidence": result.evidence,
            "dense_ranking": result.dense_ranking,
            "lexical_ranking": result.lexical_ranking,
            "fused_ranking": result.fused_ranking,
        }
    )
