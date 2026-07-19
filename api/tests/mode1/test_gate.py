"""Unit tests for the deterministic gate: the terminal outcome is a code decision.

No LLM, no corpus -- the gate is pure logic over the critique verdicts and the
surviving citations, so these tests pin the rule directly: what makes an answer, a
partial answer, or each flavour of abstention.
"""

from datetime import date

from lexme.mode1.graph.gate import gate_node
from lexme.mode1.graph.state import Mode1State, SubQuery, SubQueryState
from lexme.mode1.models import (
    AbstentionReason,
    CitationVerdict,
    Outcome,
    SubQueryVerdict,
    VerifiedCitation,
)
from lexme.retrieval.models import RetrievedBlock
from lexme.verification import VerifiedAnchor

AS_OF = date(2020, 1, 1)


def _block() -> RetrievedBlock:
    """A minimal retrieved block, standing in for real evidence."""
    return RetrievedBlock(
        norm_id="BOE-A-1994-26003",
        block_id="a9",
        title="Artículo 9",
        text="La duración del arrendamiento será libremente pactada por las partes",
        effective_date=AS_OF,
    )


def _verified() -> VerifiedCitation:
    """A minimal verified citation, standing in for a survived citation."""
    anchor = VerifiedAnchor(
        eli="https://www.boe.es/eli/es/l/1994/11/24/29",
        consolidated_html_url="https://example.test/a9",
        block_id="a9",
        title="Artículo 9",
        effective_date=AS_OF,
    )
    return VerifiedCitation(
        block_id="a9",
        text="La duración del arrendamiento será libremente pactada por las partes",
        verdict=CitationVerdict.VERIFIED_DIRECT,
        anchor=anchor,
    )


def _subquery(
    subquery_id: str,
    *,
    is_critical: bool,
    verdict: SubQueryVerdict | None,
    with_evidence: bool = True,
) -> SubQueryState:
    """A sub-query state with an optional evidence block and a verdict."""
    return SubQueryState(
        subquery=SubQuery(id=subquery_id, text="q", purpose="p", is_critical=is_critical),
        query_text="q",
        evidence=[_block()] if with_evidence else [],
        verdict=verdict,
    )


def _state(*subqueries: SubQueryState, verified: bool = True) -> Mode1State:
    """A terminal-ish state carrying sub-queries and whether a citation survived."""
    return Mode1State(
        question="q",
        vertical="vivienda",
        target_date=AS_OF,
        subqueries=list(subqueries),
        verified=[_verified()] if verified else [],
    )


def test_no_evidence_abstains() -> None:
    state = _state(
        _subquery("sq1", is_critical=True, verdict=None, with_evidence=False),
        verified=False,
    )
    assert gate_node(state)["abstention_reason"] is AbstentionReason.NO_EVIDENCE


def test_evidence_but_no_surviving_citation_abstains() -> None:
    state = _state(
        _subquery("sq1", is_critical=True, verdict=SubQueryVerdict.SUFFICIENT),
        verified=False,
    )
    assert gate_node(state)["abstention_reason"] is AbstentionReason.NO_VERIFIABLE_CITATION


def test_an_ungrounded_critical_subquery_abstains() -> None:
    state = _state(_subquery("sq1", is_critical=True, verdict=SubQueryVerdict.INSUFFICIENT))
    assert gate_node(state)["abstention_reason"] is AbstentionReason.INSUFFICIENT_CORE


def test_an_ungrounded_peripheral_subquery_yields_a_partial_answer() -> None:
    state = _state(
        _subquery("sq1", is_critical=True, verdict=SubQueryVerdict.SUFFICIENT),
        _subquery("sq2", is_critical=False, verdict=SubQueryVerdict.INSUFFICIENT),
    )
    assert gate_node(state)["outcome"] is Outcome.PARTIAL_ANSWER


def test_all_subqueries_grounded_yields_a_full_answer() -> None:
    state = _state(
        _subquery("sq1", is_critical=True, verdict=SubQueryVerdict.SUFFICIENT),
        _subquery("sq2", is_critical=False, verdict=SubQueryVerdict.SUFFICIENT),
    )
    assert gate_node(state)["outcome"] is Outcome.ANSWER
