"""The synthesis node: one grounded generation, then runtime citation verification.

Synthesis runs over the union of every sub-query's evidence and is told which
parts of the plan stayed ungrounded, so it neither invents nor fills gaps. Each
proposed citation is then re-verified against the corpus; only the survivors reach
the state. When no evidence was retrieved at all, the model is not called and the
gate abstains.
"""

from datetime import date

from lexme.mode1.graph.deps import Mode1Deps
from lexme.mode1.graph.state import Mode1State
from lexme.mode1.graph.steps import Step
from lexme.mode1.models import Mode1Synthesis, VerifiedCitation
from lexme.mode1.synthesis import synthesize
from lexme.retrieval.models import RetrievedBlock
from lexme.verification import (
    CitationResult,
    CitationVerdict,
    CorpusReader,
    EvidenceBlock,
    summarize,
    verify_citations,
)


def synthesize_node(state: Mode1State, *, deps: Mode1Deps) -> dict:
    """Synthesize a three-layer answer over the evidence, then verify its citations.

    Short-circuits with an empty result when nothing was retrieved, leaving the
    gate to abstain. Otherwise returns the proposed synthesis, the surviving
    verified citations and the per-verdict counts.
    """
    evidence = _union_evidence(state)
    if not evidence:
        return {
            "synthesis": None,
            "verified": [],
            "citation_verdicts": {},
            "current_step": Step.SYNTHESIZING,
        }

    synthesis = synthesize(deps.llm, state.question, evidence, gaps=_gaps(state))
    results = _verify(synthesis, evidence, deps.corpus, state.target_date)
    verified = [_to_verified(result) for result in results if _held(result)]
    return {
        "synthesis": synthesis,
        "verified": verified,
        "citation_verdicts": summarize(results),
        "current_step": "sintetizando",
    }


def _union_evidence(state: Mode1State) -> list[RetrievedBlock]:
    """The deduplicated union of every sub-query's evidence, in first-seen order."""
    seen: dict[tuple[str, str], RetrievedBlock] = {}
    for sub in state.subqueries:
        for block in sub.evidence:
            seen.setdefault(block.key, block)
    return list(seen.values())


def _gaps(state: Mode1State) -> list[str]:
    """The purposes of the sub-queries the critique left insufficient."""
    return [sub.subquery.purpose for sub in state.subqueries if not sub.is_sufficient]


def _verify(
    synthesis: Mode1Synthesis,
    evidence: list[RetrievedBlock],
    corpus: CorpusReader,
    target_date: date,
) -> list[CitationResult]:
    """Verify the proposed citations against the retrieved evidence."""
    blocks = [EvidenceBlock(norm_id=block.norm_id, block_id=block.block_id) for block in evidence]
    return verify_citations(synthesis.fundamento, blocks, target_date, corpus)


def _held(result: CitationResult) -> bool:
    """Whether a citation survived verification (was not discarded)."""
    return result.verdict is not CitationVerdict.DISCARDED


def _to_verified(result: CitationResult) -> VerifiedCitation:
    """Map a surviving citation result to the display citation.

    A non-discarded result always carries an anchor, so it is safe to require one.
    """
    if result.anchor is None:
        raise ValueError(f"citation {result.block_id} held with no anchor")
    return VerifiedCitation(
        block_id=result.block_id,
        text=result.text,
        verdict=result.verdict,
        anchor=result.anchor,
    )
