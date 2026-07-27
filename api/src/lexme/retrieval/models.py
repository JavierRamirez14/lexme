"""Value types for hybrid retrieval: a retrieved block and a fused result set.

The retrievers return the block's point-in-time redaction (the same text the
citation verifier will match against), so synthesis and verification see one
consistent view of each block. The intermediate rankings are kept verbatim so the
harness can measure the dense/lexical/fused deltas without extra instrumentation.
"""

from dataclasses import dataclass
from datetime import date

BlockKey = tuple[str, str]


@dataclass(frozen=True)
class RetrievedBlock:
    """A candidate block surfaced by a retriever, with its in-force redaction.

    ``text`` is the version of the block in force at the run's target date; it is
    what synthesis quotes and what verification re-checks. ``norm_label`` is the
    short name of the law it comes from, so a prompt showing blocks from several
    norms can say which is which rather than offering two "article 9"s.
    """

    norm_id: str
    norm_label: str
    block_id: str
    title: str
    text: str
    effective_date: date

    @property
    def key(self) -> BlockKey:
        """The block's identity across retrievers: ``(norm_id, block_id)``."""
        return (self.norm_id, self.block_id)


@dataclass(frozen=True)
class HybridResult:
    """The outcome of a hybrid retrieval, evidence plus the rankings that produced it.

    ``evidence`` is the fused top-k, deduplicated and in fused order -- the blocks
    handed to synthesis. ``dense_ranking`` and ``lexical_ranking`` are each
    retriever's ordered keys; ``fused_ranking`` is the RRF output as
    ``(key, score)``. The three rankings are the run's retrieval telemetry.
    """

    evidence: list[RetrievedBlock]
    dense_ranking: list[BlockKey]
    lexical_ranking: list[BlockKey]
    fused_ranking: list[tuple[BlockKey, float]]
