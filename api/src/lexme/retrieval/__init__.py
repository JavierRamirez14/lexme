"""Hybrid retrieval over the corpus: dense (pgvector) + lexical (FTS), fused by RRF.

A single entry point, :func:`hybrid_retrieve`, embeds a query, runs both
retrievers over the in-force redaction of each block, and fuses their rankings
with :func:`reciprocal_rank_fusion`. It returns the evidence set plus every
intermediate ranking, so the agentic loop and the eval harness can read the
retrieval telemetry straight from the result.
"""

from lexme.retrieval.hybrid import (
    CANDIDATE_LIMIT,
    EVIDENCE_TOP_K,
    QueryEmbedder,
    hybrid_retrieve,
)
from lexme.retrieval.models import BlockKey, HybridResult, RetrievedBlock
from lexme.retrieval.rrf import RRF_K, reciprocal_rank_fusion

__all__ = [
    "CANDIDATE_LIMIT",
    "EVIDENCE_TOP_K",
    "RRF_K",
    "BlockKey",
    "HybridResult",
    "QueryEmbedder",
    "RetrievedBlock",
    "hybrid_retrieve",
    "reciprocal_rank_fusion",
]
