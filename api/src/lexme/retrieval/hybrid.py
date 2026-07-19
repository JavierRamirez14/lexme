"""Hybrid retrieval: run the two retrievers and fuse them with our own RRF.

One query fans out to a semantic search (pgvector) and a lexical search
(Postgres FTS) over the same in-force candidate set, then reciprocal rank fusion
merges the two orderings into the evidence set. Fusing by rank, not by raw score,
lets an incomparable cosine distance and ``ts_rank`` be combined without tuning a
weight between them.
"""

from datetime import date
from typing import Protocol

import psycopg

from lexme.retrieval import repository
from lexme.retrieval.models import HybridResult, RetrievedBlock
from lexme.retrieval.rrf import reciprocal_rank_fusion

CANDIDATE_LIMIT = 10
EVIDENCE_TOP_K = 8


class QueryEmbedder(Protocol):
    """Anything that can embed a batch of texts, preserving order."""

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


def hybrid_retrieve(
    conn: psycopg.Connection,
    embedder: QueryEmbedder,
    *,
    query: str,
    vertical: str,
    target_date: date,
    candidate_limit: int = CANDIDATE_LIMIT,
    top_k: int = EVIDENCE_TOP_K,
) -> HybridResult:
    """Retrieve evidence for ``query`` by fusing dense and lexical retrieval.

    Embeds the query once, runs both retrievers up to ``candidate_limit`` each,
    fuses their rankings and returns the fused top ``top_k`` as evidence together
    with all three intermediate rankings.
    """
    embedding = embedder.embed_many([query])[0]
    dense = repository.search_dense(conn, vertical, target_date, embedding, candidate_limit)
    lexical = repository.search_lexical(conn, vertical, target_date, query, candidate_limit)

    dense_ranking = [block.key for block in dense]
    lexical_ranking = [block.key for block in lexical]
    fused = reciprocal_rank_fusion([dense_ranking, lexical_ranking])

    blocks_by_key: dict[tuple[str, str], RetrievedBlock] = {}
    for block in (*dense, *lexical):
        blocks_by_key.setdefault(block.key, block)
    evidence = [blocks_by_key[key] for key, _ in fused[:top_k]]

    return HybridResult(
        evidence=evidence,
        dense_ranking=dense_ranking,
        lexical_ranking=lexical_ranking,
        fused_ranking=fused,
    )
