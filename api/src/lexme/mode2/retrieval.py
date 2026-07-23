"""The retrieval seam Phase B reaches through for clauses with no checklist item.

Most clauses resolve their evidence deterministically -- a checklist item points
straight at its anchor blocks. A clause the checklist does not cover has to search
for its ground, and it does so through the very same hybrid retrieval Mode 1 uses.
Exposing it as a port keeps the classification testable without a database: a test
supplies a fake retriever, while production wires the pgvector-backed one.
"""

from datetime import date
from typing import Protocol

import psycopg

from lexme.retrieval import QueryEmbedder, RetrievedBlock, hybrid_retrieve


class ClauseRetriever(Protocol):
    """Retrieve corpus evidence for a clause the checklist does not index."""

    def retrieve(
        self, clause_text: str, *, vertical: str, target_date: date
    ) -> list[RetrievedBlock]:
        """Return the evidence blocks in force at ``target_date`` for ``clause_text``."""
        ...


class HybridClauseRetriever:
    """A :class:`ClauseRetriever` backed by the shared hybrid retrieval.

    Holds the corpus connection and the query embedder, and fuses dense and
    lexical retrieval exactly as Mode 1 does, so a clause off the checklist is
    grounded by the same machinery as a Mode 1 question.
    """

    def __init__(self, connection: psycopg.Connection, embedder: QueryEmbedder) -> None:
        """Bind the retriever to an open corpus connection and a query embedder."""
        self._connection = connection
        self._embedder = embedder

    def retrieve(
        self, clause_text: str, *, vertical: str, target_date: date
    ) -> list[RetrievedBlock]:
        """Fuse dense and lexical retrieval over ``clause_text`` into its evidence."""
        result = hybrid_retrieve(
            self._connection,
            self._embedder,
            query=clause_text,
            vertical=vertical,
            target_date=target_date,
        )
        return result.evidence
