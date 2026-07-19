"""The request-scoped collaborators the graph nodes need, passed as one bundle.

Nodes are bound to a :class:`Mode1Deps` when the graph is built, so the LangGraph
node signature stays ``(state) -> update`` while retrieval, the corpus and the LLM
are still injected explicitly rather than reached for globally.
"""

from dataclasses import dataclass

import psycopg

from lexme.llm import LlmClient
from lexme.retrieval import QueryEmbedder
from lexme.verification import CorpusReader


@dataclass(frozen=True)
class Mode1Deps:
    """The external collaborators of a single Mode 1 run."""

    connection: psycopg.Connection
    embedder: QueryEmbedder
    corpus: CorpusReader
    llm: LlmClient
