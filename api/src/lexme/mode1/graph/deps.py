"""The request-scoped collaborators the graph nodes need, passed as one bundle.

Nodes are bound to a :class:`Mode1Deps` when the graph is built, so the LangGraph
node signature stays ``(state) -> update`` while retrieval, the corpus and the LLM
are still injected explicitly rather than reached for globally.
"""

from dataclasses import dataclass

import psycopg

from lexme.llm import LlmClient
from lexme.mode1.branches import CriticalBranch
from lexme.mode1.notices import VersionHistory
from lexme.retrieval import QueryEmbedder
from lexme.verification import CorpusReader


@dataclass(frozen=True)
class Mode1Deps:
    """The external collaborators of a single Mode 1 run.

    ``branches`` is the vertical's data package of case variables that change the
    applicable regime; the engine reads it, never hardcodes it.
    """

    connection: psycopg.Connection
    embedder: QueryEmbedder
    corpus: CorpusReader
    history: VersionHistory
    llm: LlmClient
    branches: tuple[CriticalBranch, ...] = ()
