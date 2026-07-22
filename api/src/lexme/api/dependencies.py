"""FastAPI dependencies wiring the request-scoped collaborators of Mode 1.

Each dependency is a seam: tests override them on the app to inject a fake LLM, a
deterministic embedder or a test-database connection, so the HTTP surface can be
driven end to end without the network. One connection is opened per request and
shared by the retriever and the corpus reader through FastAPI's dependency cache.
The checkpointer is the exception: it deliberately outlives the request, because a
run paused on a disambiguating question has to still be there when the answer
arrives on the next one.
"""

from collections.abc import Iterator
from datetime import date
from functools import lru_cache
from pathlib import Path

import psycopg
from fastapi import Depends
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pgvector.psycopg import register_vector

from lexme.checklist import Checklist, checklist_path, load_checklist
from lexme.config import Settings, get_settings
from lexme.ingestion.embeddings import TeiEmbedder
from lexme.llm import LlmClient, build_llm_client
from lexme.mode1 import (
    CriticalBranch,
    Mode1Deps,
    PsycopgVersionHistory,
    VersionHistory,
    load_branches,
)
from lexme.retrieval import QueryEmbedder
from lexme.verification import CorpusReader, PsycopgCorpusReader

BRANCHES_FILENAME = "disambiguation.json"
DEFAULT_VERTICAL = "vivienda"


def get_db_connection(
    settings: Settings = Depends(get_settings),
) -> Iterator[psycopg.Connection]:
    """Yield a corpus connection with the pgvector adapter registered."""
    with psycopg.connect(settings.database_url) as connection:
        register_vector(connection)
        yield connection


def get_embedder(settings: Settings = Depends(get_settings)) -> Iterator[QueryEmbedder]:
    """Yield a TEI-backed query embedder, closing it after the request."""
    embedder = TeiEmbedder(settings.tei_url)
    try:
        yield embedder
    finally:
        embedder.close()


@lru_cache
def _default_llm_client() -> LlmClient:
    """Build the routing LLM client once and reuse it across requests."""
    return build_llm_client(get_settings())


def get_llm_client() -> LlmClient:
    """Return the shared routing LLM client."""
    return _default_llm_client()


def get_corpus_reader(
    connection: psycopg.Connection = Depends(get_db_connection),
) -> CorpusReader:
    """Return a corpus reader over the request's connection."""
    return PsycopgCorpusReader(connection)


def get_version_history(
    connection: psycopg.Connection = Depends(get_db_connection),
) -> VersionHistory:
    """Return a version-history reader over the request's connection."""
    return PsycopgVersionHistory(connection)


def get_vertical() -> str:
    """Return the vertical this deployment serves."""
    return DEFAULT_VERTICAL


@lru_cache
def _branches(verticals_dir: str, vertical: str) -> tuple[CriticalBranch, ...]:
    """Load and cache a vertical's branch package; it changes only on redeploy."""
    return load_branches(Path(verticals_dir) / vertical / BRANCHES_FILENAME)


def get_branches(
    settings: Settings = Depends(get_settings),
    vertical: str = Depends(get_vertical),
) -> tuple[CriticalBranch, ...]:
    """Return the vertical's critical-branch package, loaded from its data directory."""
    return _branches(settings.verticals_dir, vertical)


@lru_cache
def _checklist(verticals_dir: str, vertical: str) -> Checklist:
    """Load and cache a vertical's checklist package; it changes only on redeploy."""
    return load_checklist(checklist_path(verticals_dir, vertical))


def get_checklist(
    settings: Settings = Depends(get_settings),
    vertical: str = Depends(get_vertical),
) -> Checklist:
    """Return the vertical's legal-default checklist, loaded from its data directory."""
    return _checklist(settings.verticals_dir, vertical)


@lru_cache
def _checkpointer() -> BaseCheckpointSaver:
    """Build the process-wide checkpoint store once.

    In-process and in-memory: paused runs survive between requests but not a
    restart, which is the honest trade for a single-instance deployment. Swapping
    in a Postgres saver is a change to this function alone.
    """
    return InMemorySaver()


def get_checkpointer() -> BaseCheckpointSaver:
    """Return the shared checkpoint store paused runs are stored in."""
    return _checkpointer()


def get_today() -> date:
    """Return the run's clock: the date "in force now" is measured against."""
    return date.today()


def get_deps(
    connection: psycopg.Connection = Depends(get_db_connection),
    embedder: QueryEmbedder = Depends(get_embedder),
    corpus: CorpusReader = Depends(get_corpus_reader),
    history: VersionHistory = Depends(get_version_history),
    llm: LlmClient = Depends(get_llm_client),
    branches: tuple[CriticalBranch, ...] = Depends(get_branches),
) -> Mode1Deps:
    """Bundle this request's collaborators for the Mode 1 graph."""
    return Mode1Deps(
        connection=connection,
        embedder=embedder,
        corpus=corpus,
        history=history,
        llm=llm,
        branches=branches,
    )
