"""FastAPI dependencies wiring the request-scoped collaborators of Mode 1.

Each dependency is a seam: tests override them on the app to inject a fake LLM, a
deterministic embedder or a test-database connection, so the HTTP surface can be
driven end to end without the network. One connection is opened per request and
shared by the retriever and the corpus reader through FastAPI's dependency cache.
"""

from collections.abc import Iterator
from datetime import date
from functools import lru_cache

import psycopg
from fastapi import Depends
from pgvector.psycopg import register_vector

from lexme.config import Settings, get_settings
from lexme.ingestion.embeddings import TeiEmbedder
from lexme.llm import LlmClient, build_llm_client
from lexme.retrieval import QueryEmbedder
from lexme.verification import CorpusReader, PsycopgCorpusReader


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


def get_target_date() -> date:
    """Return the point-in-time date to resolve the corpus at: today."""
    return date.today()
