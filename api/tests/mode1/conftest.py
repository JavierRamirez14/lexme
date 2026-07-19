"""Fixtures for Mode 1 pipeline tests: a seeded LAU corpus and its reader."""

from datetime import date
from pathlib import Path

import psycopg
import pytest

from lexme.ingestion import repository
from lexme.ingestion.xml_parsing import parse_norm_xml
from lexme.verification import CorpusReader, PsycopgCorpusReader
from tests.conftest import DeterministicEmbedder, seed_norm

LAU_XML = (
    Path(__file__).resolve().parents[1] / "ingestion" / "fixtures" / "lau_sample.xml"
).read_text(encoding="utf-8")
LAU_NORM_ID = "BOE-A-1994-26003"
AS_OF = date(2020, 1, 1)


@pytest.fixture
def seeded_corpus(
    corpus_db: psycopg.Connection, deterministic_embedder: DeterministicEmbedder
) -> psycopg.Connection:
    """A corpus_db with the trimmed LAU norm loaded."""
    seed_norm(corpus_db, parse_norm_xml(LAU_XML), deterministic_embedder)
    return corpus_db


@pytest.fixture
def corpus_reader(seeded_corpus: psycopg.Connection) -> CorpusReader:
    """A corpus reader over the seeded LAU corpus."""
    return PsycopgCorpusReader(seeded_corpus)


def in_force_text(conn: psycopg.Connection, block_id: str) -> str:
    """The plain text of a block's redaction in force at the test's target date."""
    version = repository.get_version_in_force(conn, LAU_NORM_ID, block_id, AS_OF)
    assert version is not None
    return version.text_content
