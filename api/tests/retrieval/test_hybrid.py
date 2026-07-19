"""Integration tests for hybrid retrieval against a real Postgres corpus."""

from datetime import date
from pathlib import Path

import psycopg

from lexme.ingestion.xml_parsing import parse_norm_xml
from lexme.retrieval import hybrid_retrieve
from lexme.retrieval.hybrid import repository
from tests.conftest import DeterministicEmbedder, seed_norm

LAU_XML = (
    Path(__file__).resolve().parents[1] / "ingestion" / "fixtures" / "lau_sample.xml"
).read_text(encoding="utf-8")
AS_OF = date(2020, 1, 1)


def _seed_lau(conn: psycopg.Connection, embedder: DeterministicEmbedder) -> None:
    """Seed the trimmed LAU corpus (articles 1, 2 and 9)."""
    seed_norm(conn, parse_norm_xml(LAU_XML), embedder)


def test_hybrid_surfaces_the_article_matching_the_question(
    corpus_db: psycopg.Connection, deterministic_embedder: DeterministicEmbedder
) -> None:
    _seed_lau(corpus_db, deterministic_embedder)

    result = hybrid_retrieve(
        corpus_db,
        deterministic_embedder,
        query="¿cuál es el plazo mínimo del arrendamiento de vivienda?",
        vertical="vivienda",
        target_date=AS_OF,
    )

    evidence_ids = [block.block_id for block in result.evidence]
    assert "a9" in evidence_ids


def test_both_retrievers_contribute_visible_rankings(
    corpus_db: psycopg.Connection, deterministic_embedder: DeterministicEmbedder
) -> None:
    _seed_lau(corpus_db, deterministic_embedder)

    result = hybrid_retrieve(
        corpus_db,
        deterministic_embedder,
        query="plazo mínimo arrendamiento",
        vertical="vivienda",
        target_date=AS_OF,
    )

    assert result.dense_ranking
    assert result.lexical_ranking
    assert result.fused_ranking
    fused_keys = {key for key, _ in result.fused_ranking}
    assert fused_keys == set(result.dense_ranking) | set(result.lexical_ranking)


def test_retrieval_returns_the_in_force_redaction(
    corpus_db: psycopg.Connection, deterministic_embedder: DeterministicEmbedder
) -> None:
    _seed_lau(corpus_db, deterministic_embedder)

    blocks = repository.search_lexical(corpus_db, "vivienda", AS_OF, "plazo mínimo", limit=10)

    article_nine = next(block for block in blocks if block.block_id == "a9")
    assert article_nine.effective_date == date(2019, 3, 6)
    assert "persona jurídica" in article_nine.text


def test_dense_search_ranks_every_block_as_a_candidate(
    corpus_db: psycopg.Connection, deterministic_embedder: DeterministicEmbedder
) -> None:
    _seed_lau(corpus_db, deterministic_embedder)

    embedding = deterministic_embedder.embed_many(["arrendamiento de vivienda"])[0]
    blocks = repository.search_dense(corpus_db, "vivienda", AS_OF, embedding, limit=10)

    assert {block.block_id for block in blocks} == {"a1", "a2", "a9"}
