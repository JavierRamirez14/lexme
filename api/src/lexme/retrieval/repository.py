"""SQL for the two retrievers: dense (pgvector) and lexical (Postgres FTS).

Both search the same candidate set -- the redaction of each block in force at the
target date, one row per block -- so the semantic index and the lexical index
never rank different versions of the same article against each other. Only the
current redaction is a useful retrieval decoy; the historical ones are near
duplicates that would only crowd the ranking, so the point-in-time filter also
narrows what is searched.
"""

from datetime import date

import psycopg
from pgvector import Vector
from psycopg.rows import class_row

from lexme.retrieval.models import RetrievedBlock

_LATEST_PER_BLOCK = """
    SELECT DISTINCT ON (v.block_id)
           b.norm_id AS norm_id,
           n.label AS norm_label,
           b.block_id AS block_id,
           b.title AS title,
           v.text_content AS text_content,
           v.effective_date AS effective_date,
           v.embedding AS embedding
    FROM versions v
    JOIN blocks b ON b.id = v.block_id
    JOIN norms n ON n.id = b.norm_id
    WHERE n.vertical = %(vertical)s AND v.effective_date <= %(target_date)s
    ORDER BY v.block_id, v.effective_date DESC
"""


def search_dense(
    conn: psycopg.Connection,
    vertical: str,
    target_date: date,
    embedding: list[float],
    limit: int,
) -> list[RetrievedBlock]:
    """Return the ``limit`` blocks nearest ``embedding`` by cosine distance.

    Searches the in-force redaction of each block in ``vertical`` as of
    ``target_date``. The embedding is wrapped so it binds as a ``vector`` rather
    than a float array, which the distance operator would not accept.
    """
    query = f"""
        SELECT norm_id, norm_label, block_id, title, text_content AS text, effective_date
        FROM ({_LATEST_PER_BLOCK}) latest
        ORDER BY latest.embedding <=> %(embedding)s
        LIMIT %(limit)s
    """
    with conn.cursor(row_factory=class_row(RetrievedBlock)) as cursor:
        return cursor.execute(
            query,
            {
                "vertical": vertical,
                "target_date": target_date,
                "embedding": Vector(embedding),
                "limit": limit,
            },
        ).fetchall()


def search_lexical(
    conn: psycopg.Connection,
    vertical: str,
    target_date: date,
    query_text: str,
    limit: int,
) -> list[RetrievedBlock]:
    """Return the ``limit`` blocks best matching ``query_text`` by Spanish FTS.

    Uses the ``spanish`` text-search configuration for stemming and stop words,
    over the same in-force redaction set as :func:`search_dense`. The lexemes are
    combined with OR, not the ``plainto_tsquery`` default of AND: without a query
    planner a whole natural-language question shares few terms with any single
    article, so requiring every term would match nothing. ``ts_rank`` still
    rewards blocks that hit more terms, so recall widens without losing ordering.
    """
    query = f"""
        WITH latest AS ({_LATEST_PER_BLOCK}),
             search AS (
                 SELECT replace(
                     plainto_tsquery('spanish', %(query_text)s)::text, ' & ', ' | '
                 )::tsquery AS query
             )
        SELECT norm_id, norm_label, block_id, title, text_content AS text, effective_date
        FROM latest, search
        WHERE to_tsvector('spanish', latest.text_content) @@ search.query
        ORDER BY ts_rank(to_tsvector('spanish', latest.text_content), search.query) DESC
        LIMIT %(limit)s
    """
    with conn.cursor(row_factory=class_row(RetrievedBlock)) as cursor:
        return cursor.execute(
            query,
            {
                "vertical": vertical,
                "target_date": target_date,
                "query_text": query_text,
                "limit": limit,
            },
        ).fetchall()
