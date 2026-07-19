-- Search indexes for hybrid retrieval. The lexical retriever runs a Spanish
-- full-text match over a block's prose; the dense retriever orders by cosine
-- distance. Both index expressions must match the ones the queries use.

CREATE INDEX IF NOT EXISTS versions_fts_spanish_idx
    ON versions USING GIN (to_tsvector('spanish', text_content));

CREATE INDEX IF NOT EXISTS versions_embedding_cosine_idx
    ON versions USING hnsw (embedding vector_cosine_ops);
