-- Corpus schema: norms -> blocks -> versions, mirroring the BOE document structure.
-- Point-in-time resolution is a SQL query over versions.effective_date; every
-- historical redaction is kept, never only the current text.

CREATE TABLE IF NOT EXISTS norms (
    id                    TEXT PRIMARY KEY,
    vertical              TEXT NOT NULL,
    eli                   TEXT NOT NULL,
    title                 TEXT NOT NULL,
    consolidated_html_url TEXT NOT NULL,
    updated_at            TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS norms_vertical_idx ON norms (vertical);

CREATE TABLE IF NOT EXISTS blocks (
    id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    norm_id  TEXT NOT NULL REFERENCES norms (id) ON DELETE CASCADE,
    block_id TEXT NOT NULL,
    title    TEXT NOT NULL,
    UNIQUE (norm_id, block_id)
);

CREATE TABLE IF NOT EXISTS versions (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    block_id         BIGINT NOT NULL REFERENCES blocks (id) ON DELETE CASCADE,
    amending_norm_id TEXT NOT NULL,
    publication_date DATE NOT NULL,
    effective_date   DATE NOT NULL,
    text_content     TEXT NOT NULL,
    html_content     TEXT NOT NULL,
    embedding        VECTOR(1024) NOT NULL,
    UNIQUE (block_id, effective_date)
);
