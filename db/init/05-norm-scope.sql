-- Which norms a vertical ingests, and how much of each, is manifest data the corpus
-- has to remember: the short label the norm is presented under, and a fingerprint of
-- the manifest's selection so a widened or narrowed block list forces a re-ingest
-- even when the BOE text has not moved.

ALTER TABLE norms ADD COLUMN IF NOT EXISTS label TEXT NOT NULL DEFAULT '';
ALTER TABLE norms ADD COLUMN IF NOT EXISTS selection_digest TEXT NOT NULL DEFAULT '';
