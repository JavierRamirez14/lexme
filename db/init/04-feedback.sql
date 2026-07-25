-- Feedback: thumbs up/down votes, each carrying the full snapshot it judges.
-- Insert-only; a vote is never aggregated into a metric, only kept as a case
-- candidate for the eval harness to replay later.

CREATE TABLE IF NOT EXISTS feedback (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    mode       TEXT NOT NULL,
    vote       TEXT NOT NULL,
    snapshot   JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_mode_idx ON feedback (mode);
