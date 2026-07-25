"""SQL persistence for feedback: an insert-only case quarry."""

import psycopg
from psycopg.types.json import Jsonb

from lexme.feedback.models import FeedbackRequest


def save_feedback(conn: psycopg.Connection, feedback: FeedbackRequest) -> int:
    """Insert one feedback snapshot and return its row id."""
    row = conn.execute(
        "INSERT INTO feedback (mode, vote, snapshot) VALUES (%s, %s, %s) RETURNING id",
        (feedback.mode.value, feedback.vote.value, Jsonb(feedback.snapshot)),
    ).fetchone()
    conn.commit()
    return row[0]
