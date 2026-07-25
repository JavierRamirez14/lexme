"""The feedback endpoint: persist a vote and the full snapshot it judges."""

import psycopg
from fastapi import APIRouter, Depends, status

from lexme.api.dependencies import get_db_connection
from lexme.feedback import FeedbackReceipt, FeedbackRequest, save_feedback

router = APIRouter()


@router.post("/feedback", response_model=FeedbackReceipt, status_code=status.HTTP_201_CREATED)
def submit_feedback(
    request: FeedbackRequest,
    connection: psycopg.Connection = Depends(get_db_connection),
) -> FeedbackReceipt:
    """Store one thumbs up/down vote with its full snapshot, as a case candidate."""
    return FeedbackReceipt(id=save_feedback(connection, request))
