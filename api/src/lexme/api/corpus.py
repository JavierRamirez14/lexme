"""The corpus status endpoint: how fresh the vertical's legal text is."""

from datetime import datetime

import psycopg
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from lexme.api.dependencies import get_db_connection, get_vertical
from lexme.ingestion.repository import get_corpus_last_updated

router = APIRouter()


class CorpusStatus(BaseModel):
    """The vertical's corpus freshness, read from its newest norm update."""

    vertical: str
    updated_at: datetime | None


@router.get("/corpus/status", response_model=CorpusStatus)
def corpus_status(
    connection: psycopg.Connection = Depends(get_db_connection),
    vertical: str = Depends(get_vertical),
) -> CorpusStatus:
    """Return the freshest ``updated_at`` across the vertical's norms, if any."""
    return CorpusStatus(vertical=vertical, updated_at=get_corpus_last_updated(connection, vertical))
