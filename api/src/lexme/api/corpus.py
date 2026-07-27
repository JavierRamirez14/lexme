"""The corpus status endpoint: which norms the vertical answers from, and how fresh."""

from datetime import datetime

import psycopg
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from lexme.api.dependencies import get_db_connection, get_vertical
from lexme.ingestion.repository import get_corpus_last_updated, list_ingested_norms

router = APIRouter()


class CorpusNorm(BaseModel):
    """One ingested norm: what it is called, how much of it is held, how fresh it is."""

    norm_id: str
    label: str
    title: str
    consolidated_html_url: str
    updated_at: datetime
    blocks: int


class CorpusStatus(BaseModel):
    """The vertical's corpus: every norm it holds, and the freshest update among them.

    ``updated_at`` is the corpus-wide freshness date; ``norms`` is what that date
    summarizes, so the scope shown to a reader is the scope actually ingested
    rather than a name hardcoded in the UI.
    """

    vertical: str
    updated_at: datetime | None
    norms: list[CorpusNorm]


@router.get("/corpus/status", response_model=CorpusStatus)
def corpus_status(
    connection: psycopg.Connection = Depends(get_db_connection),
    vertical: str = Depends(get_vertical),
) -> CorpusStatus:
    """Return the vertical's ingested norms and the freshest ``updated_at`` among them."""
    return CorpusStatus(
        vertical=vertical,
        updated_at=get_corpus_last_updated(connection, vertical),
        norms=[
            CorpusNorm(
                norm_id=norm.norm_id,
                label=norm.label,
                title=norm.title,
                consolidated_html_url=norm.consolidated_html_url,
                updated_at=norm.updated_at,
                blocks=norm.blocks,
            )
            for norm in list_ingested_norms(connection, vertical)
        ],
    )
