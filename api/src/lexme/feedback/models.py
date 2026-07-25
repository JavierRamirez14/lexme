"""The feedback contract: a vote plus the full snapshot it judges.

A thumbs up or down is never a metric on its own -- it is a pointer into a case
the eval harness can later replay, so the snapshot must carry everything the
reader saw: the Mode 1 query and cited answer, or the Mode 2 document's ficha
and risk map.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, field_validator


class FeedbackMode(StrEnum):
    """Which surface the feedback was given on."""

    CONSULTATION = "consulta"
    CONTRACT = "contrato"


class FeedbackVote(StrEnum):
    """The reader's judgement: thumbs up or down."""

    UP = "positivo"
    DOWN = "negativo"


class FeedbackRequest(BaseModel):
    """A vote on one turn, carrying the exact snapshot the reader judged."""

    mode: FeedbackMode
    vote: FeedbackVote
    snapshot: dict[str, Any]

    @field_validator("snapshot")
    @classmethod
    def _reject_empty(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Reject an empty snapshot: feedback with nothing to review is useless."""
        if not value:
            raise ValueError("snapshot must not be empty")
        return value


class FeedbackReceipt(BaseModel):
    """Confirmation that a vote was stored, with the row it was stored as."""

    id: int
