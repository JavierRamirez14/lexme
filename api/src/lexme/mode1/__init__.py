"""Mode 1: natural-language question to a cited answer or an honest abstention.

The tracer-bullet path with no agency yet: hybrid retrieval, one structured
synthesis call constrained to the retrieved evidence, runtime citation
verification, and a deterministic gate that returns either a verified answer or a
first-class abstention. :func:`answer_question` is the whole path; the API layer
only adapts HTTP to it.
"""

from lexme.mode1.models import (
    Abstention,
    AbstentionReason,
    Answer,
    AskRequest,
    AskResponse,
    Mode1Synthesis,
    Outcome,
    RankedBlockRef,
    RetrievalTrace,
    VerifiedCitation,
)
from lexme.mode1.pipeline import SCOPE_REMINDER, answer_question
from lexme.mode1.synthesis import SYNTHESIS_TASK, synthesize

__all__ = [
    "SCOPE_REMINDER",
    "SYNTHESIS_TASK",
    "Abstention",
    "AbstentionReason",
    "Answer",
    "AskRequest",
    "AskResponse",
    "Mode1Synthesis",
    "Outcome",
    "RankedBlockRef",
    "RetrievalTrace",
    "VerifiedCitation",
    "answer_question",
    "synthesize",
]
