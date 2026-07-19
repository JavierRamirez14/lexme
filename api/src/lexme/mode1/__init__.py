"""Mode 1: natural-language question to a cited answer or an honest non-answer.

The agentic path: a router scope gate, flat decomposition into legal-vocabulary
sub-queries, a global self-critique loop over hybrid retrieval, one grounded
synthesis, runtime citation verification, and a deterministic gate that returns a
full answer, a partial answer, an honest abstention or a scope rejection. The
graph orchestrates; the gate, in code, decides. :func:`answer_question` runs it to
a response; :func:`stream_states` narrates it node by node for the SSE endpoint.
"""

from lexme.mode1.graph.presenter import SCOPE_REMINDER, build_agentic_trace, build_response
from lexme.mode1.models import (
    Abstention,
    AbstentionReason,
    AgenticTrace,
    Answer,
    AskRequest,
    AskResponse,
    Mode1Synthesis,
    Outcome,
    PassReport,
    QueryType,
    RankedBlockRef,
    RetrievalTrace,
    RouterRejection,
    RouterScope,
    SubQueryReport,
    SubQueryVerdict,
    VerifiedCitation,
)
from lexme.mode1.pipeline import answer_question, stream_states
from lexme.mode1.synthesis import SYNTHESIS_TASK, synthesize

__all__ = [
    "SCOPE_REMINDER",
    "SYNTHESIS_TASK",
    "Abstention",
    "AbstentionReason",
    "AgenticTrace",
    "Answer",
    "AskRequest",
    "AskResponse",
    "Mode1Synthesis",
    "Outcome",
    "PassReport",
    "QueryType",
    "RankedBlockRef",
    "RetrievalTrace",
    "RouterRejection",
    "RouterScope",
    "SubQueryReport",
    "SubQueryVerdict",
    "VerifiedCitation",
    "answer_question",
    "build_agentic_trace",
    "build_response",
    "stream_states",
    "synthesize",
]
