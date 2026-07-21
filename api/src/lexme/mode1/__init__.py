"""Mode 1: natural-language question to a cited answer or an honest non-answer.

The agentic path: a router scope gate, flat decomposition into legal-vocabulary
sub-queries anchored at a target date, at most one disambiguating question when a
critical branch of the vertical is left open, a global self-critique loop over
hybrid retrieval, one grounded synthesis, runtime citation verification, a
deterministic gate that returns a full answer, a partial answer, an honest
abstention or a scope rejection, and code-derived notices about how far the
answer's law is from today's. The graph orchestrates; the gate, in code, decides.
:class:`Mode1Run` runs one consultation, which the caller can stream and -- when
it pauses to ask -- resume on the same thread.
"""

from lexme.mode1.branches import AnswerKind, BranchesError, CriticalBranch, load_branches
from lexme.mode1.dates import DatePrecision, TargetDate, parse_target_date, resolve_target_date
from lexme.mode1.graph.deps import Mode1Deps
from lexme.mode1.graph.presenter import SCOPE_REMINDER, build_agentic_trace, build_response
from lexme.mode1.models import (
    Abstention,
    AbstentionReason,
    AgenticTrace,
    Answer,
    AskRequest,
    AskResponse,
    Clarification,
    Mode1Synthesis,
    Outcome,
    PassReport,
    QueryType,
    RankedBlockRef,
    ResumeRequest,
    RetrievalTrace,
    RouterRejection,
    RouterScope,
    SubQueryReport,
    SubQueryVerdict,
    VerifiedCitation,
)
from lexme.mode1.notices import InForceNotice, NoticeCode, VersionHistory, build_notices
from lexme.mode1.pipeline import Mode1Run, RunNotPausedError
from lexme.mode1.synthesis import SYNTHESIS_TASK, synthesize
from lexme.mode1.version_history import PsycopgVersionHistory

__all__ = [
    "SCOPE_REMINDER",
    "SYNTHESIS_TASK",
    "Abstention",
    "AbstentionReason",
    "AgenticTrace",
    "Answer",
    "AnswerKind",
    "AskRequest",
    "AskResponse",
    "BranchesError",
    "Clarification",
    "CriticalBranch",
    "DatePrecision",
    "InForceNotice",
    "Mode1Deps",
    "Mode1Run",
    "Mode1Synthesis",
    "NoticeCode",
    "Outcome",
    "PassReport",
    "PsycopgVersionHistory",
    "QueryType",
    "RankedBlockRef",
    "ResumeRequest",
    "RetrievalTrace",
    "RouterRejection",
    "RouterScope",
    "RunNotPausedError",
    "SubQueryReport",
    "SubQueryVerdict",
    "TargetDate",
    "VerifiedCitation",
    "VersionHistory",
    "build_agentic_trace",
    "build_notices",
    "build_response",
    "load_branches",
    "parse_target_date",
    "resolve_target_date",
    "synthesize",
]
