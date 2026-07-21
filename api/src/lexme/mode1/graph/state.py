"""The mutable state the LangGraph nodes read and write as the run progresses.

Every node returns a partial update of this model; LangGraph merges the updates
and revalidates, so the state is always a coherent snapshot. The UI's SSE stream
and the eval harness read their telemetry straight from here, which is why the
per-sub-query evidence and the per-pass records live on the state rather than
being recomputed at the edges.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict

from lexme.mode1.dates import DatePrecision, TargetDate
from lexme.mode1.graph.steps import Step
from lexme.mode1.models import (
    AbstentionReason,
    Mode1Synthesis,
    Outcome,
    QueryType,
    RouterScope,
    SubQueryVerdict,
    VerifiedCitation,
)
from lexme.mode1.notices import CitedBlock, InForceNotice
from lexme.retrieval.models import BlockKey, RetrievedBlock

MAX_SUBQUERIES = 4
MAX_PASSES = 3


class SubQuery(BaseModel):
    """A planned retrieval sub-query in legal vocabulary, with its role in the plan.

    ``is_critical`` marks a sub-query the answer cannot stand without; the gate
    abstains when a critical sub-query stays ungrounded but only downgrades to a
    partial answer when a peripheral one does.
    """

    id: str
    text: str
    purpose: str
    is_critical: bool


class SubQueryState(BaseModel):
    """A sub-query together with its retrieval, its live query text and its verdict.

    ``query_text`` starts as the plan's text and is replaced by the critique's
    reformulation on a re-retrieval, so the loop searches with fresh terms. The
    ranking fields are this sub-query's own hybrid-retrieval telemetry.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    subquery: SubQuery
    query_text: str
    evidence: list[RetrievedBlock] = []
    dense_ranking: list[BlockKey] = []
    lexical_ranking: list[BlockKey] = []
    fused_ranking: list[tuple[BlockKey, float]] = []
    verdict: SubQueryVerdict | None = None

    @property
    def is_sufficient(self) -> bool:
        """Whether the critique has ruled this sub-query grounded enough to cite."""
        return self.verdict is SubQueryVerdict.SUFFICIENT


class PassRecord(BaseModel):
    """The tally of one self-critique pass, kept to measure the agentic delta."""

    pass_number: int
    sufficient_ids: list[str]
    insufficient_ids: list[str]
    evidence_count: int


class Mode1State(BaseModel):
    """The whole run's state: inputs, router and plan, the loop, and the outcome.

    ``current_step`` is the human-readable stage name the SSE stream emits after
    each node. The terminal ``outcome`` and ``abstention_reason`` are set only by
    the router (for a rejection) or the gate.

    ``today`` is the run's clock, injected rather than read, and ``target_date``
    is the date the corpus is resolved at: they start equal and diverge when the
    planner reads a past date out of the question or the user answers the
    signing-date branch.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    vertical: str
    today: date
    target_date: date
    target_date_precision: DatePrecision = DatePrecision.NONE

    current_step: str = Step.START

    scope: RouterScope | None = None
    query_type: QueryType | None = None
    router_message: str | None = None

    subqueries: list[SubQueryState] = []
    assumptions: list[str] = []

    unresolved_branch_ids: list[str] = []
    clarification_asked: bool = False
    case_facts: list[str] = []

    passes: list[PassRecord] = []
    pass_number: int = 0

    synthesis: Mode1Synthesis | None = None
    verified: list[VerifiedCitation] = []
    citation_verdicts: dict[str, int] = {}

    notices: list[InForceNotice] = []

    outcome: Outcome | None = None
    abstention_reason: AbstentionReason | None = None

    @property
    def anchor(self) -> TargetDate:
        """The date the corpus is resolved at, with how precisely it was stated."""
        return TargetDate(value=self.target_date, precision=self.target_date_precision)

    @property
    def cited_blocks(self) -> list[CitedBlock]:
        """Each verified citation paired with the norm its evidence block belongs to."""
        norm_ids = {
            block.block_id: block.norm_id for sub in self.subqueries for block in sub.evidence
        }
        return [
            CitedBlock(
                norm_id=norm_ids[citation.block_id],
                block_id=citation.block_id,
                title=citation.anchor.title,
                effective_date=citation.anchor.effective_date,
            )
            for citation in self.verified
            if citation.block_id in norm_ids
        ]

    @property
    def total_evidence(self) -> int:
        """The count of evidence blocks gathered across every sub-query."""
        return sum(len(sub.evidence) for sub in self.subqueries)

    @property
    def all_sufficient(self) -> bool:
        """Whether every sub-query has been ruled sufficient by the critique."""
        return bool(self.subqueries) and all(sub.is_sufficient for sub in self.subqueries)
