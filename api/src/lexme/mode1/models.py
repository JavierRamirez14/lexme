"""The Mode 1 answer contract: what the graph proposes and what code returns.

The agentic graph classifies scope, decomposes the question into sub-queries,
retrieves and self-critiques them across passes, synthesizes a three-layer answer
and verifies every citation. The terminal outcome -- a full answer, a partial
answer, an honest abstention, an out-of-scope rejection or a pause to ask one
disambiguating question -- is a code decision over the critique verdicts and the
surviving citations, never the model's own judgement, so the outcomes and the
abstention reasons are typed here.
"""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, field_validator

from lexme.mode1.branches import AnswerKind
from lexme.mode1.notices import InForceNotice
from lexme.verification import CitationVerdict, ProposedCitation, VerifiedAnchor


class RouterScope(StrEnum):
    """Whether the question is inside the vertical's scope, decided by the router."""

    IN_SCOPE = "dentro"
    OUT_OF_SCOPE = "fuera"


class QueryType(StrEnum):
    """The router's shallow classification of intent, tuning downstream prompts."""

    INFORMATIONAL = "informativa"
    SITUATIONAL = "situacional"
    PROCEDURAL = "procedimental"


class SubQueryVerdict(StrEnum):
    """The self-critique's per-sub-query ruling: does its evidence let us cite?"""

    SUFFICIENT = "suficiente"
    INSUFFICIENT = "insuficiente"


class Mode1Synthesis(BaseModel):
    """The three-layer answer as the synthesizer proposes it, before verification.

    ``fundamento`` are the citations the model wants to make, each a block id and
    the literal text it claims that block contains. Nothing here is trusted until
    the verifier re-checks every citation against the corpus.
    """

    fundamento: list[ProposedCitation]
    explicacion: str
    accion: list[str]


class VerifiedCitation(BaseModel):
    """A citation that survived verification, with its code-hydrated anchor.

    ``text`` is the literal corpus text to show (the verified quote, or the real
    span it was repaired to); ``verdict`` records how it was resolved.
    """

    block_id: str
    text: str
    verdict: CitationVerdict
    anchor: VerifiedAnchor


class Answer(BaseModel):
    """The three-layer answer as returned, situated in time and in the user's case.

    ``fundamento`` never contains a discarded citation. ``fecha_objetivo`` is the
    date the corpus was resolved at, so the reader can never mistake the law of
    then for the law of now, and ``avisos_vigencia`` carries the code-derived
    warnings about that distance. ``asunciones`` are the interpretation choices
    stated instead of asked; ``huecos_declarados`` names what a partial answer
    could not ground, and is empty for a full answer.
    """

    fundamento: list[VerifiedCitation]
    explicacion: str
    accion: list[str]
    fecha_objetivo: date
    avisos_vigencia: list[InForceNotice] = []
    asunciones: list[str] = []
    huecos_declarados: list[str] = []


class AbstentionReason(StrEnum):
    """Why code chose to abstain -- the closed set the gate can produce."""

    NO_EVIDENCE = "sin_evidencia"
    NO_VERIFIABLE_CITATION = "sin_cita_verificable"
    INSUFFICIENT_CORE = "sin_base_nuclear"


class Abstention(BaseModel):
    """An honest non-answer: a machine reason plus user-facing message and scope."""

    reason: AbstentionReason
    message: str
    scope_reminder: str


class RouterRejection(BaseModel):
    """An out-of-scope rejection decided before any retrieval, with the scope note."""

    message: str
    scope_reminder: str


class Clarification(BaseModel):
    """The single question the graph pauses to ask before it can answer the case.

    Raised only for a critical branch of the vertical -- a fact that changes which
    legal regime applies -- and only once per run. ``answer_kind`` tells the client
    what to collect, so a date branch is answered with a date rather than prose.
    """

    branch_id: str
    question: str
    answer_kind: AnswerKind


class RankedBlockRef(BaseModel):
    """One block's identity at a position in a ranking, for retrieval telemetry."""

    norm_id: str
    block_id: str


class RetrievalTrace(BaseModel):
    """The dense, lexical and fused rankings behind one sub-query, in order.

    Exposing the intermediate rankings makes the hybrid retrieval measurable: the
    fused order is visibly a fusion of the two retriever orders.
    """

    dense: list[RankedBlockRef]
    lexical: list[RankedBlockRef]
    fused: list[RankedBlockRef]


class SubQueryReport(BaseModel):
    """One decomposed sub-query as shown to the client: intent, evidence, verdict.

    ``verdict`` is ``None`` until the first critique pass rules on it. ``evidence``
    is the fused top-k handed to synthesis; ``retrieval`` is that sub-query's own
    hybrid trace.
    """

    id: str
    text: str
    purpose: str
    is_critical: bool
    verdict: SubQueryVerdict | None = None
    evidence: list[RankedBlockRef] = []
    retrieval: RetrievalTrace | None = None


class PassReport(BaseModel):
    """One self-critique pass: which sub-queries held, which did not, and evidence.

    ``evidence_count`` is the total number of evidence blocks across all
    sub-queries at the end of this pass; comparing it across passes is the raw
    signal behind the agentic delta.
    """

    pass_number: int
    sufficient_ids: list[str]
    insufficient_ids: list[str]
    evidence_count: int


class AgenticTrace(BaseModel):
    """The measurable record of the agentic run, read by the UI and the harness.

    ``agentic_delta`` is the growth in grounded sub-queries from the first pass to
    the last -- the evidence that iterating earned something over a single shot.
    """

    query_type: QueryType | None = None
    subqueries: list[SubQueryReport] = []
    passes: list[PassReport] = []
    first_pass_sufficient: int = 0
    final_sufficient: int = 0
    agentic_delta: int = 0


class Outcome(StrEnum):
    """The deterministic states a Mode 1 request can end in.

    ``CLARIFICATION`` is the only non-terminal one: the run is paused mid-graph
    waiting for the user's answer and resumes on the same thread.
    """

    ANSWER = "respuesta"
    PARTIAL_ANSWER = "respuesta_parcial"
    ABSTENTION = "abstencion"
    ROUTER_REJECTION = "rechazo_router"
    CLARIFICATION = "desambiguacion"


class AskRequest(BaseModel):
    """A user's natural-language question for Mode 1."""

    question: str

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """Reject an empty or whitespace-only question at the API boundary."""
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class AskResponse(BaseModel):
    """The full Mode 1 result: outcome, the matching payload, and agentic telemetry.

    Exactly one of ``answer`` / ``abstention`` / ``rejection`` / ``clarification``
    is set, per ``outcome``. ``agentic`` carries the decomposition and per-pass
    telemetry for every outcome except an out-of-scope rejection (which runs
    before planning). ``thread_id`` identifies the paused run to resume it, and is
    carried on every response so a client never has to track it itself.
    ``citation_verdicts`` is the per-verdict count for this run.
    """

    outcome: Outcome
    thread_id: str
    answer: Answer | None = None
    abstention: Abstention | None = None
    rejection: RouterRejection | None = None
    clarification: Clarification | None = None
    agentic: AgenticTrace | None = None
    citation_verdicts: dict[str, int] = {}


class ResumeRequest(BaseModel):
    """A user's reply to the disambiguating question, against the paused run.

    A blank ``answer`` is a legitimate "I would rather not say": the run resumes
    with the branch's assumption stated explicitly instead of an answer.
    """

    thread_id: str
    answer: str = ""

    @field_validator("thread_id")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        """Reject an empty thread id at the API boundary."""
        if not value.strip():
            raise ValueError("thread_id must not be blank")
        return value
