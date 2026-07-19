"""The Mode 1 answer contract: what the synthesizer proposes and what code returns.

The synthesizer emits three layers plus the citations it wants to make; the
verifier and the deterministic gate turn that proposal into either a
:class:`Answer` carrying only verified citations or an :class:`Abstention`. The
outcome is a code decision, never the model's, so the outcome and the abstention
reason are typed here rather than left to prose.
"""

from enum import StrEnum

from pydantic import BaseModel, field_validator

from lexme.verification import CitationVerdict, ProposedCitation, VerifiedAnchor


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
    """The three-layer answer as returned: verified foundation, then plain layers.

    ``fundamento`` never contains a discarded citation. ``explicacion`` and
    ``accion`` are the model's prose, shown only because at least one citation
    held; the action layer is bounded to informing and negotiating.
    """

    fundamento: list[VerifiedCitation]
    explicacion: str
    accion: list[str]


class AbstentionReason(StrEnum):
    """Why code chose to abstain -- the closed set the gate can produce."""

    NO_EVIDENCE = "sin_evidencia"
    NO_VERIFIABLE_CITATION = "sin_cita_verificable"


class Abstention(BaseModel):
    """An honest non-answer: a machine reason plus user-facing message and scope."""

    reason: AbstentionReason
    message: str
    scope_reminder: str


class RankedBlockRef(BaseModel):
    """One block's identity at a position in a ranking, for retrieval telemetry."""

    norm_id: str
    block_id: str


class RetrievalTrace(BaseModel):
    """The dense, lexical and fused rankings behind an answer, in order.

    Exposing the intermediate rankings makes the hybrid retrieval measurable: the
    fused order is visibly a fusion of the two retriever orders.
    """

    dense: list[RankedBlockRef]
    lexical: list[RankedBlockRef]
    fused: list[RankedBlockRef]


class Outcome(StrEnum):
    """The deterministic terminal states of a Mode 1 query."""

    ANSWER = "respuesta"
    ABSTENTION = "abstencion"


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
    """The full Mode 1 result: outcome, one of answer/abstention, and telemetry.

    Exactly one of ``answer`` and ``abstention`` is set, per ``outcome``.
    ``citation_verdicts`` is the per-verdict count for this run.
    """

    outcome: Outcome
    answer: Answer | None = None
    abstention: Abstention | None = None
    retrieval: RetrievalTrace
    citation_verdicts: dict[str, int]
