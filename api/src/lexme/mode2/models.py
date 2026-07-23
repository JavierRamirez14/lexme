"""The Mode 2 contract-analysis contract: outcomes, the sheet, and the clauses.

A tenant uploads a lease and the pipeline reaches one honest verdict. The document
is either analyzed -- returning a code-assembled sheet, a deterministic executive
summary and the list of code-anchored clauses -- or stopped at a gate with the
reason spelled out: outside the LAU's scope, or not analyzable at all. Every gate
is a code decision; the model only proposes clause snippets and extracts raw
fields, and nothing it proposes is trusted until code has checked it.
"""

from enum import StrEnum

from pydantic import BaseModel

from lexme.mode2.risk import RiskMap


class Mode2Outcome(StrEnum):
    """The terminal states a contract analysis can reach.

    ``ANALYZED`` is the good side of both gates; the other two are honest stops,
    kept distinct so the UI never presents a scope rejection as a read failure.
    """

    ANALYZED = "analizado"
    OUT_OF_SCOPE = "fuera_de_ambito"
    NOT_ANALYZABLE = "no_analizable"


class TenancyUse(StrEnum):
    """The lease's use as the triage reads it; code, not the model, gates on it.

    ``UNDETERMINED`` is a legitimate reading of a silent contract: the pipeline
    assumes a habitual dwelling and states that assumption rather than guessing a
    regime.
    """

    HABITUAL_DWELLING = "vivienda_habitual"
    SEASONAL = "temporada"
    NON_DWELLING = "uso_distinto"
    UNDETERMINED = "indeterminado"


class RejectionReason(StrEnum):
    """Why a document did not reach a full analysis -- the closed set of stops.

    The first three end in ``NOT_ANALYZABLE``; the last two, both gate verdicts, end
    in ``OUT_OF_SCOPE``.
    """

    NOT_EXTRACTABLE = "texto_no_extraible"
    BROKEN_ANCHOR = "anclaje_roto"
    UNSUPPORTED_FORMAT = "formato_no_soportado"
    OUT_OF_SCOPE_USE = "uso_fuera_de_ambito"
    PRIOR_REDACTION = "redaccion_anterior"


class ContractSheet(BaseModel):
    """The contract's key facts as the triage extracts them: its photo.

    Every field is raw text lifted from the document; an absent field is an empty
    string rather than an invented value. ``fecha_firma`` is the signing date as
    written, parsed into a real date by code downstream, never here.
    """

    arrendador: str
    arrendatario: str
    inmueble: str
    renta: str
    duracion: str
    fianza: str
    fecha_firma: str = ""


class Clause(BaseModel):
    """One segmented clause anchored by code to a literal span of the document.

    ``text`` is the real document span the clause was validated against, not the
    model's proposal; ``start`` and ``end`` are its half-open character range in the
    normalized document. Ticket 10 fills the risk map onto these clauses.
    """

    id: str
    heading: str
    text: str
    start: int
    end: int


class SummaryItem(BaseModel):
    """One labelled highlight of the executive summary."""

    label: str
    value: str


class ExecutiveSummary(BaseModel):
    """The executive summary, assembled from the sheet by code with no extra call.

    ``items`` are the non-empty highlights of the sheet; ``clause_count`` is how
    many clauses the segmentation anchored, so the summary states the scaffolding
    the risk map will later populate.
    """

    headline: str
    items: list[SummaryItem]
    clause_count: int


class Rejection(BaseModel):
    """An honest stop: the machine reason plus a user-facing explanation."""

    reason: RejectionReason
    message: str


class ContractAnalysis(BaseModel):
    """The full Mode 2 result: the outcome and exactly the payload it implies.

    On ``ANALYZED`` the ``sheet``, ``summary``, ``clauses`` and ``risk_map`` are
    set and ``rejection`` is ``None``; on either stop only ``rejection`` is set.
    Both hold ``assumptions`` -- the choices code made out loud (a silent use
    assumed a dwelling, a missing signing date assumed today) so no gate verdict
    rests on a hidden guess.
    """

    outcome: Mode2Outcome
    sheet: ContractSheet | None = None
    summary: ExecutiveSummary | None = None
    clauses: list[Clause] = []
    risk_map: RiskMap | None = None
    rejection: Rejection | None = None
    assumptions: list[str] = []
