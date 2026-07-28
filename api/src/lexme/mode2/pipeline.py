"""Mode 2's orchestration: bytes in, one honest contract analysis out.

The flow is linear and honest to the gate: extract the text (a scan with no text
layer stops here), segment the clauses and anchor each to a literal document span
(a broken anchor stops here), triage the ficha, and let the two code gates decide
scope and time -- the scope gate reads the document text itself, because it closes
only on a span the document really declares an excluded use in. Only the good side
of the gates assembles the sheet, the
deterministic summary and the anchored clauses. The document is held in memory for
the length of one call and never persisted -- there is nowhere in this module that
writes it down.
"""

from dataclasses import dataclass
from datetime import date

from lexme.checklist import Checklist
from lexme.llm import LlmClient
from lexme.mode1.dates import parse_target_date
from lexme.mode2.anchor import AnchoredSpan, anchor_clauses
from lexme.mode2.extraction import TextExtractor, UnsupportedDocumentError
from lexme.mode2.gates import apply_gates
from lexme.mode2.models import (
    Clause,
    ContractAnalysis,
    ContractSheet,
    Mode2Outcome,
    Rejection,
    RejectionReason,
)
from lexme.mode2.retrieval import ClauseRetriever
from lexme.mode2.riskmap import build_risk_map
from lexme.mode2.scope import ScopePackage
from lexme.mode2.segmentation import SegmentationProposal, segment_document
from lexme.mode2.summary import build_summary
from lexme.mode2.triage import TriageResult, triage_document
from lexme.verification import CorpusReader

MIN_ANALYZABLE_CHARS = 40

_REJECTION_MESSAGES: dict[RejectionReason, str] = {
    RejectionReason.UNSUPPORTED_FORMAT: (
        "No puedo leer este archivo. Solo analizo contratos en PDF con texto o en Word (.docx)."
    ),
    RejectionReason.NOT_EXTRACTABLE: (
        "No he podido extraer texto del documento. Si es un escaneado o una foto, "
        "no puedo leerlo con garantías; sube el contrato en PDF con texto o en Word."
    ),
    RejectionReason.BROKEN_ANCHOR: (
        "No he podido anclar las cláusulas al texto literal del documento, así que "
        "no puedo analizarlo sin arriesgarme a citar algo que no dice."
    ),
    RejectionReason.OUT_OF_SCOPE_USE: (
        "Este contrato queda fuera de lo que puedo analizar: no es un arrendamiento "
        "de vivienda habitual sujeto a la Ley de Arrendamientos Urbanos (art. 4.2)."
    ),
    RejectionReason.PRIOR_REDACTION: (
        "Este contrato se firmó bajo una redacción anterior de la Ley de "
        "Arrendamientos Urbanos. Solo puedo analizar contratos regidos por la "
        "redacción vigente."
    ),
}


@dataclass(frozen=True)
class Mode2Deps:
    """The collaborators one contract analysis is wired to.

    ``extractor`` reads the document's text and ``llm`` runs triage, segmentation
    and classification; ``scope`` and ``checklist`` carry the vertical's gate rules
    and legal defaults; ``corpus`` resolves the norm behind each citation and
    ``retriever`` grounds clauses the checklist does not index. Every one is a port
    or data, so a test drives the whole pipeline -- risk map included -- with fakes
    and no network. ``vertical`` names which package the retrieval filters to.
    """

    llm: LlmClient
    extractor: TextExtractor
    scope: ScopePackage
    checklist: Checklist
    corpus: CorpusReader
    retriever: ClauseRetriever
    vertical: str


def analyze_contract(
    *,
    filename: str,
    content: bytes,
    deps: Mode2Deps,
    today: date,
) -> ContractAnalysis:
    """Analyze an uploaded contract, or stop honestly at the first gate it fails.

    Returns an ``ANALYZED`` result -- ficha, executive summary and code-anchored
    clauses -- for a readable, in-scope, current lease; otherwise a ``NOT_ANALYZABLE``
    result (unreadable format, no extractable text, or clauses that will not anchor)
    or an ``OUT_OF_SCOPE`` result (an art 4.2 use, or a superseded redaction).
    """
    extracted_text = _extract(deps.extractor, filename, content)
    if extracted_text is None:
        return _reject(RejectionReason.UNSUPPORTED_FORMAT)
    if len(extracted_text.strip()) < MIN_ANALYZABLE_CHARS:
        return _reject(RejectionReason.NOT_EXTRACTABLE)

    proposal = segment_document(deps.llm, extracted_text)
    spans = _anchor(extracted_text, proposal)
    if spans is None:
        return _reject(RejectionReason.BROKEN_ANCHOR)
    clauses = _build_clauses(proposal, spans)

    triage = triage_document(deps.llm, extracted_text)
    gate = apply_gates(
        use=triage.uso,
        use_evidence=triage.uso_evidencia,
        document_text=extracted_text,
        signing_date=parse_target_date(triage.fecha_firma),
        scope=deps.scope,
        today=today,
    )
    if gate.rejection is not None:
        return _reject(gate.rejection, assumptions=list(gate.assumptions))

    risk_map = build_risk_map(
        clauses,
        extracted_text,
        llm=deps.llm,
        corpus=deps.corpus,
        retriever=deps.retriever,
        checklist=deps.checklist,
        vertical=deps.vertical,
        target_date=gate.target_date,
    )
    sheet = _build_sheet(triage)
    return ContractAnalysis(
        outcome=Mode2Outcome.ANALYZED,
        sheet=sheet,
        summary=build_summary(sheet, len(clauses)),
        clauses=clauses,
        risk_map=risk_map,
        assumptions=list(gate.assumptions),
    )


def _extract(extractor: TextExtractor, filename: str, content: bytes) -> str | None:
    """Extract the document's text, or ``None`` when its format is unreadable."""
    try:
        return extractor.extract(filename, content).text
    except UnsupportedDocumentError:
        return None


def _anchor(document_text: str, proposal: SegmentationProposal) -> list[AnchoredSpan] | None:
    """Anchor every proposed clause to a literal span, or ``None`` if any fails.

    A proposal with no clauses cannot be anchored into anything, so it is treated
    as a broken anchor rather than an empty success.
    """
    if not proposal.clauses:
        return None
    return anchor_clauses(document_text, [clause.text for clause in proposal.clauses])


def _build_clauses(proposal: SegmentationProposal, spans: list[AnchoredSpan]) -> list[Clause]:
    """Pair each proposed heading with the literal span code anchored it to."""
    return [
        Clause(
            id=f"c{index}",
            heading=clause.heading,
            text=span.text,
            start=span.start,
            end=span.end,
        )
        for index, (clause, span) in enumerate(zip(proposal.clauses, spans, strict=True), start=1)
    ]


def _build_sheet(triage: TriageResult) -> ContractSheet:
    """Project the triage result onto the user-facing ficha."""
    return ContractSheet(
        arrendador=triage.arrendador,
        arrendatario=triage.arrendatario,
        inmueble=triage.inmueble,
        renta=triage.renta,
        duracion=triage.duracion,
        fianza=triage.fianza,
        fecha_firma=triage.fecha_firma,
    )


def _reject(reason: RejectionReason, *, assumptions: list[str] | None = None) -> ContractAnalysis:
    """Build the honest-stop result for a rejection reason and any stated assumptions."""
    outcome = (
        Mode2Outcome.OUT_OF_SCOPE
        if reason in (RejectionReason.OUT_OF_SCOPE_USE, RejectionReason.PRIOR_REDACTION)
        else Mode2Outcome.NOT_ANALYZABLE
    )
    return ContractAnalysis(
        outcome=outcome,
        rejection=Rejection(reason=reason, message=_REJECTION_MESSAGES[reason]),
        assumptions=assumptions or [],
    )
