"""Mode 2: an uploaded lease to a code-anchored analysis or an honest stop.

The pipeline extracts the document's text (a scan with no text layer stops as
``no_analizable``), asks the model to segment the clauses and anchors each to a
literal span of the document in code (a broken anchor stops the same way), triages
the ficha in one call, and lets two code gates decide scope (art 4.2) and time
(only the current redaction). The good side of the gates returns the ficha, a
deterministic executive summary and the anchored clauses; the document is never
persisted. The model proposes; the gates, in code, decide.
"""

from lexme.mode2.anchor import AnchoredSpan, anchor_clauses
from lexme.mode2.extraction import (
    ExtractedText,
    PdfWordExtractor,
    TextExtractor,
    UnsupportedDocumentError,
)
from lexme.mode2.gates import GateOutcome, apply_gates
from lexme.mode2.models import (
    Clause,
    ContractAnalysis,
    ContractSheet,
    ExecutiveSummary,
    Mode2Outcome,
    Rejection,
    RejectionReason,
    SummaryItem,
    TenancyUse,
)
from lexme.mode2.pipeline import Mode2Deps, analyze_contract
from lexme.mode2.scope import ScopeError, ScopePackage, load_scope
from lexme.mode2.segmentation import (
    SEGMENTATION_TASK,
    ProposedClause,
    SegmentationProposal,
    segment_document,
)
from lexme.mode2.summary import build_summary
from lexme.mode2.triage import TRIAGE_TASK, TriageResult, triage_document

__all__ = [
    "SEGMENTATION_TASK",
    "TRIAGE_TASK",
    "AnchoredSpan",
    "Clause",
    "ContractAnalysis",
    "ContractSheet",
    "ExecutiveSummary",
    "ExtractedText",
    "GateOutcome",
    "Mode2Deps",
    "Mode2Outcome",
    "PdfWordExtractor",
    "ProposedClause",
    "Rejection",
    "RejectionReason",
    "ScopeError",
    "ScopePackage",
    "SegmentationProposal",
    "SummaryItem",
    "TenancyUse",
    "TextExtractor",
    "TriageResult",
    "UnsupportedDocumentError",
    "analyze_contract",
    "anchor_clauses",
    "apply_gates",
    "build_summary",
    "load_scope",
    "segment_document",
    "triage_document",
]
