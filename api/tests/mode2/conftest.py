"""Fixtures and builders for Mode 2 tests: fakes, scope data and real documents.

The contract pipeline reads through three seams -- the text extractor, the LLM and
the scope package -- so a test programs all three here and lets extraction,
anchoring, the gates and the deterministic summary run for real. :func:`contract_server`
does the same over the HTTP surface, substituting only those seams and the clock;
no database is touched because Mode 2 persists nothing. The document builders emit
real PDF and ``.docx`` bytes so the library-backed extractor can be tested against
genuine files.
"""

import io
from collections.abc import Iterator, Sequence
from datetime import date

import pytest
from docx import Document
from pypdf import PdfWriter

from lexme.api.dependencies import (
    get_clause_retriever,
    get_corpus_reader,
    get_llm_client,
    get_scope,
    get_text_extractor,
    get_today,
)
from lexme.blocks import BlockRef
from lexme.checklist import (
    Checklist,
    ChecklistCitation,
    ChecklistItem,
    RuleCharacter,
    SilenceTone,
)
from lexme.llm import FakeLlmClient
from lexme.main import app
from lexme.mode2 import ExtractedText, ScopePackage, TenancyUse
from lexme.mode2.classify import ClauseClassification
from lexme.mode2.mapping import ClauseMap, DocumentMapping
from lexme.mode2.risk import ProposedClauseLevel
from lexme.mode2.segmentation import ProposedClause, SegmentationProposal
from lexme.mode2.triage import TriageResult
from lexme.retrieval import RetrievedBlock
from lexme.verification import ProposedCitation, ResolvedBlock, VerifiedAnchor

TODAY = date(2024, 6, 1)
CURRENT_REDACTION_FROM = date(2019, 3, 6)
NORM_ID = "BOE-A-1994-26003"


SEASONAL_DECLARATION = "El inmueble se arrienda con finalidad de temporada estival"

USE_EVIDENCE_MARKERS = {
    TenancyUse.SEASONAL: ("temporada", "vacacion", "turistic", "estival"),
    TenancyUse.NON_DWELLING: ("uso distinto del de vivienda", "local comercial", "oficina"),
}


def seasonal_document(body: str) -> str:
    """``body`` prefixed with a heading and a span declaring a seasonal let."""
    return f"CONTRATO DE ARRENDAMIENTO\n{SEASONAL_DECLARATION}.\n{body}"


def scope_package(
    *,
    excluded_uses: frozenset[TenancyUse] = frozenset(
        {TenancyUse.SEASONAL, TenancyUse.NON_DWELLING}
    ),
    current_redaction_effective_from: date = CURRENT_REDACTION_FROM,
) -> ScopePackage:
    """The vivienda scope rules, with the folded markers the real package ships."""
    return ScopePackage(
        current_redaction_effective_from=current_redaction_effective_from,
        excluded_uses=excluded_uses,
        use_evidence_markers=USE_EVIDENCE_MARKERS,
    )


@pytest.fixture
def scope() -> ScopePackage:
    """The vivienda scope rules: art 4.2 exclusions and the current-redaction edge."""
    return scope_package()


class FakeCorpus:
    """A :class:`CorpusReader` over an in-memory ``block_id -> ResolvedBlock`` map."""

    def __init__(self, blocks: dict[str, ResolvedBlock] | None = None) -> None:
        self._blocks = dict(blocks or {})

    def resolve_block(self, norm_id: str, block_id: str, target_date: date) -> ResolvedBlock | None:
        return self._blocks.get(block_id)


class FakeRetriever:
    """A :class:`ClauseRetriever` that returns programmed blocks and records queries."""

    def __init__(self, blocks: Sequence[RetrievedBlock] = ()) -> None:
        self._blocks = list(blocks)
        self.queries: list[str] = []

    def retrieve(
        self, clause_text: str, *, vertical: str, target_date: date
    ) -> list[RetrievedBlock]:
        self.queries.append(clause_text)
        return list(self._blocks)


def resolved_block(block_id: str, text: str, *, title: str = "Artículo") -> ResolvedBlock:
    """A resolved corpus block with a hydrated anchor, for the fake corpus."""
    return ResolvedBlock(
        text=text,
        anchor=VerifiedAnchor(
            norm_id=NORM_ID,
            norm_label="LAU",
            eli=f"https://www.boe.es/eli/es/l/1994/11/24/29/{block_id}",
            consolidated_html_url="https://www.boe.es/buscar/act.php?id=BOE-A-1994-26003",
            block_id=block_id,
            title=title,
            effective_date=date(2023, 5, 26),
        ),
    )


def checklist_item(
    item_id: str,
    right: str,
    block_id: str,
    citation_text: str,
    *,
    character: RuleCharacter = RuleCharacter.IMPERATIVE,
    silence_tone: SilenceTone = SilenceTone.EX_LEGE_INFORMATIVE,
    absence_template: str = "La ley te lo reconoce aunque el contrato calle.",
) -> ChecklistItem:
    """One checklist item wired to a single anchor block and its citation."""
    return ChecklistItem(
        id=item_id,
        right=right,
        anchors=(block_id,),
        character=character,
        silence_tone=silence_tone,
        absence_template=absence_template,
        citation=ChecklistCitation(block_id=block_id, text=citation_text),
    )


def checklist_of(*items: ChecklistItem) -> Checklist:
    """A checklist bound to the vivienda norm from the given items."""
    return Checklist(vertical="vivienda", norm_id=NORM_ID, items=tuple(items))


def mapping_of(*entries: tuple[str, bool, list[str]]) -> DocumentMapping:
    """A document mapping from ``(clause_id, evaluable, chk_ids)`` tuples."""
    return DocumentMapping(
        clauses=[
            ClauseMap(clause_id=clause_id, evaluable=evaluable, chk_ids=chk_ids)
            for clause_id, evaluable, chk_ids in entries
        ]
    )


def classification_of(
    level: ProposedClauseLevel,
    *,
    explanation: str = "Explicación de la cláusula.",
    what_you_can_do: list[str] | None = None,
    citation: tuple[str, str] | None = None,
) -> ClauseClassification:
    """A clause classification, its citation given as ``(block_id, text)`` or omitted."""
    return ClauseClassification(
        level=level,
        explanation=explanation,
        what_you_can_do=what_you_can_do or [],
        citation=(
            ProposedCitation(block_ref=block_ref(citation[0]), text=citation[1])
            if citation
            else None
        ),
    )


def block_ref(block_id: str, norm_id: str = NORM_ID) -> str:
    """The norm-qualified reference the vivienda norm's blocks are cited by."""
    return str(BlockRef(norm_id=norm_id, block_id=block_id))


class FakeExtractor:
    """A :class:`TextExtractor` that returns programmed text or raises on demand.

    Program it with the text an upload should yield; set ``error`` to make every
    extraction raise, standing in for an unreadable format.
    """

    def __init__(self, text: str = "", *, error: Exception | None = None) -> None:
        self.text = text
        self.error = error

    def extract(self, filename: str, content: bytes) -> ExtractedText:
        if self.error is not None:
            raise self.error
        return ExtractedText(text=self.text, page_count=1)


def segmentation_of(*clauses: tuple[str, str]) -> SegmentationProposal:
    """A segmentation proposal from ``(heading, text)`` clause tuples."""
    return SegmentationProposal(
        clauses=[ProposedClause(heading=heading, text=text) for heading, text in clauses]
    )


def triage_of(
    *,
    arrendador: str = "Juan Pérez",
    arrendatario: str = "Ana García",
    inmueble: str = "Calle Mayor 1, Madrid",
    renta: str = "800 euros al mes",
    duracion: str = "cinco años",
    fianza: str = "una mensualidad",
    fecha_firma: str = "",
    uso: TenancyUse = TenancyUse.HABITUAL_DWELLING,
    uso_evidencia: str = "",
) -> TriageResult:
    """A triage result with sensible defaults a test overrides field by field."""
    return TriageResult(
        arrendador=arrendador,
        arrendatario=arrendatario,
        inmueble=inmueble,
        renta=renta,
        duracion=duracion,
        fianza=fianza,
        fecha_firma=fecha_firma,
        uso=uso,
        uso_evidencia=uso_evidencia,
    )


def _override_mode2_seams(
    fake_llm: FakeLlmClient,
    scope: ScopePackage,
    corpus: FakeCorpus,
    retriever: FakeRetriever,
    checklist: Checklist,
) -> None:
    """Point every Mode 2 leaf dependency at a fake, so the bundle composes them."""
    from lexme.api.dependencies import get_checklist

    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_scope] = lambda: scope
    app.dependency_overrides[get_corpus_reader] = lambda: corpus
    app.dependency_overrides[get_clause_retriever] = lambda: retriever
    app.dependency_overrides[get_checklist] = lambda: checklist
    app.dependency_overrides[get_today] = lambda: TODAY


@pytest.fixture
def contract_server(
    fake_llm: FakeLlmClient,
    scope: ScopePackage,
    live_server: str,
) -> Iterator[tuple[str, FakeLlmClient, FakeExtractor]]:
    """Override the endpoint's Mode 2 seams and yield the URL, fake LLM and extractor.

    Overriding the leaf dependencies (not the bundle) lets ``get_mode2_deps``
    compose the fakes exactly as it composes the real collaborators in production.
    The checklist defaults to empty, so a lease that clears the gates reaches an
    empty risk map after one mapping call, keeping the gate and anchor assertions
    focused; the risk-map behaviour is covered by the pipeline-level tests.
    """
    extractor = FakeExtractor()
    _override_mode2_seams(fake_llm, scope, FakeCorpus(), FakeRetriever(), checklist_of())
    app.dependency_overrides[get_text_extractor] = lambda: extractor
    try:
        yield live_server, fake_llm, extractor
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def contract_server_real_extraction(
    fake_llm: FakeLlmClient,
    scope: ScopePackage,
    live_server: str,
) -> Iterator[tuple[str, FakeLlmClient]]:
    """Like :func:`contract_server` but keeps the real PDF/Word extractor.

    Only the LLM, scope, corpus, retriever, checklist and clock are faked, so a
    posted PDF or ``.docx`` travels through the genuine library-backed extraction
    over HTTP.
    """
    _override_mode2_seams(fake_llm, scope, FakeCorpus(), FakeRetriever(), checklist_of())
    try:
        yield live_server, fake_llm
    finally:
        app.dependency_overrides.clear()


def build_text_pdf(lines: list[str]) -> bytes:
    """Build a minimal single-page PDF whose text layer is ``lines``.

    Hand-assembled rather than rendered, so the tests need no PDF toolkit; it is a
    real, extractable born-digital PDF.
    """
    operations = "BT /F1 12 Tf 72 720 Td 14 TL\n"
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        operations += f"({escaped}) Tj T*\n"
    operations += "ET"
    stream = operations.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % index + obj + b"\nendobj\n")
    xref_position = out.tell()
    out.write(b"xref\n0 %d\n" % (len(objects) + 1))
    out.write(b"0000000000 65535 f \n")
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF"
        % (len(objects) + 1, xref_position)
    )
    return out.getvalue()


def build_blank_pdf() -> bytes:
    """Build a single-page PDF with no text layer, standing in for a scan."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def build_docx(paragraphs: list[str]) -> bytes:
    """Build a ``.docx`` document whose paragraphs are ``paragraphs``."""
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()
