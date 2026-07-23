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
from collections.abc import Iterator
from datetime import date

import pytest
from docx import Document
from pypdf import PdfWriter

from lexme.api.dependencies import (
    get_llm_client,
    get_scope,
    get_text_extractor,
    get_today,
)
from lexme.llm import FakeLlmClient
from lexme.main import app
from lexme.mode2 import ExtractedText, ScopePackage, TenancyUse
from lexme.mode2.segmentation import ProposedClause, SegmentationProposal
from lexme.mode2.triage import TriageResult

TODAY = date(2024, 6, 1)
CURRENT_REDACTION_FROM = date(2019, 3, 6)


@pytest.fixture
def scope() -> ScopePackage:
    """The vivienda scope rules: art 4.2 exclusions and the current-redaction edge."""
    return ScopePackage(
        current_redaction_effective_from=CURRENT_REDACTION_FROM,
        excluded_uses=frozenset({TenancyUse.SEASONAL, TenancyUse.NON_DWELLING}),
    )


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
    )


@pytest.fixture
def contract_server(
    fake_llm: FakeLlmClient,
    scope: ScopePackage,
    live_server: str,
) -> Iterator[tuple[str, FakeLlmClient, FakeExtractor]]:
    """Override the endpoint's Mode 2 seams and yield the URL, fake LLM and extractor.

    Overriding the leaf dependencies (not the bundle) lets ``get_mode2_deps``
    compose the fakes exactly as it composes the real collaborators in production.
    """
    extractor = FakeExtractor()
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_scope] = lambda: scope
    app.dependency_overrides[get_text_extractor] = lambda: extractor
    app.dependency_overrides[get_today] = lambda: TODAY
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

    Only the LLM, scope and clock are faked, so a posted PDF or ``.docx`` travels
    through the genuine library-backed extraction over HTTP.
    """
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_scope] = lambda: scope
    app.dependency_overrides[get_today] = lambda: TODAY
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
