"""The library-backed extractor over real PDF and Word bytes."""

import pytest

from lexme.mode2 import PdfWordExtractor, UnsupportedDocumentError
from tests.mode2.conftest import build_blank_pdf, build_docx, build_text_pdf


def test_a_digital_pdf_yields_its_text_layer() -> None:
    pdf = build_text_pdf(["PRIMERA. El plazo sera de cinco anos.", "SEGUNDA. Renta mensual."])

    extracted = PdfWordExtractor().extract("contrato.pdf", pdf)

    assert "cinco anos" in extracted.text
    assert extracted.page_count == 1


def test_a_word_document_yields_its_paragraph_text() -> None:
    docx = build_docx(["PRIMERA. Duración de cinco años.", "SEGUNDA. Renta de 800 euros."])

    extracted = PdfWordExtractor().extract("contrato.docx", docx)

    assert "cinco años" in extracted.text
    assert "800 euros" in extracted.text


def test_a_scanned_pdf_with_no_text_layer_yields_empty_text() -> None:
    scan = build_blank_pdf()

    extracted = PdfWordExtractor().extract("escaneado.pdf", scan)

    assert extracted.text.strip() == ""


def test_an_unknown_extension_is_unsupported() -> None:
    with pytest.raises(UnsupportedDocumentError):
        PdfWordExtractor().extract("contrato.txt", b"texto plano")


def test_bytes_that_are_not_a_pdf_are_unsupported() -> None:
    with pytest.raises(UnsupportedDocumentError):
        PdfWordExtractor().extract("contrato.pdf", b"this is not a pdf")
