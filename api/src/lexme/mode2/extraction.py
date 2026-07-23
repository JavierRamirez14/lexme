"""Deterministic text extraction from a born-digital PDF or a Word document.

A lease reaches the pipeline as bytes; only its text is analyzable. A born-digital
PDF and a ``.docx`` carry an extractable text layer, so code reads it without a
model. A scan or a photo carries none: the extractor returns whatever little text
is there and the pipeline rejects an empty extraction as ``no_analizable`` -- an
honest "I cannot read this", never a guess over pixels. The bytes are read in
memory and never written anywhere.
"""

import io
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from pypdf import PdfReader
from pypdf.errors import PyPdfError

PDF_EXTENSIONS = (".pdf",)
WORD_EXTENSIONS = (".docx",)


class UnsupportedDocumentError(ValueError):
    """Raised when a document's format is not one the extractor can read.

    Signals a boundary problem -- an unknown extension or bytes that are not the
    document they claim to be -- which the pipeline turns into a ``no_analizable``
    outcome rather than propagating.
    """


@dataclass(frozen=True)
class ExtractedText:
    """The text pulled from a document, with how many pages it spanned."""

    text: str
    page_count: int


@runtime_checkable
class TextExtractor(Protocol):
    """The port the pipeline reads a document's text through.

    Tests substitute a fake returning known text (or a near-empty extraction for a
    scan); :class:`PdfWordExtractor` is the real, library-backed one.
    """

    def extract(self, filename: str, content: bytes) -> ExtractedText:
        """Extract the text of ``content``, using ``filename`` to pick the format.

        Raises :class:`UnsupportedDocumentError` when the format is unknown or the
        bytes cannot be opened as that format.
        """
        ...


class PdfWordExtractor:
    """A :class:`TextExtractor` over born-digital PDFs and ``.docx`` documents.

    The extension picks the reader. Neither reader performs OCR, so a scanned PDF
    yields little or no text -- which is the signal the pipeline reads as
    unreadable, not an error to hide.
    """

    def extract(self, filename: str, content: bytes) -> ExtractedText:
        """Extract text from a PDF or Word document by its filename extension."""
        lowered = filename.lower()
        if lowered.endswith(PDF_EXTENSIONS):
            return self._extract_pdf(content)
        if lowered.endswith(WORD_EXTENSIONS):
            return self._extract_word(content)
        raise UnsupportedDocumentError(
            f"unsupported document '{filename}': only PDF and .docx are read"
        )

    def _extract_pdf(self, content: bytes) -> ExtractedText:
        """Read a PDF's embedded text layer page by page."""
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = list(reader.pages)
        except (PyPdfError, ValueError) as error:
            raise UnsupportedDocumentError(f"cannot open bytes as a PDF: {error}") from error
        text = "\n".join(page.extract_text() or "" for page in pages)
        return ExtractedText(text=text, page_count=len(pages))

    def _extract_word(self, content: bytes) -> ExtractedText:
        """Read a ``.docx`` document's paragraph text."""
        try:
            document = Document(io.BytesIO(content))
        except (PackageNotFoundError, ValueError) as error:
            raise UnsupportedDocumentError(f"cannot open bytes as a .docx: {error}") from error
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return ExtractedText(text=text, page_count=1)
