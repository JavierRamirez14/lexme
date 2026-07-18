"""Parse a BOE consolidated-law XML document into domain types.

The ``/id/{id}`` endpoint returns the norm's metadata plus its full text with
every historical version of every block. This module keeps only the ``precepto``
blocks (the legal articles) and turns them into a :class:`ConsolidatedNorm`.
"""

import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime

from lexme.ingestion.models import Block, BlockVersion, ConsolidatedNorm, NormMetadata

PRECEPTO_TYPE = "precepto"
UPDATED_AT_FORMAT = "%Y%m%dT%H%M%SZ"
BOE_DATE_FORMAT = "%Y%m%d"


class BoeXmlError(ValueError):
    """Raised when a BOE XML document is missing a field the corpus requires."""


def parse_norm_xml(xml_text: str) -> ConsolidatedNorm:
    """Parse the XML of a consolidated norm into a :class:`ConsolidatedNorm`.

    Only ``tipo="precepto"`` blocks are kept; preambles, headers and signatures
    are dropped. Raises :class:`BoeXmlError` if any required field is missing.
    """
    root = ET.fromstring(xml_text.encode("utf-8"))
    data = _require(root, "data")
    metadata = _parse_metadata(_require(data, "metadatos"))
    texto = _require(data, "texto")
    blocks = tuple(
        _parse_block(block)
        for block in texto.findall("bloque")
        if block.get("tipo") == PRECEPTO_TYPE
    )
    if not blocks:
        raise BoeXmlError(f"norm {metadata.norm_id} has no precepto blocks")
    return ConsolidatedNorm(metadata=metadata, blocks=blocks)


def _parse_metadata(metadatos: ET.Element) -> NormMetadata:
    """Build :class:`NormMetadata` from the ``<metadatos>`` element."""
    return NormMetadata(
        norm_id=_require_text(metadatos, "identificador"),
        eli=_require_text(metadatos, "url_eli"),
        title=_require_text(metadatos, "titulo"),
        consolidated_html_url=_require_text(metadatos, "url_html_consolidada"),
        updated_at=_parse_updated_at(_require_text(metadatos, "fecha_actualizacion")),
    )


def _parse_block(block: ET.Element) -> Block:
    """Build a :class:`Block` (with all its versions) from a ``<bloque>`` element."""
    block_id = block.get("id")
    title = block.get("titulo")
    if not block_id or not title:
        raise BoeXmlError(f"bloque missing id or titulo: {block.attrib}")
    versions = tuple(_parse_version(version, block_id) for version in block.findall("version"))
    if not versions:
        raise BoeXmlError(f"bloque {block_id} has no versions")
    return Block(block_id=block_id, title=title, versions=versions)


def _parse_version(version: ET.Element, block_id: str) -> BlockVersion:
    """Build a :class:`BlockVersion` from a ``<version>`` element."""
    return BlockVersion(
        amending_norm_id=_require_attr(version, "id_norma", block_id),
        publication_date=_parse_date(_require_attr(version, "fecha_publicacion", block_id)),
        effective_date=_parse_date(_require_attr(version, "fecha_vigencia", block_id)),
        text_content=_extract_text(version),
        html_content=_extract_html(version),
    )


def _extract_text(version: ET.Element) -> str:
    """Plain text of the precepto itself, excluding footnote blockquotes."""
    paragraphs = ("".join(child.itertext()).strip() for child in version if child.tag == "p")
    return "\n".join(paragraph for paragraph in paragraphs if paragraph)


def _extract_html(version: ET.Element) -> str:
    """Original inner markup of the version, concatenated child by child."""
    return "".join(ET.tostring(child, encoding="unicode") for child in version).strip()


def _parse_updated_at(raw: str) -> datetime:
    """Parse ``fecha_actualizacion`` (``YYYYMMDDThhmmssZ``, UTC) into a datetime."""
    try:
        return datetime.strptime(raw, UPDATED_AT_FORMAT).replace(tzinfo=UTC)
    except ValueError as error:
        raise BoeXmlError(f"invalid fecha_actualizacion {raw!r}") from error


def _parse_date(raw: str) -> date:
    """Parse a BOE ``YYYYMMDD`` date string into a :class:`date`."""
    try:
        return datetime.strptime(raw, BOE_DATE_FORMAT).date()
    except ValueError as error:
        raise BoeXmlError(f"invalid BOE date {raw!r}") from error


def _require(parent: ET.Element, tag: str) -> ET.Element:
    """Return the first child with ``tag`` or raise :class:`BoeXmlError`."""
    child = parent.find(tag)
    if child is None:
        raise BoeXmlError(f"missing <{tag}> in <{parent.tag}>")
    return child


def _require_text(parent: ET.Element, tag: str) -> str:
    """Return the non-empty text of ``parent/tag`` or raise :class:`BoeXmlError`."""
    child = _require(parent, tag)
    text = (child.text or "").strip()
    if not text:
        raise BoeXmlError(f"empty <{tag}> in <{parent.tag}>")
    return text


def _require_attr(element: ET.Element, name: str, block_id: str) -> str:
    """Return attribute ``name`` of ``element`` or raise :class:`BoeXmlError`."""
    value = element.get(name)
    if not value:
        raise BoeXmlError(f"version in bloque {block_id} missing attribute {name!r}")
    return value
