"""Parse a BOE consolidated-law XML document into domain types.

The ``/id/{id}`` endpoint returns the norm's metadata plus its full text with
every historical version of every block. This module keeps only the ``precepto``
blocks (the legal articles) and turns them into a :class:`ConsolidatedNorm`. A
caller ingesting part of a norm names the blocks it wants, and the rest are never
parsed: a malformed version in an article the vertical does not use -- the BOE has
a few -- must not be able to fail the whole ingest.
"""

import logging
import xml.etree.ElementTree as ET
from collections.abc import Collection
from datetime import UTC, date, datetime

from lexme.ingestion.models import Block, BlockVersion, ConsolidatedNorm, NormMetadata

logger = logging.getLogger(__name__)

PRECEPTO_TYPE = "precepto"
UPDATED_AT_FORMAT = "%Y%m%dT%H%M%SZ"
BOE_DATE_FORMAT = "%Y%m%d"


class BoeXmlError(ValueError):
    """Raised when a BOE XML document is missing a field the corpus requires."""


def parse_norm_xml(xml_text: str, keep_blocks: Collection[str] | None = None) -> ConsolidatedNorm:
    """Parse the XML of a consolidated norm into a :class:`ConsolidatedNorm`.

    Only ``tipo="precepto"`` blocks are kept; preambles, headers and signatures
    are dropped. ``keep_blocks`` narrows that further to the named block ids, in
    document order, and blocks outside it are not parsed at all. Raises
    :class:`BoeXmlError` if a parsed block is missing a required field, or if
    nothing survives.
    """
    root = ET.fromstring(xml_text.encode("utf-8"))
    data = _require(root, "data")
    metadata = _parse_metadata(_require(data, "metadatos"))
    texto = _require(data, "texto")
    blocks = tuple(
        _parse_block(block)
        for block in texto.findall("bloque")
        if block.get("tipo") == PRECEPTO_TYPE
        and (keep_blocks is None or block.get("id") in keep_blocks)
    )
    if not blocks:
        raise BoeXmlError(f"norm {metadata.norm_id} has no precepto blocks to ingest")
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
    versions = _one_version_per_date(
        [_parse_version(version, block_id) for version in block.findall("version")], block_id
    )
    if not versions:
        raise BoeXmlError(f"bloque {block_id} has no versions")
    return Block(block_id=block_id, title=title, versions=versions)


def _one_version_per_date(versions: list[BlockVersion], block_id: str) -> tuple[BlockVersion, ...]:
    """Collapse redactions that share an effective date, keeping the last listed.

    A block has exactly one redaction in force at any date -- that is what makes
    the point-in-time query well defined -- but the BOE occasionally publishes the
    same amendment twice with the same ``fecha_vigencia`` and a hair's difference in
    text. The document order is chronological, so the last one listed is the text
    that stands.
    """
    by_date: dict[date, BlockVersion] = {}
    for version in versions:
        if version.effective_date in by_date:
            logger.warning(
                "bloque %s has two redactions in force on %s; keeping the last listed",
                block_id,
                version.effective_date.isoformat(),
            )
        by_date[version.effective_date] = version
    return tuple(by_date.values())


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
