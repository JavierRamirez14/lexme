"""Immutable domain types mirroring the BOE consolidated-law structure.

A norm (a law) is made of blocks (articles), and each block carries the full
history of its versions (redactions), each tagged with the date it took effect.
"""

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class NormMetadata:
    """Governance metadata for a consolidated norm.

    ``updated_at`` is the BOE ``fecha_actualizacion`` and drives incremental
    re-ingestion: a norm is only reprocessed when this timestamp, or the manifest
    selection it was ingested under, changes.
    """

    norm_id: str
    eli: str
    title: str
    consolidated_html_url: str
    updated_at: datetime


@dataclass(frozen=True)
class BlockVersion:
    """One historical redaction of a block.

    ``text_content`` is the article's own prose (used for embedding); it excludes
    the footnote blockquotes that describe which later norm amended the article.
    ``html_content`` keeps the original markup for faithful citation rendering.
    """

    amending_norm_id: str
    publication_date: date
    effective_date: date
    text_content: str
    html_content: str


@dataclass(frozen=True)
class Block:
    """A citable unit of a norm (typically one article), with its full history."""

    block_id: str
    title: str
    versions: tuple[BlockVersion, ...]


@dataclass(frozen=True)
class ConsolidatedNorm:
    """A norm's metadata together with every precepto block it contains."""

    metadata: NormMetadata
    blocks: tuple[Block, ...]
