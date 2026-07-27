"""State types for citation verification: the verdict enum, the citation the
synthesizer emits, the corpus port and the per-citation result.

The synthesizer only ever emits ``(block_ref, text)``; every anchor field (norm,
ELI, URL, effective date, title) is hydrated here by code from the corpus, so an
anchor can never contradict the block it points at. The reference is norm
qualified because the corpus holds several norms and a bare block id would leave
"article 9" ambiguous between them.
"""

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class CitationVerdict(StrEnum):
    """The closed set of outcomes for a citation -- the single source of citation metrics.

    Every citation ends at exactly one of these. The string values are the
    telemetry contract read by the SSE stream and the eval harness.
    """

    VERIFIED_DIRECT = "verificada_directa"
    REPAIRED_SNAP = "reparada_snap"
    REPAIRED_ANCHOR = "reparada_anclaje"
    DISCARDED = "descartada"


class ProposedCitation(BaseModel):
    """A citation as the synthesizer emits it: a block reference and literal text.

    ``block_ref`` is the ``norm:block`` token of a block retrieved as evidence in
    this run, copied from the evidence rather than assembled: one token cannot be
    mismatched the way a separate norm and block id could. The model never emits
    the anchor; code hydrates it from the corpus.
    """

    block_ref: str
    text: str


class VerifiedAnchor(BaseModel):
    """A citation's anchor, hydrated by code from the corpus and point-in-time state.

    ``norm_id`` and ``norm_label`` name the law the block belongs to, so a reader
    can tell two norms' article 9 apart and follow the right consolidated text.
    """

    norm_id: str
    norm_label: str
    eli: str
    consolidated_html_url: str
    block_id: str
    title: str
    effective_date: date


class CitationResult(BaseModel):
    """The outcome of verifying one citation.

    ``block_ref`` echoes the reference the citation ended up anchored to -- the one
    proposed, or the one it was re-anchored to. ``text`` is the literal text to
    show: the verified quote, or -- after a repair -- the real corpus text it was
    snapped to. ``anchor`` is present for every non-discarded verdict.
    ``snap_similarity`` is set only for repairs.
    """

    verdict: CitationVerdict
    block_ref: str
    text: str
    anchor: VerifiedAnchor | None = None
    snap_similarity: float | None = None


@dataclass(frozen=True)
class ResolvedBlock:
    """A block's point-in-time redaction together with its hydrated anchor."""

    text: str
    anchor: VerifiedAnchor


@runtime_checkable
class CorpusReader(Protocol):
    """The corpus port the verifier reads through.

    A single call both resolves the point-in-time redaction and hydrates the
    anchor, so the verifier never assembles an anchor from LLM-supplied fields.
    """

    def resolve_block(self, norm_id: str, block_id: str, target_date: date) -> ResolvedBlock | None:
        """Return the block's redaction in force at ``target_date`` with its anchor.

        Returns ``None`` when the norm/block is unknown or has no redaction in
        force at that date.
        """
        ...
