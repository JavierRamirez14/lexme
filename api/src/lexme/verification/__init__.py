"""Runtime citation verifier: no cited text reaches the user unverified.

A shared, model-free guardrail for both modes. It re-verifies each citation
against the corpus at the run's point-in-time date by strict equality after
conservative typographic normalization, repairs near-misses only by deterministic
snap to the real corpus text, and hydrates anchors by code. Every citation ends
at exactly one :class:`CitationVerdict`, the single source of citation telemetry.

Callers depend on this package's surface; tests inject any object satisfying
:class:`CorpusReader`. :class:`PsycopgCorpusReader` is the corpus-backed one.
"""

from lexme.verification.corpus import PsycopgCorpusReader
from lexme.verification.models import (
    CitationResult,
    CitationVerdict,
    CorpusReader,
    EvidenceBlock,
    ProposedCitation,
    ResolvedBlock,
    VerifiedAnchor,
)
from lexme.verification.verifier import summarize, verify_citations

__all__ = [
    "CitationResult",
    "CitationVerdict",
    "CorpusReader",
    "EvidenceBlock",
    "ProposedCitation",
    "PsycopgCorpusReader",
    "ResolvedBlock",
    "VerifiedAnchor",
    "summarize",
    "verify_citations",
]
