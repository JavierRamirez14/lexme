"""The corpus-backed :class:`CorpusReader`, wrapping the SQL repository.

It composes the point-in-time redaction (text and effective date) with the norm
anchor (ELI, URL, title) into a single :class:`ResolvedBlock`, so the verifier
gets both resolution and anchor hydration through one call.
"""

from datetime import date

import psycopg

from lexme.ingestion import repository
from lexme.verification.models import ResolvedBlock, VerifiedAnchor


class PsycopgCorpusReader:
    """A :class:`~lexme.verification.models.CorpusReader` over a Postgres connection."""

    def __init__(self, connection: psycopg.Connection) -> None:
        """Bind the reader to an open corpus connection."""
        self._connection = connection

    def resolve_block(self, norm_id: str, block_id: str, target_date: date) -> ResolvedBlock | None:
        """Resolve the block at ``target_date`` and hydrate its anchor, or ``None``.

        The prose (``text_content``) is what citations quote, so it is what the
        verifier matches against.
        """
        version = repository.get_version_in_force(self._connection, norm_id, block_id, target_date)
        if version is None:
            return None
        anchor = repository.get_citation_anchor(self._connection, norm_id, block_id)
        if anchor is None:
            return None
        return ResolvedBlock(
            text=version.text_content,
            anchor=VerifiedAnchor(
                eli=anchor.eli,
                consolidated_html_url=anchor.consolidated_html_url,
                block_id=anchor.block_id,
                title=anchor.title,
                effective_date=version.effective_date,
            ),
        )
