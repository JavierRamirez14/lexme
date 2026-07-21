"""The corpus-backed :class:`~lexme.mode1.notices.VersionHistory`.

Kept apart from the notice rules so those stay pure date arithmetic, testable
without a database.
"""

from datetime import date

import psycopg

from lexme.ingestion import repository


class PsycopgVersionHistory:
    """A version history over a Postgres corpus connection."""

    def __init__(self, connection: psycopg.Connection) -> None:
        """Bind the reader to an open corpus connection."""
        self._connection = connection

    def effective_dates(self, norm_id: str, block_id: str) -> list[date]:
        """Return the effective dates of every stored redaction of the block, oldest first."""
        return repository.get_effective_dates(self._connection, norm_id, block_id)
