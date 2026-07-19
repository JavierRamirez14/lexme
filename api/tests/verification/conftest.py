"""An in-memory corpus reader: the high seam the verifier tests inject at.

Tests program blocks and their version history here and feed proposed citations
(including corrupt ones) straight into the verifier -- no database, no model, no
network. The reader answers point-in-time exactly as the SQL repository does:
the redaction with the latest effective date not past the target date.
"""

from dataclasses import dataclass
from datetime import date

import pytest

from lexme.verification.models import ResolvedBlock, VerifiedAnchor


@dataclass(frozen=True)
class _Redaction:
    """One version of a block: the date it took effect and its prose."""

    effective_date: date
    text: str


@dataclass
class _BlockRecord:
    """A block's anchor fields plus its ordered redaction history."""

    eli: str
    consolidated_html_url: str
    title: str
    redactions: list[_Redaction]


class FakeCorpusReader:
    """A network-free :class:`~lexme.verification.models.CorpusReader` for tests."""

    def __init__(self) -> None:
        self._blocks: dict[tuple[str, str], _BlockRecord] = {}

    def add_block(
        self,
        norm_id: str,
        block_id: str,
        *,
        versions: list[tuple[date, str]],
        eli: str = "https://www.boe.es/eli/es/l/1994/11/24/29",
        url: str = "https://www.boe.es/buscar/act.php?id=BOE-A-1994-26003",
        title: str | None = None,
    ) -> None:
        """Register a block with one or more (effective_date, text) redactions."""
        self._blocks[(norm_id, block_id)] = _BlockRecord(
            eli=eli,
            consolidated_html_url=url,
            title=title or f"Artículo {block_id}",
            redactions=[_Redaction(effective, text) for effective, text in versions],
        )

    def resolve_block(self, norm_id: str, block_id: str, target_date: date) -> ResolvedBlock | None:
        """Return the redaction in force at ``target_date`` with its anchor, or ``None``."""
        record = self._blocks.get((norm_id, block_id))
        if record is None:
            return None
        in_force = [r for r in record.redactions if r.effective_date <= target_date]
        if not in_force:
            return None
        redaction = max(in_force, key=lambda r: r.effective_date)
        return ResolvedBlock(
            text=redaction.text,
            anchor=VerifiedAnchor(
                eli=record.eli,
                consolidated_html_url=record.consolidated_html_url,
                block_id=block_id,
                title=record.title,
                effective_date=redaction.effective_date,
            ),
        )


@pytest.fixture
def corpus() -> FakeCorpusReader:
    """An empty in-memory corpus for a test to populate."""
    return FakeCorpusReader()
