"""Network-free fixtures for checklist tests: an in-memory corpus and the real path.

The corpus answers point-in-time exactly as the SQL repository does, so the
validator runs against it without a database. Tests seed it from curated LAU
excerpts and feed the shipped checklist through unchanged.
"""

from datetime import date
from pathlib import Path

import pytest

from lexme.verification.models import ResolvedBlock, VerifiedAnchor

REPO_ROOT = Path(__file__).resolve().parents[3]
VIVIENDA_DIR = REPO_ROOT / "verticales" / "vivienda"
CHECKLIST_PATH = VIVIENDA_DIR / "checklist.json"

CORPUS_EFFECTIVE_DATE = date(2023, 5, 25)
TARGET_DATE = date(2025, 1, 1)

_ELI = "https://www.boe.es/eli/es/l/1994/11/24/29"
_URL = "https://www.boe.es/buscar/act.php?id=BOE-A-1994-26003"


class FakeCorpusReader:
    """An in-memory :class:`~lexme.verification.models.CorpusReader` for tests.

    Each block holds one redaction effective from a single date; ``resolve_block``
    returns it for any target date on or after that date, and ``None`` otherwise.
    """

    def __init__(self) -> None:
        self._blocks: dict[tuple[str, str], tuple[date, str]] = {}

    def add_block(self, norm_id: str, block_id: str, text: str, *, effective_date: date) -> None:
        """Register a single redaction for a block."""
        self._blocks[(norm_id, block_id)] = (effective_date, text)

    def resolve_block(self, norm_id: str, block_id: str, target_date: date) -> ResolvedBlock | None:
        """Return the block's redaction in force at ``target_date`` with its anchor, or ``None``."""
        record = self._blocks.get((norm_id, block_id))
        if record is None:
            return None
        effective_date, text = record
        if effective_date > target_date:
            return None
        return ResolvedBlock(
            text=text,
            anchor=VerifiedAnchor(
                norm_id=norm_id,
                norm_label="LAU",
                eli=_ELI,
                consolidated_html_url=_URL,
                block_id=block_id,
                title=f"Artículo {block_id.lstrip('a')}",
                effective_date=effective_date,
            ),
        )


def build_corpus(norm_id: str, excerpts: dict[str, str]) -> FakeCorpusReader:
    """Seed a :class:`FakeCorpusReader` with one redaction per excerpt."""
    corpus = FakeCorpusReader()
    for block_id, text in excerpts.items():
        corpus.add_block(norm_id, block_id, text, effective_date=CORPUS_EFFECTIVE_DATE)
    return corpus


@pytest.fixture
def corpus() -> FakeCorpusReader:
    """An empty in-memory corpus for a test to populate."""
    return FakeCorpusReader()
