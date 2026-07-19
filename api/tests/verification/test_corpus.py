"""Tests that PsycopgCorpusReader composes point-in-time text with the anchor.

The SQL is exercised by the repository's own integration tests; here the
repository calls are stubbed to check only the composition -- effective date from
the resolved version, ELI and title from the anchor.
"""

from datetime import date

import pytest

from lexme.ingestion import repository
from lexme.ingestion.repository import CitationAnchor, VersionInForce
from lexme.verification.corpus import PsycopgCorpusReader

TARGET = date(2020, 1, 1)
VERSION = VersionInForce(
    amending_norm_id="BOE-A-2019-1",
    effective_date=date(2019, 3, 6),
    text_content="El plazo mínimo será de cinco años.",
    html_content="<p>El plazo mínimo será de cinco años.</p>",
)
ANCHOR = CitationAnchor(
    eli="https://www.boe.es/eli/es/l/1994/11/24/29",
    consolidated_html_url="https://www.boe.es/buscar/act.php?id=BOE-A-1994-26003",
    block_id="a9",
    title="Artículo 9",
)


def test_resolve_block_merges_version_text_and_anchor_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repository, "get_version_in_force", lambda *_: VERSION)
    monkeypatch.setattr(repository, "get_citation_anchor", lambda *_: ANCHOR)

    resolved = PsycopgCorpusReader(object()).resolve_block("BOE-A-1994-26003", "a9", TARGET)

    assert resolved is not None
    assert resolved.text == "El plazo mínimo será de cinco años."
    assert resolved.anchor.eli == ANCHOR.eli
    assert resolved.anchor.title == "Artículo 9"
    assert resolved.anchor.effective_date == date(2019, 3, 6)


def test_resolve_block_returns_none_without_a_version_in_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repository, "get_version_in_force", lambda *_: None)

    resolved = PsycopgCorpusReader(object()).resolve_block("BOE-A-1994-26003", "a9", TARGET)

    assert resolved is None
