"""Shared helpers for the reference-set tests: an in-memory corpus and checklists."""

from datetime import date

from lexme.checklist import (
    Checklist,
    ChecklistCitation,
    ChecklistItem,
    RuleCharacter,
    SilenceTone,
)
from lexme.verification.models import ResolvedBlock, VerifiedAnchor

_NORM_ID = "BOE-A-1994-26003"


def make_item(item_id: str, silence: SilenceTone = SilenceTone.FAVORABLE) -> ChecklistItem:
    """Build a minimal checklist item for tests, keyed only by id and silence tone."""
    return ChecklistItem(
        id=item_id,
        right=f"right {item_id}",
        anchors=("a9",),
        character=RuleCharacter.IMPERATIVE,
        silence_tone=silence,
        absence_template="...",
        citation=ChecklistCitation(block_id="a9", text="t"),
    )


def make_checklist(*items: ChecklistItem) -> Checklist:
    """Wrap ``items`` into a checklist bound to the LAU norm id."""
    return Checklist(vertical="vivienda", norm_id=_NORM_ID, items=tuple(items))


class FakeCorpus:
    """An in-memory :class:`~lexme.verification.CorpusReader` keyed by block id.

    Program it with ``{block_id: text}``; every known block resolves to that text
    with a stub anchor, and an unknown block resolves to ``None``.
    """

    def __init__(self, texts: dict[str, str], norm_id: str = "BOE-A-1994-26003") -> None:
        """Build the fake over a ``{block_id: text}`` map for one norm."""
        self._texts = texts
        self._norm_id = norm_id

    def resolve_block(self, norm_id: str, block_id: str, target_date: date) -> ResolvedBlock | None:
        """Resolve ``block_id`` to its programmed text, or ``None`` if unknown."""
        if norm_id != self._norm_id or block_id not in self._texts:
            return None
        return ResolvedBlock(
            text=self._texts[block_id],
            anchor=VerifiedAnchor(
                eli=f"eli/{block_id}",
                consolidated_html_url=f"https://boe.es/{block_id}",
                block_id=block_id,
                title=f"Artículo {block_id}",
                effective_date=date(2023, 1, 1),
            ),
        )
