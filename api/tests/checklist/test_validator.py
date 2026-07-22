"""The validator flags any anchor that does not resolve or citation that does not verify."""

from datetime import date

from lexme.checklist import (
    Checklist,
    ChecklistCitation,
    ChecklistItem,
    RuleCharacter,
    SilenceTone,
    validate_checklist,
)
from tests.checklist.conftest import CORPUS_EFFECTIVE_DATE, TARGET_DATE, FakeCorpusReader

NORM_ID = "BOE-A-1994-26003"
_BLOCK_TEXT = "El arrendador está obligado a realizar todas las reparaciones necesarias."
_CITATION_TEXT = "El arrendador está obligado a realizar todas las reparaciones necesarias"


def _item(
    block_id: str, citation_text: str, *, anchors: tuple[str, ...] | None = None
) -> ChecklistItem:
    """A single-anchor checklist item citing ``citation_text`` from ``block_id``."""
    return ChecklistItem(
        id="CHK-13",
        right="Reparaciones a cargo del arrendador.",
        anchors=anchors or (block_id,),
        character=RuleCharacter.IMPERATIVE,
        silence_tone=SilenceTone.EX_LEGE_WITH_BURDEN,
        absence_template="La ley obliga al arrendador a conservar la vivienda habitable.",
        citation=ChecklistCitation(block_id=block_id, text=citation_text),
    )


def _checklist(*items: ChecklistItem) -> Checklist:
    """Bundle ``items`` into a checklist against the LAU norm."""
    return Checklist(vertical="vivienda", norm_id=NORM_ID, items=items)


def test_passes_when_every_anchor_resolves_and_citation_verifies(
    corpus: FakeCorpusReader,
) -> None:
    corpus.add_block(NORM_ID, "a21", _BLOCK_TEXT, effective_date=CORPUS_EFFECTIVE_DATE)

    report = validate_checklist(_checklist(_item("a21", _CITATION_TEXT)), corpus, TARGET_DATE)

    assert report.is_valid
    assert report.findings == ()


def test_flags_an_anchor_that_does_not_resolve(corpus: FakeCorpusReader) -> None:
    item = _item("a21", _CITATION_TEXT, anchors=("a21", "a6"))
    corpus.add_block(NORM_ID, "a21", _BLOCK_TEXT, effective_date=CORPUS_EFFECTIVE_DATE)

    report = validate_checklist(_checklist(item), corpus, TARGET_DATE)

    assert not report.is_valid
    (finding,) = report.findings
    assert finding.block_id == "a6"
    assert "does not resolve" in finding.message


def test_flags_a_citation_that_does_not_verify(corpus: FakeCorpusReader) -> None:
    corpus.add_block(NORM_ID, "a21", _BLOCK_TEXT, effective_date=CORPUS_EFFECTIVE_DATE)
    corrupt = _item("a21", "texto que no aparece en el articulo veintiuno de la ley")

    report = validate_checklist(_checklist(corrupt), corpus, TARGET_DATE)

    assert not report.is_valid
    (finding,) = report.findings
    assert finding.item_id == "CHK-13"
    assert "does not verify" in finding.message


def test_flags_a_redaction_not_yet_in_force_at_the_target_date(
    corpus: FakeCorpusReader,
) -> None:
    corpus.add_block(NORM_ID, "a21", _BLOCK_TEXT, effective_date=date(2030, 1, 1))

    report = validate_checklist(_checklist(_item("a21", _CITATION_TEXT)), corpus, TARGET_DATE)

    assert not report.is_valid
