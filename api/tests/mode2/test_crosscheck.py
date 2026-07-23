"""The checklist × contract cross-check in isolation: pure code, no model.

These pin the code that turns a contract's silence into absence whites: an untouched
right becomes an absence finding carrying its static template and tone, a touched
right produces none, and the closing meta-rule is never an omission. The citation
anchor is hydrated from the corpus for display.
"""

from datetime import date

from lexme.checklist import RuleCharacter, SilenceTone
from lexme.mode2.crosscheck import cross_check
from lexme.mode2.risk import ClauseFinding, CoverageStatus, RiskLevel
from tests.mode2.conftest import (
    FakeCorpus,
    checklist_item,
    checklist_of,
    resolved_block,
)

TARGET_DATE = date(2024, 1, 1)

CHK_A = checklist_item(
    "CHK-A",
    "Plazo mínimo",
    "a9",
    "prórroga obligatoria",
    silence_tone=SilenceTone.EX_LEGE_WITH_BURDEN,
    absence_template="La ley te lo garantiza; avisa con 30 días.",
)
CHK_B = checklist_item(
    "CHK-B",
    "Actualización de renta",
    "a18",
    "en defecto de pacto",
    silence_tone=SilenceTone.FAVORABLE,
)
CHK_META = checklist_item(
    "CHK-META",
    "Nulidad de renuncias",
    "a6",
    "son nulas",
    character=RuleCharacter.IMPERATIVE,
    silence_tone=SilenceTone.NOT_APPLICABLE,
)

CHECKLIST = checklist_of(CHK_A, CHK_B, CHK_META)
CORPUS = FakeCorpus(
    {
        "a9": resolved_block("a9", "El plazo se prorroga obligatoriamente por anualidades."),
        "a18": resolved_block("a18", "En defecto de pacto expreso no se actualiza la renta."),
        "a6": resolved_block("a6", "Son nulas las estipulaciones en perjuicio del inquilino."),
    }
)


def _finding(
    clause_id: str,
    chk_ids: list[str],
    *,
    coverage: CoverageStatus = CoverageStatus.EVALUADA,
    level: RiskLevel | None = RiskLevel.CORRECTO,
) -> ClauseFinding:
    return ClauseFinding(
        clause_id=clause_id,
        heading="H",
        snippet="texto",
        start=0,
        end=5,
        coverage=coverage,
        level=level,
        chk_ids=chk_ids,
    )


def test_a_right_no_clause_touches_becomes_a_white_with_its_template_and_tone() -> None:
    findings = [_finding("c1", ["CHK-A"])]

    absences = cross_check(CHECKLIST, findings, CORPUS, TARGET_DATE)

    assert [a.item_id for a in absences] == ["CHK-B"]
    absence = absences[0]
    assert absence.level is RiskLevel.AUSENTE
    assert absence.silence_tone == SilenceTone.FAVORABLE.value


def test_the_meta_rule_is_never_an_omission() -> None:
    absences = cross_check(CHECKLIST, [], CORPUS, TARGET_DATE)

    assert "CHK-META" not in {a.item_id for a in absences}
    assert {a.item_id for a in absences} == {"CHK-A", "CHK-B"}


def test_the_white_citation_anchor_is_hydrated_from_the_corpus() -> None:
    absences = cross_check(CHECKLIST, [], CORPUS, TARGET_DATE)

    burden = next(a for a in absences if a.item_id == "CHK-A")
    assert burden.citation is not None
    assert burden.citation.anchor.eli
    assert burden.citation.text == "prórroga obligatoria"


def test_an_unresolvable_anchor_yields_a_white_without_a_citation() -> None:
    absences = cross_check(CHECKLIST, [], FakeCorpus(), TARGET_DATE)

    assert all(a.citation is None for a in absences)
    assert {a.item_id for a in absences} == {"CHK-A", "CHK-B"}


def test_an_informative_clause_mapped_to_a_right_does_not_suppress_its_white() -> None:
    findings = [
        _finding("c1", ["CHK-A"], coverage=CoverageStatus.INFORMATIVA, level=None),
    ]

    absences = cross_check(CHECKLIST, findings, CORPUS, TARGET_DATE)

    assert "CHK-A" in {a.item_id for a in absences}


def test_a_right_addressed_only_by_an_inconclusive_clause_yields_no_white() -> None:
    findings = [
        _finding("c1", ["CHK-A"], coverage=CoverageStatus.NO_CONCLUYENTE, level=None),
    ]

    absences = cross_check(CHECKLIST, findings, CORPUS, TARGET_DATE)

    assert "CHK-A" not in {a.item_id for a in absences}
