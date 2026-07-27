"""The risk map assembled end to end, with the LLM, corpus and retriever faked.

These drive :func:`build_risk_map` -- the code that maps, classifies, cross-checks
and counts -- against a small checklist and a fake corpus, so the classification
LLM, the real citation verifier and the pure-code cross-check all run. They cover
the spectrum the ticket asks for: a red with a verified citation, an orange, the
absence whites the silence cross-check produces, the degradation of a discarded
citation, off-checklist clauses through retrieval, full coverage and the absence
of any aggregate verdict.
"""

from datetime import date

from lexme.checklist import RuleCharacter, SilenceTone
from lexme.llm import FakeLlmClient
from lexme.mode2.classify import CLASSIFICATION_TASK
from lexme.mode2.mapping import MAPPING_TASK, DocumentMapping
from lexme.mode2.models import Clause
from lexme.mode2.risk import CoverageStatus, ProposedClauseLevel, RiskLevel
from lexme.mode2.riskmap import build_risk_map
from lexme.retrieval import RetrievedBlock
from tests.mode2.conftest import (
    FakeCorpus,
    FakeRetriever,
    checklist_item,
    checklist_of,
    classification_of,
    mapping_of,
    resolved_block,
)

TARGET_DATE = date(2024, 1, 1)

A9_TEXT = (
    "El plazo del arrendamiento se prorrogará obligatoriamente por plazos anuales "
    "hasta que alcance una duración mínima de cinco años."
)
A25_TEXT = (
    "El arrendatario tendrá derecho de adquisición preferente sobre la vivienda "
    "arrendada en caso de que el arrendador decida venderla."
)
A6_TEXT = (
    "Son nulas, y se tendrán por no puestas, las estipulaciones que modifiquen en "
    "perjuicio del arrendatario las normas del presente Título."
)

CHK_A = checklist_item(
    "CHK-A",
    "Plazo mínimo con prórroga obligatoria",
    "a9",
    "se prorrogará obligatoriamente por plazos anuales",
    character=RuleCharacter.IMPERATIVE,
    silence_tone=SilenceTone.EX_LEGE_WITH_BURDEN,
    absence_template=(
        "La ley te garantiza el plazo mínimo aunque el contrato lo calle. "
        "Ten en cuenta que debes avisar con 30 días de antelación para no renovar."
    ),
)
CHK_B = checklist_item(
    "CHK-B",
    "Derecho de adquisición preferente",
    "a25",
    "derecho de adquisición preferente",
    character=RuleCharacter.DISPOSITIVE,
    silence_tone=SilenceTone.EX_LEGE_INFORMATIVE,
)
CHK_META = checklist_item(
    "CHK-META",
    "Nulidad de renuncias en perjuicio del inquilino",
    "a6",
    "Son nulas, y se tendrán por no puestas",
    silence_tone=SilenceTone.NOT_APPLICABLE,
)

CHECKLIST = checklist_of(CHK_A, CHK_B, CHK_META)
CORPUS = FakeCorpus(
    {
        "a9": resolved_block("a9", A9_TEXT),
        "a25": resolved_block("a25", A25_TEXT),
        "a6": resolved_block("a6", A6_TEXT),
    }
)


def _clause(clause_id: str, heading: str, text: str) -> Clause:
    return Clause(id=clause_id, heading=heading, text=text, start=0, end=len(text))


def _build(
    clauses: list[Clause],
    mapping: DocumentMapping,
    classifications: list,
    *,
    retriever: FakeRetriever | None = None,
    checklist=CHECKLIST,
):
    fake = FakeLlmClient()
    fake.queue(MAPPING_TASK, mapping)
    for classification in classifications:
        fake.queue(CLASSIFICATION_TASK, classification)
    risk_map = build_risk_map(
        clauses,
        "documento completo",
        llm=fake,
        corpus=CORPUS,
        retriever=retriever or FakeRetriever(),
        checklist=checklist,
        vertical="vivienda",
        target_date=TARGET_DATE,
    )
    return risk_map, fake


def test_a_clause_contradicting_an_imperative_norm_is_red_with_a_verified_citation() -> None:
    clauses = [
        _clause("c1", "Duración", "La duración la decide el arrendador cada mes a su antojo.")
    ]
    mapping = mapping_of(("c1", True, ["CHK-A"]))
    classification = classification_of(
        ProposedClauseLevel.ILEGAL,
        citation=("a9", "se prorrogará obligatoriamente por plazos anuales"),
    )

    risk_map, _ = _build(clauses, mapping, [classification])

    finding = risk_map.clause_findings[0]
    assert finding.coverage is CoverageStatus.EVALUADA
    assert finding.level is RiskLevel.ILEGAL
    assert finding.citation is not None
    assert finding.citation.anchor.eli


def test_a_clause_worsening_a_dispositive_default_is_orange_with_a_verified_citation() -> None:
    clauses = [
        _clause("c1", "Venta", "El inquilino renuncia a cualquier derecho de compra preferente.")
    ]
    mapping = mapping_of(("c1", True, ["CHK-B"]))
    classification = classification_of(
        ProposedClauseLevel.PEOR_QUE_DEFAULT,
        citation=("a25", "derecho de adquisición preferente"),
    )

    risk_map, _ = _build(clauses, mapping, [classification])

    finding = risk_map.clause_findings[0]
    assert finding.coverage is CoverageStatus.EVALUADA
    assert finding.level is RiskLevel.PEOR_QUE_DEFAULT
    assert finding.citation is not None


def test_a_right_no_clause_touches_produces_its_white_with_template_and_deadline() -> None:
    clauses = [
        _clause("c1", "Objeto", "El presente contrato tiene por objeto la vivienda descrita.")
    ]
    mapping = mapping_of(("c1", True, []))
    classification = classification_of(ProposedClauseLevel.NEGOCIABLE, citation=None)

    risk_map, _ = _build(clauses, mapping, [classification])

    absent_ids = [finding.item_id for finding in risk_map.absence_findings]
    assert absent_ids == ["CHK-A", "CHK-B"]
    burden = next(f for f in risk_map.absence_findings if f.item_id == "CHK-A")
    assert burden.level is RiskLevel.AUSENTE
    assert "30 días" in burden.explanation
    assert burden.citation is not None
    assert burden.citation.anchor.eli


def test_a_touched_right_produces_no_white() -> None:
    clauses = [_clause("c1", "Duración", "La duración será de tres años prorrogables.")]
    mapping = mapping_of(("c1", True, ["CHK-A"]))
    classification = classification_of(
        ProposedClauseLevel.CORRECTO,
        citation=("a9", "se prorrogará obligatoriamente por plazos anuales"),
    )

    risk_map, _ = _build(clauses, mapping, [classification])

    absent_ids = [finding.item_id for finding in risk_map.absence_findings]
    assert "CHK-A" not in absent_ids


def test_a_discarded_citation_degrades_a_level_to_inconclusive_keeping_the_signal() -> None:
    clauses = [
        _clause("c1", "Duración", "El arrendador puede recuperar la vivienda cuando quiera.")
    ]
    mapping = mapping_of(("c1", True, ["CHK-A"]))
    classification = classification_of(
        ProposedClauseLevel.ILEGAL,
        explanation="Esta cláusula parece dejar la duración al arbitrio del arrendador.",
        citation=("a9", "el arrendador puede recuperar la vivienda cuando quiera"),
    )

    risk_map, _ = _build(clauses, mapping, [classification])

    finding = risk_map.clause_findings[0]
    assert finding.coverage is CoverageStatus.NO_CONCLUYENTE
    assert finding.level is None
    assert finding.citation is None
    assert "arbitrio del arrendador" in finding.explanation


def test_a_negotiable_clause_needs_no_citation_to_stand() -> None:
    clauses = [_clause("c1", "Mascotas", "Queda prohibida la tenencia de animales en la vivienda.")]
    mapping = mapping_of(("c1", True, []))
    classification = classification_of(ProposedClauseLevel.NEGOCIABLE, citation=None)

    risk_map, _ = _build(clauses, mapping, [classification])

    finding = risk_map.clause_findings[0]
    assert finding.coverage is CoverageStatus.EVALUADA
    assert finding.level is RiskLevel.NEGOCIABLE
    assert finding.citation is None


def test_an_off_checklist_clause_governed_elsewhere_is_marked_out_of_scope() -> None:
    clauses = [
        _clause("c1", "Datos", "El arrendatario consiente el tratamiento de sus datos personales.")
    ]
    mapping = mapping_of(("c1", True, []))
    classification = classification_of(
        ProposedClauseLevel.FUERA_DE_AMBITO,
        explanation="La protección de datos se rige por el RGPD, ajeno a la LAU.",
    )

    risk_map, _ = _build(clauses, mapping, [classification])

    finding = risk_map.clause_findings[0]
    assert finding.coverage is CoverageStatus.FUERA_DE_AMBITO
    assert finding.level is None
    assert "RGPD" in finding.out_of_scope_matter


def test_a_clause_off_the_checklist_is_grounded_by_hybrid_retrieval() -> None:
    clauses = [_clause("c1", "Compra", "Si se vende la vivienda podrás comprarla tú primero.")]
    mapping = mapping_of(("c1", True, []))
    classification = classification_of(
        ProposedClauseLevel.CORRECTO,
        citation=("a25", "derecho de adquisición preferente"),
    )
    retriever = FakeRetriever(
        [
            RetrievedBlock(
                norm_id=CHECKLIST.norm_id,
                norm_label="LAU",
                block_id="a25",
                title="Art 25",
                text=A25_TEXT,
                effective_date=TARGET_DATE,
            )
        ]
    )

    risk_map, _ = _build(clauses, mapping, [classification], retriever=retriever)

    finding = risk_map.clause_findings[0]
    assert retriever.queries == ["Si se vende la vivienda podrás comprarla tú primero."]
    assert finding.coverage is CoverageStatus.EVALUADA
    assert finding.level is RiskLevel.CORRECTO
    assert finding.citation is not None


def test_an_informative_clause_is_placed_without_a_model_call() -> None:
    clauses = [_clause("c1", "Partes", "De una parte Juan Pérez y de otra Ana García.")]
    mapping = mapping_of(("c1", False, []))

    risk_map, fake = _build(clauses, mapping, [])

    finding = risk_map.clause_findings[0]
    assert finding.coverage is CoverageStatus.INFORMATIVA
    assert finding.level is None
    assert [call.task for call in fake.calls] == [MAPPING_TASK]


def test_every_clause_appears_in_coverage_and_the_map_carries_no_aggregate_verdict() -> None:
    clauses = [
        _clause("c1", "Duración", "La duración la decide el arrendador a su antojo."),
        _clause("c2", "Partes", "De una parte Juan y de otra Ana."),
        _clause("c3", "Mascotas", "Se prohíbe tener animales."),
    ]
    mapping = mapping_of(("c1", True, ["CHK-A"]), ("c2", False, []), ("c3", True, []))
    classifications = [
        classification_of(
            ProposedClauseLevel.ILEGAL,
            citation=("a9", "se prorrogará obligatoriamente por plazos anuales"),
        ),
        classification_of(ProposedClauseLevel.NEGOCIABLE, citation=None),
    ]

    risk_map, _ = _build(clauses, mapping, classifications)

    assert len(risk_map.clause_findings) == 3
    assert {f.clause_id for f in risk_map.clause_findings} == {"c1", "c2", "c3"}
    assert all(f.coverage in CoverageStatus for f in risk_map.clause_findings)
    assert risk_map.coverage_counts["evaluada"] == 2
    assert risk_map.coverage_counts["informativa"] == 1
    assert risk_map.level_counts["ilegal"] == 1
    assert risk_map.level_counts["ausente"] == len(risk_map.absence_findings)
    assert "verdict" not in risk_map.model_dump()
    assert not hasattr(risk_map, "score")
