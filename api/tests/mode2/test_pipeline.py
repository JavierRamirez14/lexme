"""The contract pipeline end to end, with the extractor, LLM and scope faked."""

from datetime import date

from lexme.llm import FakeLlmClient
from lexme.mode2 import Mode2Deps, Mode2Outcome, RejectionReason, ScopePackage, TenancyUse
from lexme.mode2.mapping import MAPPING_TASK
from lexme.mode2.pipeline import analyze_contract
from lexme.mode2.segmentation import SEGMENTATION_TASK
from lexme.mode2.triage import TRIAGE_TASK
from tests.mode2.conftest import (
    SEASONAL_DECLARATION,
    FakeCorpus,
    FakeExtractor,
    FakeRetriever,
    checklist_of,
    mapping_of,
    scope_package,
    seasonal_document,
    segmentation_of,
    triage_of,
)

INFORMATIVE_THREE = mapping_of(("c1", False, []), ("c2", False, []), ("c3", False, []))

TODAY = date(2024, 6, 1)
DOCUMENT = (
    "CONTRATO DE ARRENDAMIENTO DE VIVIENDA\n"
    "PRIMERA. Duración. El plazo del arrendamiento será de cinco años.\n"
    "SEGUNDA. Renta. La renta mensual se fija en 800 euros pagaderos por adelantado.\n"
    "TERCERA. Fianza. El arrendatario entrega una mensualidad en concepto de fianza."
)
CLAUSES = (
    ("Duración", "El plazo del arrendamiento será de cinco años."),
    ("Renta", "La renta mensual se fija en 800 euros pagaderos por adelantado."),
    ("Fianza", "El arrendatario entrega una mensualidad en concepto de fianza."),
)
SEASONAL_DOCUMENT = seasonal_document(DOCUMENT)


def _scope() -> ScopePackage:
    return scope_package()


def _deps(fake: FakeLlmClient, *, text: str = DOCUMENT) -> Mode2Deps:
    return Mode2Deps(
        llm=fake,
        extractor=FakeExtractor(text),
        scope=_scope(),
        checklist=checklist_of(),
        corpus=FakeCorpus(),
        retriever=FakeRetriever(),
        vertical="vivienda",
    )


def _analyze(fake: FakeLlmClient, deps: Mode2Deps):
    return analyze_contract(filename="contrato.pdf", content=b"ignored", deps=deps, today=TODAY)


def test_a_readable_in_scope_lease_is_analyzed_with_ficha_summary_and_clauses() -> None:
    fake = FakeLlmClient()
    fake.queue(SEGMENTATION_TASK, segmentation_of(*CLAUSES))
    fake.queue(TRIAGE_TASK, triage_of(fecha_firma="2023-01-01"))
    fake.queue(MAPPING_TASK, INFORMATIVE_THREE)

    analysis = _analyze(fake, _deps(fake))

    assert analysis.outcome is Mode2Outcome.ANALYZED
    assert analysis.sheet is not None
    assert analysis.summary is not None
    assert analysis.risk_map is not None
    assert len(analysis.clauses) == 3
    assert analysis.clauses[0].text == "El plazo del arrendamiento será de cinco años."


def test_a_scan_with_no_extractable_text_is_not_analyzable() -> None:
    fake = FakeLlmClient()

    analysis = _analyze(fake, _deps(fake, text="   "))

    assert analysis.outcome is Mode2Outcome.NOT_ANALYZABLE
    assert analysis.rejection is not None
    assert analysis.rejection.reason is RejectionReason.NOT_EXTRACTABLE
    assert not fake.calls


def test_a_clause_that_does_not_anchor_makes_the_document_not_analyzable() -> None:
    fake = FakeLlmClient()
    fake.queue(
        SEGMENTATION_TASK,
        segmentation_of(
            ("Duración", "El plazo del arrendamiento será de cinco años."),
            ("Falsa", "El arrendador puede recuperar la vivienda cuando quiera sin preaviso."),
        ),
    )

    analysis = _analyze(fake, _deps(fake))

    assert analysis.outcome is Mode2Outcome.NOT_ANALYZABLE
    assert analysis.rejection.reason is RejectionReason.BROKEN_ANCHOR


def test_a_lease_the_document_declares_seasonal_is_out_of_scope() -> None:
    fake = FakeLlmClient()
    fake.queue(SEGMENTATION_TASK, segmentation_of(*CLAUSES))
    fake.queue(
        TRIAGE_TASK,
        triage_of(
            uso=TenancyUse.SEASONAL,
            uso_evidencia=SEASONAL_DECLARATION,
            fecha_firma="2023-01-01",
        ),
    )

    analysis = _analyze(fake, _deps(fake, text=SEASONAL_DOCUMENT))

    assert analysis.outcome is Mode2Outcome.OUT_OF_SCOPE
    assert analysis.rejection.reason is RejectionReason.OUT_OF_SCOPE_USE
    assert analysis.clauses == []


def test_a_seasonal_reading_with_no_span_behind_it_is_analyzed_and_stated() -> None:
    fake = FakeLlmClient()
    fake.queue(SEGMENTATION_TASK, segmentation_of(*CLAUSES))
    fake.queue(TRIAGE_TASK, triage_of(uso=TenancyUse.SEASONAL, fecha_firma="2023-01-01"))
    fake.queue(MAPPING_TASK, INFORMATIVE_THREE)

    analysis = _analyze(fake, _deps(fake))

    assert analysis.outcome is Mode2Outcome.ANALYZED
    assert analysis.risk_map is not None
    assert any("no declara" in assumption for assumption in analysis.assumptions)


def test_a_lease_under_a_prior_redaction_is_out_of_scope() -> None:
    fake = FakeLlmClient()
    fake.queue(SEGMENTATION_TASK, segmentation_of(*CLAUSES))
    fake.queue(TRIAGE_TASK, triage_of(fecha_firma="2017-05-04"))

    analysis = _analyze(fake, _deps(fake))

    assert analysis.outcome is Mode2Outcome.OUT_OF_SCOPE
    assert analysis.rejection.reason is RejectionReason.PRIOR_REDACTION


def test_a_lease_without_a_signing_date_assumes_today_and_states_it() -> None:
    fake = FakeLlmClient()
    fake.queue(SEGMENTATION_TASK, segmentation_of(*CLAUSES))
    fake.queue(TRIAGE_TASK, triage_of(fecha_firma=""))
    fake.queue(MAPPING_TASK, INFORMATIVE_THREE)

    analysis = _analyze(fake, _deps(fake))

    assert analysis.outcome is Mode2Outcome.ANALYZED
    assert any("hoy" in assumption for assumption in analysis.assumptions)


def test_the_summary_is_assembled_without_an_extra_model_call() -> None:
    fake = FakeLlmClient()
    fake.queue(SEGMENTATION_TASK, segmentation_of(*CLAUSES))
    fake.queue(TRIAGE_TASK, triage_of(fecha_firma="2023-01-01"))
    fake.queue(MAPPING_TASK, INFORMATIVE_THREE)

    analysis = _analyze(fake, _deps(fake))

    assert analysis.summary is not None
    tasks = [call.task for call in fake.calls]
    assert tasks == [SEGMENTATION_TASK, TRIAGE_TASK, MAPPING_TASK]
