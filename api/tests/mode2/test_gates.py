"""The two code gates: art 4.2 scope, then the temporal edge."""

from datetime import date

from lexme.mode1.dates import DatePrecision, TargetDate
from lexme.mode2.gates import (
    ASSUMED_TODAY_ASSUMPTION,
    UNDETERMINED_USE_ASSUMPTION,
    UNEVIDENCED_EXCLUSION_ASSUMPTION,
    GateOutcome,
    apply_gates,
)
from lexme.mode2.models import RejectionReason, TenancyUse
from lexme.mode2.scope import ScopePackage
from tests.mode2.conftest import scope_package

TODAY = date(2024, 6, 1)
BOUNDARY = date(2019, 3, 6)

DWELLING_DOCUMENT = (
    "CONTRATO DE ARRENDAMIENTO DE VIVIENDA\n"
    "El presente contrato se pacta por una duración de once meses, sin derecho a "
    "prórroga alguna."
)
SEASONAL_DOCUMENT = (
    "CONTRATO DE ARRENDAMIENTO\n"
    "El inmueble se arrienda con finalidad de temporada estival, sin constituir la "
    "vivienda habitual del arrendatario."
)
SEASONAL_SPAN = "con finalidad de temporada estival"
OFFICE_DOCUMENT = (
    "CONTRATO DE ARRENDAMIENTO PARA USO DISTINTO DEL DE VIVIENDA\n"
    "El local se destina a oficina y actividad profesional del arrendatario."
)
OFFICE_SPAN = "El local se destina a oficina"


def _scope() -> ScopePackage:
    return scope_package(current_redaction_effective_from=BOUNDARY)


def _day(value: date) -> TargetDate:
    return TargetDate(value=value, precision=DatePrecision.DAY)


def _gates(
    *,
    use: TenancyUse,
    use_evidence: str = "",
    document_text: str = DWELLING_DOCUMENT,
    signing_date: TargetDate | None = None,
) -> GateOutcome:
    return apply_gates(
        use=use,
        use_evidence=use_evidence,
        document_text=document_text,
        signing_date=signing_date,
        scope=_scope(),
        today=TODAY,
    )


def test_a_current_dwelling_lease_clears_both_gates() -> None:
    outcome = _gates(use=TenancyUse.HABITUAL_DWELLING, signing_date=_day(date(2023, 1, 1)))

    assert outcome.rejection is None
    assert outcome.target_date == date(2023, 1, 1)


def test_a_seasonal_lease_the_document_declares_is_out_of_scope_under_article_4_2() -> None:
    outcome = _gates(
        use=TenancyUse.SEASONAL,
        use_evidence=SEASONAL_SPAN,
        document_text=SEASONAL_DOCUMENT,
        signing_date=_day(date(2023, 1, 1)),
    )

    assert outcome.rejection is RejectionReason.OUT_OF_SCOPE_USE


def test_a_non_dwelling_lease_the_document_declares_is_out_of_scope() -> None:
    outcome = _gates(
        use=TenancyUse.NON_DWELLING,
        use_evidence=OFFICE_SPAN,
        document_text=OFFICE_DOCUMENT,
        signing_date=_day(date(2023, 1, 1)),
    )

    assert outcome.rejection is RejectionReason.OUT_OF_SCOPE_USE


def test_a_seasonal_reading_with_no_span_behind_it_is_analyzed_and_stated() -> None:
    outcome = _gates(use=TenancyUse.SEASONAL, signing_date=_day(date(2023, 1, 1)))

    assert outcome.rejection is None
    assert UNEVIDENCED_EXCLUSION_ASSUMPTION in outcome.assumptions


def test_a_span_the_document_does_not_contain_does_not_close_the_gate() -> None:
    outcome = _gates(
        use=TenancyUse.SEASONAL,
        use_evidence="se arrienda con finalidad de temporada estival",
        document_text=DWELLING_DOCUMENT,
        signing_date=_day(date(2023, 1, 1)),
    )

    assert outcome.rejection is None
    assert UNEVIDENCED_EXCLUSION_ASSUMPTION in outcome.assumptions


def test_a_term_clause_is_a_real_span_and_still_declares_no_seasonal_use() -> None:
    outcome = _gates(
        use=TenancyUse.SEASONAL,
        use_evidence="duración de once meses, sin derecho a prórroga alguna",
        signing_date=_day(date(2023, 1, 1)),
    )

    assert outcome.rejection is None
    assert UNEVIDENCED_EXCLUSION_ASSUMPTION in outcome.assumptions


def test_a_span_too_short_to_declare_anything_does_not_close_the_gate() -> None:
    outcome = _gates(
        use=TenancyUse.SEASONAL,
        use_evidence="temporada",
        document_text=SEASONAL_DOCUMENT,
        signing_date=_day(date(2023, 1, 1)),
    )

    assert outcome.rejection is None


def test_a_short_but_real_declaration_still_closes_the_gate() -> None:
    outcome = _gates(
        use=TenancyUse.SEASONAL,
        use_evidence="uso vacacional",
        document_text="CONTRATO\nEl inmueble se cede para uso vacacional del arrendatario.",
        signing_date=_day(date(2023, 1, 1)),
    )

    assert outcome.rejection is RejectionReason.OUT_OF_SCOPE_USE


def test_a_span_that_only_differs_in_typography_still_closes_the_gate() -> None:
    outcome = _gates(
        use=TenancyUse.SEASONAL,
        use_evidence="con  finalidad\nde temporada estival",
        document_text=SEASONAL_DOCUMENT,
        signing_date=_day(date(2023, 1, 1)),
    )

    assert outcome.rejection is RejectionReason.OUT_OF_SCOPE_USE


def test_a_lease_under_a_prior_redaction_is_out_of_temporal_scope() -> None:
    outcome = _gates(use=TenancyUse.HABITUAL_DWELLING, signing_date=_day(date(2017, 5, 4)))

    assert outcome.rejection is RejectionReason.PRIOR_REDACTION


def test_a_lease_signed_on_the_boundary_day_is_in_scope() -> None:
    outcome = _gates(use=TenancyUse.HABITUAL_DWELLING, signing_date=_day(BOUNDARY))

    assert outcome.rejection is None


def test_a_missing_signing_date_is_assumed_today_and_stated() -> None:
    outcome = _gates(use=TenancyUse.HABITUAL_DWELLING, signing_date=None)

    assert outcome.rejection is None
    assert outcome.target_date == TODAY
    assert ASSUMED_TODAY_ASSUMPTION in outcome.assumptions


def test_an_undetermined_use_is_assumed_a_dwelling_and_stated() -> None:
    outcome = _gates(use=TenancyUse.UNDETERMINED, signing_date=_day(date(2023, 1, 1)))

    assert outcome.rejection is None
    assert UNDETERMINED_USE_ASSUMPTION in outcome.assumptions


def test_the_scope_gate_precedes_the_temporal_gate() -> None:
    outcome = _gates(
        use=TenancyUse.NON_DWELLING,
        use_evidence=OFFICE_SPAN,
        document_text=OFFICE_DOCUMENT,
        signing_date=_day(date(2010, 1, 1)),
    )

    assert outcome.rejection is RejectionReason.OUT_OF_SCOPE_USE


def test_an_unevidenced_exclusion_still_reaches_the_temporal_gate() -> None:
    outcome = _gates(use=TenancyUse.SEASONAL, signing_date=_day(date(2010, 1, 1)))

    assert outcome.rejection is RejectionReason.PRIOR_REDACTION
