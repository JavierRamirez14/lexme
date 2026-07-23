"""The two code gates: art 4.2 scope, then the temporal edge."""

from datetime import date

from lexme.mode1.dates import DatePrecision, TargetDate
from lexme.mode2.gates import (
    ASSUMED_TODAY_ASSUMPTION,
    UNDETERMINED_USE_ASSUMPTION,
    apply_gates,
)
from lexme.mode2.models import RejectionReason, TenancyUse
from lexme.mode2.scope import ScopePackage

TODAY = date(2024, 6, 1)
BOUNDARY = date(2019, 3, 6)


def _scope() -> ScopePackage:
    return ScopePackage(
        current_redaction_effective_from=BOUNDARY,
        excluded_uses=frozenset({TenancyUse.SEASONAL, TenancyUse.NON_DWELLING}),
    )


def _day(value: date) -> TargetDate:
    return TargetDate(value=value, precision=DatePrecision.DAY)


def test_a_current_dwelling_lease_clears_both_gates() -> None:
    outcome = apply_gates(
        use=TenancyUse.HABITUAL_DWELLING,
        signing_date=_day(date(2023, 1, 1)),
        scope=_scope(),
        today=TODAY,
    )

    assert outcome.rejection is None
    assert outcome.target_date == date(2023, 1, 1)


def test_a_seasonal_lease_is_out_of_scope_under_article_4_2() -> None:
    outcome = apply_gates(
        use=TenancyUse.SEASONAL,
        signing_date=_day(date(2023, 1, 1)),
        scope=_scope(),
        today=TODAY,
    )

    assert outcome.rejection is RejectionReason.OUT_OF_SCOPE_USE


def test_a_lease_under_a_prior_redaction_is_out_of_temporal_scope() -> None:
    outcome = apply_gates(
        use=TenancyUse.HABITUAL_DWELLING,
        signing_date=_day(date(2017, 5, 4)),
        scope=_scope(),
        today=TODAY,
    )

    assert outcome.rejection is RejectionReason.PRIOR_REDACTION


def test_a_lease_signed_on_the_boundary_day_is_in_scope() -> None:
    outcome = apply_gates(
        use=TenancyUse.HABITUAL_DWELLING,
        signing_date=_day(BOUNDARY),
        scope=_scope(),
        today=TODAY,
    )

    assert outcome.rejection is None


def test_a_missing_signing_date_is_assumed_today_and_stated() -> None:
    outcome = apply_gates(
        use=TenancyUse.HABITUAL_DWELLING,
        signing_date=None,
        scope=_scope(),
        today=TODAY,
    )

    assert outcome.rejection is None
    assert outcome.target_date == TODAY
    assert ASSUMED_TODAY_ASSUMPTION in outcome.assumptions


def test_an_undetermined_use_is_assumed_a_dwelling_and_stated() -> None:
    outcome = apply_gates(
        use=TenancyUse.UNDETERMINED,
        signing_date=_day(date(2023, 1, 1)),
        scope=_scope(),
        today=TODAY,
    )

    assert outcome.rejection is None
    assert UNDETERMINED_USE_ASSUMPTION in outcome.assumptions


def test_the_scope_gate_precedes_the_temporal_gate() -> None:
    outcome = apply_gates(
        use=TenancyUse.NON_DWELLING,
        signing_date=_day(date(2010, 1, 1)),
        scope=_scope(),
        today=TODAY,
    )

    assert outcome.rejection is RejectionReason.OUT_OF_SCOPE_USE
