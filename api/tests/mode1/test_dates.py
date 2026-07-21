"""Tests for the deterministic parsing of the dates that anchor a run in time."""

from datetime import date

from lexme.mode1.dates import DatePrecision, parse_target_date, resolve_target_date

TODAY = date(2026, 7, 21)


def test_an_iso_date_is_parsed_to_the_day() -> None:
    parsed = parse_target_date("2017-05-04")

    assert parsed is not None
    assert parsed.value == date(2017, 5, 4)
    assert parsed.precision is DatePrecision.DAY


def test_a_spanish_ordered_date_is_parsed_as_day_month_year() -> None:
    parsed = parse_target_date("04/05/2017")

    assert parsed is not None
    assert parsed.value == date(2017, 5, 4)


def test_a_bare_year_resolves_to_its_first_day_and_is_marked_imprecise() -> None:
    parsed = parse_target_date("2017")

    assert parsed is not None
    assert parsed.value == date(2017, 1, 1)
    assert parsed.is_year_only


def test_surrounding_whitespace_does_not_prevent_parsing() -> None:
    parsed = parse_target_date("  2017-05-04  ")

    assert parsed is not None
    assert parsed.value == date(2017, 5, 4)


def test_prose_is_not_a_date() -> None:
    assert parse_target_date("no me acuerdo") is None


def test_an_impossible_calendar_date_is_not_a_date() -> None:
    assert parse_target_date("2017-02-30") is None


def test_an_empty_reference_resolves_to_today() -> None:
    resolved = resolve_target_date("", TODAY)

    assert resolved.value == TODAY
    assert resolved.precision is DatePrecision.NONE


def test_an_unparseable_reference_resolves_to_today() -> None:
    assert resolve_target_date("cuando firmé", TODAY).value == TODAY


def test_a_past_reference_resolves_to_that_date() -> None:
    assert resolve_target_date("2017", TODAY).value == date(2017, 1, 1)


def test_a_future_reference_is_clamped_to_today() -> None:
    resolved = resolve_target_date("2030-01-01", TODAY)

    assert resolved.value == TODAY
    assert resolved.precision is DatePrecision.NONE
