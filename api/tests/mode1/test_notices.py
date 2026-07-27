"""Tests for the deterministic in-force notices.

Every notice here is a rule in code over dates the corpus already holds, so these
tests need no model and no judge: given a target date, a cited redaction and the
block's version history, the notices are fully determined.
"""

from datetime import date

from lexme.mode1.dates import DatePrecision, TargetDate
from lexme.mode1.notices import CitedBlock, NoticeCode, build_notices

NORM = "BOE-A-1994-26003"
TODAY = date(2026, 7, 21)


class StubHistory:
    """A version history programmed with the effective dates of each block."""

    def __init__(self, dates_by_block: dict[str, list[date]]) -> None:
        """Bind the stub to a ``{block_id: [effective_date, ...]}`` mapping."""
        self._dates_by_block = dates_by_block

    def effective_dates(self, norm_id: str, block_id: str) -> list[date]:
        """Return the programmed effective dates for ``block_id``."""
        return self._dates_by_block.get(block_id, [])


def cited(block_id: str, effective_date: date, title: str = "Artículo 9") -> CitedBlock:
    """A cited block resolved at ``effective_date``."""
    return CitedBlock(norm_id=NORM, block_id=block_id, title=title, effective_date=effective_date)


def day(value: date) -> TargetDate:
    """A target date the user pinned down to the day."""
    return TargetDate(value=value, precision=DatePrecision.DAY)


def year(value: int) -> TargetDate:
    """A target date the user named only as a year."""
    return TargetDate(value=date(value, 1, 1), precision=DatePrecision.YEAR)


def codes(notices: list) -> list[NoticeCode]:
    """The codes of ``notices``, in order."""
    return [notice.code for notice in notices]


def test_answering_at_today_with_the_current_redaction_raises_no_notice() -> None:
    history = StubHistory({"a9": [date(2013, 6, 6)]})

    notices = build_notices(
        [cited("a9", date(2013, 6, 6))], target_date=day(TODAY), today=TODAY, history=history
    )

    assert notices == []


def test_a_past_target_date_is_always_announced() -> None:
    history = StubHistory({"a9": [date(2013, 6, 6)]})

    notices = build_notices(
        [cited("a9", date(2013, 6, 6))],
        target_date=day(date(2017, 3, 1)),
        today=TODAY,
        history=history,
    )

    assert codes(notices)[0] is NoticeCode.HISTORICAL_TARGET_DATE
    assert "2017" in notices[0].message


def test_a_citation_amended_after_the_target_date_is_flagged_as_superseded() -> None:
    history = StubHistory({"a9": [date(2013, 6, 6), date(2019, 3, 6)]})

    notices = build_notices(
        [cited("a9", date(2013, 6, 6))],
        target_date=day(date(2017, 3, 1)),
        today=TODAY,
        history=history,
    )

    assert NoticeCode.SUPERSEDED_REDACTION in codes(notices)
    superseded = next(n for n in notices if n.code is NoticeCode.SUPERSEDED_REDACTION)
    assert "Artículo 9" in superseded.message
    assert superseded.block_ref == f"{NORM}:a9"


def test_a_version_not_yet_in_force_today_does_not_supersede_anything() -> None:
    history = StubHistory({"a9": [date(2013, 6, 6), date(2027, 1, 1)]})

    notices = build_notices(
        [cited("a9", date(2013, 6, 6))], target_date=day(TODAY), today=TODAY, history=history
    )

    assert notices == []


def test_a_redaction_amended_just_before_the_target_date_is_flagged_as_recent() -> None:
    amended = date(2026, 5, 1)
    history = StubHistory({"a9": [date(2013, 6, 6), amended]})

    notices = build_notices(
        [cited("a9", amended)], target_date=day(TODAY), today=TODAY, history=history
    )

    assert codes(notices) == [NoticeCode.RECENT_AMENDMENT]
    assert "2026" in notices[0].message


def test_a_long_standing_redaction_is_not_flagged_as_recent() -> None:
    history = StubHistory({"a9": [date(2013, 6, 6)]})

    notices = build_notices(
        [cited("a9", date(2013, 6, 6))], target_date=day(TODAY), today=TODAY, history=history
    )

    assert NoticeCode.RECENT_AMENDMENT not in codes(notices)


def test_a_block_cited_twice_raises_one_notice_per_rule() -> None:
    history = StubHistory({"a9": [date(2013, 6, 6), date(2019, 3, 6)]})

    notices = build_notices(
        [cited("a9", date(2013, 6, 6)), cited("a9", date(2013, 6, 6))],
        target_date=day(date(2017, 3, 1)),
        today=TODAY,
        history=history,
    )

    assert codes(notices).count(NoticeCode.SUPERSEDED_REDACTION) == 1


def test_each_amended_article_gets_its_own_superseded_notice() -> None:
    history = StubHistory(
        {"a9": [date(2013, 6, 6), date(2019, 3, 6)], "a10": [date(2013, 6, 6), date(2019, 3, 6)]}
    )

    notices = build_notices(
        [cited("a9", date(2013, 6, 6)), cited("a10", date(2013, 6, 6), title="Artículo 10")],
        target_date=day(date(2017, 3, 1)),
        today=TODAY,
        history=history,
    )

    assert codes(notices).count(NoticeCode.SUPERSEDED_REDACTION) == 2


def test_a_past_target_date_is_announced_even_without_citations() -> None:
    notices = build_notices(
        [], target_date=day(date(2017, 3, 1)), today=TODAY, history=StubHistory({})
    )

    assert codes(notices) == [NoticeCode.HISTORICAL_TARGET_DATE]


def test_a_bare_year_that_straddles_an_amendment_is_flagged_as_ambiguous() -> None:
    history = StubHistory({"a9": [date(2013, 6, 6), date(2019, 3, 6)]})

    notices = build_notices(
        [cited("a9", date(2013, 6, 6))], target_date=year(2019), today=TODAY, history=history
    )

    ambiguous = next(n for n in notices if n.code is NoticeCode.AMBIGUOUS_TARGET_YEAR)
    assert ambiguous.block_ref == f"{NORM}:a9"
    assert "2019" in ambiguous.message
    assert "6 de marzo de 2019" in ambiguous.message


def test_a_bare_year_with_no_amendment_inside_it_is_not_flagged_as_ambiguous() -> None:
    history = StubHistory({"a9": [date(2013, 6, 6), date(2019, 3, 6)]})

    notices = build_notices(
        [cited("a9", date(2013, 6, 6))], target_date=year(2017), today=TODAY, history=history
    )

    assert NoticeCode.AMBIGUOUS_TARGET_YEAR not in codes(notices)


def test_a_date_pinned_to_the_day_is_never_flagged_as_ambiguous() -> None:
    history = StubHistory({"a9": [date(2013, 6, 6), date(2019, 3, 6)]})

    notices = build_notices(
        [cited("a9", date(2013, 6, 6))],
        target_date=day(date(2019, 1, 15)),
        today=TODAY,
        history=history,
    )

    assert NoticeCode.AMBIGUOUS_TARGET_YEAR not in codes(notices)
