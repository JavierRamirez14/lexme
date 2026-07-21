"""Deterministic parsing of the dates that anchor a Mode 1 run in time.

Two dates reach the run as free text: the temporal reference the planner reads
out of the question ("firmé en 2017"), and the date a user types when the graph
stops to ask which contract date applies. Both are parsed here by code, so a
vague or impossible value degrades to a documented fallback instead of silently
resolving the corpus at the wrong redaction.

How precisely the user pinned the date is carried alongside it rather than
thrown away: "en 2019" and "el 4 de mayo de 2019" resolve to different-quality
answers, because a reform landed mid-2019, and only the run that knows the anchor
was a bare year can warn about it.
"""

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

_DAY_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")
_YEAR_LENGTH = 4


class DatePrecision(StrEnum):
    """How precisely the user pinned down the date the answer is situated at."""

    DAY = "dia"
    YEAR = "anio"
    NONE = "ninguna"


@dataclass(frozen=True)
class TargetDate:
    """The date to resolve the corpus at, with how precisely it was stated.

    A ``YEAR`` precision means ``value`` is the first day of a year the user named
    without a day: the earliest redaction that year can mean, so the answer never
    claims a reform that had not yet entered into force for part of it.
    """

    value: date
    precision: DatePrecision

    @property
    def is_year_only(self) -> bool:
        """Whether the user named a year but not a day within it."""
        return self.precision is DatePrecision.YEAR


def parse_target_date(text: str) -> TargetDate | None:
    """Parse ``text`` as a target date, or return ``None`` when it is not one.

    Accepts an ISO date (``2017-05-04``), the Spanish written order
    (``04/05/2017`` or ``04-05-2017``) and a bare year (``2017``).
    """
    candidate = text.strip()
    if not candidate:
        return None
    if len(candidate) == _YEAR_LENGTH and candidate.isdigit():
        return _year_start(int(candidate))
    for pattern in _DAY_FORMATS:
        try:
            parsed = datetime.strptime(candidate, pattern).date()
        except ValueError:
            continue
        return TargetDate(value=parsed, precision=DatePrecision.DAY)
    return None


def resolve_target_date(reference: str, today: date) -> TargetDate:
    """Resolve a free-text temporal ``reference`` into the date to answer at.

    Falls back to ``today`` when the reference is absent or unparseable, and
    clamps a future reference to ``today``: the corpus holds no redaction that is
    not yet in force, so answering "in force at a future date" would be a promise
    the data cannot keep.
    """
    parsed = parse_target_date(reference)
    if parsed is None or parsed.value > today:
        return TargetDate(value=today, precision=DatePrecision.NONE)
    return parsed


def _year_start(year: int) -> TargetDate | None:
    """The first day of ``year``, or ``None`` when it is out of calendar range."""
    try:
        return TargetDate(value=date(year, 1, 1), precision=DatePrecision.YEAR)
    except ValueError:
        return None
