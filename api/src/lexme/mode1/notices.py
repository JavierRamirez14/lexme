"""Deterministic notices about how far the answer's law is from today's law.

Every notice is a rule in code over dates the corpus already holds -- the run's
target date, the redaction each citation resolved to, and the block's version
history. The model never writes one and never decides whether one applies: a
warning that the law has since changed is exactly the claim a fluent generator is
most likely to drop, so it is not left to the generator at all.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel

from lexme.mode1.dates import TargetDate

RECENT_AMENDMENT_WINDOW = timedelta(days=365)

_MONTHS = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)


class NoticeCode(StrEnum):
    """The closed set of in-force notices -- the contract the UI styles against."""

    HISTORICAL_TARGET_DATE = "fecha_objetivo_pasada"
    AMBIGUOUS_TARGET_YEAR = "anio_objetivo_ambiguo"
    SUPERSEDED_REDACTION = "redaccion_superada"
    RECENT_AMENDMENT = "modificacion_reciente"


class InForceNotice(BaseModel):
    """One notice: its machine code, its user-facing text and the block it concerns.

    ``block_id`` is ``None`` for a notice about the run as a whole rather than
    about a single citation.
    """

    code: NoticeCode
    message: str
    block_id: str | None = None


@dataclass(frozen=True)
class CitedBlock:
    """A block the answer cites, resolved to the redaction in force at the target date."""

    norm_id: str
    block_id: str
    title: str
    effective_date: date


class VersionHistory(Protocol):
    """The corpus port that lists every redaction a block has ever had."""

    def effective_dates(self, norm_id: str, block_id: str) -> list[date]:
        """Return the effective dates of every stored version of the block."""
        ...


def build_notices(
    citations: Sequence[CitedBlock],
    *,
    target_date: TargetDate,
    today: date,
    history: VersionHistory,
) -> list[InForceNotice]:
    """Derive the in-force notices for a run, in descending importance.

    The run-wide notice that the answer is situated in the past comes first, then
    the warning that a bare year straddles a reform, then one notice per cited
    article amended since the target date, then one per article amended shortly
    before it. A block cited more than once yields at most one notice per rule.
    """
    anchor = target_date.value
    notices: list[InForceNotice] = []
    if anchor < today:
        notices.append(_historical_notice(anchor))

    unique = _deduplicate(citations)
    notices.extend(_ambiguous_year_notices(unique, target_date, history))
    notices.extend(_superseded_notices(unique, anchor, today, history))
    notices.extend(_recent_amendment_notices(unique, anchor))
    return notices


def _deduplicate(citations: Sequence[CitedBlock]) -> list[CitedBlock]:
    """The cited blocks in first-seen order, one entry per block."""
    seen: dict[tuple[str, str], CitedBlock] = {}
    for citation in citations:
        seen.setdefault((citation.norm_id, citation.block_id), citation)
    return list(seen.values())


def _historical_notice(target_date: date) -> InForceNotice:
    """Announce that the whole answer is situated at a past date."""
    return InForceNotice(
        code=NoticeCode.HISTORICAL_TARGET_DATE,
        message=(
            f"Esta respuesta refleja el derecho vigente el {_spell(target_date)}, "
            "no el de hoy. Si tu situación es actual, vuelve a preguntar sin fecha."
        ),
    )


def _ambiguous_year_notices(
    citations: Iterable[CitedBlock], target_date: TargetDate, history: VersionHistory
) -> list[InForceNotice]:
    """One notice per cited article amended inside the bare year the user named.

    A year without a day resolves to its first of January, so a reform landing
    later that same year would be answered with the earlier redaction and no other
    rule would say so: the article was not superseded *after* the period asked
    about, it changed *within* it.
    """
    if not target_date.is_year_only:
        return []
    year_end = date(target_date.value.year, 12, 31)
    notices = []
    for citation in citations:
        amendments = _amendments_between(
            citation, after=target_date.value, until=year_end, history=history
        )
        if not amendments:
            continue
        notices.append(
            InForceNotice(
                code=NoticeCode.AMBIGUOUS_TARGET_YEAR,
                block_id=citation.block_id,
                message=(
                    f"Has situado la consulta en {target_date.value.year} sin día concreto, y el "
                    f"{citation.title} cambió el {_spell(min(amendments))}, dentro de ese mismo "
                    "año. Si tu contrato es posterior a esa fecha, te aplica la otra redacción: "
                    "dime el día exacto para afinarlo."
                ),
            )
        )
    return notices


def _superseded_notices(
    citations: Iterable[CitedBlock], target_date: date, today: date, history: VersionHistory
) -> list[InForceNotice]:
    """One notice per cited article whose text was amended between the target date and today."""
    notices = []
    for citation in citations:
        amendments = _amendments_between(
            citation, after=citation.effective_date, until=today, history=history
        )
        if not amendments:
            continue
        notices.append(
            InForceNotice(
                code=NoticeCode.SUPERSEDED_REDACTION,
                block_id=citation.block_id,
                message=(
                    f"El {citation.title} que cito es la redacción en vigor el "
                    f"{_spell(target_date)}; se modificó después ({_spell(min(amendments))}) "
                    "y hoy dice otra cosa."
                ),
            )
        )
    return notices


def _recent_amendment_notices(
    citations: Iterable[CitedBlock], target_date: date
) -> list[InForceNotice]:
    """One notice per cited article amended shortly before the target date."""
    threshold = target_date - RECENT_AMENDMENT_WINDOW
    return [
        InForceNotice(
            code=NoticeCode.RECENT_AMENDMENT,
            block_id=citation.block_id,
            message=(
                f"El {citation.title} se modificó hace poco ({_spell(citation.effective_date)}). "
                "Si tu contrato es anterior, puede seguir rigiéndose por la redacción previa."
            ),
        )
        for citation in citations
        if threshold <= citation.effective_date <= target_date
    ]


def _amendments_between(
    citation: CitedBlock, *, after: date, until: date, history: VersionHistory
) -> list[date]:
    """The effective dates of the block's versions that came into force in ``(after, until]``."""
    return [
        effective
        for effective in history.effective_dates(citation.norm_id, citation.block_id)
        if after < effective <= until
    ]


def _spell(value: date) -> str:
    """Render a date the way the answers read it: ``4 de mayo de 2017``."""
    return f"{value.day} de {_MONTHS[value.month - 1]} de {value.year}"
