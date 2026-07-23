"""The two code gates before a contract is analyzed: scope, then time.

The model reads the lease's use and signing date; code decides whether either
puts the contract out of the LAU's reach. The scope gate applies art 4.2 (a
seasonal or non-dwelling lease is out); the temporal gate keeps the analysis to
the redaction it actually supports (a contract signed under a superseded redaction
is out). A missing signing date is not a stop -- it is assumed to be today and the
assumption is stated -- so an absent date never becomes a silent guess about which
law applies.
"""

from dataclasses import dataclass
from datetime import date

from lexme.mode1.dates import TargetDate
from lexme.mode2.models import RejectionReason, TenancyUse
from lexme.mode2.scope import ScopePackage

UNDETERMINED_USE_ASSUMPTION = (
    "El contrato no deja claro el uso del inmueble; se asume vivienda habitual. "
    "Si fuese un alquiler de temporada o de uso distinto, el régimen aplicable "
    "sería otro."
)
ASSUMED_TODAY_ASSUMPTION = (
    "El contrato no indica fecha de firma; se asume la fecha de hoy y, con ella, "
    "la redacción vigente de la ley."
)


@dataclass(frozen=True)
class GateOutcome:
    """The gates' verdict: a rejection or a pass, plus the date and assumptions used.

    ``rejection`` is ``None`` when the contract cleared both gates. ``target_date``
    is the date the temporal gate judged against -- the signing date, or today when
    it was absent. ``assumptions`` are the choices stated out loud so no verdict
    rests on a hidden guess.
    """

    rejection: RejectionReason | None
    target_date: date
    assumptions: tuple[str, ...]


def apply_gates(
    *,
    use: TenancyUse,
    signing_date: TargetDate | None,
    scope: ScopePackage,
    today: date,
) -> GateOutcome:
    """Run the scope gate then the temporal gate over a contract's use and date.

    Rejects an art 4.2 use before considering time. Assumes today when the signing
    date is absent, recording the assumption, and rejects a contract signed under a
    superseded redaction. A cleared contract returns no rejection and the date the
    temporal gate used.
    """
    if scope.excludes(use):
        return GateOutcome(RejectionReason.OUT_OF_SCOPE_USE, today, ())

    assumptions: list[str] = []
    if use is TenancyUse.UNDETERMINED:
        assumptions.append(UNDETERMINED_USE_ASSUMPTION)

    if signing_date is None:
        target_date = today
        assumptions.append(ASSUMED_TODAY_ASSUMPTION)
    else:
        target_date = signing_date.value

    if scope.is_prior_redaction(target_date):
        return GateOutcome(RejectionReason.PRIOR_REDACTION, target_date, tuple(assumptions))
    return GateOutcome(None, target_date, tuple(assumptions))
