"""The two code gates before a contract is analyzed: scope, then time.

The model reads the lease's use and signing date; code decides whether either
puts the contract out of the LAU's reach. The scope gate applies art 4.2 (a
seasonal or non-dwelling lease is out); the temporal gate keeps the analysis to
the redaction it actually supports (a contract signed under a superseded redaction
is out). Neither stop rests on the model's word: the scope gate closes only when
the document itself declares the excluded destination, in a span code finds
literally in the text. Without that span the reading degrades to
``indeterminado``, the contract is analyzed and the assumption is stated -- the
same treatment an absent signing date gets, and for the same reason. The
asymmetry sets the direction: refusing an abusive lease costs a tenant every
illegal clause in it, while analyzing a genuine seasonal let costs only some less
pertinent findings.
"""

from dataclasses import dataclass
from datetime import date

from lexme.mode1.dates import TargetDate
from lexme.mode2.models import RejectionReason, TenancyUse
from lexme.mode2.scope import ScopePackage
from lexme.verification.normalization import normalize

MIN_USE_EVIDENCE_CHARS = 12

UNDETERMINED_USE_ASSUMPTION = (
    "El contrato no deja claro el uso del inmueble; se asume vivienda habitual. "
    "Si fuese un alquiler de temporada o de uso distinto, el régimen aplicable "
    "sería otro."
)
UNEVIDENCED_EXCLUSION_ASSUMPTION = (
    "El contrato no declara en ninguna parte una finalidad de temporada, vacacional "
    "o de uso distinto del de vivienda, así que lo analizo como vivienda habitual: "
    "un plazo corto no convierte por sí solo el arrendamiento de una vivienda en uno "
    "de temporada, porque la ley distingue por el destino del inmueble y no por su "
    "duración."
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
    use_evidence: str,
    document_text: str,
    signing_date: TargetDate | None,
    scope: ScopePackage,
    today: date,
) -> GateOutcome:
    """Run the scope gate then the temporal gate over a contract's use and date.

    Rejects an art 4.2 use before considering time, but only when ``use_evidence``
    is a span of ``document_text`` declaring that destination; an excluded use with
    nothing behind it is analyzed as an undetermined one and the assumption is
    stated. Assumes today when the signing date is absent, recording the assumption,
    and rejects a contract signed under a superseded redaction. A cleared contract
    returns no rejection and the date the temporal gate used.
    """
    resolved_use, use_assumption = _resolve_use(use, use_evidence, document_text, scope)
    if scope.excludes(resolved_use):
        return GateOutcome(RejectionReason.OUT_OF_SCOPE_USE, today, ())

    assumptions: list[str] = []
    if use_assumption is not None:
        assumptions.append(use_assumption)

    if signing_date is None:
        target_date = today
        assumptions.append(ASSUMED_TODAY_ASSUMPTION)
    else:
        target_date = signing_date.value

    if scope.is_prior_redaction(target_date):
        return GateOutcome(RejectionReason.PRIOR_REDACTION, target_date, tuple(assumptions))
    return GateOutcome(None, target_date, tuple(assumptions))


def _resolve_use(
    use: TenancyUse,
    use_evidence: str,
    document_text: str,
    scope: ScopePackage,
) -> tuple[TenancyUse, str | None]:
    """Resolve the use the analysis proceeds under, and the assumption it owes.

    An excluded use survives only when the document declares it; otherwise it
    degrades to ``UNDETERMINED``, which the scope gate never excludes, and carries
    the assumption saying so. A use the scope does not exclude passes through
    untouched, with the silent-contract assumption when it was undetermined to
    begin with.
    """
    if not scope.excludes(use):
        return use, UNDETERMINED_USE_ASSUMPTION if use is TenancyUse.UNDETERMINED else None
    if _declares(use, use_evidence, document_text, scope):
        return use, None
    return TenancyUse.UNDETERMINED, UNEVIDENCED_EXCLUSION_ASSUMPTION


def _declares(use: TenancyUse, use_evidence: str, document_text: str, scope: ScopePackage) -> bool:
    """Whether the document really declares ``use`` in the span the model returned.

    The span must be long enough to say something, literally present in the
    document (normalized on both sides, so typographic drift does not lose a real
    declaration) and carry one of the vertical's markers for that use. The marker is
    what keeps a term clause out: a span about duration is real text and still
    declares nothing about destination.
    """
    needle = normalize(use_evidence)
    if len(needle) < MIN_USE_EVIDENCE_CHARS:
        return False
    if needle not in normalize(document_text):
        return False
    return scope.declares(use, needle)
