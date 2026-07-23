"""The Mode 2 risk map: the spectrum of findings, never a single verdict.

Every clause the pipeline anchored ends as one :class:`ClauseFinding` carrying a
discrete coverage status, so a reader can always tell "evaluated and fine" from
"skipped". A clause that was actually judged also carries a :class:`RiskLevel`;
the rest name why they were not (informative, out of scope, inconclusive). The
protections the contract stays silent about surface as :class:`AbsenceFinding`
whites, the differentiator a flat RAG cannot produce. The map counts its findings
by level and by coverage but deliberately holds no aggregate score: the unit of
judgement is the finding, and an aggregate would reintroduce the false calm this
design exists to kill.
"""

from enum import StrEnum

from pydantic import BaseModel

from lexme.mode1.models import VerifiedCitation


class RiskLevel(StrEnum):
    """The five-colour risk spectrum, each level defined by its tie to the norm.

    ``ILEGAL`` contradicts an imperative article; ``PEOR_QUE_DEFAULT`` picks the
    worse branch of a dispositive default; ``NEGOCIABLE`` is a burden the LAU is
    silent on; ``CORRECTO`` was checked and conforms. ``AUSENTE`` is the odd one
    out -- it grades an absence, not a clause, so only an :class:`AbsenceFinding`
    ever carries it.
    """

    ILEGAL = "ilegal"
    PEOR_QUE_DEFAULT = "peor_que_default"
    NEGOCIABLE = "negociable"
    AUSENTE = "ausente"
    CORRECTO = "correcto"


class CoverageStatus(StrEnum):
    """Why a clause is where it is on the map, so coverage is never a silent hole.

    ``EVALUADA`` carries a :class:`RiskLevel`; ``INFORMATIVA`` has no evaluable
    legal content (parties, inventory); ``FUERA_DE_AMBITO`` is governed by a norm
    outside the corpus (named, not guessed); ``NO_CONCLUYENTE`` is legally
    relevant but could not be grounded -- including a level whose citation the
    verifier discarded, kept as a signal rather than dropped.
    """

    EVALUADA = "evaluada"
    INFORMATIVA = "informativa"
    FUERA_DE_AMBITO = "fuera_de_ambito"
    NO_CONCLUYENTE = "no_concluyente"


class ProposedClauseLevel(StrEnum):
    """A clause verdict as the classifier proposes it, before code places it.

    The first four map onto the evaluated :class:`RiskLevel` spectrum; the last
    two are coverage escapes the model may reach for when the LAU does not govern
    the clause or when it lacks the ground to assign a level.
    """

    ILEGAL = "ilegal"
    PEOR_QUE_DEFAULT = "peor_que_default"
    NEGOCIABLE = "negociable"
    CORRECTO = "correcto"
    FUERA_DE_AMBITO = "fuera_de_ambito"
    NO_CONCLUYENTE = "no_concluyente"


_SEVERITY_ORDER: tuple[RiskLevel, ...] = (
    RiskLevel.CORRECTO,
    RiskLevel.NEGOCIABLE,
    RiskLevel.PEOR_QUE_DEFAULT,
    RiskLevel.ILEGAL,
)

_LEVELS_REQUIRING_CITATION: frozenset[RiskLevel] = frozenset(
    {RiskLevel.ILEGAL, RiskLevel.PEOR_QUE_DEFAULT, RiskLevel.CORRECTO}
)


def severity(level: RiskLevel) -> int:
    """Rank a clause level so the cross-check can pick the worst of several.

    ``AUSENTE`` is not a clause level and has no place in this ordering; ranking
    it is a programming error, so it raises rather than sorting silently.
    """
    if level is RiskLevel.AUSENTE:
        raise ValueError("AUSENTE grades an absence, not a clause, and has no severity")
    return _SEVERITY_ORDER.index(level)


def requires_citation(level: RiskLevel) -> bool:
    """Whether a clause at ``level`` must carry a verified norm citation to stand.

    The three levels that assert something about the law (illegal, worse than the
    default, conforming) need the article behind them; ``NEGOCIABLE`` rests on the
    LAU's silence and cites nothing.
    """
    return level in _LEVELS_REQUIRING_CITATION


class ClauseFinding(BaseModel):
    """One contract clause placed on the map: its coverage, and a level if judged.

    ``snippet`` is the literal document span code anchored the clause to, with its
    half-open ``start``/``end`` range, so the UI can highlight it in place.
    ``level`` is set only when ``coverage`` is ``EVALUADA``; ``citation`` is the
    verified norm behind that level. ``out_of_scope_matter`` names the governing
    norm when ``coverage`` is ``FUERA_DE_AMBITO``. ``chk_ids`` records which
    checklist items the clause reflects, feeding the code cross-check.
    """

    clause_id: str
    heading: str
    snippet: str
    start: int
    end: int
    coverage: CoverageStatus
    level: RiskLevel | None = None
    explanation: str = ""
    what_you_can_do: list[str] = []
    citation: VerifiedCitation | None = None
    out_of_scope_matter: str = ""
    chk_ids: list[str] = []
    related_clause_ids: list[str] = []


class AbsenceFinding(BaseModel):
    """A protection the contract leaves unsaid: the white of the spectrum.

    ``right`` is the checklist item's plain-language protection, standing in for
    the missing clause; ``explanation`` is the item's static absence text (which
    already spells out any consequence and deadline for the burdened rights).
    ``citation`` is the law that grants the right, its literalness guaranteed at
    build time and its anchor hydrated from the corpus for display.
    """

    item_id: str
    right: str
    level: RiskLevel = RiskLevel.AUSENTE
    silence_tone: str
    explanation: str
    citation: VerifiedCitation | None = None


class RiskMap(BaseModel):
    """The whole spectrum for one contract: every clause, every absence, no score.

    ``clause_findings`` holds one entry per anchored clause; ``absence_findings``
    the whites the cross-check produced. ``level_counts`` and ``coverage_counts``
    are the header recount, keyed by the enum string values. There is, by design,
    no field that aggregates the findings into a single verdict.
    """

    clause_findings: list[ClauseFinding] = []
    absence_findings: list[AbsenceFinding] = []
    level_counts: dict[str, int] = {}
    coverage_counts: dict[str, int] = {}
