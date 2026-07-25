"""The Mode 2 clause bank: labelled clauses curated once, seeded from public sources.

The bank is the ground truth the contract assembler samples from. Each clause carries
its constructed level -- illegal and worse-than-default seeded from the official
consumer-protection report on abusive tenancy clauses, conforming from public
templates, negotiable for a burden the LAU is silent on -- and the checklist items
it reflects. The bank is validated once against the vertical's checklist so it cannot
silently stop covering a protection: every checklist item is exemplified, the three
citing levels are present, and at least one off-checklist negotiable clause exists.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError, model_validator

from lexme.checklist import Checklist, ChecklistItem, SilenceTone
from lexme.mode2.risk import RiskLevel
from lexme.refset.models import CLAUSE_LEVELS

CLAUSE_BANK_FILENAME = "clause_bank.json"

_LEVELS_REQUIRED_IN_BANK: tuple[RiskLevel, ...] = (
    RiskLevel.ILEGAL,
    RiskLevel.PEOR_QUE_DEFAULT,
    RiskLevel.CORRECTO,
)


def clause_bank_path(verticals_dir: str | Path, vertical: str) -> Path:
    """Return the path to a vertical's clause bank inside the data directory."""
    return Path(verticals_dir) / vertical / "refset" / CLAUSE_BANK_FILENAME


def absent_items(checklist: Checklist, covered_chk_ids: set[str]) -> list[ChecklistItem]:
    """Return the checklist items that can be absent and no clause covers.

    An item can be absent unless it is the closing meta-rule (silence tone
    ``NOT_APPLICABLE``), which never produces an absence finding. This is the one
    rule the coverage check and the contract assembler both read, so it lives once.
    """
    return [
        item
        for item in checklist.items
        if item.silence_tone is not SilenceTone.NOT_APPLICABLE and item.id not in covered_chk_ids
    ]


class ClauseBankError(ValueError):
    """Raised when a clause bank file is missing or structurally malformed."""


class BankClause(BaseModel):
    """One labelled clause in the bank, ready to be sampled into a contract.

    ``expected_level`` is the clause's constructed label, one of the four clause
    levels. ``chk_ids`` are the checklist items it reflects; a negotiable clause is
    off the checklist and carries none. ``source`` records the provenance of the
    label (the consumer report, a public template, the checklist).
    """

    id: str
    heading: str
    text: str
    expected_level: RiskLevel
    chk_ids: list[str] = []
    source: str
    note: str = ""

    @model_validator(mode="after")
    def _consistent_level(self) -> "BankClause":
        """Reject ``AUSENTE`` and a negotiable clause that claims checklist items."""
        if self.expected_level not in CLAUSE_LEVELS:
            raise ValueError(
                f"clause '{self.id}': '{self.expected_level.value}' is not a clause level"
            )
        if self.expected_level is RiskLevel.NEGOCIABLE and self.chk_ids:
            raise ValueError(
                f"clause '{self.id}': a negotiable clause is off the checklist "
                "and carries no chk_ids"
            )
        return self


class ClauseBank(BaseModel):
    """A vertical's whole clause bank, bound to the norm its labels are read against."""

    vertical: str
    norm_id: str
    clauses: list[BankClause]

    @model_validator(mode="after")
    def _unique_ids(self) -> "ClauseBank":
        """Reject two clauses sharing an id, which would make sampling ambiguous."""
        seen: set[str] = set()
        for clause in self.clauses:
            if clause.id in seen:
                raise ValueError(f"duplicate clause id '{clause.id}'")
            seen.add(clause.id)
        return self

    def by_id(self, clause_id: str) -> BankClause:
        """Return the clause with ``clause_id`` or raise :class:`KeyError`."""
        for clause in self.clauses:
            if clause.id == clause_id:
                return clause
        raise KeyError(clause_id)


def load_clause_bank(path: Path) -> ClauseBank:
    """Read and validate a clause bank file into a :class:`ClauseBank`.

    Raises :class:`ClauseBankError` on an unreadable file, invalid JSON, a bad
    level, a negotiable clause with checklist items, or a duplicate clause id.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ClauseBankError(f"clause bank not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ClauseBankError(f"clause bank {path} is not valid JSON: {error}") from error
    try:
        return ClauseBank.model_validate(raw)
    except ValidationError as error:
        raise ClauseBankError(f"clause bank {path} is invalid: {error}") from error


@dataclass(frozen=True)
class CoverageFinding:
    """One gap between the clause bank and what a reference set needs of it."""

    message: str


@dataclass(frozen=True)
class CoverageReport:
    """The result of checking a bank against its checklist: every gap, at once."""

    findings: tuple[CoverageFinding, ...]

    @property
    def is_valid(self) -> bool:
        """Whether the bank exemplifies everything the reference set requires."""
        return not self.findings


def validate_bank_coverage(bank: ClauseBank, checklist: Checklist) -> CoverageReport:
    """Check the bank covers what the reference set needs of it, against ``checklist``.

    Collects a finding when a clause cites an unknown checklist item, when a
    checklist item that can produce an absence is exemplified by no clause, when a
    citing level (illegal, worse-than-default, conforming) is missing, or when no
    off-checklist negotiable clause exists.
    """
    known = {item.id for item in checklist.items}
    cited = {chk_id for clause in bank.clauses for chk_id in clause.chk_ids}
    levels_present = {clause.expected_level for clause in bank.clauses}

    findings: list[CoverageFinding] = []
    findings.extend(_unknown_item_findings(bank, known))
    findings.extend(_uncovered_item_findings(checklist, cited))
    findings.extend(_missing_level_findings(levels_present))
    findings.extend(_negotiable_findings(bank))
    return CoverageReport(findings=tuple(findings))


def _unknown_item_findings(bank: ClauseBank, known: set[str]) -> list[CoverageFinding]:
    """A finding for each clause that cites a checklist item that does not exist."""
    findings = []
    for clause in bank.clauses:
        for chk_id in clause.chk_ids:
            if chk_id not in known:
                findings.append(
                    CoverageFinding(f"clause '{clause.id}' cites unknown checklist item '{chk_id}'")
                )
    return findings


def _uncovered_item_findings(checklist: Checklist, cited: set[str]) -> list[CoverageFinding]:
    """A finding for each checklist item that can be absent but no clause exemplifies."""
    return [
        CoverageFinding(f"checklist item '{item.id}' has no clause in the bank")
        for item in absent_items(checklist, cited)
    ]


def _missing_level_findings(levels_present: set[RiskLevel]) -> list[CoverageFinding]:
    """A finding for each citing level the bank fails to exemplify."""
    return [
        CoverageFinding(f"the bank has no '{level.value}' clause")
        for level in _LEVELS_REQUIRED_IN_BANK
        if level not in levels_present
    ]


def _negotiable_findings(bank: ClauseBank) -> list[CoverageFinding]:
    """A finding when the bank exemplifies no off-checklist negotiable clause."""
    if any(clause.expected_level is RiskLevel.NEGOCIABLE for clause in bank.clauses):
        return []
    return [CoverageFinding("the bank has no off-checklist negotiable clause")]
