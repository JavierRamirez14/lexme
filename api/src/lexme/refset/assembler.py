"""Assemble a synthetic contract from bank clauses, with its ground truth for free.

The assembler is pure code: it lays labelled bank clauses into one document and,
because it places every clause itself, it knows each clause's exact character span --
so the segmentation layer's reference boundaries fall out of construction rather than
being labelled by hand. The clauses it does not include are the deliberate omissions:
every checklist protection no included clause reflects becomes an expected white,
the differentiator a flat contract review cannot produce.
"""

from collections.abc import Sequence
from datetime import datetime

from lexme.checklist import Checklist
from lexme.refset.clause_bank import BankClause, absent_items
from lexme.refset.models import (
    Candidate,
    CaseKind,
    ClauseGroundTruth,
    ExpectedAbsence,
    Mode2ReferenceCase,
    Provenance,
)

_GENERATOR_NAME = "assembler"
_CLAUSE_SEPARATOR = "\n\n"
_HEADING_SEPARATOR = "\n"


class AssemblyError(ValueError):
    """Raised when a contract cannot be assembled from the given clauses."""


def assemble_contract(
    case_id: str,
    clauses: Sequence[BankClause],
    checklist: Checklist,
    *,
    now: datetime,
    header: str = "",
) -> Candidate:
    """Assemble ``clauses`` into a pending Mode 2 candidate with full ground truth.

    Lays the clauses into a single document under an optional ``header``, recording
    each clause's known level and its exact character span, and derives the expected
    absences: every checklist item, other than the closing meta-rule, that no
    included clause covers. ``now`` stamps the provenance.

    Raises :class:`AssemblyError` when ``clauses`` is empty.
    """
    if not clauses:
        raise AssemblyError(f"contract '{case_id}': at least one clause is required")

    document, clause_truths = _lay_out(case_id, clauses, header)
    absences = _expected_absences(clauses, checklist)
    case = Mode2ReferenceCase(
        id=case_id,
        document=document,
        clauses=clause_truths,
        expected_absences=absences,
    )
    provenance = Provenance(
        generator=_GENERATOR_NAME,
        sources=[clause.id for clause in clauses],
        generated_at=now,
    )
    return Candidate(kind=CaseKind.MODE2, provenance=provenance, mode2=case)


def _lay_out(
    case_id: str, clauses: Sequence[BankClause], header: str
) -> tuple[str, list[ClauseGroundTruth]]:
    """Concatenate the clauses into a document, tracking each body's span.

    Each clause is rendered as its heading line followed by its body; the body's
    half-open ``[start, end)`` range indexes the returned document exactly.
    """
    parts: list[str] = []
    offset = 0
    if header.strip():
        block = header.strip() + _CLAUSE_SEPARATOR
        parts.append(block)
        offset += len(block)

    truths: list[ClauseGroundTruth] = []
    for index, clause in enumerate(clauses, start=1):
        heading_line = clause.heading + _HEADING_SEPARATOR
        parts.append(heading_line)
        offset += len(heading_line)

        start = offset
        parts.append(clause.text)
        offset += len(clause.text)
        end = offset

        parts.append(_CLAUSE_SEPARATOR)
        offset += len(_CLAUSE_SEPARATOR)

        truths.append(
            ClauseGroundTruth(
                clause_id=f"{case_id}-c{index}",
                heading=clause.heading,
                text=clause.text,
                start=start,
                end=end,
                expected_level=clause.expected_level,
                chk_ids=list(clause.chk_ids),
                source=clause.source,
            )
        )
    return "".join(parts), truths


def _expected_absences(
    clauses: Sequence[BankClause], checklist: Checklist
) -> list[ExpectedAbsence]:
    """Derive the whites: the checklist items no included clause covers."""
    covered = {chk_id for clause in clauses for chk_id in clause.chk_ids}
    return [
        ExpectedAbsence(item_id=item.id, right=item.right)
        for item in absent_items(checklist, covered)
    ]
