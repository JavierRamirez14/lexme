"""Mode 2 eval cases: a synthetic contract and its per-clause, per-absence truth.

A case is the materialized form of an accepted reference contract, one JSON file
per case under the vertical's ``refset/modo2`` directory, versioned in the repo and
reviewable in a pull request. It carries the assembled document, each clause's known
level and character span, and the checklist items the contract deliberately omits.
The harness reads only this shape; it never sees, and never branches on, which case
it is running. The loader validates at the boundary and trusts the data within.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from lexme.mode2.risk import RiskLevel

CLAUSE_LEVELS: frozenset[RiskLevel] = frozenset(
    {RiskLevel.ILEGAL, RiskLevel.PEOR_QUE_DEFAULT, RiskLevel.NEGOCIABLE, RiskLevel.CORRECTO}
)


class Mode2CasesError(ValueError):
    """Raised when a Mode 2 case file or directory is missing or malformed."""


@dataclass(frozen=True)
class ClauseTruth:
    """One assembled clause: its heading, its span and its constructed level.

    ``start`` and ``end`` are the half-open character range of the clause body in
    the assembled document, the reference boundaries the segmentation layer scores
    against. ``expected_level`` is the clause's known placement on the spectrum, one
    of the four clause levels and never ``AUSENTE``. ``chk_ids`` names the checklist
    items the clause reflects, and is empty for a clause off the checklist.
    """

    clause_id: str
    heading: str
    text: str
    start: int
    end: int
    expected_level: RiskLevel
    chk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AbsenceTruth:
    """A checklist protection the assembled contract deliberately omits (a white).

    ``item_id`` is the omitted checklist item and ``right`` its plain-language
    protection; the expected level is always ``AUSENTE`` and is left implicit.
    """

    item_id: str
    right: str


@dataclass(frozen=True)
class Mode2EvalCase:
    """One Mode 2 evaluation case: a contract with its clause and absence truth.

    ``document`` is the assembled contract text the clause spans index into;
    ``clauses`` carry the per-clause level and boundaries; ``expected_absences`` are
    the checklist items no clause covers, the whites the cross-check must surface.
    ``expected_outcome`` is the analysis outcome the case was written to reach --
    ``"analizado"`` or the compound ``"<outcome>:<rejection_reason>"`` a gate stop
    produces -- and is never inferred: a case that does not declare it is rejected
    at load time, so a contract that is silently rejected can never disappear from
    the outcome-match denominator the way it used to.
    """

    id: str
    document: str
    expected_outcome: str
    clauses: tuple[ClauseTruth, ...] = ()
    expected_absences: tuple[AbsenceTruth, ...] = ()


def load_mode2_cases(path: Path) -> tuple[Mode2EvalCase, ...]:
    """Load the Mode 2 case set at ``path``, ordered by case id.

    ``path`` may be a single ``*.json`` case file or a directory of them. Raises
    :class:`Mode2CasesError` when the path is missing, empty, holds malformed JSON,
    or two cases share an id.
    """
    files = _case_files(path)
    cases = [_read_case(file) for file in files]
    _reject_duplicate_ids(cases)
    return tuple(sorted(cases, key=lambda case: case.id))


def _case_files(path: Path) -> list[Path]:
    """Return the case files at ``path``, or fail if there are none."""
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            raise Mode2CasesError(f"no '*.json' cases found in directory {path}")
        return files
    if path.is_file():
        return [path]
    raise Mode2CasesError(f"cases path not found: {path}")


def _read_case(file: Path) -> Mode2EvalCase:
    """Read and validate one case file into a :class:`Mode2EvalCase`."""
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise Mode2CasesError(f"case {file} is not valid JSON: {error}") from error

    if not isinstance(raw, dict):
        raise Mode2CasesError(f"case {file} must be a JSON object")

    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise Mode2CasesError(f"case {file} missing a non-empty 'id'")

    document = raw.get("document")
    if not isinstance(document, str) or not document.strip():
        raise Mode2CasesError(f"case {file} missing a non-empty 'document'")

    expected_outcome = raw.get("expected_outcome")
    if not isinstance(expected_outcome, str) or not expected_outcome:
        raise Mode2CasesError(f"case {file} missing a non-empty 'expected_outcome'")

    return Mode2EvalCase(
        id=case_id,
        document=document,
        expected_outcome=expected_outcome,
        clauses=_read_clauses(raw, file, document),
        expected_absences=_read_absences(raw, file),
    )


def _read_clauses(raw: dict[str, object], file: Path, document: str) -> tuple[ClauseTruth, ...]:
    """Extract the ``clauses`` array, validating each clause's span and level."""
    entries = raw.get("clauses", [])
    if not isinstance(entries, list):
        raise Mode2CasesError(f"case {file} 'clauses' must be an array")
    return tuple(_read_clause(entry, file, document) for entry in entries)


def _read_clause(entry: object, file: Path, document: str) -> ClauseTruth:
    """Validate one clause entry into a :class:`ClauseTruth`."""
    if not isinstance(entry, dict):
        raise Mode2CasesError(f"case {file} has a non-object clause: {entry!r}")
    clause_id = _require_str(entry, "clause_id", file)
    start = _require_span_bound(entry, "start", file, document)
    end = _require_span_bound(entry, "end", file, document)
    if end <= start:
        raise Mode2CasesError(
            f"case {file} clause '{clause_id}' has end {end} not after start {start}"
        )
    return ClauseTruth(
        clause_id=clause_id,
        heading=_optional_str(entry, "heading", file),
        text=_optional_str(entry, "text", file),
        start=start,
        end=end,
        expected_level=_read_level(entry, file, clause_id),
        chk_ids=_read_chk_ids(entry, file, clause_id),
    )


def _read_level(entry: dict[str, object], file: Path, clause_id: str) -> RiskLevel:
    """Read a clause's ``expected_level``, rejecting any non-clause level."""
    raw_level = entry.get("expected_level")
    if not isinstance(raw_level, str):
        raise Mode2CasesError(f"case {file} clause '{clause_id}' missing 'expected_level'")
    try:
        level = RiskLevel(raw_level)
    except ValueError as error:
        raise Mode2CasesError(
            f"case {file} clause '{clause_id}' has unknown level '{raw_level}'"
        ) from error
    if level not in CLAUSE_LEVELS:
        raise Mode2CasesError(
            f"case {file} clause '{clause_id}' level '{raw_level}' is not a clause level"
        )
    return level


def _read_chk_ids(entry: dict[str, object], file: Path, clause_id: str) -> tuple[str, ...]:
    """Read a clause's optional ``chk_ids`` array of non-empty strings."""
    chk_ids = entry.get("chk_ids", [])
    if not isinstance(chk_ids, list):
        raise Mode2CasesError(f"case {file} clause '{clause_id}' 'chk_ids' must be an array")
    for chk_id in chk_ids:
        if not isinstance(chk_id, str) or not chk_id:
            raise Mode2CasesError(
                f"case {file} clause '{clause_id}' has a non-string chk_id: {chk_id!r}"
            )
    return tuple(chk_ids)


def _read_absences(raw: dict[str, object], file: Path) -> tuple[AbsenceTruth, ...]:
    """Extract the ``expected_absences`` array of ``(item_id, right)`` whites."""
    entries = raw.get("expected_absences", [])
    if not isinstance(entries, list):
        raise Mode2CasesError(f"case {file} 'expected_absences' must be an array")
    absences = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise Mode2CasesError(f"case {file} has a non-object absence: {entry!r}")
        absences.append(
            AbsenceTruth(
                item_id=_require_str(entry, "item_id", file),
                right=_optional_str(entry, "right", file),
            )
        )
    return tuple(absences)


def _require_str(entry: dict[str, object], key: str, file: Path) -> str:
    """Read a required non-empty string field, or raise."""
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise Mode2CasesError(f"case {file} entry missing a non-empty '{key}'")
    return value


def _optional_str(entry: dict[str, object], key: str, file: Path) -> str:
    """Read an optional string field, defaulting to the empty string when absent.

    A present value of the wrong type is rejected rather than coerced, so the
    loader stays fail-loud like the rest of the boundary validation.
    """
    value = entry.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise Mode2CasesError(f"case {file} field '{key}' must be a string when present")
    return value


def _require_span_bound(entry: dict[str, object], key: str, file: Path, document: str) -> int:
    """Read a required integer span bound and reject one outside the document."""
    value = entry.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise Mode2CasesError(f"case {file} clause bound '{key}' must be an integer")
    if value < 0 or value > len(document):
        raise Mode2CasesError(
            f"case {file} clause bound '{key}'={value} is outside the document [0, {len(document)}]"
        )
    return value


def _reject_duplicate_ids(cases: list[Mode2EvalCase]) -> None:
    """Fail loudly when two cases share an id, so a run cannot silently drop one."""
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise Mode2CasesError(f"duplicate case id: {case.id!r}")
        seen.add(case.id)
