"""The closed list of case variables that change which legal regime applies.

Which facts about a tenant's case flip the applicable regime is vertical
knowledge, not engine knowledge, so it ships as data inside the vertical package
alongside its manifest. The engine only ever sees an ordered list of branches: a
question to ask, the kind of answer expected, whether the answer re-anchors the
run in time, and the assumption to state out loud when the user does not answer.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

_ANSWER_PLACEHOLDER = "{answer}"


class BranchesError(ValueError):
    """Raised when a vertical's branch package is missing or malformed."""


class AnswerKind(StrEnum):
    """What the user is expected to reply with, so the client can offer the right input."""

    DATE = "fecha"
    TEXT = "texto"


@dataclass(frozen=True)
class CriticalBranch:
    """One case variable whose value changes the applicable regime.

    ``sets_target_date`` marks the branch whose answer re-anchors the run in time
    (the contract's signing date); such a branch must be answerable with a date.
    ``fact_template`` turns the user's reply into the sentence synthesis reads, so
    the phrasing of a resolved branch is vertical data too. ``assumption`` is what
    the answer states explicitly when the user does not resolve the branch, so an
    unanswered question never becomes a silent guess.
    """

    id: str
    question: str
    answer_kind: AnswerKind
    sets_target_date: bool
    fact_template: str
    assumption: str

    def state_fact(self, answer: str) -> str:
        """Render the user's ``answer`` as the case fact synthesis is told about."""
        return self.fact_template.format(answer=answer)


def load_branches(path: Path) -> tuple[CriticalBranch, ...]:
    """Read and validate a vertical's branch package, in declared order.

    The declared order is the priority order: only the first unresolved branch is
    ever asked about. Raises :class:`BranchesError` on any structural problem.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BranchesError(f"branch package not found: {path}") from error
    except json.JSONDecodeError as error:
        raise BranchesError(f"branch package {path} is not valid JSON: {error}") from error

    entries = raw.get("branches") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise BranchesError(f"branch package {path} missing a non-empty 'branches' array")

    branches = tuple(_read_branch(entry, path) for entry in entries)
    _reject_duplicates(branches, path)
    return branches


def _read_branch(entry: object, path: Path) -> CriticalBranch:
    """Validate one branch entry into a :class:`CriticalBranch`."""
    if not isinstance(entry, dict):
        raise BranchesError(f"branch package {path} has a non-object branch: {entry!r}")

    branch = CriticalBranch(
        id=_read_text(entry, "id", path),
        question=_read_text(entry, "question", path),
        answer_kind=_read_answer_kind(entry, path),
        sets_target_date=bool(entry.get("sets_target_date", False)),
        fact_template=_read_text(entry, "fact_template", path),
        assumption=_read_text(entry, "assumption", path),
    )
    if branch.sets_target_date and branch.answer_kind is not AnswerKind.DATE:
        raise BranchesError(
            f"branch package {path}: branch '{branch.id}' sets the target date "
            f"but expects a '{branch.answer_kind}' answer"
        )
    if _ANSWER_PLACEHOLDER not in branch.fact_template:
        raise BranchesError(
            f"branch package {path}: branch '{branch.id}' has a fact_template "
            f"without the {_ANSWER_PLACEHOLDER} placeholder"
        )
    return branch


def _read_text(entry: dict, field: str, path: Path) -> str:
    """Read a required non-empty string field from a branch entry."""
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BranchesError(f"branch package {path}: branch missing a non-empty '{field}'")
    return value.strip()


def _read_answer_kind(entry: dict, path: Path) -> AnswerKind:
    """Read the branch's expected answer kind, rejecting unknown values."""
    value = entry.get("answer_kind")
    try:
        return AnswerKind(value)
    except ValueError as error:
        raise BranchesError(
            f"branch package {path}: unknown answer_kind {value!r}; "
            f"expected one of {[kind.value for kind in AnswerKind]}"
        ) from error


def _reject_duplicates(branches: tuple[CriticalBranch, ...], path: Path) -> None:
    """Fail when two branches share an id, which would make the priority order ambiguous."""
    seen: set[str] = set()
    for branch in branches:
        if branch.id in seen:
            raise BranchesError(f"branch package {path}: duplicate branch id '{branch.id}'")
        seen.add(branch.id)
