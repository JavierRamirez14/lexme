"""Eval cases: hand-authored Mode 1 questions and the corpus blocks they should ground on.

A case is a data file outside the engine, one JSON per case, versioned in the repo
and reviewable in a pull request. The engine only sees a question and the gold
block ids a good answer must cite; it never branches on which case it is running.
The generated-by-construction set arrives later and produces the same shape.
"""

import json
from dataclasses import dataclass
from pathlib import Path


class CasesError(ValueError):
    """Raised when a cases file or directory is missing or malformed."""


@dataclass(frozen=True)
class EvalCase:
    """One Mode 1 evaluation case: a question and the blocks it should ground on.

    ``gold_block_ids`` are the corpus blocks a correct answer must cite, the
    reference recall is measured against. ``expected_outcome`` is the outcome the
    case was written to reach, or ``None`` when the case pins no outcome.
    """

    id: str
    question: str
    gold_block_ids: tuple[str, ...] = ()
    expected_outcome: str | None = None


def load_cases(path: Path) -> tuple[EvalCase, ...]:
    """Load the case set at ``path``, ordered by case id.

    ``path`` may be a single ``*.json`` case file or a directory of them. Raises
    :class:`CasesError` when the path is missing, empty, holds malformed JSON, or
    two cases share an id.
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
            raise CasesError(f"no '*.json' cases found in directory {path}")
        return files
    if path.is_file():
        return [path]
    raise CasesError(f"cases path not found: {path}")


def _read_case(file: Path) -> EvalCase:
    """Read and validate one case file into an :class:`EvalCase`."""
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CasesError(f"case {file} is not valid JSON: {error}") from error

    if not isinstance(raw, dict):
        raise CasesError(f"case {file} must be a JSON object")

    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise CasesError(f"case {file} missing a non-empty 'id'")

    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise CasesError(f"case {file} missing a non-empty 'question'")

    gold = _read_gold_block_ids(raw, file)

    expected_outcome = raw.get("expected_outcome")
    if expected_outcome is not None and not isinstance(expected_outcome, str):
        raise CasesError(f"case {file} 'expected_outcome' must be a string when present")

    return EvalCase(
        id=case_id,
        question=question,
        gold_block_ids=gold,
        expected_outcome=expected_outcome,
    )


def _read_gold_block_ids(raw: dict, file: Path) -> tuple[str, ...]:
    """Extract the optional ``gold_block_ids`` array of non-empty strings."""
    gold = raw.get("gold_block_ids", [])
    if not isinstance(gold, list):
        raise CasesError(f"case {file} 'gold_block_ids' must be an array")
    for block_id in gold:
        if not isinstance(block_id, str) or not block_id:
            raise CasesError(f"case {file} has a non-string 'gold_block_ids' entry: {block_id!r}")
    return tuple(gold)


def _reject_duplicate_ids(cases: list[EvalCase]) -> None:
    """Fail loudly when two cases share an id, so a run cannot silently drop one."""
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise CasesError(f"duplicate case id: {case.id!r}")
        seen.add(case.id)
