"""The candidate store and the human review filter that guards the reference set.

Every generated case lands here as a pending candidate. The invariant this module
exists to hold is simple: a case reaches the versioned set the harness consumes
*only* by being accepted. Generation writes candidates; acceptance stamps who
reviewed it and when, then materializes it into the set carrying its provenance;
rejection records the reason and materializes nothing. The set on disk is therefore
never anything but reviewed cases, and each one says where it came from.
"""

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from lexme.refset.models import Candidate, CaseKind, ReviewStatus

CANDIDATES_DIRNAME = "candidates"
MODE1_SUITE_DIRNAME = "modo1"
MODE2_SUITE_DIRNAME = "modo2"


class StoreError(ValueError):
    """Raised when a store operation is invalid: unknown id, or a bad status change."""


class CandidateStore:
    """A file-backed store of candidates plus the review gate into the eval set.

    ``refset_dir`` holds the candidates and the accepted Mode 2 contracts;
    ``eval_dir`` holds the harness suites, where accepted Mode 1 cases are
    materialized so ``eval`` consumes them directly.
    """

    def __init__(self, refset_dir: Path, eval_dir: Path) -> None:
        """Bind the store to a vertical's refset and eval directories."""
        self._refset_dir = refset_dir
        self._eval_dir = eval_dir
        self._candidates_dir = refset_dir / CANDIDATES_DIRNAME

    def write(self, candidate: Candidate) -> Path:
        """Persist ``candidate`` as pending, returning its file path.

        Raises :class:`StoreError` if a candidate with the same id has already been
        accepted, so a regeneration cannot silently desync the materialized set.
        """
        existing = self._read_if_present(candidate.id)
        if existing is not None and existing.status is ReviewStatus.ACCEPTED:
            raise StoreError(f"candidate '{candidate.id}' is already accepted; cannot overwrite")
        self._candidates_dir.mkdir(parents=True, exist_ok=True)
        path = self._candidate_path(candidate.id)
        path.write_text(candidate.model_dump_json(indent=2), encoding="utf-8")
        return path

    def get(self, candidate_id: str) -> Candidate:
        """Load the candidate with ``candidate_id`` or raise :class:`StoreError`."""
        candidate = self._read_if_present(candidate_id)
        if candidate is None:
            raise StoreError(f"unknown candidate '{candidate_id}'")
        return candidate

    def list_pending(self, kind: CaseKind | None = None) -> list[Candidate]:
        """Return the pending candidates, optionally of one kind, ordered by id."""
        pending = [
            candidate
            for candidate in self._iter_candidates()
            if candidate.status is ReviewStatus.PENDING and (kind is None or candidate.kind is kind)
        ]
        return sorted(pending, key=lambda candidate: candidate.id)

    def accept(self, candidate_id: str, *, reviewed_by: str, now: datetime, note: str = "") -> Path:
        """Accept a pending candidate and materialize it into the eval set.

        Stamps the reviewer and time onto the candidate's provenance, records the
        acceptance on the candidate file, then writes the case -- carrying that
        provenance -- into the harness suite. Returns the materialized path.

        Raises :class:`StoreError` if the candidate is unknown or not pending.
        """
        candidate = self._require_pending(candidate_id)
        reviewed = _stamp_review(candidate, ReviewStatus.ACCEPTED, reviewed_by, now, note)
        self._candidate_path(candidate_id).write_text(
            reviewed.model_dump_json(indent=2), encoding="utf-8"
        )
        return self._materialize(reviewed)

    def reject(self, candidate_id: str, *, reviewed_by: str, reason: str, now: datetime) -> None:
        """Reject a pending candidate, recording the reason and materializing nothing.

        Raises :class:`StoreError` if the candidate is unknown or not pending.
        """
        candidate = self._require_pending(candidate_id)
        reviewed = _stamp_review(candidate, ReviewStatus.REJECTED, reviewed_by, now, reason)
        self._candidate_path(candidate_id).write_text(
            reviewed.model_dump_json(indent=2), encoding="utf-8"
        )

    def materialized_path(self, candidate: Candidate) -> Path:
        """The path an accepted ``candidate`` is (or would be) materialized to."""
        if candidate.kind is CaseKind.MODE1:
            return self._eval_dir / MODE1_SUITE_DIRNAME / f"{candidate.id}.json"
        return self._refset_dir / MODE2_SUITE_DIRNAME / f"{candidate.id}.json"

    def _require_pending(self, candidate_id: str) -> Candidate:
        """Load a candidate and assert it is pending, else raise."""
        candidate = self.get(candidate_id)
        if candidate.status is not ReviewStatus.PENDING:
            raise StoreError(
                f"candidate '{candidate_id}' is '{candidate.status.value}', not pending"
            )
        return candidate

    def _materialize(self, candidate: Candidate) -> Path:
        """Write an accepted candidate's case, with provenance, into the set."""
        payload = candidate.mode1 or candidate.mode2
        assert payload is not None  # guaranteed by the candidate invariant
        data = payload.model_dump(mode="json")
        data["provenance"] = candidate.provenance.model_dump(mode="json")
        path = self.materialized_path(candidate)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _iter_candidates(self) -> Iterator[Candidate]:
        """Yield every stored candidate, skipping nothing silently."""
        if not self._candidates_dir.is_dir():
            return
        for path in sorted(self._candidates_dir.glob("*.json")):
            yield Candidate.model_validate_json(path.read_text(encoding="utf-8"))

    def _read_if_present(self, candidate_id: str) -> Candidate | None:
        """Load a candidate by id, or ``None`` if no file exists for it."""
        path = self._candidate_path(candidate_id)
        if not path.is_file():
            return None
        return Candidate.model_validate_json(path.read_text(encoding="utf-8"))

    def _candidate_path(self, candidate_id: str) -> Path:
        """The candidate file path for ``candidate_id``."""
        return self._candidates_dir / f"{candidate_id}.json"


def _stamp_review(
    candidate: Candidate, status: ReviewStatus, reviewed_by: str, now: datetime, note: str
) -> Candidate:
    """Return a copy of ``candidate`` with its review status and provenance stamped."""
    provenance = candidate.provenance.model_copy(
        update={"reviewed_by": reviewed_by, "reviewed_at": now, "review_note": note}
    )
    return candidate.model_copy(update={"status": status, "provenance": provenance})
