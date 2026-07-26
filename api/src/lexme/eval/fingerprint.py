"""The configuration fingerprint: what a run's numbers are comparable across.

A run's metrics only compare to another run's when the models, the prompts, the
vertical's configuration, the corpus and the case set behind them are identical.
This distils those into one short, stable hash: two runs of the same system
declare the same fingerprint, and any change to a model, a prompt, a vertical data
file, the ingested corpus or the dataset changes it, so a regression can always be
attributed to what moved.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

import lexme
from lexme.eval.cases import EvalCase
from lexme.llm import TaskRegistry

if TYPE_CHECKING:
    from lexme.eval.mode2.cases import Mode2EvalCase

VERTICAL_CONFIG_FILENAMES = (
    "manifest.json",
    "checklist.json",
    "disambiguation.json",
    "scope.json",
)

PROMPT_SOURCE_FILES = (
    "mode1/graph/router.py",
    "mode1/graph/planner.py",
    "mode1/graph/critique.py",
    "mode1/synthesis.py",
)

MODE2_PROMPT_SOURCE_FILES = (
    "mode2/segmentation.py",
    "mode2/triage.py",
    "mode2/mapping.py",
    "mode2/classify.py",
)

_MISSING_CORPUS_DIGEST = "empty"
_DIGEST_LENGTH = 16


class TaskFingerprint(BaseModel):
    """The pinned provider, model and temperature bound to one task."""

    provider: str
    model: str
    temperature: float


class ConfigFingerprint(BaseModel):
    """The full configuration a run was produced under, plus its combined hash.

    ``models`` is the per-task model pin, ``prompts_hash`` the hash of the prompt
    source, ``vertical_config_hash`` the vertical's data files, ``corpus_hash`` the
    ingested corpus's content digest and ``dataset_hash`` the case set. ``fingerprint``
    is the short hash over all of them: the single value two runs are compared as
    same-or-different by.
    """

    models: dict[str, TaskFingerprint]
    vertical: str
    prompts_hash: str
    vertical_config_hash: str
    corpus_hash: str
    dataset_hash: str
    fingerprint: str


def build_fingerprint(
    registry: TaskRegistry,
    vertical: str,
    vertical_dir: Path,
    corpus_digest: str | None,
    prompts_digest: str,
    dataset_digest: str,
) -> ConfigFingerprint:
    """Assemble the configuration fingerprint from its sources.

    ``registry`` supplies the per-task model pins, ``vertical_dir`` the vertical's
    data files, ``corpus_digest`` the ingested corpus's content hash (``None`` for
    an empty corpus), ``prompts_digest`` the prompt-source hash and ``dataset_digest``
    the case-set hash. The combined ``fingerprint`` is deterministic in these inputs
    and independent of dict iteration order.
    """
    models = {task: _task_fingerprint(registry, task) for task in registry.tasks()}
    vertical_config_hash = _hash_vertical_config(vertical_dir)
    corpus_hash = corpus_digest or _MISSING_CORPUS_DIGEST
    combined = _digest(
        {
            "models": {task: model.model_dump() for task, model in models.items()},
            "vertical": vertical,
            "prompts_hash": prompts_digest,
            "vertical_config_hash": vertical_config_hash,
            "corpus_hash": corpus_hash,
            "dataset_hash": dataset_digest,
        }
    )
    return ConfigFingerprint(
        models=models,
        vertical=vertical,
        prompts_hash=prompts_digest,
        vertical_config_hash=vertical_config_hash,
        corpus_hash=corpus_hash,
        dataset_hash=dataset_digest,
        fingerprint=combined,
    )


def compute_prompts_digest() -> str:
    """Hash the source of the modules that own the Mode 1 agentic prompts.

    Editing a router, planner, critique or synthesis prompt changes the source of
    its module and so changes this digest, which is what makes a prompt change a
    detectable configuration change. Line endings are normalized so a checkout's
    newline style does not move the hash.
    """
    return _hash_prompt_sources(PROMPT_SOURCE_FILES)


def compute_mode2_prompts_digest() -> str:
    """Hash the source of the modules that own the Mode 2 pipeline prompts.

    Editing the segmentation, triage, mapping or classification prompt changes its
    module's source and so this digest, so a Mode 2 prompt change is a detectable
    configuration change, independent of the Mode 1 prompt digest.
    """
    return _hash_prompt_sources(MODE2_PROMPT_SOURCE_FILES)


def _hash_prompt_sources(relatives: Sequence[str]) -> str:
    """Hash the given prompt-owning source files, newline-normalized, by path."""
    package_dir = Path(lexme.__file__).parent
    payload: dict[str, str] = {}
    for relative in relatives:
        path = package_dir / relative
        if path.is_file():
            payload[relative] = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return _digest(payload)


def compute_dataset_digest(cases: Sequence[EvalCase]) -> str:
    """Hash the case set a run was measured over, in case-id order.

    Two runs over different questions or gold blocks are different experiments, not
    a regression pair; folding the dataset into the fingerprint keeps that honest.
    """
    payload = [
        {
            "id": case.id,
            "question": case.question,
            "gold_block_ids": list(case.gold_block_ids),
            "expected_outcome": case.expected_outcome,
            "key_points": [
                {"claim": point.claim, "block_id": point.block_id} for point in case.key_points
            ],
            "target_date": case.target_date.isoformat() if case.target_date else None,
        }
        for case in sorted(cases, key=lambda case: case.id)
    ]
    return _digest(payload)


def compute_mode2_dataset_digest(cases: "Sequence[Mode2EvalCase]") -> str:
    """Hash the Mode 2 contract set a run was measured over, in case-id order.

    Folds the document, each clause's span and level, and the expected absences into
    one digest, so a run over a different or edited contract set is a different
    experiment rather than a regression pair.
    """
    payload = [
        {
            "id": case.id,
            "document": case.document,
            "clauses": [
                {
                    "clause_id": clause.clause_id,
                    "start": clause.start,
                    "end": clause.end,
                    "expected_level": clause.expected_level.value,
                    "chk_ids": list(clause.chk_ids),
                }
                for clause in case.clauses
            ],
            "expected_absences": [
                {"item_id": absence.item_id} for absence in case.expected_absences
            ],
        }
        for case in sorted(cases, key=lambda case: case.id)
    ]
    return _digest(payload)


def _task_fingerprint(registry: TaskRegistry, task: str) -> TaskFingerprint:
    """Read one task's pinned model out of the registry."""
    model = registry.resolve(task)
    return TaskFingerprint(
        provider=model.provider,
        model=model.model,
        temperature=model.temperature,
    )


def _hash_vertical_config(vertical_dir: Path) -> str:
    """Hash the vertical's data files, canonicalized and in a fixed order.

    Each known config file present in ``vertical_dir`` is parsed and re-serialized
    canonically before hashing, so a whitespace-only edit does not change the hash
    but any change to the data does. A missing file contributes nothing.
    """
    payload: dict[str, object] = {}
    for filename in VERTICAL_CONFIG_FILENAMES:
        path = vertical_dir / filename
        if path.is_file():
            payload[filename] = json.loads(path.read_text(encoding="utf-8"))
    return _digest(payload)


def _digest(payload: object) -> str:
    """Return a short hex digest of ``payload``'s canonical JSON form."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
