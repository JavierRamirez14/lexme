"""Tests for the configuration fingerprint: same config hashes equal, changes differ."""

import json
from pathlib import Path

from lexme.eval.cases import ClarificationAnswer, EvalCase, KeyPoint
from lexme.eval.fingerprint import (
    ConfigFingerprint,
    build_fingerprint,
    compute_dataset_digest,
    compute_mode2_dataset_digest,
    compute_prompts_digest,
)
from lexme.eval.mode2.cases import Mode2EvalCase
from lexme.llm import TaskModel, TaskRegistry


def _registry(model: str = "model-a") -> TaskRegistry:
    """A one-task registry pinned to ``model``."""
    return TaskRegistry(
        _by_task={
            "mode1_synthesis": TaskModel(
                task="mode1_synthesis", provider="openrouter", model=model, temperature=0.0
            )
        }
    )


def _vertical_dir(tmp_path: Path, checklist: dict | None = None) -> Path:
    """A vertical data directory carrying a manifest and optional checklist."""
    directory = tmp_path / "vivienda"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"vertical": "vivienda", "norms": [{"norm_id": "BOE-A-1994-26003"}]}),
        encoding="utf-8",
    )
    if checklist is not None:
        (directory / "checklist.json").write_text(json.dumps(checklist), encoding="utf-8")
    return directory


def _build(
    registry: TaskRegistry,
    vertical_dir: Path,
    *,
    corpus: str | None = "corpus-digest",
    prompts: str = "prompts-digest",
    dataset: str = "dataset-digest",
) -> ConfigFingerprint:
    """Build a fingerprint with defaulted digests, so a test varies one input at a time."""
    return build_fingerprint(registry, "vivienda", vertical_dir, corpus, prompts, dataset)


def test_two_runs_of_the_same_config_declare_the_same_fingerprint(tmp_path: Path) -> None:
    vertical_dir = _vertical_dir(tmp_path)

    assert (
        _build(_registry(), vertical_dir).fingerprint
        == _build(_registry(), vertical_dir).fingerprint
    )


def test_a_model_change_changes_the_fingerprint(tmp_path: Path) -> None:
    vertical_dir = _vertical_dir(tmp_path)

    before = _build(_registry("model-a"), vertical_dir)
    after = _build(_registry("model-b"), vertical_dir)

    assert before.fingerprint != after.fingerprint
    assert after.models["mode1_synthesis"].model == "model-b"


def test_a_corpus_change_changes_the_fingerprint(tmp_path: Path) -> None:
    vertical_dir = _vertical_dir(tmp_path)

    before = _build(_registry(), vertical_dir, corpus="digest-1")
    after = _build(_registry(), vertical_dir, corpus="digest-2")

    assert before.fingerprint != after.fingerprint


def test_a_prompt_change_changes_the_fingerprint(tmp_path: Path) -> None:
    vertical_dir = _vertical_dir(tmp_path)

    before = _build(_registry(), vertical_dir, prompts="prompts-1")
    after = _build(_registry(), vertical_dir, prompts="prompts-2")

    assert before.prompts_hash != after.prompts_hash
    assert before.fingerprint != after.fingerprint


def test_a_dataset_change_changes_the_fingerprint(tmp_path: Path) -> None:
    vertical_dir = _vertical_dir(tmp_path)

    before = _build(_registry(), vertical_dir, dataset="dataset-1")
    after = _build(_registry(), vertical_dir, dataset="dataset-2")

    assert before.dataset_hash != after.dataset_hash
    assert before.fingerprint != after.fingerprint


def test_a_vertical_config_change_changes_the_fingerprint(tmp_path: Path) -> None:
    plain = _vertical_dir(tmp_path / "plain")
    with_checklist = _vertical_dir(tmp_path / "edited", checklist={"items": ["fianza"]})

    before = _build(_registry(), plain)
    after = _build(_registry(), with_checklist)

    assert before.vertical_config_hash != after.vertical_config_hash
    assert before.fingerprint != after.fingerprint


def test_an_empty_corpus_still_fingerprints(tmp_path: Path) -> None:
    fingerprint = _build(_registry(), _vertical_dir(tmp_path), corpus=None)

    assert fingerprint.corpus_hash == "empty"
    assert fingerprint.fingerprint


def test_the_prompts_digest_reads_the_real_prompt_sources() -> None:
    first = compute_prompts_digest()
    second = compute_prompts_digest()

    assert first == second
    assert first


def test_the_dataset_digest_is_order_independent_and_content_sensitive() -> None:
    a = EvalCase(id="a", question="q1", gold_block_refs=("BOE-A-1994-26003:x",))
    b = EvalCase(id="b", question="q2")

    assert compute_dataset_digest([a, b]) == compute_dataset_digest([b, a])
    assert compute_dataset_digest([a, b]) != compute_dataset_digest([a])


def test_editing_a_key_point_changes_the_dataset_digest() -> None:
    def case(claim: str) -> EvalCase:
        return EvalCase(
            id="a",
            question="q1",
            gold_block_refs=("BOE-A-1994-26003:a36",),
            key_points=(KeyPoint(claim=claim, block_ref="BOE-A-1994-26003:a36"),),
        )

    assert compute_dataset_digest([case("una mensualidad")]) != compute_dataset_digest(
        [case("dos mensualidades")]
    )


def test_editing_a_pinned_clarification_answer_changes_the_dataset_digest() -> None:
    def case(answer: str) -> EvalCase:
        return EvalCase(
            id="a",
            question="q1",
            clarification_answers=(ClarificationAnswer(branch_id="fecha_firma", answer=answer),),
        )

    assert compute_dataset_digest([case("12/06/2025")]) != compute_dataset_digest(
        [case("01/01/2024")]
    )
    assert compute_dataset_digest([case("12/06/2025")]) != compute_dataset_digest(
        [EvalCase(id="a", question="q1")]
    )


def test_editing_a_mode2_case_expected_outcome_changes_the_dataset_digest() -> None:
    def case(expected_outcome: str) -> Mode2EvalCase:
        return Mode2EvalCase(id="a", document="d", expected_outcome=expected_outcome)

    assert compute_mode2_dataset_digest([case("analizado")]) != compute_mode2_dataset_digest(
        [case("fuera_de_ambito:uso_fuera_de_ambito")]
    )
