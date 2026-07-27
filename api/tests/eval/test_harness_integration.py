"""End-to-end harness test: the real Mode 1 system behind the fake LLM.

This is the criterion that ``eval`` runs the minimal set against the real system
and emits an artifact with a full configuration fingerprint. Real hybrid retrieval,
real citation verification and the real gate run against the seeded LAU corpus;
only the model is faked. The guardrail re-verifies the displayed citation against
the same corpus and passes, because the honest path never shows a non-literal quote.

The disambiguation gate is exercised here too, because resuming a paused case is
only real when the checkpoint, the interrupt and the branch package are all real:
a case that brings a written answer reaches a final answer on the same thread, a
case that brings none stops at the pause, and the guardrail still rules on the
answer a resumed case ended with.
"""

from datetime import UTC, date, datetime
from pathlib import Path

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver

from lexme.eval.artifact import read_artifact
from lexme.eval.cases import ClarificationAnswer, EvalCase
from lexme.eval.fingerprint import (
    ConfigFingerprint,
    build_fingerprint,
    compute_dataset_digest,
    compute_prompts_digest,
)
from lexme.eval.metrics import Disambiguation
from lexme.eval.runner import Mode1CaseRunner, run_suite
from lexme.ingestion.repository import get_corpus_digest
from lexme.llm import FakeLlmClient, load_task_registry
from lexme.mode1 import CriticalBranch, Mode1Deps, Outcome, PsycopgVersionHistory
from lexme.mode1.models import QueryType
from lexme.verification import CorpusReader
from tests.conftest import DeterministicEmbedder
from tests.mode1.conftest import AS_OF, REPO_ROOT, VERTICAL, program_single_sufficient

QUESTION = "¿cuál es el plazo mínimo del arrendamiento de vivienda?"
QUOTE = "La duración del arrendamiento será libremente pactada por las partes"
SIGNING_BRANCH = "fecha_firma"
SIGNED_ON = "04/05/2017"
NOW = datetime(2026, 7, 25, tzinfo=UTC)


def _runner(
    connection: psycopg.Connection,
    embedder: DeterministicEmbedder,
    corpus: CorpusReader,
    llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
    branches: tuple[CriticalBranch, ...] = (),
) -> Mode1CaseRunner:
    """A real Mode 1 case runner over the seeded corpus and the fake LLM."""
    deps = Mode1Deps(
        connection=connection,
        embedder=embedder,
        corpus=corpus,
        history=PsycopgVersionHistory(connection),
        llm=llm,
        branches=branches,
    )
    return Mode1CaseRunner(deps=deps, checkpointer=checkpointer, vertical=VERTICAL)


def _pausing_case(
    case_id: str, *, pinned: tuple[ClarificationAnswer, ...] = ()
) -> tuple[EvalCase, ...]:
    """A one-case set whose question the planner leaves the signing branch open on."""
    return (
        EvalCase(
            id=case_id,
            question=QUESTION,
            gold_block_refs=("BOE-A-1994-26003:a9",),
            clarification_answers=pinned,
        ),
    )


def _fingerprint() -> ConfigFingerprint:
    """A fixed fingerprint, for the runs whose subject is the disambiguation gate."""
    return ConfigFingerprint(
        models={},
        vertical=VERTICAL,
        prompts_hash="prompts",
        vertical_config_hash="cfg",
        corpus_hash="corpus",
        dataset_hash="dataset",
        fingerprint="fp-under-test",
    )


def _program_pausing_run(llm: FakeLlmClient, citation: tuple[str, str]) -> None:
    """Program a situational run the clarify node stops on, then grounds after resume."""
    program_single_sufficient(
        llm,
        query_text=QUESTION,
        citation=citation,
        query_type=QueryType.SITUATIONAL,
        unresolved=(SIGNING_BRANCH,),
    )


def test_the_harness_runs_a_case_end_to_end_and_emits_a_passing_artifact(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
    tmp_path: Path,
) -> None:
    program_single_sufficient(fake_llm, query_text=QUESTION, citation=("a9", QUOTE))
    cases = (EvalCase(id="plazo", question=QUESTION, gold_block_refs=("BOE-A-1994-26003:a9",)),)
    runner = _runner(seeded_corpus, deterministic_embedder, corpus_reader, fake_llm, checkpointer)
    fingerprint = build_fingerprint(
        load_task_registry(),
        VERTICAL,
        REPO_ROOT / "verticales" / VERTICAL,
        get_corpus_digest(seeded_corpus, VERTICAL),
        compute_prompts_digest(),
        compute_dataset_digest(cases),
    )

    artifact = run_suite("modo1", cases, runner, corpus_reader, AS_OF, fingerprint, NOW)

    assert artifact.passed is True
    assert artifact.cases[0].outcome == Outcome.ANSWER.value
    assert artifact.metrics.mean_recall == 1.0
    assert artifact.fingerprint.corpus_hash is not None
    assert artifact.fingerprint.models["mode1_synthesis"].model

    out = tmp_path / "run.json"
    artifact.write(out)
    assert read_artifact(out) == artifact


def test_a_paused_case_resumes_on_the_same_thread_and_is_measured_on_its_final_answer(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
    vivienda_branches: tuple[CriticalBranch, ...],
) -> None:
    _program_pausing_run(fake_llm, citation=("a9", QUOTE))
    cases = _pausing_case(
        "plazo", pinned=(ClarificationAnswer(branch_id=SIGNING_BRANCH, answer=SIGNED_ON),)
    )
    runner = _runner(
        seeded_corpus,
        deterministic_embedder,
        corpus_reader,
        fake_llm,
        checkpointer,
        vivienda_branches,
    )

    artifact = run_suite("modo1", cases, runner, corpus_reader, AS_OF, _fingerprint(), NOW)

    (result,) = artifact.cases
    assert result.outcome == Outcome.ANSWER.value
    assert result.disambiguation == Disambiguation.RESUMED.value
    assert artifact.metrics.disambiguation_rate == 1.0
    assert artifact.passed is True


def test_a_resumed_case_answers_at_the_date_its_pinned_answer_anchors_it_to(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
    vivienda_branches: tuple[CriticalBranch, ...],
) -> None:
    _program_pausing_run(fake_llm, citation=("a9", QUOTE))
    runner = _runner(
        seeded_corpus,
        deterministic_embedder,
        corpus_reader,
        fake_llm,
        checkpointer,
        vivienda_branches,
    )

    case_run = runner.run(
        QUESTION, AS_OF, (ClarificationAnswer(branch_id=SIGNING_BRANCH, answer=SIGNED_ON),)
    )

    assert case_run.response.answer is not None
    assert case_run.response.answer.fecha_objetivo == date(2017, 5, 4)


def test_a_paused_case_with_no_pinned_answer_ends_as_a_disambiguation(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
    vivienda_branches: tuple[CriticalBranch, ...],
) -> None:
    _program_pausing_run(fake_llm, citation=("a9", QUOTE))
    runner = _runner(
        seeded_corpus,
        deterministic_embedder,
        corpus_reader,
        fake_llm,
        checkpointer,
        vivienda_branches,
    )

    artifact = run_suite(
        "modo1", _pausing_case("plazo"), runner, corpus_reader, AS_OF, _fingerprint(), NOW
    )

    (result,) = artifact.cases
    assert result.outcome == Outcome.CLARIFICATION.value
    assert result.disambiguation == Disambiguation.UNANSWERED.value
    assert artifact.metrics.disambiguation == {"directo": 0, "reanudado": 0, "sin_respuesta": 1}


def test_the_corpus_digest_is_stable_and_reflects_content(
    seeded_corpus: psycopg.Connection,
) -> None:
    first = get_corpus_digest(seeded_corpus, VERTICAL)
    second = get_corpus_digest(seeded_corpus, VERTICAL)

    assert first is not None
    assert first == second
    assert get_corpus_digest(seeded_corpus, "otra-vertical") is None
