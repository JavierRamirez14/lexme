"""End-to-end harness test: the real Mode 1 system behind the fake LLM.

This is the criterion that ``eval`` runs the minimal set against the real system
and emits an artifact with a full configuration fingerprint. Real hybrid retrieval,
real citation verification and the real gate run against the seeded LAU corpus;
only the model is faked. The guardrail re-verifies the displayed citation against
the same corpus and passes, because the honest path never shows a non-literal quote.
"""

from datetime import UTC, datetime
from pathlib import Path

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver

from lexme.eval.artifact import read_artifact
from lexme.eval.cases import EvalCase
from lexme.eval.fingerprint import (
    build_fingerprint,
    compute_dataset_digest,
    compute_prompts_digest,
)
from lexme.eval.runner import Mode1CaseRunner, run_suite
from lexme.ingestion.repository import get_corpus_digest
from lexme.llm import FakeLlmClient, load_task_registry
from lexme.mode1 import Mode1Deps, Outcome, PsycopgVersionHistory
from lexme.verification import CorpusReader
from tests.conftest import DeterministicEmbedder
from tests.mode1.conftest import AS_OF, REPO_ROOT, VERTICAL, program_single_sufficient

QUESTION = "¿cuál es el plazo mínimo del arrendamiento de vivienda?"
QUOTE = "La duración del arrendamiento será libremente pactada por las partes"
NOW = datetime(2026, 7, 25, tzinfo=UTC)


def _runner(
    connection: psycopg.Connection,
    embedder: DeterministicEmbedder,
    corpus: CorpusReader,
    llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
) -> Mode1CaseRunner:
    """A real Mode 1 case runner over the seeded corpus and the fake LLM."""
    deps = Mode1Deps(
        connection=connection,
        embedder=embedder,
        corpus=corpus,
        history=PsycopgVersionHistory(connection),
        llm=llm,
        branches=(),
    )
    return Mode1CaseRunner(deps=deps, checkpointer=checkpointer, vertical=VERTICAL)


def test_the_harness_runs_a_case_end_to_end_and_emits_a_passing_artifact(
    seeded_corpus: psycopg.Connection,
    deterministic_embedder: DeterministicEmbedder,
    corpus_reader: CorpusReader,
    fake_llm: FakeLlmClient,
    checkpointer: BaseCheckpointSaver,
    tmp_path: Path,
) -> None:
    program_single_sufficient(fake_llm, query_text=QUESTION, citation=("a9", QUOTE))
    cases = (EvalCase(id="plazo", question=QUESTION, gold_block_ids=("a9",)),)
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


def test_the_corpus_digest_is_stable_and_reflects_content(
    seeded_corpus: psycopg.Connection,
) -> None:
    first = get_corpus_digest(seeded_corpus, VERTICAL)
    second = get_corpus_digest(seeded_corpus, VERTICAL)

    assert first is not None
    assert first == second
    assert get_corpus_digest(seeded_corpus, "otra-vertical") is None
