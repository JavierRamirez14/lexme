"""The harness that runs a Mode 2 case set end-to-end and assembles its artifact.

The harness drives each synthetic contract through a :class:`Mode2CaseRunner` --
the seam behind which the real Mode 2 pipeline lives -- applies the citation
guardrail to the risk map, measures the quality metrics and folds everything into
one artifact. The runner is a seam so the harness itself is tested with the real
pipeline behind a fake LLM, and never needs a live model of its own. The reference
document is fed straight to the pipeline through a literal text extractor, so the
clause spans the pipeline anchors share the reference's own character coordinates.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, runtime_checkable

from lexme.checklist import Checklist
from lexme.eval.fingerprint import ConfigFingerprint
from lexme.eval.guardrail import GuardrailViolation
from lexme.eval.mode2.artifact import Mode2RunArtifact, build_mode2_artifact
from lexme.eval.mode2.cases import Mode2EvalCase
from lexme.eval.mode2.guardrail import check_mode2_citations
from lexme.eval.mode2.metrics import aggregate_mode2, build_mode2_case_result
from lexme.llm import LlmClient
from lexme.mode2 import (
    ClauseRetriever,
    ContractAnalysis,
    ExtractedText,
    Mode2Deps,
    ScopePackage,
    analyze_contract,
)
from lexme.verification import CorpusReader

_REFERENCE_FILENAME = "reference.txt"


@runtime_checkable
class Mode2CaseRunner(Protocol):
    """Runs one reference contract through the pipeline and returns its analysis."""

    def run(self, document: str, target_date: date) -> ContractAnalysis:
        """Analyze ``document`` situated at ``target_date`` and return the analysis."""
        ...


@dataclass(frozen=True)
class _LiteralExtractor:
    """A :class:`TextExtractor` that returns a fixed text verbatim, ignoring bytes.

    The reference contract is already plain text whose clause spans index into it,
    so the eval feeds it straight in rather than round-tripping through a real
    document format; anchoring then lands on the reference's own coordinates.
    """

    text: str

    def extract(self, filename: str, content: bytes) -> ExtractedText:
        """Return the fixed reference text as a single-page extraction."""
        return ExtractedText(text=self.text, page_count=1)


@dataclass(frozen=True)
class PipelineMode2CaseRunner:
    """A :class:`Mode2CaseRunner` backed by the real Mode 2 pipeline per case.

    Holds every pipeline collaborator except the extractor, which is bound per run
    to the case's own document so the pipeline anchors clauses into the reference's
    character coordinates. The ``target_date`` fixes the point-in-time clock the
    gates and the corpus are resolved at.
    """

    llm: LlmClient
    scope: ScopePackage
    checklist: Checklist
    corpus: CorpusReader
    retriever: ClauseRetriever
    vertical: str

    def run(self, document: str, target_date: date) -> ContractAnalysis:
        """Analyze the reference ``document`` through the real pipeline."""
        deps = Mode2Deps(
            llm=self.llm,
            extractor=_LiteralExtractor(document),
            scope=self.scope,
            checklist=self.checklist,
            corpus=self.corpus,
            retriever=self.retriever,
            vertical=self.vertical,
        )
        return analyze_contract(
            filename=_REFERENCE_FILENAME, content=b"", deps=deps, today=target_date
        )


def run_mode2_suite(
    suite: str,
    cases: Sequence[Mode2EvalCase],
    runner: Mode2CaseRunner,
    corpus: CorpusReader,
    norm_id: str,
    default_date: date,
    fingerprint: ConfigFingerprint,
    created_at: datetime,
) -> Mode2RunArtifact:
    """Run every contract, apply the guardrail and metrics, and build the artifact.

    Each case is analyzed at ``default_date``; the guardrail re-verifies the risk
    map's citations against ``corpus`` under ``norm_id`` at the same date. The
    returned artifact's ``passed`` is false if any displayed citation broke the
    literality invariant.
    """
    results = []
    hard_failures: list[GuardrailViolation] = []
    for case in cases:
        analysis = runner.run(case.document, default_date)
        violations = check_mode2_citations(case.id, analysis, corpus, norm_id, default_date)
        hard_failures.extend(violations)
        results.append(build_mode2_case_result(case, analysis, violations))
    metrics = aggregate_mode2(results)
    return build_mode2_artifact(suite, created_at, fingerprint, results, metrics, hard_failures)
