"""The scope gate over the shipped reference contract, driven by the real pipeline.

``contrato-abusivo-01`` is an ordinary permanent lease whose first clause is an
illegal eleven-month, no-renewal term -- the clause a model most easily mistakes
for a seasonal let. These tests program that mistake and pin the two directions the
gate keeps apart: an exclusion with no declaration behind it is analyzed anyway,
and a declared one is still refused.
"""

from datetime import date
from pathlib import Path

from lexme.eval.mode2.cases import Mode2EvalCase, load_mode2_cases
from lexme.eval.mode2.metrics import aggregate_mode2, build_mode2_case_result
from lexme.eval.mode2.runner import PipelineMode2CaseRunner
from lexme.llm import FakeLlmClient
from lexme.mode2.mapping import MAPPING_TASK
from lexme.mode2.models import Mode2Outcome, RejectionReason, TenancyUse
from lexme.mode2.segmentation import SEGMENTATION_TASK
from lexme.mode2.triage import TRIAGE_TASK
from tests.mode2.conftest import (
    FakeCorpus,
    FakeRetriever,
    checklist_of,
    mapping_of,
    scope_package,
    segmentation_of,
    triage_of,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CASE_FILE = REPO_ROOT / "verticales" / "vivienda" / "refset" / "modo2" / "contrato-abusivo-01.json"
TODAY = date(2024, 6, 1)
VERTICAL = "vivienda"

# The term clause a model reads as a seasonal let, quoted from the case itself.
TERM_SPAN = "duración de once meses, sin derecho a prórroga alguna"


def _case() -> Mode2EvalCase:
    """The shipped reference case, loaded from the file the eval run reads."""
    (case,) = load_mode2_cases(CASE_FILE)
    return case


def _llm(
    case: Mode2EvalCase,
    *,
    use_evidence: str,
    use: TenancyUse = TenancyUse.SEASONAL,
) -> FakeLlmClient:
    """A model that segments the case's own clauses and calls the lease seasonal.

    Every clause is mapped as non-evaluable, so the risk map is assembled without
    classification calls: these tests are about the gate, not about the levels.
    """
    fake = FakeLlmClient()
    fake.queue(
        SEGMENTATION_TASK,
        segmentation_of(*[(clause.heading, clause.text) for clause in case.clauses]),
    )
    fake.queue(
        TRIAGE_TASK,
        triage_of(
            duracion="once meses",
            uso=use,
            uso_evidencia=use_evidence,
            fecha_firma="2023-01-01",
        ),
    )
    fake.queue(
        MAPPING_TASK,
        mapping_of(*[(f"c{index}", False, []) for index in range(1, len(case.clauses) + 1)]),
    )
    return fake


def _runner(fake: FakeLlmClient) -> PipelineMode2CaseRunner:
    return PipelineMode2CaseRunner(
        llm=fake,
        scope=scope_package(),
        checklist=checklist_of(),
        corpus=FakeCorpus(),
        retriever=FakeRetriever(),
        vertical=VERTICAL,
    )


def test_the_abusive_contract_is_analyzed_despite_a_seasonal_misreading() -> None:
    case = _case()

    analysis = _runner(_llm(case, use_evidence="")).run(case.document, TODAY)

    assert analysis.outcome is Mode2Outcome.ANALYZED
    assert analysis.risk_map is not None
    assert len(analysis.clauses) == len(case.clauses)


def test_its_own_eleven_month_term_is_not_evidence_of_a_seasonal_let() -> None:
    case = _case()

    analysis = _runner(_llm(case, use_evidence=TERM_SPAN)).run(case.document, TODAY)

    assert TERM_SPAN in case.document  # the span is real; it just declares no destination
    assert analysis.outcome is Mode2Outcome.ANALYZED


def test_the_analyzed_contract_matches_its_declared_outcome_and_delimits_every_clause() -> None:
    case = _case()

    analysis = _runner(_llm(case, use_evidence="")).run(case.document, TODAY)
    metrics = aggregate_mode2([build_mode2_case_result(case, analysis)])

    assert metrics.outcome_match_rate == 1.0
    assert metrics.segmentation_delimited_rate == 1.0
    assert metrics.problematic_total_e2e == 5


def test_a_contract_that_declares_a_non_dwelling_use_is_still_refused() -> None:
    declaration = "El local se destina a oficina y actividad profesional del arrendatario."
    case = _case()
    document = f"{case.document}\nUSO\n{declaration}\n"
    fake = _llm(case, use_evidence=declaration, use=TenancyUse.NON_DWELLING)

    analysis = _runner(fake).run(document, TODAY)

    assert analysis.outcome is Mode2Outcome.OUT_OF_SCOPE
    assert analysis.rejection is not None
    assert analysis.rejection.reason is RejectionReason.OUT_OF_SCOPE_USE
