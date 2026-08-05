"""End-to-end Mode 2 harness: the real risk-map pipeline behind the fake LLM.

The criterion is that ``run_mode2_suite`` drives a synthetic contract through the
real segmentation, anchoring, classification and cross-check -- only the model is
faked -- measures the false-tranquility numbers and emits a passing artifact whose
finding citations re-verify against the corpus. The reference clause spans are
computed from the document, so the segmentation layer matches at IoU 1.
"""

from datetime import UTC, date, datetime
from pathlib import Path

from lexme.eval.fingerprint import (
    build_fingerprint,
    compute_mode2_dataset_digest,
    compute_mode2_prompts_digest,
)
from lexme.eval.mode2 import (
    ClauseTruth,
    Mode2EvalCase,
    PipelineMode2CaseRunner,
    read_mode2_artifact,
    run_mode2_suite,
)
from lexme.eval.mode2.cases import AbsenceTruth
from lexme.llm import FakeLlmClient, load_task_registry
from lexme.mode2.classify import CLASSIFICATION_TASK
from lexme.mode2.mapping import MAPPING_TASK
from lexme.mode2.risk import ProposedClauseLevel, RiskLevel
from lexme.mode2.scope import ScopePackage
from lexme.mode2.segmentation import SEGMENTATION_TASK
from lexme.mode2.triage import TRIAGE_TASK
from tests.mode2.conftest import (
    NORM_ID,
    FakeCorpus,
    FakeRetriever,
    checklist_item,
    checklist_of,
    classification_of,
    mapping_of,
    resolved_block,
    segmentation_of,
    triage_of,
)

TODAY = date(2024, 6, 1)
NOW = datetime(2026, 7, 26, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[4]
VERTICAL = "vivienda"

RENT_CLAUSE = "La renta se actualiza un diez por ciento anual con independencia del IPC."
TERM_CLAUSE = "Queda prohibida la tenencia de animales en la vivienda arrendada."
DOCUMENT = (
    "CONTRATO DE ARRENDAMIENTO DE VIVIENDA HABITUAL\n\n"
    f"PRIMERA. Actualización. {RENT_CLAUSE}\n\n"
    f"SEGUNDA. Mascotas. {TERM_CLAUSE}\n"
)

A18_TEXT = "El arrendador solo podrá actualizar la renta conforme al índice de referencia anual."
A36_TEXT = (
    "A la celebración del contrato será obligatoria la exigencia de una mensualidad de renta."
)

CHK_RENTA = checklist_item(
    "CHK-RENTA",
    "Actualización de renta topada al índice",
    "a18",
    "solo podrá actualizar la renta",
)
CHK_FIANZA = checklist_item(
    "CHK-FIANZA",
    "Fianza de una mensualidad",
    "a36",
    "una mensualidad de renta",
)
CHECKLIST = checklist_of(CHK_RENTA, CHK_FIANZA)
CORPUS = FakeCorpus(
    {"a18": resolved_block("a18", A18_TEXT), "a36": resolved_block("a36", A36_TEXT)}
)


def _span(text: str) -> tuple[int, int]:
    """The half-open character range of ``text`` inside the reference document."""
    start = DOCUMENT.index(text)
    return start, start + len(text)


def _case() -> Mode2EvalCase:
    """A two-clause reference contract: a worse-than-default rent clause and a burden."""
    rent_start, rent_end = _span(RENT_CLAUSE)
    term_start, term_end = _span(TERM_CLAUSE)
    return Mode2EvalCase(
        id="contract-01",
        document=DOCUMENT,
        expected_outcome="analizado",
        clauses=(
            ClauseTruth(
                clause_id="c1",
                heading="Actualización",
                text=RENT_CLAUSE,
                start=rent_start,
                end=rent_end,
                expected_level=RiskLevel.PEOR_QUE_DEFAULT,
                chk_ids=("CHK-RENTA",),
            ),
            ClauseTruth(
                clause_id="c2",
                heading="Mascotas",
                text=TERM_CLAUSE,
                start=term_start,
                end=term_end,
                expected_level=RiskLevel.NEGOCIABLE,
            ),
        ),
        expected_absences=(AbsenceTruth(item_id="CHK-FIANZA", right="Fianza de una mensualidad"),),
    )


def _programmed_llm() -> FakeLlmClient:
    """A fake model that segments, triages, maps and classifies the two clauses."""
    fake = FakeLlmClient()
    fake.queue(
        SEGMENTATION_TASK,
        segmentation_of(("Actualización", RENT_CLAUSE), ("Mascotas", TERM_CLAUSE)),
    )
    fake.queue(TRIAGE_TASK, triage_of(fecha_firma="2023-01-01"))
    fake.queue(MAPPING_TASK, mapping_of(("c1", True, ["CHK-RENTA"]), ("c2", True, [])))
    fake.queue(
        CLASSIFICATION_TASK,
        classification_of(
            ProposedClauseLevel.PEOR_QUE_DEFAULT,
            citation=("a18", "solo podrá actualizar la renta"),
        ),
    )
    fake.queue(
        CLASSIFICATION_TASK, classification_of(ProposedClauseLevel.NEGOCIABLE, citation=None)
    )
    return fake


def _fingerprint(cases: tuple[Mode2EvalCase, ...]):
    return build_fingerprint(
        load_task_registry(),
        VERTICAL,
        REPO_ROOT / "verticales" / VERTICAL,
        None,
        compute_mode2_prompts_digest(),
        compute_mode2_dataset_digest(cases),
    )


def _runner(fake: FakeLlmClient) -> PipelineMode2CaseRunner:
    return PipelineMode2CaseRunner(
        llm=fake,
        scope=ScopePackage(
            current_redaction_effective_from=date(2019, 3, 6),
            excluded_uses=frozenset(),
            use_evidence_markers={},
        ),
        checklist=CHECKLIST,
        corpus=CORPUS,
        retriever=FakeRetriever(),
        vertical=VERTICAL,
    )


def test_the_harness_runs_a_contract_end_to_end_and_emits_a_passing_artifact(
    tmp_path: Path,
) -> None:
    cases = (_case(),)
    fake = _programmed_llm()

    artifact = run_mode2_suite(
        "modo2", cases, _runner(fake), CORPUS, NORM_ID, TODAY, _fingerprint(cases), NOW
    )

    assert artifact.passed is True
    metrics = artifact.metrics
    assert metrics.segmentation_delimited_rate == 1.0
    assert metrics.problematic_total == 1
    assert metrics.problematic_detected == 1
    assert metrics.recall_problematic == 1.0
    assert metrics.false_tranquility_rate == 0.0
    assert metrics.absence_recall == 1.0
    assert metrics.confusion["peor_que_default"]["peor_que_default"] == 1

    out = tmp_path / "run.json"
    artifact.write(out)
    assert read_mode2_artifact(out) == artifact


def test_repeating_the_contract_suite_bands_the_headline_over_the_repetitions() -> None:
    cases = (_case(),)
    fake = _programmed_llm()
    fake.queue(
        SEGMENTATION_TASK,
        segmentation_of(("Actualización", RENT_CLAUSE), ("Mascotas", TERM_CLAUSE)),
    )
    fake.queue(TRIAGE_TASK, triage_of(fecha_firma="2023-01-01"))
    fake.queue(MAPPING_TASK, mapping_of(("c1", True, ["CHK-RENTA"]), ("c2", True, [])))
    # The second repetition reassures on the same worse-than-default clause.
    fake.queue(
        CLASSIFICATION_TASK,
        classification_of(
            ProposedClauseLevel.CORRECTO, citation=("a18", "solo podrá actualizar la renta")
        ),
    )
    fake.queue(
        CLASSIFICATION_TASK, classification_of(ProposedClauseLevel.NEGOCIABLE, citation=None)
    )

    artifact = run_mode2_suite(
        "modo2", cases, _runner(fake), CORPUS, NORM_ID, TODAY, _fingerprint(cases), NOW, 2
    )

    assert artifact.repetitions is not None
    band = artifact.repetitions.band("recall_problematic_e2e")
    assert band is not None
    assert band.values == [1.0, 0.0]
    assert (band.low, band.high) == (0.0, 1.0)


def test_a_worse_than_default_clause_called_correct_is_caught_as_false_tranquility(
    tmp_path: Path,
) -> None:
    cases = (_case(),)
    fake = FakeLlmClient()
    fake.queue(
        SEGMENTATION_TASK,
        segmentation_of(("Actualización", RENT_CLAUSE), ("Mascotas", TERM_CLAUSE)),
    )
    fake.queue(TRIAGE_TASK, triage_of(fecha_firma="2023-01-01"))
    fake.queue(MAPPING_TASK, mapping_of(("c1", True, ["CHK-RENTA"]), ("c2", True, [])))
    # The model wrongly reassures on the worse-than-default rent clause.
    fake.queue(
        CLASSIFICATION_TASK,
        classification_of(
            ProposedClauseLevel.CORRECTO, citation=("a18", "solo podrá actualizar la renta")
        ),
    )
    fake.queue(
        CLASSIFICATION_TASK, classification_of(ProposedClauseLevel.NEGOCIABLE, citation=None)
    )

    artifact = run_mode2_suite(
        "modo2", cases, _runner(fake), CORPUS, NORM_ID, TODAY, _fingerprint(cases), NOW
    )

    assert artifact.passed is True  # the (wrong) citation is still literal
    assert artifact.metrics.recall_problematic == 0.0
    assert artifact.metrics.false_tranquility_events == 1
    assert artifact.metrics.false_tranquility_rate == 1.0
