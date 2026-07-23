"""Acceptance tests over the HTTP surface of Mode 2 with the seams faked.

The whole contract pipeline runs against a live server -- real extraction of the
posted bytes, real anchoring, the code gates and the deterministic summary -- with
only the LLM, the clock, the scope package and (for the honest-stop cases) the
extractor substituted. They assert the output contract on ``/contract/analyze``:
an analyzed ficha with anchored clauses, an out-of-scope stop, a not-analyzable
stop, and a broken anchor turned into a rejection rather than a shown clause.
"""

import httpx

from lexme.llm import FakeLlmClient
from lexme.mode2 import TenancyUse
from lexme.mode2.segmentation import SEGMENTATION_TASK
from lexme.mode2.triage import TRIAGE_TASK
from tests.mode2.conftest import (
    FakeExtractor,
    build_blank_pdf,
    build_docx,
    build_text_pdf,
    segmentation_of,
    triage_of,
)

DOCUMENT = (
    "CONTRATO DE ARRENDAMIENTO DE VIVIENDA\n"
    "PRIMERA. Duración. El plazo del arrendamiento será de cinco años.\n"
    "SEGUNDA. Renta. La renta mensual se fija en 800 euros pagaderos por adelantado."
)
CLAUSES = (
    ("Duración", "El plazo del arrendamiento será de cinco años."),
    ("Renta", "La renta mensual se fija en 800 euros pagaderos por adelantado."),
)


def _post(base_url: str) -> httpx.Response:
    return httpx.post(
        f"{base_url}/contract/analyze",
        files={"file": ("contrato.pdf", b"ignored-bytes", "application/pdf")},
        timeout=10,
    )


def test_a_readable_lease_is_analyzed_with_anchored_clauses(
    contract_server: tuple[str, FakeLlmClient, FakeExtractor],
) -> None:
    base_url, fake_llm, extractor = contract_server
    extractor.text = DOCUMENT
    fake_llm.queue(SEGMENTATION_TASK, segmentation_of(*CLAUSES))
    fake_llm.queue(TRIAGE_TASK, triage_of(fecha_firma="2023-01-01"))

    response = _post(base_url)

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "analizado"
    assert body["summary"]["headline"]
    clauses = body["clauses"]
    assert len(clauses) == 2
    assert clauses[0]["text"] == "El plazo del arrendamiento será de cinco años."


def test_a_seasonal_lease_returns_an_out_of_scope_stop(
    contract_server: tuple[str, FakeLlmClient, FakeExtractor],
) -> None:
    base_url, fake_llm, extractor = contract_server
    extractor.text = DOCUMENT
    fake_llm.queue(SEGMENTATION_TASK, segmentation_of(*CLAUSES))
    fake_llm.queue(TRIAGE_TASK, triage_of(uso=TenancyUse.SEASONAL, fecha_firma="2023-01-01"))

    response = _post(base_url)

    body = response.json()
    assert body["outcome"] == "fuera_de_ambito"
    assert body["clauses"] == []
    assert body["rejection"]["reason"] == "uso_fuera_de_ambito"
    assert body["rejection"]["message"]


def test_a_prior_redaction_lease_returns_an_out_of_scope_stop(
    contract_server: tuple[str, FakeLlmClient, FakeExtractor],
) -> None:
    base_url, fake_llm, extractor = contract_server
    extractor.text = DOCUMENT
    fake_llm.queue(SEGMENTATION_TASK, segmentation_of(*CLAUSES))
    fake_llm.queue(TRIAGE_TASK, triage_of(fecha_firma="2017-05-04"))

    response = _post(base_url)

    body = response.json()
    assert body["outcome"] == "fuera_de_ambito"
    assert body["rejection"]["reason"] == "redaccion_anterior"


def test_a_scan_with_no_text_returns_a_not_analyzable_stop(
    contract_server: tuple[str, FakeLlmClient, FakeExtractor],
) -> None:
    base_url, _, extractor = contract_server
    extractor.text = "  "

    response = _post(base_url)

    body = response.json()
    assert body["outcome"] == "no_analizable"
    assert body["rejection"]["reason"] == "texto_no_extraible"


def test_a_broken_anchor_returns_a_not_analyzable_stop_not_a_clause(
    contract_server: tuple[str, FakeLlmClient, FakeExtractor],
) -> None:
    base_url, fake_llm, extractor = contract_server
    extractor.text = DOCUMENT
    fake_llm.queue(
        SEGMENTATION_TASK,
        segmentation_of(
            ("Duración", "El plazo del arrendamiento será de cinco años."),
            ("Falsa", "El arrendador recupera la vivienda cuando quiera sin ningún preaviso."),
        ),
    )

    response = _post(base_url)

    body = response.json()
    assert body["outcome"] == "no_analizable"
    assert body["rejection"]["reason"] == "anclaje_roto"
    assert body["clauses"] == []


def test_a_missing_signing_date_is_analyzed_on_a_stated_assumption(
    contract_server: tuple[str, FakeLlmClient, FakeExtractor],
) -> None:
    base_url, fake_llm, extractor = contract_server
    extractor.text = DOCUMENT
    fake_llm.queue(SEGMENTATION_TASK, segmentation_of(*CLAUSES))
    fake_llm.queue(TRIAGE_TASK, triage_of(fecha_firma=""))

    response = _post(base_url)

    body = response.json()
    assert body["outcome"] == "analizado"
    assert any("hoy" in assumption for assumption in body["assumptions"])


def test_a_real_pdf_upload_is_read_and_analyzed_over_the_api(
    contract_server_real_extraction: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = contract_server_real_extraction
    clause = "El plazo del arrendamiento sera de cinco anos."
    pdf = build_text_pdf(["PRIMERA. Duracion.", clause])
    fake_llm.queue(SEGMENTATION_TASK, segmentation_of(("Duración", clause)))
    fake_llm.queue(TRIAGE_TASK, triage_of(fecha_firma="2023-01-01"))

    response = httpx.post(
        f"{base_url}/contract/analyze",
        files={"file": ("contrato.pdf", pdf, "application/pdf")},
        timeout=10,
    )

    body = response.json()
    assert body["outcome"] == "analizado"
    assert body["clauses"][0]["text"] == clause


def test_a_real_word_upload_is_read_and_analyzed_over_the_api(
    contract_server_real_extraction: tuple[str, FakeLlmClient],
) -> None:
    base_url, fake_llm = contract_server_real_extraction
    clause = "La renta mensual se fija en 800 euros pagaderos por adelantado."
    docx = build_docx(["PRIMERA. Renta.", clause])
    fake_llm.queue(SEGMENTATION_TASK, segmentation_of(("Renta", clause)))
    fake_llm.queue(TRIAGE_TASK, triage_of(fecha_firma="2023-01-01"))

    response = httpx.post(
        f"{base_url}/contract/analyze",
        files={
            "file": (
                "contrato.docx",
                docx,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
        timeout=10,
    )

    body = response.json()
    assert body["outcome"] == "analizado"
    assert body["clauses"][0]["text"] == clause


def test_a_scanned_pdf_upload_is_not_analyzable_over_the_api(
    contract_server_real_extraction: tuple[str, FakeLlmClient],
) -> None:
    base_url, _ = contract_server_real_extraction
    scan = build_blank_pdf()

    response = httpx.post(
        f"{base_url}/contract/analyze",
        files={"file": ("escaneado.pdf", scan, "application/pdf")},
        timeout=10,
    )

    body = response.json()
    assert body["outcome"] == "no_analizable"
    assert body["rejection"]["reason"] == "texto_no_extraible"
