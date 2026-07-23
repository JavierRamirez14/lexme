"""Anchoring proposed clauses to literal spans of the document."""

from lexme.mode2.anchor import MIN_CLAUSE_CHARS, anchor_clauses

DOCUMENT = (
    "PRIMERA. Duración. El plazo del arrendamiento será de cinco años.\n"
    "SEGUNDA. Renta. La renta mensual se fija en 800 euros pagaderos por adelantado.\n"
    "TERCERA. Fianza. El arrendatario entrega una mensualidad en concepto de fianza."
)


def test_a_verbatim_clause_anchors_to_its_literal_span() -> None:
    spans = anchor_clauses(DOCUMENT, ["El plazo del arrendamiento será de cinco años."])

    assert spans is not None
    assert spans[0].text == "El plazo del arrendamiento será de cinco años."


def test_a_clause_with_typographic_drift_still_anchors() -> None:
    proposed = "La renta mensual se fija en 800 euros pagaderos por adelantado."

    spans = anchor_clauses(DOCUMENT, [proposed.replace("800", "800 ").replace("  ", " ")])

    assert spans is not None
    assert "800 euros" in spans[0].text


def test_a_fabricated_clause_breaks_the_whole_anchoring() -> None:
    fabricated = "El arrendador podrá desalojar al inquilino sin preaviso en cualquier momento."

    assert anchor_clauses(DOCUMENT, [fabricated]) is None


def test_one_broken_clause_fails_the_whole_document() -> None:
    real = "El arrendatario entrega una mensualidad en concepto de fianza."
    fabricated = "El contrato se prorroga automáticamente por diez años sin excepción."

    assert anchor_clauses(DOCUMENT, [real, fabricated]) is None


def test_a_too_short_snippet_does_not_anchor() -> None:
    assert len("Duración.") < MIN_CLAUSE_CHARS
    assert anchor_clauses(DOCUMENT, ["Duración."]) is None


def test_each_clause_reports_its_span_offsets() -> None:
    spans = anchor_clauses(DOCUMENT, ["La renta mensual se fija en 800 euros pagaderos"])

    assert spans is not None
    start, end = spans[0].start, spans[0].end
    assert end > start
