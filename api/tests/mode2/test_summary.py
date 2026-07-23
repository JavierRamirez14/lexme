"""The deterministic executive summary assembled from the ficha."""

from lexme.mode2 import build_summary
from lexme.mode2.models import ContractSheet


def _sheet(**overrides: str) -> ContractSheet:
    base = {
        "arrendador": "Juan Pérez",
        "arrendatario": "Ana García",
        "inmueble": "Calle Mayor 1",
        "renta": "800 euros",
        "duracion": "cinco años",
        "fianza": "una mensualidad",
        "fecha_firma": "2023-01-01",
    }
    base.update(overrides)
    return ContractSheet(**base)


def test_the_headline_names_both_parties() -> None:
    summary = build_summary(_sheet(), clause_count=3)

    assert "Juan Pérez" in summary.headline
    assert "Ana García" in summary.headline


def test_the_clause_count_is_carried_through() -> None:
    summary = build_summary(_sheet(), clause_count=7)

    assert summary.clause_count == 7


def test_an_empty_field_is_dropped_rather_than_shown_blank() -> None:
    summary = build_summary(_sheet(fianza=""), clause_count=1)

    labels = [item.label for item in summary.items]
    assert "Fianza" not in labels
    assert "Renta" in labels


def test_a_missing_party_falls_back_to_a_neutral_placeholder() -> None:
    summary = build_summary(_sheet(arrendador=""), clause_count=1)

    assert "Ana García" in summary.headline
    assert "Juan Pérez" not in summary.headline
