"""The executive summary, assembled from the sheet by code with no extra LLM call.

The summary is a deterministic fold of the ficha: a headline naming the parties
and one highlight per non-empty field, plus the count of anchored clauses. It runs
only on the good side of the gates, so it never summarizes a contract the system
declined to analyze, and it adds no model call -- the same facts, rearranged for
reading.
"""

from lexme.mode2.models import ContractSheet, ExecutiveSummary, SummaryItem

_UNKNOWN_PARTY = "parte no identificada"


def build_summary(sheet: ContractSheet, clause_count: int) -> ExecutiveSummary:
    """Fold ``sheet`` and the clause count into an executive summary.

    Only fields the triage actually filled become highlights; an empty field is
    dropped rather than shown blank. The headline always names both parties, using
    a neutral placeholder when one was not identified.
    """
    items = [
        SummaryItem(label=label, value=value)
        for label, value in (
            ("Inmueble", sheet.inmueble),
            ("Renta", sheet.renta),
            ("Duración", sheet.duracion),
            ("Fianza", sheet.fianza),
            ("Fecha de firma", sheet.fecha_firma),
        )
        if value.strip()
    ]
    return ExecutiveSummary(
        headline=_headline(sheet),
        items=items,
        clause_count=clause_count,
    )


def _headline(sheet: ContractSheet) -> str:
    """Name both parties, falling back to a neutral placeholder for a missing one."""
    landlord = sheet.arrendador.strip() or _UNKNOWN_PARTY
    tenant = sheet.arrendatario.strip() or _UNKNOWN_PARTY
    return f"Contrato de arrendamiento entre {landlord} y {tenant}"
