"""Phase A: one batch call that maps every clause to the checklist.

The whole document goes to the model at once, because the mapping needs
document-wide context -- an additional-guarantee clause only makes sense next to
the deposit clause, and leases cross-reference themselves. For each clause the
model answers three things: is there evaluable legal content or is it merely
informative, which checklist items (0..n) does it touch, and which other clauses
it references. It classifies and points; it never judges. The per-clause verdict
in Phase B reads this map to know which items' anchor blocks to hand the judge.
"""

from collections.abc import Sequence

from pydantic import BaseModel

from lexme.checklist import Checklist
from lexme.llm import LlmClient, Message
from lexme.mode2.models import Clause

MAPPING_TASK = "mode2_clause_mapping"

_SYSTEM_PROMPT = (
    "Eres un asistente que relaciona las cláusulas de un contrato de arrendamiento "
    "con un checklist de derechos que la ley reconoce al inquilino. NO juzgas si una "
    "cláusula es buena o mala; solo la clasificas y la enlazas. Para cada cláusula, "
    "identificada por su 'clause_id', devuelves:\n"
    "- evaluable: true si la cláusula tiene contenido jurídico que evaluar; false si "
    "es meramente informativa (datos de las partes, inventario, encabezados).\n"
    "- chk_ids: la lista de identificadores del checklist (CHK-01..CHK-20) que la "
    "cláusula toca, o lista vacía si no toca ninguno. Una cláusula puede tocar "
    "varios ítems o ninguno.\n"
    "- related_clause_ids: los 'clause_id' de otras cláusulas que esta cláusula "
    "referencia o necesita para entenderse, o lista vacía.\n"
    "Devuelve exactamente una entrada por cada cláusula que se te da, con su "
    "'clause_id' literal. No inventes cláusulas ni identificadores de checklist que "
    "no estén en las listas que se te proporcionan."
)


class ClauseMap(BaseModel):
    """How one clause relates to the checklist, as the mapping call reports it.

    ``evaluable`` gates whether Phase B judges the clause at all; a false value
    sends it straight to the informative coverage status with no model call.
    ``chk_ids`` are the checklist items the clause touches, deterministically
    resolving its evidence when non-empty.
    """

    clause_id: str
    evaluable: bool
    chk_ids: list[str] = []
    related_clause_ids: list[str] = []


class DocumentMapping(BaseModel):
    """The mapping call's whole-document answer: one :class:`ClauseMap` per clause."""

    clauses: list[ClauseMap]


def map_clauses(
    llm: LlmClient,
    document_text: str,
    clauses: Sequence[Clause],
    checklist: Checklist,
) -> DocumentMapping:
    """Run the batch mapping over the document, its clauses and the checklist.

    Returns one :class:`ClauseMap` per clause as the model proposes it; the
    caller reconciles the answer against the real clause ids, so a missing or
    stray entry degrades to a safe default rather than trusting the model's set.
    """
    messages = [
        Message("system", _SYSTEM_PROMPT),
        Message("user", _render_prompt(document_text, clauses, checklist)),
    ]
    return llm.complete_structured(MAPPING_TASK, messages, DocumentMapping)


def _render_prompt(document_text: str, clauses: Sequence[Clause], checklist: Checklist) -> str:
    """Build the user message: the document, its clauses and the checklist index."""
    clause_lines = "\n".join(
        f"- {clause.id} ({clause.heading}): {clause.text}" for clause in clauses
    )
    checklist_lines = "\n".join(f"- {item.id}: {item.right}" for item in checklist.items)
    return (
        f"Documento completo:\n{document_text}\n\n"
        f"Cláusulas a mapear:\n{clause_lines}\n\n"
        f"Checklist de derechos del inquilino:\n{checklist_lines}"
    )
