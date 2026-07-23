"""Phase B: judge one clause against its evidence, then verify what it cited.

Each evaluable clause gets its own call. The evidence is assembled first, in code:
a clause the mapping tied to checklist items is handed those items' anchor blocks
directly; a clause off the checklist is grounded by hybrid retrieval. The model
receives the clause, its related clauses, that evidence and the character of any
item it touches, and returns a level with its five fields. Code then re-verifies
the citation: a level that must cite the law (illegal, worse-than-default,
conforming) but whose citation the verifier discards degrades to inconclusive,
keeping the signal, rather than showing a level with no verified ground.
"""

from collections.abc import Sequence
from datetime import date

from pydantic import BaseModel

from lexme.checklist import ChecklistItem
from lexme.llm import LlmClient, Message
from lexme.mode1.models import VerifiedCitation
from lexme.mode2.mapping import ClauseMap
from lexme.mode2.models import Clause
from lexme.mode2.retrieval import ClauseRetriever
from lexme.mode2.risk import (
    ClauseFinding,
    CoverageStatus,
    ProposedClauseLevel,
    RiskLevel,
    requires_citation,
)
from lexme.retrieval import RetrievedBlock
from lexme.verification import (
    CitationResult,
    CitationVerdict,
    CorpusReader,
    EvidenceBlock,
    ProposedCitation,
    verify_citations,
)

CLASSIFICATION_TASK = "mode2_clause_classification"

_LEVEL_BY_PROPOSAL: dict[ProposedClauseLevel, RiskLevel] = {
    ProposedClauseLevel.ILEGAL: RiskLevel.ILEGAL,
    ProposedClauseLevel.PEOR_QUE_DEFAULT: RiskLevel.PEOR_QUE_DEFAULT,
    ProposedClauseLevel.NEGOCIABLE: RiskLevel.NEGOCIABLE,
    ProposedClauseLevel.CORRECTO: RiskLevel.CORRECTO,
}

_SYSTEM_PROMPT = (
    "Eres un asistente que evalúa una cláusula de un contrato de arrendamiento de "
    "vivienda desde la posición del inquilino, comparándola con la Ley de "
    "Arrendamientos Urbanos. Devuelves un único veredicto con estos campos:\n"
    "- level: uno de\n"
    "  'ilegal' (la cláusula contradice una norma imperativa; es nula),\n"
    "  'peor_que_default' (la ley fija un default dispositivo y la cláusula elige la "
    "rama peor para el inquilino),\n"
    "  'negociable' (la LAU no regula esta materia y la cláusula impone una carga al "
    "inquilino),\n"
    "  'correcto' (evaluada y conforme, o mejor que el default para el inquilino),\n"
    "  'fuera_de_ambito' (la materia se rige por normas ajenas a la LAU; nómbralas),\n"
    "  'no_concluyente' (jurídicamente relevante pero sin base suficiente para "
    "asignar nivel; di qué falta).\n"
    "- explanation: en lenguaje llano, qué dice la cláusula y qué significa para el "
    "inquilino.\n"
    "- what_you_can_do: pasos acotados a informarse, negociar o vigilar plazos; nunca "
    "redactes documentos ni prometas resultados. Lista vacía si no procede.\n"
    "- citation: SOLO para 'ilegal', 'peor_que_default' o 'correcto', la norma que "
    "fundamenta el nivel: 'block_id' de un bloque de la evidencia y 'text' copiado "
    "LITERALMENTE de ese bloque. Nunca cites un bloque que no esté en la evidencia ni "
    "inventes texto. Para 'negociable', 'fuera_de_ambito' o 'no_concluyente' deja "
    "citation en null: no cites un artículo que no regula la materia."
)


class ClauseClassification(BaseModel):
    """The model's verdict on one clause, before code re-verifies its citation.

    ``level`` is the proposed placement on the spectrum or a coverage escape;
    ``citation`` is present only for the levels that must cite the law, and is
    unverified until the caller runs it through the citation verifier.
    """

    level: ProposedClauseLevel
    explanation: str
    what_you_can_do: list[str] = []
    citation: ProposedCitation | None = None


def classify_clause(
    clause: Clause,
    clause_map: ClauseMap,
    *,
    related: Sequence[Clause],
    llm: LlmClient,
    corpus: CorpusReader,
    retriever: ClauseRetriever,
    items_by_id: dict[str, ChecklistItem],
    norm_id: str,
    vertical: str,
    target_date: date,
) -> ClauseFinding:
    """Classify one evaluable clause and return its verified finding.

    Gathers evidence -- the anchor blocks of the checklist items the clause
    touches, or hybrid retrieval when it touches none -- runs the classification
    call, then verifies the proposed citation. A citation-bearing level whose
    citation does not survive degrades to ``NO_CONCLUYENTE`` with the signal kept.
    """
    items = [items_by_id[chk_id] for chk_id in clause_map.chk_ids if chk_id in items_by_id]
    evidence = _gather_evidence(clause, items, corpus, retriever, norm_id, vertical, target_date)
    classification = _classify(llm, clause, related, evidence, items)
    return _resolve_finding(clause, clause_map, classification, evidence, corpus, target_date)


def _gather_evidence(
    clause: Clause,
    items: Sequence[ChecklistItem],
    corpus: CorpusReader,
    retriever: ClauseRetriever,
    norm_id: str,
    vertical: str,
    target_date: date,
) -> list[RetrievedBlock]:
    """Assemble the clause's evidence: item anchor blocks, or hybrid retrieval.

    A clause tied to checklist items is grounded deterministically on those
    items' anchor blocks; a clause off the checklist falls back to hybrid
    retrieval over its own text.
    """
    if not items:
        return retriever.retrieve(clause.text, vertical=vertical, target_date=target_date)
    return _resolve_anchor_blocks(items, corpus, norm_id, target_date)


def _resolve_anchor_blocks(
    items: Sequence[ChecklistItem],
    corpus: CorpusReader,
    norm_id: str,
    target_date: date,
) -> list[RetrievedBlock]:
    """Resolve every anchor block of the given items into evidence, de-duplicated."""
    blocks: dict[str, RetrievedBlock] = {}
    for item in items:
        for anchor in item.anchors:
            if anchor in blocks:
                continue
            resolved = corpus.resolve_block(norm_id, anchor, target_date)
            if resolved is None:
                continue
            blocks[anchor] = RetrievedBlock(
                norm_id=norm_id,
                block_id=anchor,
                title=resolved.anchor.title,
                text=resolved.text,
                effective_date=resolved.anchor.effective_date,
            )
    return list(blocks.values())


def _classify(
    llm: LlmClient,
    clause: Clause,
    related: Sequence[Clause],
    evidence: Sequence[RetrievedBlock],
    items: Sequence[ChecklistItem],
) -> ClauseClassification:
    """Run the classification call over the clause, its context and its evidence."""
    messages = [
        Message("system", _SYSTEM_PROMPT),
        Message("user", _render_prompt(clause, related, evidence, items)),
    ]
    return llm.complete_structured(CLASSIFICATION_TASK, messages, ClauseClassification)


def _render_prompt(
    clause: Clause,
    related: Sequence[Clause],
    evidence: Sequence[RetrievedBlock],
    items: Sequence[ChecklistItem],
) -> str:
    """Build the user message: the clause, its context, evidence and item character."""
    prompt = f"Cláusula a evaluar ({clause.heading}):\n{clause.text}"
    if related:
        related_lines = "\n".join(f"- ({rel.heading}) {rel.text}" for rel in related)
        prompt += f"\n\nCláusulas relacionadas:\n{related_lines}"
    if items:
        item_lines = "\n".join(
            f"- {item.id} ({item.character.value}): {item.right}" for item in items
        )
        prompt += f"\n\nDerechos del checklist que toca esta cláusula:\n{item_lines}"
    if evidence:
        blocks = "\n\n".join(
            f"block_id: {block.block_id}\n{block.title}\n{block.text}" for block in evidence
        )
        prompt += f"\n\nEvidencia (redacción vigente):\n{blocks}"
    else:
        prompt += (
            "\n\nNo se ha recuperado ninguna norma de la LAU para esta cláusula. "
            "Si la LAU no regula la materia, clasifícala en consecuencia."
        )
    return prompt


def _resolve_finding(
    clause: Clause,
    clause_map: ClauseMap,
    classification: ClauseClassification,
    evidence: Sequence[RetrievedBlock],
    corpus: CorpusReader,
    target_date: date,
) -> ClauseFinding:
    """Turn the model's verdict into a finding, verifying any citation it made."""
    if classification.level is ProposedClauseLevel.FUERA_DE_AMBITO:
        return _finding(
            clause,
            clause_map,
            CoverageStatus.FUERA_DE_AMBITO,
            explanation=classification.explanation,
            out_of_scope_matter=classification.explanation,
        )
    if classification.level is ProposedClauseLevel.NO_CONCLUYENTE:
        return _finding(
            clause,
            clause_map,
            CoverageStatus.NO_CONCLUYENTE,
            explanation=classification.explanation,
            what_you_can_do=classification.what_you_can_do,
        )

    level = _LEVEL_BY_PROPOSAL[classification.level]
    citation = _verify(classification.citation, evidence, corpus, target_date)
    if requires_citation(level) and citation is None:
        return _finding(
            clause,
            clause_map,
            CoverageStatus.NO_CONCLUYENTE,
            explanation=classification.explanation,
            what_you_can_do=classification.what_you_can_do,
        )
    return _finding(
        clause,
        clause_map,
        CoverageStatus.EVALUADA,
        level=level,
        explanation=classification.explanation,
        what_you_can_do=classification.what_you_can_do,
        citation=citation,
    )


def _verify(
    citation: ProposedCitation | None,
    evidence: Sequence[RetrievedBlock],
    corpus: CorpusReader,
    target_date: date,
) -> VerifiedCitation | None:
    """Verify one proposed citation against the evidence, or ``None`` if none holds.

    A missing citation, or one the verifier discards, yields ``None`` so the
    caller can decide whether the level survives without it.
    """
    if citation is None:
        return None
    blocks = [EvidenceBlock(norm_id=block.norm_id, block_id=block.block_id) for block in evidence]
    (result,) = verify_citations([citation], blocks, target_date, corpus)
    if result.verdict is CitationVerdict.DISCARDED:
        return None
    return _to_verified(result)


def _to_verified(result: CitationResult) -> VerifiedCitation:
    """Map a surviving citation result to the display citation with its anchor."""
    if result.anchor is None:
        raise ValueError(f"citation {result.block_id} held with no anchor")
    return VerifiedCitation(
        block_id=result.block_id,
        text=result.text,
        verdict=result.verdict,
        anchor=result.anchor,
    )


def _finding(
    clause: Clause,
    clause_map: ClauseMap,
    coverage: CoverageStatus,
    **fields: object,
) -> ClauseFinding:
    """Build a clause finding from the clause's identity, span and mapping."""
    return ClauseFinding(
        clause_id=clause.id,
        heading=clause.heading,
        snippet=clause.text,
        start=clause.start,
        end=clause.end,
        coverage=coverage,
        chk_ids=list(clause_map.chk_ids),
        related_clause_ids=list(clause_map.related_clause_ids),
        **fields,
    )
