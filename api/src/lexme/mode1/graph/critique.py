"""The self-critique node: one ruling per sub-query, plus reformulations.

A single structured call judges each sub-query against the criterion "do these
blocks let me answer it with a literal citation?" -- not "do they look relevant?".
Insufficient sub-queries get a reformulation (new terms, a suspected article) that
the next retrieval pass will use. Each pass is tallied into a :class:`PassRecord`
so the agentic delta (grounded sub-queries, first pass vs. last) is measurable
without extra instrumentation.
"""

from pydantic import BaseModel

from lexme.blocks import BlockRef
from lexme.llm import LlmClient, Message
from lexme.mode1.graph.deps import Mode1Deps
from lexme.mode1.graph.state import Mode1State, PassRecord, SubQueryState
from lexme.mode1.graph.steps import Step
from lexme.mode1.models import SubQueryVerdict
from lexme.retrieval.models import RetrievedBlock

CRITIQUE_TASK = "mode1_selfcritique"

_SYSTEM_PROMPT = (
    "Eres el crítico de un asistente jurídico sobre alquiler de vivienda. Para cada "
    "sub-consulta recibes los bloques recuperados, que pueden venir de varias normas "
    "estatales, y dictaminas si permiten responderla CON UNA CITA "
    "LITERAL del texto (no si 'parecen relevantes'):\n"
    "- verdict: 'suficiente' o 'insuficiente'.\n"
    "- reformulation: si es 'insuficiente', una reformulación de la consulta con "
    "términos nuevos o el artículo que sospechas; si es 'suficiente', vacío.\n"
    "Devuelve un dictamen por cada sub-consulta, identificada por su id."
)


class SubQueryCritique(BaseModel):
    """The critique's ruling on one sub-query, with a reformulation if insufficient."""

    subquery_id: str
    verdict: SubQueryVerdict
    reformulation: str = ""


class Critique(BaseModel):
    """The critique's rulings across every sub-query judged this pass."""

    verdicts: list[SubQueryCritique]


def critique_node(state: Mode1State, *, deps: Mode1Deps) -> dict:
    """Judge every sub-query, apply reformulations and record the pass tally.

    A sub-query already ruled sufficient keeps that verdict. Insufficient ones
    take the critique's reformulation as their next query text. Appends this
    pass's tally to the state's pass history.
    """
    critique = _critique(deps.llm, state)
    rulings = {ruling.subquery_id: ruling for ruling in critique.verdicts}
    judged = [_apply(sub, rulings.get(sub.subquery.id)) for sub in state.subqueries]
    return {
        "subqueries": judged,
        "passes": [*state.passes, _record(state.pass_number, judged)],
        "current_step": Step.ITERATING,
    }


def _critique(llm: LlmClient, state: Mode1State) -> Critique:
    """Run the critique task over the sub-queries still awaiting a sufficient ruling."""
    pending = [sub for sub in state.subqueries if not sub.is_sufficient]
    messages = [
        Message("system", _SYSTEM_PROMPT),
        Message("user", _render(pending)),
    ]
    return llm.complete_structured(CRITIQUE_TASK, messages, Critique)


def _apply(sub: SubQueryState, ruling: SubQueryCritique | None) -> SubQueryState:
    """Fold one critique ruling into a sub-query's verdict and next query text."""
    if sub.is_sufficient or ruling is None:
        return sub
    reformulation = ruling.reformulation.strip()
    next_text = (
        reformulation
        if ruling.verdict is SubQueryVerdict.INSUFFICIENT and reformulation
        else sub.query_text
    )
    return sub.model_copy(update={"verdict": ruling.verdict, "query_text": next_text})


def _record(pass_number: int, subqueries: list[SubQueryState]) -> PassRecord:
    """Tally the pass: which sub-queries hold, which do not, and the evidence gathered."""
    sufficient = [sub.subquery.id for sub in subqueries if sub.is_sufficient]
    insufficient = [sub.subquery.id for sub in subqueries if not sub.is_sufficient]
    evidence_count = sum(len(sub.evidence) for sub in subqueries)
    return PassRecord(
        pass_number=pass_number,
        sufficient_ids=sufficient,
        insufficient_ids=insufficient,
        evidence_count=evidence_count,
        evidence_block_refs=_accumulated_block_refs(subqueries),
    )


def _accumulated_block_refs(subqueries: list[SubQueryState]) -> list[str]:
    """The distinct evidence block references across every sub-query, first-seen order."""
    seen: dict[str, None] = {}
    for sub in subqueries:
        for block in sub.evidence:
            seen.setdefault(str(BlockRef(norm_id=block.norm_id, block_id=block.block_id)), None)
    return list(seen)


def _render(subqueries: list[SubQueryState]) -> str:
    """Render the pending sub-queries and their evidence for the critique prompt."""
    return "\n\n".join(_render_one(sub) for sub in subqueries)


def _render_one(sub: SubQueryState) -> str:
    """Render one sub-query with its retrieved evidence blocks."""
    blocks = "\n".join(_render_block(block) for block in sub.evidence) or "(sin bloques)"
    return f"sub-consulta {sub.subquery.id}: {sub.query_text}\nEvidencia:\n{blocks}"


def _render_block(block: RetrievedBlock) -> str:
    """Render one evidence block as its citable reference, source norm, title and text."""
    ref = BlockRef(norm_id=block.norm_id, block_id=block.block_id)
    return f"- block_ref {ref} ({block.norm_label}, {block.title}): {block.text}"
