"""The planner node: flat decomposition into legal-vocabulary sub-queries.

Translating the user's plain wording into legal terms is the main recall lever
("me echan del piso" shares no terms with art. 27 LAU), so the planner emits one
to four retrieval sub-queries in legal vocabulary, each with a purpose and a
criticality flag. It also surfaces the non-critical assumptions it made about the
query instead of stopping to ask, which the answer then states explicitly.
"""

from pydantic import BaseModel

from lexme.llm import LlmClient, Message
from lexme.mode1.graph.deps import Mode1Deps
from lexme.mode1.graph.state import MAX_SUBQUERIES, Mode1State, SubQuery, SubQueryState
from lexme.mode1.graph.steps import Step

PLANNING_TASK = "mode1_planning"

_SYSTEM_PROMPT = (
    "Eres el planificador de un asistente sobre la Ley de Arrendamientos Urbanos "
    "(LAU estatal, vivienda). Recibes la pregunta de un inquilino en lenguaje llano "
    f"y la descompones en entre 1 y {MAX_SUBQUERIES} sub-consultas de recuperación, "
    "cada una redactada en VOCABULARIO LEGAL (los términos que aparecerían en el "
    "articulado), no en lenguaje llano. Cada sub-consulta lleva:\n"
    "- text: la consulta en términos legales para buscar en el corpus.\n"
    "- purpose: qué parte de la pregunta resuelve, en una frase.\n"
    "- is_critical: true si la respuesta no se sostiene sin esta sub-consulta; "
    "false si es periférica o complementaria.\n"
    "Una pregunta informativa simple da un plan de una sola sub-consulta crítica. "
    "En 'assumptions' enumera las suposiciones no críticas que haces sobre el caso "
    "(p. ej. 'asumo vivienda habitual y no de temporada'); deja la lista vacía si "
    "no asumes nada relevante."
)


class PlannedSubQuery(BaseModel):
    """One sub-query as the planner proposes it, before ids are assigned."""

    text: str
    purpose: str
    is_critical: bool


class Plan(BaseModel):
    """The planner's output: the sub-queries and the explicit assumptions made."""

    subqueries: list[PlannedSubQuery]
    assumptions: list[str] = []


def planner_node(state: Mode1State, *, deps: Mode1Deps) -> dict:
    """Decompose the question into sub-queries and record the assumptions made.

    Clamps the plan to at most :data:`MAX_SUBQUERIES` and guarantees at least one
    critical sub-query, falling back to the raw question so retrieval always has a
    query to run.
    """
    plan = _plan(deps.llm, state.question)
    subqueries = _to_states(plan.subqueries) or [_fallback(state.question)]
    return {
        "subqueries": subqueries,
        "assumptions": plan.assumptions,
        "current_step": Step.PLANNING,
    }


def _plan(llm: LlmClient, question: str) -> Plan:
    """Run the planning task over the raw question."""
    messages = [
        Message("system", _SYSTEM_PROMPT),
        Message("user", question),
    ]
    return llm.complete_structured(PLANNING_TASK, messages, Plan)


def _to_states(planned: list[PlannedSubQuery]) -> list[SubQueryState]:
    """Assign stable ids and wrap each planned sub-query as retrieval state."""
    states: list[SubQueryState] = []
    for index, item in enumerate(planned[:MAX_SUBQUERIES], start=1):
        subquery = SubQuery(
            id=f"sq{index}",
            text=item.text,
            purpose=item.purpose,
            is_critical=item.is_critical,
        )
        states.append(SubQueryState(subquery=subquery, query_text=item.text))
    return states


def _fallback(question: str) -> SubQueryState:
    """A single critical sub-query over the raw question, used when a plan is empty."""
    subquery = SubQuery(
        id="sq1",
        text=question,
        purpose="Responder la pregunta tal como se planteó.",
        is_critical=True,
    )
    return SubQueryState(subquery=subquery, query_text=question)
