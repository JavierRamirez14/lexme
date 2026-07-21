"""The planner node: sub-queries, the target date, and the branches left open.

Translating the user's plain wording into legal terms is the main recall lever
("me echan del piso" shares no terms with art. 27 LAU), so the planner emits one
to four retrieval sub-queries in legal vocabulary, each with a purpose and a
criticality flag. The same reading of the question also yields the two things
that situate the answer: the temporal reference that anchors it ("firmé en 2017")
and which of the vertical's critical branches the question leaves open. The model
only reports what it read; code resolves the date and picks the branch to ask
about, so neither depends on the model's arithmetic or its sense of priority.
"""

from datetime import date

from pydantic import BaseModel

from lexme.llm import LlmClient, Message
from lexme.mode1.branches import CriticalBranch
from lexme.mode1.dates import resolve_target_date
from lexme.mode1.graph.deps import Mode1Deps
from lexme.mode1.graph.state import MAX_SUBQUERIES, Mode1State, SubQuery, SubQueryState
from lexme.mode1.graph.steps import Step

PLANNING_TASK = "mode1_planning"

_BASE_PROMPT = (
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

_DATE_PROMPT = (
    "En 'target_date_reference' escribe la fecha a la que hay que situar la "
    "respuesta EN FORMATO AAAA-MM-DD, si la pregunta se ancla en el pasado "
    "('firmé en 2017', 'mi contrato es de marzo de 2019'). Si solo se menciona el "
    "año, escribe únicamente el año (p. ej. '2017'). Si la pregunta no menciona "
    "ninguna fecha, deja el campo vacío: se responderá con el derecho de hoy "
    "({today})."
)

_BRANCHES_PROMPT = (
    "Estas variables del caso cambian el régimen legal aplicable. En "
    "'unresolved_branch_ids' enumera los identificadores de las que la pregunta NO "
    "deja claras. Si la pregunta ya las resuelve, o es una consulta informativa "
    "general que no depende del caso concreto, deja la lista vacía:\n{branches}"
)


class PlannedSubQuery(BaseModel):
    """One sub-query as the planner proposes it, before ids are assigned."""

    text: str
    purpose: str
    is_critical: bool


class Plan(BaseModel):
    """The planner's reading of the question: sub-queries, date, assumptions, gaps.

    ``target_date_reference`` is free text the model copied out of the question;
    code, not the model, turns it into a date.
    """

    subqueries: list[PlannedSubQuery]
    assumptions: list[str] = []
    target_date_reference: str = ""
    unresolved_branch_ids: list[str] = []


def planner_node(state: Mode1State, *, deps: Mode1Deps) -> dict:
    """Decompose the question, anchor it in time, and note the branches left open.

    Clamps the plan to at most :data:`MAX_SUBQUERIES` and guarantees at least one
    critical sub-query, falling back to the raw question so retrieval always has a
    query to run.
    """
    plan = _plan(deps.llm, state.question, deps.branches, state.today)
    subqueries = _to_states(plan.subqueries) or [_fallback(state.question)]
    anchor = resolve_target_date(plan.target_date_reference, state.today)
    return {
        "subqueries": subqueries,
        "assumptions": plan.assumptions,
        "target_date": anchor.value,
        "target_date_precision": anchor.precision,
        "unresolved_branch_ids": _declared_order(plan.unresolved_branch_ids, deps.branches),
        "current_step": Step.PLANNING,
    }


def _plan(llm: LlmClient, question: str, branches: tuple[CriticalBranch, ...], today: date) -> Plan:
    """Run the planning task over the raw question."""
    messages = [
        Message("system", _system_prompt(branches, today)),
        Message("user", question),
    ]
    return llm.complete_structured(PLANNING_TASK, messages, Plan)


def _system_prompt(branches: tuple[CriticalBranch, ...], today: date) -> str:
    """Compose the planning prompt with the run's clock and the vertical's branches."""
    sections = [_BASE_PROMPT, _DATE_PROMPT.format(today=today.isoformat())]
    if branches:
        listing = "\n".join(f"- {branch.id}: {branch.question}" for branch in branches)
        sections.append(_BRANCHES_PROMPT.format(branches=listing))
    return "\n".join(sections)


def _declared_order(reported_ids: list[str], branches: tuple[CriticalBranch, ...]) -> list[str]:
    """Keep the ids the vertical actually declares, in the vertical's priority order.

    Reordering here is what makes "which question gets asked" a property of the
    vertical package rather than of the order the model happened to list them in.
    """
    reported = set(reported_ids)
    return [branch.id for branch in branches if branch.id in reported]


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
