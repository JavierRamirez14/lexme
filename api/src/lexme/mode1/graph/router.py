"""The router node: the scope gate plus a shallow intent classification.

One structured call decides whether the question is inside the vertical's scope
(state law on renting a home) and, if so, its query type. Out of scope is the
most likely failure with a single vertical, and answering it from a corpus that
does not cover the matter would be false authority -- the worst outcome -- so a
rejection here terminates the graph before any retrieval runs.
"""

from pydantic import BaseModel

from lexme.llm import LlmClient, Message
from lexme.mode1.graph.deps import Mode1Deps
from lexme.mode1.graph.state import Mode1State
from lexme.mode1.graph.steps import Step
from lexme.mode1.models import Outcome, QueryType, RouterScope

ROUTER_TASK = "mode1_router"

DEFAULT_REJECTION = (
    "Tu pregunta queda fuera de lo que puedo cubrir: solo respondo sobre el alquiler "
    "de vivienda en la legislación estatal española."
)

_SYSTEM_PROMPT = (
    "Eres el enrutador de un asistente sobre el alquiler de vivienda en la "
    "legislación estatal española. El corpus cubre la Ley de Arrendamientos "
    "Urbanos, el contrato de arrendamiento del Código Civil, el desahucio en la "
    "Ley de Enjuiciamiento Civil y la Ley por el derecho a la vivienda. "
    "Clasificas la pregunta del usuario en dos ejes:\n"
    "- scope: 'dentro' si trata del alquiler de vivienda regido por esas normas "
    "estatales, incluido el proceso de desahucio; 'fuera' si trata de normativa "
    "autonómica, otros ámbitos legales (laboral, tráfico, penal...), o cuestiones "
    "no jurídicas. Ante la duda entre un vertical distinto y el alquiler de "
    "vivienda, responde 'fuera': contestar con una norma que no cubre la materia "
    "sería falsa autoridad.\n"
    "- query_type: 'informativa' (qué dice la ley en general), 'situacional' (un "
    "caso concreto del usuario) o 'procedimental' (cómo hacer un trámite).\n"
    "Si scope es 'fuera', escribe en 'rejection_message' una frase honesta y breve "
    "que explique que no cubres eso. Si es 'dentro', deja 'rejection_message' vacío."
)


class RouterDecision(BaseModel):
    """The router's structured verdict: scope, intent and an out-of-scope message."""

    scope: RouterScope
    query_type: QueryType
    rejection_message: str = ""


def router_node(state: Mode1State, *, deps: Mode1Deps) -> dict:
    """Classify scope and intent; on an out-of-scope verdict, end with a rejection.

    Returns the state update carrying the scope, the query type and -- when out of
    scope -- the terminal rejection outcome and its message.
    """
    decision = _decide(deps.llm, state.question)
    update: dict = {
        "scope": decision.scope,
        "query_type": decision.query_type,
        "current_step": Step.ROUTING,
    }
    if decision.scope is RouterScope.OUT_OF_SCOPE:
        update["outcome"] = Outcome.ROUTER_REJECTION
        update["router_message"] = decision.rejection_message.strip() or DEFAULT_REJECTION
    return update


def _decide(llm: LlmClient, question: str) -> RouterDecision:
    """Run the router task over the raw question."""
    messages = [
        Message("system", _SYSTEM_PROMPT),
        Message("user", question),
    ]
    return llm.complete_structured(ROUTER_TASK, messages, RouterDecision)
