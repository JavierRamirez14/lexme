"""Projection of the internal graph state onto the public Mode 1 response.

Two projections share the sub-query and pass telemetry: the live SSE snapshot
(the agentic trace on its own, emitted after each node) and the final response
(the trace plus the answer, abstention or rejection the gate chose). Keeping the
mapping here lets the state carry rich, non-serialized objects while the API
stays a stable, lean contract.
"""

from lexme.mode1.graph.state import Mode1State, PassRecord, SubQueryState
from lexme.mode1.models import (
    Abstention,
    AbstentionReason,
    AgenticTrace,
    Answer,
    AskResponse,
    Clarification,
    Outcome,
    PassReport,
    RankedBlockRef,
    RetrievalTrace,
    RouterRejection,
    SubQueryReport,
)
from lexme.retrieval.models import BlockKey

SCOPE_REMINDER = (
    "Solo cubro la Ley de Arrendamientos Urbanos estatal (vivienda). "
    "No cubro normativa autonómica ni otros ámbitos."
)

_NO_EVIDENCE_MESSAGE = (
    "No he encontrado ninguna base en la LAU para responder a tu pregunta. "
    "Prueba a reformularla centrándote en el arrendamiento de vivienda."
)
_NO_VERIFIABLE_CITATION_MESSAGE = (
    "He encontrado texto relacionado, pero no he podido respaldar la respuesta "
    "con una cita literal verificada, así que prefiero no responder."
)


def build_agentic_trace(state: Mode1State) -> AgenticTrace:
    """Project the decomposition and per-pass telemetry into the public trace."""
    passes = [_pass_report(record) for record in state.passes]
    first = len(state.passes[0].sufficient_ids) if state.passes else 0
    final = len(state.passes[-1].sufficient_ids) if state.passes else 0
    return AgenticTrace(
        query_type=state.query_type,
        subqueries=[_subquery_report(sub) for sub in state.subqueries],
        passes=passes,
        first_pass_sufficient=first,
        final_sufficient=final,
        agentic_delta=final - first,
    )


def build_response(state: Mode1State, thread_id: str) -> AskResponse:
    """Build the final response from a completed run's state.

    An out-of-scope rejection carries no agentic trace (it ran before planning);
    every other outcome carries the trace plus its matching payload.
    """
    if state.outcome is Outcome.ROUTER_REJECTION:
        return AskResponse(
            outcome=Outcome.ROUTER_REJECTION,
            thread_id=thread_id,
            rejection=RouterRejection(
                message=state.router_message or "",
                scope_reminder=SCOPE_REMINDER,
            ),
        )

    trace = build_agentic_trace(state)
    if state.outcome is Outcome.ABSTENTION:
        return AskResponse(
            outcome=Outcome.ABSTENTION,
            thread_id=thread_id,
            abstention=_abstention(state),
            agentic=trace,
            citation_verdicts=state.citation_verdicts,
        )

    return AskResponse(
        outcome=state.outcome or Outcome.ANSWER,
        thread_id=thread_id,
        answer=_answer(state),
        agentic=trace,
        citation_verdicts=state.citation_verdicts,
    )


def build_clarification_response(
    state: Mode1State, clarification: Clarification, thread_id: str
) -> AskResponse:
    """Build the paused response carrying the one question the run is waiting on.

    The agentic trace travels with it, so the client can keep showing the work
    done so far while the user answers.
    """
    return AskResponse(
        outcome=Outcome.CLARIFICATION,
        thread_id=thread_id,
        clarification=clarification,
        agentic=build_agentic_trace(state),
    )


def _answer(state: Mode1State) -> Answer:
    """Assemble the answer, situated in time and listing assumptions and gaps."""
    synthesis = state.synthesis
    return Answer(
        fundamento=state.verified,
        explicacion=synthesis.explicacion if synthesis else "",
        accion=synthesis.accion if synthesis else [],
        fecha_objetivo=state.target_date,
        avisos_vigencia=state.notices,
        asunciones=state.assumptions,
        huecos_declarados=_declared_gaps(state),
    )


def _declared_gaps(state: Mode1State) -> list[str]:
    """The purposes of the peripheral sub-queries left ungrounded, for a partial answer."""
    if state.outcome is not Outcome.PARTIAL_ANSWER:
        return []
    return [
        sub.subquery.purpose
        for sub in state.subqueries
        if not sub.subquery.is_critical and not sub.is_sufficient
    ]


def _abstention(state: Mode1State) -> Abstention:
    """Build the abstention payload matching the gate's reason."""
    reason = state.abstention_reason or AbstentionReason.NO_VERIFIABLE_CITATION
    return Abstention(
        reason=reason,
        message=_abstention_message(state, reason),
        scope_reminder=SCOPE_REMINDER,
    )


def _abstention_message(state: Mode1State, reason: AbstentionReason) -> str:
    """Compose the user-facing abstention message for a given reason."""
    if reason is AbstentionReason.NO_EVIDENCE:
        return _NO_EVIDENCE_MESSAGE
    if reason is AbstentionReason.NO_VERIFIABLE_CITATION:
        return _NO_VERIFIABLE_CITATION_MESSAGE
    return _insufficient_core_message(state)


def _insufficient_core_message(state: Mode1State) -> str:
    """Explain which core parts of the question could not be grounded."""
    missing = [
        sub.subquery.purpose
        for sub in state.subqueries
        if sub.subquery.is_critical and not sub.is_sufficient
    ]
    detail = "; ".join(missing) if missing else "la parte central de tu pregunta"
    return (
        f"No he encontrado base suficiente en la LAU para lo esencial de tu consulta "
        f"({detail}), así que prefiero no responder antes que hacerlo sin apoyo. "
        "Prueba a acotar la pregunta o reformularla."
    )


def _subquery_report(sub: SubQueryState) -> SubQueryReport:
    """Project one sub-query's state into its public report, with its retrieval trace."""
    return SubQueryReport(
        id=sub.subquery.id,
        text=sub.subquery.text,
        purpose=sub.subquery.purpose,
        is_critical=sub.subquery.is_critical,
        verdict=sub.verdict,
        evidence=[_ref(block.key) for block in sub.evidence],
        retrieval=_retrieval_trace(sub),
    )


def _retrieval_trace(sub: SubQueryState) -> RetrievalTrace | None:
    """The sub-query's dense/lexical/fused rankings, or ``None`` before retrieval."""
    if not sub.dense_ranking and not sub.lexical_ranking:
        return None
    return RetrievalTrace(
        dense=[_ref(key) for key in sub.dense_ranking],
        lexical=[_ref(key) for key in sub.lexical_ranking],
        fused=[_ref(key) for key, _ in sub.fused_ranking],
    )


def _pass_report(record: PassRecord) -> PassReport:
    """Project one internal pass record into its public report."""
    return PassReport(
        pass_number=record.pass_number,
        sufficient_ids=record.sufficient_ids,
        insufficient_ids=record.insufficient_ids,
        evidence_count=record.evidence_count,
    )


def _ref(key: BlockKey) -> RankedBlockRef:
    """Build a ranked block reference from a ``(norm_id, block_id)`` key."""
    norm_id, block_id = key
    return RankedBlockRef(norm_id=norm_id, block_id=block_id)
