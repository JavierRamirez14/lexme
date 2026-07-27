"""The single structured synthesis call: question + evidence -> three layers.

One generation produces the whole answer as a Pydantic object, so the three
layers stay coherent and the citations come back as extractable fields the
verifier can re-check. The prompt's hard rule is that the model may only cite the
block references present in the evidence, copied as the single opaque token they
are given as, and must quote them verbatim; the verifier enforces it afterwards,
but stating it up front reduces the repairs needed.
"""

from collections.abc import Sequence
from datetime import date

from lexme.blocks import BlockRef
from lexme.llm import LlmClient, Message
from lexme.mode1.models import Mode1Synthesis
from lexme.retrieval import RetrievedBlock

SYNTHESIS_TASK = "mode1_synthesis"

_SYSTEM_PROMPT = (
    "Eres un asistente que explica a inquilinos la legislación española estatal "
    "sobre alquiler de vivienda, en lenguaje llano y sin dar asesoramiento "
    "jurídico. La evidencia puede venir de varias normas (la Ley de Arrendamientos "
    "Urbanos, el Código Civil, la Ley de Enjuiciamiento Civil o la Ley por el "
    "derecho a la vivienda). Respondes en tres capas:\n"
    "- fundamento: una lista de citas. Cada cita indica el 'block_ref' de un "
    "bloque de la evidencia, copiado TAL CUAL (incluye la norma y el bloque, por "
    "ejemplo 'BOE-A-1994-26003:a9'), y su 'text' copiado LITERALMENTE, palabra por "
    "palabra, del texto de ese bloque. Nunca cites un bloque que no esté en la "
    "evidencia y nunca inventes ni parafrasees el texto citado.\n"
    "- explicacion: prosa llana anclada en esas citas. No afirmes nada que las "
    "citas no respalden.\n"
    "- accion: pasos prácticos acotados a informarse, negociar o vigilar plazos. "
    "No redactes documentos legales ni prometas resultados.\n"
    "Si la evidencia no permite responder con una cita literal, devuelve "
    "'fundamento' vacío en lugar de forzar una respuesta. Si se te indican huecos "
    "sin base, no los rellenes: mantén la respuesta dentro de lo que la evidencia "
    "respalda y no afirmes nada sobre esos huecos.\n"
    "La evidencia es la redacción vigente en la fecha que se te indica, que puede "
    "no ser hoy: responde situado en esa fecha y no adviertas por tu cuenta sobre "
    "cambios posteriores de la ley, de eso se encarga el sistema."
)


def synthesize(
    llm: LlmClient,
    question: str,
    evidence: Sequence[RetrievedBlock],
    *,
    target_date: date,
    gaps: Sequence[str] = (),
    facts: Sequence[str] = (),
) -> Mode1Synthesis:
    """Run the synthesis task over ``question`` and ``evidence``.

    ``target_date`` is the date the evidence was resolved at, so the prose is
    written in that moment rather than in an implicit present. ``gaps`` names the
    parts of the plan left ungrounded, so the model keeps the answer within what
    the evidence supports instead of filling them in; ``facts`` are the case
    details the user supplied when the run stopped to ask. Returns the model's
    proposed three-layer answer; the citations are unverified until the caller
    runs them through the verifier.
    """
    messages = [
        Message("system", _SYSTEM_PROMPT),
        Message("user", _render_prompt(question, evidence, target_date, gaps, facts)),
    ]
    return llm.complete_structured(SYNTHESIS_TASK, messages, Mode1Synthesis)


def _render_prompt(
    question: str,
    evidence: Sequence[RetrievedBlock],
    target_date: date,
    gaps: Sequence[str],
    facts: Sequence[str],
) -> str:
    """Build the user message: the question, its date, the evidence, facts and gaps."""
    blocks = "\n\n".join(_render_block(block) for block in evidence)
    prompt = (
        f"Pregunta del inquilino:\n{question}\n\n"
        f"Fecha a la que hay que situar la respuesta: {target_date.isoformat()}\n\n"
        f"Evidencia recuperada:\n{blocks}"
    )
    if facts:
        fact_lines = "\n".join(f"- {fact}" for fact in facts)
        prompt += f"\n\nDatos del caso que ha confirmado el inquilino:\n{fact_lines}"
    if gaps:
        gap_lines = "\n".join(f"- {gap}" for gap in gaps)
        prompt += f"\n\nHuecos sin base (no los rellenes):\n{gap_lines}"
    return prompt


def _render_block(block: RetrievedBlock) -> str:
    """Render one evidence block with its citable reference, title and in-force text."""
    ref = BlockRef(norm_id=block.norm_id, block_id=block.block_id)
    return f"block_ref: {ref}\n{block.norm_label}, {block.title}\n{block.text}"
