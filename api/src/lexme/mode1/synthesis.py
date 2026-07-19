"""The single structured synthesis call: question + evidence -> three layers.

One generation produces the whole answer as a Pydantic object, so the three
layers stay coherent and the citations come back as extractable fields the
verifier can re-check. The prompt's hard rule is that the model may only cite the
block ids present in the evidence, and must quote them verbatim; the verifier
enforces it afterwards, but stating it up front reduces the repairs needed.
"""

from collections.abc import Sequence

from lexme.llm import LlmClient, Message
from lexme.mode1.models import Mode1Synthesis
from lexme.retrieval import RetrievedBlock

SYNTHESIS_TASK = "mode1_synthesis"

_SYSTEM_PROMPT = (
    "Eres un asistente que explica la legislación española de arrendamientos "
    "urbanos (LAU, ámbito estatal) a inquilinos, en lenguaje llano y sin dar "
    "asesoramiento jurídico. Respondes en tres capas:\n"
    "- fundamento: una lista de citas. Cada cita indica el 'block_id' de un "
    "bloque de la evidencia y su 'text' copiado LITERALMENTE, palabra por "
    "palabra, del texto de ese bloque. Nunca cites un bloque que no esté en la "
    "evidencia y nunca inventes ni parafrasees el texto citado.\n"
    "- explicacion: prosa llana anclada en esas citas. No afirmes nada que las "
    "citas no respalden.\n"
    "- accion: pasos prácticos acotados a informarse, negociar o vigilar plazos. "
    "No redactes documentos legales ni prometas resultados.\n"
    "Si la evidencia no permite responder con una cita literal, devuelve "
    "'fundamento' vacío en lugar de forzar una respuesta."
)


def synthesize(llm: LlmClient, question: str, evidence: Sequence[RetrievedBlock]) -> Mode1Synthesis:
    """Run the synthesis task over ``question`` and ``evidence``.

    Returns the model's proposed three-layer answer. The reply is parsed into
    :class:`Mode1Synthesis` by the LLM interface; the citations it contains are
    unverified until the caller runs them through the verifier.
    """
    messages = [
        Message("system", _SYSTEM_PROMPT),
        Message("user", _render_prompt(question, evidence)),
    ]
    return llm.complete_structured(SYNTHESIS_TASK, messages, Mode1Synthesis)


def _render_prompt(question: str, evidence: Sequence[RetrievedBlock]) -> str:
    """Build the user message: the question followed by the retrieved evidence."""
    blocks = "\n\n".join(_render_block(block) for block in evidence)
    return f"Pregunta del inquilino:\n{question}\n\nEvidencia recuperada:\n{blocks}"


def _render_block(block: RetrievedBlock) -> str:
    """Render one evidence block with its citable id, title and in-force text."""
    return f"block_id: {block.block_id}\n{block.title}\n{block.text}"
