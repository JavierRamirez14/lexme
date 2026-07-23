"""The segmentation call: the document split into clauses the code then anchors.

One structured call proposes the lease's clauses, each a short heading and the
text copied literally from the document. The literalness is the model's only job
here; whether the text is actually in the document is decided afterwards by
:mod:`lexme.mode2.anchor`, so the prompt asks for verbatim spans but the guarantee
is enforced in code.
"""

from pydantic import BaseModel

from lexme.llm import LlmClient, Message

SEGMENTATION_TASK = "mode2_segmentation"

_SYSTEM_PROMPT = (
    "Eres un asistente que segmenta un contrato de arrendamiento en sus cláusulas. "
    "Divides el documento en una lista ordenada de cláusulas. Para cada cláusula "
    "devuelves:\n"
    "- heading: un título breve que la identifique (por ejemplo 'Duración', "
    "'Renta', 'Fianza').\n"
    "- text: el texto de la cláusula copiado LITERALMENTE, palabra por palabra, del "
    "documento. No lo resumas, no lo parafrasees, no inventes texto que no esté en "
    "el documento. Copia un fragmento contiguo tal cual aparece.\n"
    "No añadas cláusulas que no estén en el documento. Si una parte del documento no "
    "es una cláusula (encabezados, firmas), no la incluyas."
)


class ProposedClause(BaseModel):
    """One clause as the model proposes it: a heading and a literal text span."""

    heading: str
    text: str


class SegmentationProposal(BaseModel):
    """The model's proposed clause list, before code anchors each to the document."""

    clauses: list[ProposedClause]


def segment_document(llm: LlmClient, document_text: str) -> SegmentationProposal:
    """Run the segmentation task over ``document_text``.

    Returns the proposed clauses; none is trusted until the caller anchors it to a
    literal span of the document.
    """
    messages = [
        Message("system", _SYSTEM_PROMPT),
        Message("user", document_text),
    ]
    return llm.complete_structured(SEGMENTATION_TASK, messages, SegmentationProposal)
