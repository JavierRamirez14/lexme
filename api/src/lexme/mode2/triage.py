"""The triage call: one pass that reads the contract's ficha and its use.

A single structured call extracts the sheet fields a tenant recognizes -- the
parties, the property, the rent, the term, the deposit, the signing date -- plus
the lease's use as a raw classification and the span the document declares it in.
The use is a signal, not a verdict: the art 4.2 scope gate is decided by code in
:mod:`lexme.mode2.gates`, never by the model, and it closes only on the span, which
code checks against the document. So the model is asked to classify by the declared
destination and never by the term: a short fixed term over a dwelling is still a
dwelling, and reading it as a seasonal let is the exact trick an eleven-month,
no-renewal clause is written to play. Every field is text lifted from the document;
an absent field comes back empty rather than invented.
"""

from pydantic import BaseModel

from lexme.llm import LlmClient, Message
from lexme.mode2.models import TenancyUse

TRIAGE_TASK = "mode2_triage"

_SYSTEM_PROMPT = (
    "Eres un asistente que hace el triaje de un contrato de arrendamiento y "
    "extrae su ficha. Devuelves, copiando del documento y sin inventar nada:\n"
    "- arrendador: quién alquila (nombre o identificación tal como aparece).\n"
    "- arrendatario: el inquilino.\n"
    "- inmueble: la vivienda o inmueble arrendado (dirección o descripción).\n"
    "- renta: la renta pactada tal como aparece.\n"
    "- duracion: la duración o plazo del contrato.\n"
    "- fianza: la fianza o garantía.\n"
    "- fecha_firma: la fecha de firma en formato AAAA-MM-DD si aparece; cadena "
    "vacía si el contrato no indica fecha.\n"
    "- uso: clasifica el uso del inmueble por el destino que el documento declara, "
    "nunca por su duración. Un plazo corto, una fecha de vencimiento o la ausencia "
    "de prórroga no convierten una vivienda en un alquiler de temporada: la ley "
    "distingue por el destino del inmueble. Devuelve 'vivienda_habitual' si el "
    "inmueble se destina a vivienda del arrendatario; 'temporada' solo si el "
    "documento declara una finalidad de temporada, vacacional o turística; "
    "'uso_distinto' solo si declara un uso distinto del de vivienda (local, "
    "oficina, industria); 'indeterminado' si el documento no declara el destino.\n"
    "- uso_evidencia: si has devuelto 'temporada' o 'uso_distinto', copia "
    "literalmente del documento el fragmento que declara esa finalidad; cadena "
    "vacía en cualquier otro caso, incluido el caso en que ese fragmento no exista. "
    "Un plazo, una fecha o una duración nunca son esa evidencia.\n"
    "Si un dato no aparece en el documento, devuelve una cadena vacía en ese campo; "
    "no lo deduzcas ni lo inventes."
)


class TriageResult(BaseModel):
    """The contract's ficha and its raw use classification as the model reads them.

    The sheet fields are text; ``uso`` is a signal the scope gate reads, not a
    scope decision, and ``uso_evidencia`` is the span the document declares an
    excluded use in -- empty whenever there is none, which is what keeps the gate
    from closing on the model's word alone. ``fecha_firma`` is text too, parsed
    into a date by code.
    """

    arrendador: str = ""
    arrendatario: str = ""
    inmueble: str = ""
    renta: str = ""
    duracion: str = ""
    fianza: str = ""
    fecha_firma: str = ""
    uso: TenancyUse = TenancyUse.UNDETERMINED
    uso_evidencia: str = ""


def triage_document(llm: LlmClient, document_text: str) -> TriageResult:
    """Run the triage task over ``document_text`` and return the extracted ficha."""
    messages = [
        Message("system", _SYSTEM_PROMPT),
        Message("user", document_text),
    ]
    return llm.complete_structured(TRIAGE_TASK, messages, TriageResult)
