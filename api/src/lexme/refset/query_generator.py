"""Generate Mode 1 reference cases by walking the corpus backwards.

The corpus is read in reverse: a seed names one to three related blocks and the
outcome the case should reach, and the model writes the user-language question and
extracts, from those blocks' own text, the key points a good answer must contain.
The gold blocks are the seed's blocks by construction -- never the model's choice --
so recall has a correct reference without a human labelling it. Adversarial seeds
(a matter outside the corpus, a question with no sufficient ground) reach the same
shape with an empty reference. Every generated case is a pending candidate: nothing
here enters the set until a human accepts it.
"""

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel

from lexme.llm import LlmClient, Message
from lexme.mode1.models import Outcome
from lexme.refset.models import (
    Candidate,
    CaseKind,
    KeyPoint,
    Mode1ReferenceCase,
    Provenance,
)
from lexme.verification import CorpusReader

QUERY_GENERATION_TASK = "refset_query_generation"
_GENERATOR_NAME = "query_generator"

_OUTCOMES_REQUIRING_GROUND: frozenset[Outcome] = frozenset({Outcome.ANSWER, Outcome.PARTIAL_ANSWER})


class GenerationError(ValueError):
    """Raised when a seed is malformed or the model breaks the by-construction contract."""


class QueryStyle(StrEnum):
    """How the query should read, a phrasing hint that does not change the outcome.

    ``CORE`` is a plain informational question; ``MULTI_ARTICLE`` spans several
    blocks to exercise the planner; ``FALSE_PREMISE`` embeds a false claim the
    answer must correct; ``INSUFFICIENT`` asks something the corpus cannot fully
    ground; ``OUT_OF_SCOPE`` asks about a matter the corpus does not cover.
    """

    CORE = "nucleo"
    MULTI_ARTICLE = "multiarticulo"
    FALSE_PREMISE = "premisa_falsa"
    INSUFFICIENT = "sin_base"
    OUT_OF_SCOPE = "fuera_de_ambito"


class QuerySeed(BaseModel):
    """The construction inputs for one Mode 1 case: blocks, target outcome, style.

    ``block_ids`` are the gold blocks by construction; an out-of-scope seed leaves
    them empty and sets ``topic`` instead, naming the foreign matter to ask about.
    ``target_date`` pins the point-in-time clock for a time-sensitive case.
    """

    id: str
    block_ids: list[str] = []
    expected_outcome: Outcome
    style: QueryStyle = QueryStyle.CORE
    topic: str = ""
    target_date: date | None = None


class _GeneratedKeyPoint(BaseModel):
    """A key point as the model proposes it, before code checks its block."""

    claim: str
    block_id: str


class GeneratedQuery(BaseModel):
    """The model's proposal: a user question and the key points it extracted.

    Nothing here is trusted until code has checked every key point cites a seed
    block and the reference matches the seed's outcome.
    """

    question: str
    key_points: list[_GeneratedKeyPoint] = []


_SYSTEM_PROMPT = (
    "Eres un generador de casos de evaluación para un asistente jurídico sobre la Ley "
    "de Arrendamientos Urbanos. A partir de uno o varios artículos de la ley, redactas "
    "UNA consulta en el lenguaje natural de un inquilino y extraes los 'puntos clave': "
    "las afirmaciones jurídicas que una buena respuesta debe contener, cada una copiada "
    "o parafraseada del texto del artículo y anclada a su 'block_id'. Devuelves un objeto "
    "JSON con 'question' y 'key_points' (lista de objetos con 'claim' y 'block_id'). "
    "Reglas: la consulta suena a persona real, no cita artículos por número; cada "
    "'block_id' de un punto clave DEBE ser uno de los bloques que se te dan; nunca "
    "inventes un bloque ni un dato que no esté en el texto."
)

_STYLE_DIRECTIVES: dict[QueryStyle, str] = {
    QueryStyle.CORE: (
        "Redacta una consulta informativa directa sobre la materia de estos bloques."
    ),
    QueryStyle.MULTI_ARTICLE: (
        "Redacta una consulta que solo se responda bien combinando TODOS los bloques, "
        "para ejercitar la planificación en varios sub-pasos."
    ),
    QueryStyle.FALSE_PREMISE: (
        "Redacta una consulta que dé por cierta una premisa FALSA que contradice el "
        "texto (por ejemplo una cifra o un plazo equivocados); los puntos clave son la "
        "afirmación correcta del artículo que desmiente esa premisa."
    ),
    QueryStyle.INSUFFICIENT: (
        "Redacta una consulta cuya respuesta NO esté suficientemente cubierta por estos "
        "bloques, de modo que el asistente deba abstenerse. Deja 'key_points' vacío."
    ),
    QueryStyle.OUT_OF_SCOPE: (
        "Redacta una consulta sobre una materia AJENA a la Ley de Arrendamientos Urbanos "
        "(la que se indica como tema), para que el asistente declare que está fuera de su "
        "ámbito. Deja 'key_points' vacío."
    ),
}


def generate_query_case(
    seed: QuerySeed,
    *,
    corpus: CorpusReader,
    llm: LlmClient,
    norm_id: str,
    now: datetime,
    resolve_date: date,
    model: str | None = None,
) -> Candidate:
    """Generate one pending Mode 1 candidate from ``seed``.

    Resolves the seed's blocks against ``corpus`` at ``resolve_date`` for their
    text, asks ``llm`` for a question and key points, then builds the reference
    case with the seed's blocks as gold by construction. ``now`` stamps the
    provenance and ``model`` records the pinned model behind the generation.

    Raises :class:`GenerationError` if the seed is malformed, a seed block does
    not resolve, or the model cites a block outside the seed or returns a
    reference that contradicts the seed's outcome.
    """
    _validate_seed(seed)
    blocks = _resolve_blocks(seed, corpus, norm_id, resolve_date)
    proposal = _propose(seed, blocks, llm)
    key_points = _accept_key_points(seed, proposal)
    case = Mode1ReferenceCase(
        id=seed.id,
        question=proposal.question.strip(),
        gold_block_ids=list(seed.block_ids),
        expected_outcome=seed.expected_outcome,
        key_points=key_points,
        target_date=seed.target_date,
    )
    provenance = Provenance(
        generator=_GENERATOR_NAME,
        sources=list(seed.block_ids),
        model=model,
        generated_at=now,
    )
    return Candidate(kind=CaseKind.MODE1, provenance=provenance, mode1=case)


def _validate_seed(seed: QuerySeed) -> None:
    """Check the seed's shape against its outcome before any model call."""
    if seed.style is QueryStyle.OUT_OF_SCOPE:
        if seed.block_ids:
            raise GenerationError(f"seed '{seed.id}': an out-of-scope seed has no gold blocks")
        if not seed.topic.strip():
            raise GenerationError(f"seed '{seed.id}': an out-of-scope seed needs a 'topic'")
        return
    if not seed.block_ids:
        raise GenerationError(f"seed '{seed.id}': a grounded seed needs at least one block")


def _resolve_blocks(
    seed: QuerySeed, corpus: CorpusReader, norm_id: str, resolve_date: date
) -> dict[str, str]:
    """Resolve each seed block's text at ``resolve_date``, failing on a broken block."""
    blocks: dict[str, str] = {}
    for block_id in seed.block_ids:
        resolved = corpus.resolve_block(norm_id, block_id, resolve_date)
        if resolved is None:
            raise GenerationError(
                f"seed '{seed.id}': block '{block_id}' does not resolve in {norm_id} "
                f"at {resolve_date.isoformat()}"
            )
        blocks[block_id] = resolved.text
    return blocks


def _propose(seed: QuerySeed, blocks: dict[str, str], llm: LlmClient) -> GeneratedQuery:
    """Run the generation call over the seed's blocks and style directive."""
    messages = [
        Message("system", _SYSTEM_PROMPT),
        Message("user", _render_prompt(seed, blocks)),
    ]
    return llm.complete_structured(QUERY_GENERATION_TASK, messages, GeneratedQuery)


def _render_prompt(seed: QuerySeed, blocks: dict[str, str]) -> str:
    """Build the user message: the style directive and the seed's block texts."""
    prompt = _STYLE_DIRECTIVES[seed.style]
    if seed.topic.strip():
        prompt += f"\n\nTema ajeno a tratar: {seed.topic.strip()}"
    if blocks:
        rendered = "\n\n".join(f"block_id: {block_id}\n{text}" for block_id, text in blocks.items())
        prompt += f"\n\nBloques de la ley:\n{rendered}"
    return prompt


def _accept_key_points(seed: QuerySeed, proposal: GeneratedQuery) -> list[KeyPoint]:
    """Accept the proposed key points, enforcing the by-construction guarantees.

    Every key point must cite a seed block, and the count of key points must match
    what the seed's outcome allows: none for a rejection or abstention, at least
    one for an answer.
    """
    gold = set(seed.block_ids)
    for point in proposal.key_points:
        if point.block_id not in gold:
            raise GenerationError(
                f"seed '{seed.id}': model cited block '{point.block_id}' outside the seed"
            )
    key_points = [
        KeyPoint(claim=point.claim.strip(), block_id=point.block_id)
        for point in proposal.key_points
    ]
    _validate_reference(seed, key_points)
    return key_points


def _validate_reference(seed: QuerySeed, key_points: list[KeyPoint]) -> None:
    """Reject a reference that contradicts the outcome the seed was built for."""
    if seed.expected_outcome is Outcome.ROUTER_REJECTION and key_points:
        raise GenerationError(
            f"seed '{seed.id}': an out-of-scope rejection cannot carry key points"
        )
    if seed.expected_outcome in _OUTCOMES_REQUIRING_GROUND and not key_points:
        raise GenerationError(f"seed '{seed.id}': an answer case needs at least one key point")
