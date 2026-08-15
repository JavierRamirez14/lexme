"""The with-reference judge: claim-by-claim grounding of a Mode 1 answer.

The layer metrics are automatic and cheap; the judge is the one expensive,
model-driven measurement and it is confined to the end-to-end output. To keep it
auditable it never opines in the abstract: it collates the answer against the
case's written key points and against the very articles the answer cited. Three
readings come back structured -- which key points the answer covers, whether each
legal claim it makes is supported by a cited article, and a short clarity rubric --
so the hallucination number is judged claim by claim, not answer by answer.

The articles it collates against are the ones in force at the date the answer
itself declares, not at the date the run was launched: an answer given for 2023
was written from the redaction of 2023, and grading it against today's would count
as unsupported a claim its own cited article backs. The citation guardrail reads
that date the same way, and for the same reason.

The judge model is pinned, runs at temperature 0 and belongs to a different family
than the generator whose answer it grades, to keep self-preference bias out of the
score; :func:`assert_judge_distinct_from_generator` enforces that at wiring time.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from lexme.blocks import BlockRef
from lexme.eval.cases import EvalCase
from lexme.eval.guardrail import read_verification_date
from lexme.llm import LlmClient, Message, TaskRegistry
from lexme.llm.protocol import StructuredOutputError
from lexme.mode1 import SYNTHESIS_TASK, AskResponse
from lexme.verification import CorpusReader

logger = logging.getLogger(__name__)

JUDGE_TASK = "judge"

MIN_CLARITY = 1
MAX_CLARITY = 5

_REF_WRAPPERS = "[] "


class JudgeConfigError(ValueError):
    """Raised when the judge configuration breaks a with-reference invariant."""


class KeyPointCoverage(BaseModel):
    """The judge's ruling on whether the answer covers one reference key point.

    ``block_ref`` ties the ruling back to the key point's gold block; ``evidence``
    quotes where in the answer the point is made, empty when it is not covered.
    """

    block_ref: str
    covered: bool
    evidence: str = ""


class ClaimAssessment(BaseModel):
    """The judge's ruling on one legal claim the answer makes.

    ``supported`` is whether a cited article backs the claim; ``supporting_block_ref``
    names that article, or is ``None`` for an unsupported claim -- the unit the
    hallucination rate counts.
    """

    claim: str
    supported: bool
    supporting_block_ref: str | None = None


class JudgeVerdict(BaseModel):
    """The judge's structured reading of one answer against its reference.

    ``key_points`` rules on completeness, ``claims`` on per-claim support, and
    ``clarity`` scores the plain-language rubric from ``MIN_CLARITY`` to
    ``MAX_CLARITY``.
    """

    key_points: list[KeyPointCoverage]
    claims: list[ClaimAssessment]
    clarity: int = Field(ge=MIN_CLARITY, le=MAX_CLARITY)


class JudgeMetrics(BaseModel):
    """The numbers derived in code from one :class:`JudgeVerdict`.

    ``completeness`` is the fraction of reference key points the answer covers,
    ``None`` when the case declares none. ``unsupported_claim_rate`` is the fraction
    of the answer's legal claims no cited article backs -- the hallucination metric,
    ``None`` when the judge found no claims. ``clarity`` echoes the rubric score.
    """

    completeness: float | None
    unsupported_claim_rate: float | None
    clarity: int
    key_points_total: int
    key_points_covered: int
    claims_total: int
    claims_unsupported: int


@runtime_checkable
class Judge(Protocol):
    """Grades one answered case against its reference key points and cited articles."""

    def judge(self, case: EvalCase, response: AskResponse, case_date: date) -> JudgeVerdict | None:
        """Return the verdict, or ``None`` when the case is not judgeable.

        ``case_date`` is the date the case was launched at, and only backs up the
        date the answer itself declares.
        """
        ...


_SYSTEM_PROMPT = (
    "Eres un juez de calidad de respuestas jurídicas sobre el alquiler de vivienda "
    "en España. No opinas en abstracto: cotejas la respuesta contra una referencia "
    "dada. Recibes la "
    "pregunta, la respuesta en lenguaje llano, los artículos que la respuesta ha "
    "citado (con su texto) y una lista de puntos clave de referencia que una buena "
    "respuesta debe contener. Devuelves tres lecturas:\n"
    "- key_points: por cada punto clave de referencia (identificado por su "
    "'block_ref'), si la respuesta lo cubre (covered) y, si lo cubre, una cita "
    "breve de dónde (evidence).\n"
    "- claims: enumera las afirmaciones jurídicas que hace la respuesta; por cada "
    "una, si algún artículo citado la sostiene (supported) y cuál "
    "(supporting_block_ref), o supported=false si ninguno la respalda.\n"
    "- clarity: un entero de 1 a 5 sobre lo clara que es en lenguaje llano.\n"
    "Juzga afirmación a afirmación: una afirmación jurídica sin artículo que la "
    "respalde es 'supported=false', aunque suene razonable.\n"
    "Cada referencia a un bloque ('block_ref', 'supporting_block_ref') se escribe "
    "con el identificador exacto en formato 'norma:bloque', sin corchetes y sin "
    "acompañarlo del texto del punto clave."
)


@dataclass(frozen=True)
class LlmJudge:
    """A :class:`Judge` backed by the pinned, temperature-0 judge task.

    Resolves the full text of each cited article from ``corpus`` at the date the
    answer declares, so the judge weighs each claim against the article itself --
    the redaction the answer was written from -- and not just the shown quote. A
    case with no key points or no answer is not judgeable and returns ``None``
    without a model call.
    """

    llm: LlmClient
    corpus: CorpusReader

    def judge(self, case: EvalCase, response: AskResponse, case_date: date) -> JudgeVerdict | None:
        """Grade ``response`` against ``case``'s key points, or ``None`` if not judgeable.

        The cited articles are read at the date the answer declares, with
        ``case_date`` as the fallback: a disambiguation reply can move a case's
        clock back years, and grading its answer against the law of today would
        mark as unsupported the claims its own citations carry.

        A reply that does not parse into a verdict leaves the case unjudged rather
        than ending the run: the judge is one soft metric among many, and a free-tier
        model that answers in prose must not cost the run every other case's numbers.
        Provider and configuration failures still propagate.
        """
        if not case.key_points or response.answer is None:
            return None
        cited = self._cited_articles(response, read_verification_date(response, case_date))
        messages = [
            Message("system", _SYSTEM_PROMPT),
            Message("user", _render_prompt(case, response, cited)),
        ]
        try:
            verdict = self.llm.complete_structured(JUDGE_TASK, messages, JudgeVerdict)
        except StructuredOutputError as error:
            logger.warning("case '%s' left unjudged: %s", case.id, error)
            return None
        return _anchor_verdict(verdict, [*case.gold_block_refs, *cited])

    def _cited_articles(self, response: AskResponse, verified_at: date) -> dict[str, str]:
        """Resolve each cited block's text as it stood at ``verified_at``, keyed by reference."""
        texts: dict[str, str] = {}
        assert response.answer is not None  # guarded by the caller
        for citation in response.answer.fundamento:
            cited = BlockRef.parse(citation.block_ref)
            if cited is None:
                continue
            resolved = self.corpus.resolve_block(cited.norm_id, cited.block_id, verified_at)
            if resolved is not None:
                texts[citation.block_ref] = resolved.text
        return texts


def _anchor_verdict(verdict: JudgeVerdict, known_refs: Sequence[str]) -> JudgeVerdict:
    """Resolve every reference the verdict names to one of the blocks it was shown.

    Free-tier judges write the reference back loosely: bracketed, echoed together
    with the key point's text, or as a bare article number. A ruling whose anchor
    cannot be read points at nothing, so each one is matched against the closed set
    of blocks the prompt actually carried. A claim whose backing cannot be resolved
    keeps none, and a key point keeps whatever the judge wrote, so a judge naming a
    block the case never mentioned stays visible instead of being tidied away.
    """
    key_points = [
        point.model_copy(
            update={"block_ref": _resolve_ref(point.block_ref, known_refs) or point.block_ref}
        )
        for point in verdict.key_points
    ]
    claims = [
        claim.model_copy(
            update={"supporting_block_ref": _resolve_ref(claim.supporting_block_ref, known_refs)}
        )
        for claim in verdict.claims
    ]
    return verdict.model_copy(update={"key_points": key_points, "claims": claims})


def _resolve_ref(raw: str | None, known_refs: Sequence[str]) -> str | None:
    """The known block reference ``raw`` names, or ``None`` when it names none.

    A bare block id resolves only when a single known norm carries it: with several
    norms in the corpus, guessing which "article 9" was meant is exactly the
    ambiguity qualified references exist to prevent.
    """
    if not raw:
        return None
    known = list(dict.fromkeys(known_refs))
    if raw in known:
        return raw
    quoted = _longest_quoted(raw, known)
    if quoted is not None:
        return quoted
    by_block_id: dict[str, set[str]] = {}
    for ref in known:
        parsed = BlockRef.parse(ref)
        if parsed is not None:
            by_block_id.setdefault(parsed.block_id, set()).add(ref)
    named = by_block_id.get(raw.strip(_REF_WRAPPERS), set())
    if len(named) == 1:
        return named.pop()
    logger.warning("judge named a block outside the case's references: %r", raw[:80])
    return None


def _longest_quoted(raw: str, known: Sequence[str]) -> str | None:
    """The one known reference quoted inside ``raw``, or ``None`` if that is ambiguous.

    A reference that is a prefix of another quoted one is discarded: ``:a9`` is
    inside ``:a90``, and dropping the anchor over that would lose a legible ruling.
    """
    quoted = [ref for ref in known if ref in raw]
    maximal = [ref for ref in quoted if not any(ref != other and ref in other for other in quoted)]
    return maximal[0] if len(maximal) == 1 else None


def build_judge_metrics(verdict: JudgeVerdict) -> JudgeMetrics:
    """Derive the completeness, hallucination and clarity numbers from a verdict."""
    covered = sum(1 for point in verdict.key_points if point.covered)
    total_points = len(verdict.key_points)
    unsupported = sum(1 for claim in verdict.claims if not claim.supported)
    total_claims = len(verdict.claims)
    return JudgeMetrics(
        completeness=(covered / total_points) if total_points else None,
        unsupported_claim_rate=(unsupported / total_claims) if total_claims else None,
        clarity=verdict.clarity,
        key_points_total=total_points,
        key_points_covered=covered,
        claims_total=total_claims,
        claims_unsupported=unsupported,
    )


def assert_judge_distinct_from_generator(
    registry: TaskRegistry, generator_task: str = SYNTHESIS_TASK
) -> None:
    """Fail loudly unless the judge is a pinned, temperature-0 model of another family.

    The judge must not be the same model that wrote the answer it grades, or the
    score carries self-preference bias; it must run at temperature 0 so the verdict
    is reproducible. Raises :class:`JudgeConfigError` on any breach.
    """
    judge = registry.resolve(JUDGE_TASK)
    generator = registry.resolve(generator_task)
    if judge.provider == generator.provider and judge.model == generator.model:
        raise JudgeConfigError(
            f"judge model '{judge.model}' is the same as the generator "
            f"('{generator_task}'); the judge must be a different family"
        )
    if judge.temperature != 0.0:
        raise JudgeConfigError(
            f"judge temperature must be 0 for a reproducible verdict, got {judge.temperature}"
        )


def _render_prompt(case: EvalCase, response: AskResponse, cited: dict[str, str]) -> str:
    """Build the judge's user message from the answer, cited articles and reference."""
    assert response.answer is not None  # guarded by the caller
    articles = (
        "\n\n".join(f"[{block_ref}]\n{text}" for block_ref, text in cited.items())
        or "(la respuesta no citó ningún artículo)"
    )
    key_points = "\n".join(f"- [{point.block_ref}] {point.claim}" for point in case.key_points)
    return (
        f"Pregunta:\n{case.question}\n\n"
        f"Respuesta (explicación en lenguaje llano):\n{response.answer.explicacion}\n\n"
        f"Artículos citados por la respuesta:\n{articles}\n\n"
        f"Puntos clave de referencia que debe contener:\n{key_points}"
    )
