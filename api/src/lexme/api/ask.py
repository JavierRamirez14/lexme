"""The Mode 1 query endpoint: a question in, a cited answer or abstention out.

A thin HTTP adapter over :func:`lexme.mode1.answer_question`; all of retrieval,
synthesis, verification and the deterministic gate live in the Mode 1 package.
The endpoint only resolves the request's collaborators and picks the vertical.
"""

from datetime import date

import psycopg
from fastapi import APIRouter, Depends

from lexme.api.dependencies import (
    get_corpus_reader,
    get_db_connection,
    get_embedder,
    get_llm_client,
    get_target_date,
)
from lexme.llm import LlmClient
from lexme.mode1 import AskRequest, AskResponse, answer_question
from lexme.retrieval import QueryEmbedder
from lexme.verification import CorpusReader

DEFAULT_VERTICAL = "vivienda"

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    connection: psycopg.Connection = Depends(get_db_connection),
    embedder: QueryEmbedder = Depends(get_embedder),
    corpus: CorpusReader = Depends(get_corpus_reader),
    llm: LlmClient = Depends(get_llm_client),
    target_date: date = Depends(get_target_date),
) -> AskResponse:
    """Answer a Mode 1 question over the vivienda corpus, or abstain honestly."""
    return answer_question(
        request.question,
        connection=connection,
        embedder=embedder,
        corpus=corpus,
        llm=llm,
        vertical=DEFAULT_VERTICAL,
        target_date=target_date,
    )
