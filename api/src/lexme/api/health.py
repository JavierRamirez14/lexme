"""Health endpoint reporting connectivity to the database and TEI."""

import logging

import httpx
import psycopg
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from lexme.config import Settings, get_settings

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT_SECONDS = 2

router = APIRouter()


def is_database_reachable(settings: Settings = Depends(get_settings)) -> bool:
    """Return whether the database answers a trivial query within the timeout."""
    try:
        with psycopg.connect(
            settings.database_url, connect_timeout=HEALTH_CHECK_TIMEOUT_SECONDS
        ) as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.Error:
        logger.exception("database health check failed")
        return False


def is_tei_reachable(settings: Settings = Depends(get_settings)) -> bool:
    """Return whether the TEI embeddings server answers its health endpoint."""
    try:
        response = httpx.get(f"{settings.tei_url}/health", timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
        return response.status_code == 200
    except httpx.HTTPError:
        logger.exception("tei health check failed")
        return False


@router.get("/health")
def health(
    database_ok: bool = Depends(is_database_reachable),
    tei_ok: bool = Depends(is_tei_reachable),
) -> JSONResponse:
    """Report overall readiness plus a per-dependency breakdown.

    Returns ``200`` with ``status: ok`` only when every dependency is reachable;
    otherwise ``503`` with ``status: degraded`` and the failing dependency
    marked ``error``.
    """
    is_healthy = database_ok and tei_ok
    body = {
        "status": "ok" if is_healthy else "degraded",
        "db": "ok" if database_ok else "error",
        "tei": "ok" if tei_ok else "error",
    }
    return JSONResponse(content=body, status_code=200 if is_healthy else 503)
