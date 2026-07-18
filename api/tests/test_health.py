"""Integration tests for GET /health against a real HTTP server."""

from collections.abc import Iterator
from contextlib import contextmanager

import httpx

from lexme.api.health import is_database_reachable, is_tei_reachable
from lexme.main import app


@contextmanager
def overriding_health_checks(*, database_ok: bool, tei_ok: bool) -> Iterator[None]:
    """Override both health probes with fixed booleans for the duration of the block."""
    app.dependency_overrides[is_database_reachable] = lambda: database_ok
    app.dependency_overrides[is_tei_reachable] = lambda: tei_ok
    try:
        yield
    finally:
        app.dependency_overrides.clear()


def test_health_reports_ok_when_dependencies_are_reachable(live_server: str) -> None:
    with overriding_health_checks(database_ok=True, tei_ok=True):
        response = httpx.get(f"{live_server}/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok", "tei": "ok"}


def test_health_reports_degraded_when_database_is_unreachable(live_server: str) -> None:
    with overriding_health_checks(database_ok=False, tei_ok=True):
        response = httpx.get(f"{live_server}/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "db": "error", "tei": "ok"}


def test_health_reports_degraded_when_tei_is_unreachable(live_server: str) -> None:
    with overriding_health_checks(database_ok=True, tei_ok=False):
        response = httpx.get(f"{live_server}/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "db": "ok", "tei": "error"}
