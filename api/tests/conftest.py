"""Shared test fixtures: a real uvicorn server on a loopback socket."""

import os
import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn

os.environ.setdefault("DATABASE_URL", "postgresql://localhost/lexme_test")

from lexme.llm import FakeLlmClient
from lexme.main import app

STARTUP_TIMEOUT_SECONDS = 10
SHUTDOWN_TIMEOUT_SECONDS = 5
CONNECT_PROBE_TIMEOUT_SECONDS = 0.25
CONNECT_RETRY_INTERVAL_SECONDS = 0.05


def _free_port() -> int:
    """Return an OS-assigned free TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_until_accepting_connections(host: str, port: int) -> None:
    """Block until a TCP connection to host:port succeeds, or raise TimeoutError."""
    stop_at = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < stop_at:
        try:
            with socket.create_connection((host, port), timeout=CONNECT_PROBE_TIMEOUT_SECONDS):
                return
        except OSError:
            time.sleep(CONNECT_RETRY_INTERVAL_SECONDS)
    raise TimeoutError(f"server did not start accepting connections on {host}:{port}")


@pytest.fixture
def fake_llm() -> FakeLlmClient:
    """The suite's single LLM substitution point: a deterministic, network-free client.

    Every component that needs a model is wired through :class:`LlmClient`, so a
    test programs replies per task here and lets retrieval, verification, the
    checklist cross-check and the code gates run for real.
    """
    return FakeLlmClient()


@pytest.fixture
def live_server() -> Iterator[str]:
    """Run the app in a real uvicorn server on a loopback port for the test.

    Yields the base URL and tears the server down afterwards.
    """
    host = "127.0.0.1"
    port = _free_port()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    _wait_until_accepting_connections(host, port)

    yield f"http://{host}:{port}"

    server.should_exit = True
    thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
