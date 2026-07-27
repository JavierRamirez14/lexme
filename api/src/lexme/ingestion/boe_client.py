"""HTTP client for the BOE Legislación Consolidada API."""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://www.boe.es/datosabiertos/api/legislacion-consolidada"
DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 15.0


class BoeClient:
    """Fetches consolidated norms from the BOE open-data API as XML.

    The ``/id/{norm_id}`` endpoint is the only one that bundles metadata and the
    complete text with every historical version in a single call, so this client
    exposes just that.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Build a client. ``transport`` is a seam for injecting a test double.

        Connections are not pooled between norms. Ingestion spends minutes
        embedding one norm before fetching the next, by which time a kept-alive
        connection has been dropped at the other end and reusing it stalls until
        the read times out; dialing fresh costs one handshake per norm.
        """
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Accept": "application/xml"},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=0),
        )

    def fetch_norm(self, norm_id: str) -> str:
        """Return the XML of the consolidated norm ``norm_id``.

        A transport failure is retried up to :data:`MAX_ATTEMPTS` times, backing off
        between attempts. The older codes run to several megabytes and the BOE
        stops answering a burst of them for a while; retrying at once only extends
        the burst, so the wait grows with each attempt. Raises
        :class:`httpx.HTTPStatusError` on a non-2xx response and re-raises the
        transport error once the attempts are spent.
        """
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.get(f"/id/{norm_id}")
                response.raise_for_status()
                return response.text
            except httpx.TransportError as error:
                if attempt == MAX_ATTEMPTS:
                    raise
                wait = BACKOFF_SECONDS * attempt
                logger.warning(
                    "fetching norm %s failed (attempt %d/%d), retrying in %.0fs: %s",
                    norm_id,
                    attempt,
                    MAX_ATTEMPTS,
                    wait,
                    error,
                )
                self._sleep(wait)
        raise AssertionError("unreachable: the loop either returns or raises")

    def _sleep(self, seconds: float) -> None:
        """Wait between retries; overridden in tests so they stay fast."""
        time.sleep(seconds)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "BoeClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
