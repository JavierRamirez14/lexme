"""HTTP client for the BOE Legislación Consolidada API."""

import httpx

DEFAULT_BASE_URL = "https://www.boe.es/datosabiertos/api/legislacion-consolidada"
DEFAULT_TIMEOUT_SECONDS = 30.0


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
        """Build a client. ``transport`` is a seam for injecting a test double."""
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Accept": "application/xml"},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    def fetch_norm(self, norm_id: str) -> str:
        """Return the XML of the consolidated norm ``norm_id``.

        Raises :class:`httpx.HTTPStatusError` on a non-2xx response.
        """
        response = self._client.get(f"/id/{norm_id}")
        response.raise_for_status()
        return response.text

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "BoeClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
