"""Shared HTTP plumbing for provider adapters.

Every provider adapter owns an ``httpx`` client, guards against a missing key,
and runs the same request flow: build a body, POST it, raise on a bad status,
extract the reply. This base holds that flow; a concrete adapter supplies only the
endpoint, the request body and how to read the reply out of the response.
"""

import httpx

from lexme.llm.provider import ProviderRequest

DEFAULT_TIMEOUT_SECONDS = 60.0


class HttpProviderAdapter:
    """Base for adapters that reach a provider over its HTTP API.

    Subclasses implement :meth:`_endpoint`, :meth:`_build_body` and
    :meth:`_extract_text`; they never touch the HTTP client directly.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str,
        headers: dict[str, str],
        transport: httpx.BaseTransport | None,
        timeout: float,
    ) -> None:
        """Build the adapter's HTTP client.

        Raises :class:`ValueError` if ``api_key`` is empty, so a misconfigured
        deployment fails at wiring time rather than on the first call.
        ``transport`` is a seam for injecting a test double.
        """
        if not api_key:
            raise ValueError(f"{type(self).__name__} requires a non-empty api_key")
        self._client = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )

    def complete(self, request: ProviderRequest) -> str:
        """Return the model's reply text for ``request``.

        Raises :class:`httpx.HTTPStatusError` on a non-2xx response and
        :class:`~lexme.llm.protocol.ProviderResponseError` if the body carries no
        completion.
        """
        response = self._client.post(self._endpoint(request), json=self._build_body(request))
        response.raise_for_status()
        return self._extract_text(response.json())

    def _endpoint(self, request: ProviderRequest) -> str:
        """Return the request path for ``request``."""
        raise NotImplementedError

    def _build_body(self, request: ProviderRequest) -> dict:
        """Translate ``request`` into the provider's JSON payload."""
        raise NotImplementedError

    def _extract_text(self, payload: dict) -> str:
        """Read the reply text out of a decoded response body."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "HttpProviderAdapter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
