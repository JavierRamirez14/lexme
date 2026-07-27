"""Shared HTTP plumbing for provider adapters.

Every provider adapter owns an ``httpx`` client, guards against a missing key,
and runs the same request flow: build a body, POST it, raise on a bad status,
extract the reply. This base holds that flow; a concrete adapter supplies only the
endpoint, the request body and how to read the reply out of the response.

The flow retries a rate-limited or transiently-unavailable call with backoff:
the free tiers this interface targets return ``429`` in bursts, and a harness run
fires many calls back to back, so without backoff a run dies on its first ``429``.
A dropped connection is retried on the same terms -- a free-tier endpoint closing
a response mid-body says nothing about the request, and losing a whole harness run
to one severed socket is the same failure as losing it to one ``429``. The wait
honours the standard ``Retry-After`` header and any provider-specific hint a
subclass reads from the error body (see :meth:`_body_retry_hint`), and otherwise
falls back to exponential backoff.
"""

import logging
import time

import httpx

from lexme.llm.provider import ProviderRequest

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60.0

#: Statuses worth retrying: rate limit and transient upstream unavailability.
RETRYABLE_STATUS = frozenset({429, 503})
#: How many times to retry before giving up and raising the status error.
MAX_RETRIES = 8
#: Exponential-backoff base and ceiling, in seconds, when the provider gives no hint.
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 65.0


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

        Retries a ``429``/``503`` and a dropped connection with backoff up to
        :data:`MAX_RETRIES` times, then raises :class:`httpx.HTTPStatusError` on any
        non-2xx response, :class:`httpx.TransportError` if the connection kept
        failing, and :class:`~lexme.llm.protocol.ProviderResponseError` if the body
        carries no completion.
        """
        endpoint = self._endpoint(request)
        body = self._build_body(request)
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._client.post(endpoint, json=body)
            except httpx.TransportError as error:
                if attempt >= MAX_RETRIES:
                    raise
                self._wait_before_retry(str(error), self._backoff(attempt), attempt)
                continue
            if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                self._wait_before_retry(
                    f"HTTP {response.status_code}", self._retry_delay(response, attempt), attempt
                )
                continue
            response.raise_for_status()
            return self._extract_text(response.json())
        raise AssertionError("unreachable: the loop returns or raises on the last attempt")

    def _wait_before_retry(self, cause: str, delay: float, attempt: int) -> None:
        """Log why the call is being retried and sleep for ``delay`` seconds."""
        logger.warning(
            "%s got %s; retry %d/%d in %.1fs",
            type(self).__name__,
            cause,
            attempt + 1,
            MAX_RETRIES,
            delay,
        )
        time.sleep(delay)

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """Seconds to wait before retrying, always at least the exponential backoff.

        Backs off exponentially with the attempt, and honours a provider hint (a
        ``Retry-After`` header or Gemini's ``RetryInfo.retryDelay``) only when it
        asks to wait *longer*. Flooring at the backoff matters: a hint of ``0s``
        would otherwise retry instantly and burn the whole retry budget before the
        rate-limit window clears. Everything is capped at :data:`MAX_BACKOFF_SECONDS`.
        """
        backoff = self._backoff(attempt)
        hinted = self._provider_hint(response)
        delay = max(hinted, backoff) if hinted is not None else backoff
        return min(delay, MAX_BACKOFF_SECONDS)

    def _backoff(self, attempt: int) -> float:
        """The exponential backoff for ``attempt``, capped at the ceiling."""
        return min(BASE_BACKOFF_SECONDS * 2**attempt, MAX_BACKOFF_SECONDS)

    def _provider_hint(self, response: httpx.Response) -> float | None:
        """Read a retry delay the provider suggested, or ``None`` if it gave none.

        Reads the standard ``Retry-After`` header here; a provider whose hint lives
        in the error body overrides :meth:`_body_retry_hint`.
        """
        header = response.headers.get("retry-after")
        if header and header.isdigit():
            return float(header)
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        return self._body_retry_hint(payload)

    def _body_retry_hint(self, payload: dict) -> float | None:
        """A retry delay read from a provider's error body, or ``None``.

        The base reads none; a concrete adapter overrides this to parse its
        provider's shape, keeping that schema out of the shared flow.
        """
        return None

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
