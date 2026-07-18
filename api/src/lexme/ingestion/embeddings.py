"""Client for the Text Embeddings Inference (TEI) server serving BGE-M3."""

import httpx

EMBEDDING_DIMENSIONS = 1024
DEFAULT_BATCH_SIZE = 32
DEFAULT_TIMEOUT_SECONDS = 60.0


class TeiEmbedder:
    """Embeds text via a TEI ``/embed`` endpoint, in bounded batches.

    Requests are split into batches of at most ``batch_size`` inputs so a large
    call cannot exceed the TEI container's per-request limits.
    """

    def __init__(
        self,
        base_url: str,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        transport: httpx.BaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Build an embedder. ``transport`` is a seam for injecting a test double."""
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self._batch_size = batch_size

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text, preserving order.

        Raises :class:`httpx.HTTPStatusError` on a non-2xx response.
        """
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.post("/embed", json={"inputs": batch})
            response.raise_for_status()
            embeddings.extend(response.json())
        return embeddings

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "TeiEmbedder":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
