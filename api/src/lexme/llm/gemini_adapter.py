"""Adapter for the Gemini API (Google AI Studio free tier), over plain HTTP.

Talks to the ``generateContent`` REST endpoint with ``httpx``; it pulls in no
Google SDK. Structured output is requested with ``responseMimeType`` only; the
interface derives and validates the schema.
"""

import httpx

from lexme.llm.http_adapter import DEFAULT_TIMEOUT_SECONDS, HttpProviderAdapter
from lexme.llm.protocol import ProviderResponseError
from lexme.llm.provider import ProviderRequest

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
JSON_MIME_TYPE = "application/json"
_ROLE_MAP = {"user": "user", "assistant": "model"}


class GeminiAdapter(HttpProviderAdapter):
    """Completes requests against a Gemini model via ``generateContent``."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.BaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Build the adapter. The API key travels in the ``x-goog-api-key`` header."""
        super().__init__(
            api_key,
            base_url=base_url,
            headers={"x-goog-api-key": api_key},
            transport=transport,
            timeout=timeout,
        )

    def _endpoint(self, request: ProviderRequest) -> str:
        return f"/models/{request.model}:generateContent"

    def _build_body(self, request: ProviderRequest) -> dict:
        contents = [
            {"role": _ROLE_MAP[message.role], "parts": [{"text": message.content}]}
            for message in request.messages
            if message.role != "system"
        ]
        system_text = "\n\n".join(
            message.content for message in request.messages if message.role == "system"
        )
        generation_config: dict = {"temperature": request.temperature}
        if request.json_mode:
            generation_config["responseMimeType"] = JSON_MIME_TYPE

        body: dict = {"contents": contents, "generationConfig": generation_config}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        return body

    def _extract_text(self, payload: dict) -> str:
        candidates = payload.get("candidates")
        if not candidates:
            raise ProviderResponseError(f"Gemini response has no candidates: {payload}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts)
        if not text:
            raise ProviderResponseError(f"Gemini candidate has no text: {payload}")
        return text
