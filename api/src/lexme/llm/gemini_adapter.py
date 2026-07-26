"""Adapter for the Gemini API (Google AI Studio free tier), over plain HTTP.

Talks to the ``generateContent`` REST endpoint with ``httpx``; it pulls in no
Google SDK. Structured output rides on the prompt (the interface appends the
schema instruction) rather than the model's native JSON mode, which truncates
replies for these flash models; :meth:`_build_body` explains the trade-off.
"""

import re

import httpx

from lexme.llm.http_adapter import DEFAULT_TIMEOUT_SECONDS, HttpProviderAdapter
from lexme.llm.protocol import ProviderResponseError
from lexme.llm.provider import ProviderRequest

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_ROLE_MAP = {"user": "user", "assistant": "model"}
_RETRY_DELAY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*s")


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
        # No responseMimeType JSON mode. These flash models' native JSON mode
        # returns *truncated* objects for our schemas -- it drops the closing brace
        # and stops (finishReason STOP), so the reply is invalid JSON. In plain
        # text mode the same call returns a complete, valid object. `json_mode` is
        # therefore honoured through the prompt: complete_structured always appends
        # the schema instruction ("a single JSON object and nothing else"), and the
        # parser tolerates a stray ```json fence.
        #
        # We also send no thinkingConfig. Disabling thinking (thinkingBudget 0) is
        # rejected outright by the flash-lite models (HTTP 400), and the daily free
        # quota is counted in requests, not tokens, so leaving thinking on costs us
        # nothing against the limit. The trade-off is a little run-to-run variance
        # at temperature 0, which the harness tolerates.
        generation_config: dict = {"temperature": request.temperature}

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

    def _body_retry_hint(self, payload: dict) -> float | None:
        """Parse Gemini's ``RetryInfo.retryDelay`` (e.g. ``"27s"``) out of a 429 body."""
        error = payload.get("error")
        details = error.get("details", []) if isinstance(error, dict) else []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            match = _RETRY_DELAY_PATTERN.fullmatch(str(detail.get("retryDelay", "")).strip())
            if match:
                return float(match.group(1))
        return None
