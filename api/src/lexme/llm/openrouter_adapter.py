"""Adapter for OpenRouter's free models, over its OpenAI-compatible HTTP API.

Talks to ``/chat/completions`` with ``httpx``; it pulls in no OpenAI SDK.
Structured output is requested with ``response_format`` JSON mode; the interface
derives and validates the schema.
"""

import httpx

from lexme.llm.http_adapter import DEFAULT_TIMEOUT_SECONDS, HttpProviderAdapter
from lexme.llm.protocol import ProviderResponseError
from lexme.llm.provider import ProviderRequest

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"


class OpenRouterAdapter(HttpProviderAdapter):
    """Completes requests against an OpenRouter model via chat completions."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.BaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Build the adapter. The API key travels as an ``Authorization`` bearer token."""
        super().__init__(
            api_key,
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
            timeout=timeout,
        )

    def _endpoint(self, request: ProviderRequest) -> str:
        return CHAT_COMPLETIONS_ENDPOINT

    def _build_body(self, request: ProviderRequest) -> dict:
        body: dict = {
            "model": request.model,
            "temperature": request.temperature,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
        }
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def _extract_text(self, payload: dict) -> str:
        """The reply's text, falling back to ``reasoning`` when ``content`` is empty.

        A free-tier reasoning model can spend its whole completion budget thinking
        and return ``content: null`` with the reply in ``reasoning``. Treating that
        as a provider failure would end a suite mid-run over one bad reply, so the
        reasoning text is returned instead and left for the structured-output layer
        to reject as the malformed model output it is.
        """
        choices = payload.get("choices")
        if not choices:
            raise ProviderResponseError(f"OpenRouter response has no choices: {payload}")
        message = choices[0].get("message", {})
        content = message.get("content") or message.get("reasoning")
        if not content:
            raise ProviderResponseError(f"OpenRouter choice has no content: {payload}")
        return content
