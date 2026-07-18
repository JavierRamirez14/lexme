"""Tests for the OpenRouter adapter, driven by a mock HTTP transport."""

import json

import httpx
import pytest

from lexme.llm.openrouter_adapter import OpenRouterAdapter
from lexme.llm.protocol import Message, ProviderResponseError
from lexme.llm.provider import ProviderRequest


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _reply(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def test_posts_chat_completions_with_bearer_auth_and_model_in_body() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_reply("verdict"))

    with OpenRouterAdapter("secret", transport=_transport(handler)) as adapter:
        text = adapter.complete(
            ProviderRequest("deepseek/deepseek-r1:free", 0.0, [Message("user", "hi")])
        )

    assert text == "verdict"
    assert seen["path"].endswith("/chat/completions")
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "deepseek/deepseek-r1:free"


def test_passes_messages_verbatim_and_sets_json_mode() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_reply("ok"))

    messages = [Message("system", "be terse"), Message("user", "why")]
    with OpenRouterAdapter("k", transport=_transport(handler)) as adapter:
        adapter.complete(ProviderRequest("m", 0.1, messages, json_mode=True))

    body = seen["body"]
    assert body["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "why"},
    ]
    assert body["temperature"] == 0.1
    assert body["response_format"] == {"type": "json_object"}


def test_omits_response_format_when_not_in_json_mode() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_reply("ok"))

    with OpenRouterAdapter("k", transport=_transport(handler)) as adapter:
        adapter.complete(ProviderRequest("m", 0.0, [Message("user", "hi")]))

    assert "response_format" not in seen["body"]


def test_raises_on_a_response_without_choices() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    request = ProviderRequest("m", 0.0, [Message("user", "hi")])
    with (
        OpenRouterAdapter("k", transport=_transport(handler)) as adapter,
        pytest.raises(ProviderResponseError),
    ):
        adapter.complete(request)


def test_empty_api_key_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="api_key"):
        OpenRouterAdapter("")
