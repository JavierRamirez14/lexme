"""Tests for the Gemini adapter, driven by a mock HTTP transport."""

import json

import httpx
import pytest

from lexme.llm.gemini_adapter import GeminiAdapter
from lexme.llm.protocol import Message, ProviderResponseError
from lexme.llm.provider import ProviderRequest


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _reply(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_posts_to_generate_content_with_the_model_in_the_path() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["api_key"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json=_reply("hola"))

    with GeminiAdapter("secret-key", transport=_transport(handler)) as adapter:
        text = adapter.complete(ProviderRequest("gemini-2.5-flash", 0.0, [Message("user", "hi")]))

    assert text == "hola"
    assert seen["path"].endswith("/models/gemini-2.5-flash:generateContent")
    assert seen["api_key"] == "secret-key"


def test_maps_roles_and_lifts_system_messages_to_instruction() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_reply("ok"))

    messages = [
        Message("system", "you are a judge"),
        Message("user", "grade this"),
        Message("assistant", "prior"),
    ]
    with GeminiAdapter("k", transport=_transport(handler)) as adapter:
        adapter.complete(ProviderRequest("gemini-2.5-flash", 0.3, messages, json_mode=True))

    body = seen["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == "you are a judge"
    assert [content["role"] for content in body["contents"]] == ["user", "model"]
    assert body["generationConfig"]["temperature"] == 0.3


def test_never_requests_native_json_mode_which_truncates_this_model() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_reply("ok"))

    request = ProviderRequest("gemini-2.5-flash", 0.0, [Message("user", "hi")], json_mode=True)
    with GeminiAdapter("k", transport=_transport(handler)) as adapter:
        adapter.complete(request)

    assert "responseMimeType" not in seen["body"]["generationConfig"]


def test_raises_on_a_response_without_candidates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    request = ProviderRequest("gemini-2.5-flash", 0.0, [Message("user", "hi")])
    with (
        GeminiAdapter("k", transport=_transport(handler)) as adapter,
        pytest.raises(ProviderResponseError),
    ):
        adapter.complete(request)


def test_raises_httpstatuserror_on_non_2xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    request = ProviderRequest("gemini-2.5-flash", 0.0, [Message("user", "hi")])
    with (
        GeminiAdapter("k", transport=_transport(handler)) as adapter,
        pytest.raises(httpx.HTTPStatusError),
    ):
        adapter.complete(request)


def test_retries_a_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lexme.llm.http_adapter.time.sleep", lambda _seconds: None)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(
                429,
                json={"error": {"details": [{"retryDelay": "1s"}]}},
            )
        return httpx.Response(200, json=_reply("done"))

    with GeminiAdapter("k", transport=_transport(handler)) as adapter:
        text = adapter.complete(ProviderRequest("gemini-2.5-flash", 0.0, [Message("user", "hi")]))

    assert text == "done"
    assert len(calls) == 3


def test_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from lexme.llm.http_adapter import MAX_RETRIES

    monkeypatch.setattr("lexme.llm.http_adapter.time.sleep", lambda _seconds: None)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json={"error": "rate limited"})

    request = ProviderRequest("gemini-2.5-flash", 0.0, [Message("user", "hi")])
    with (
        GeminiAdapter("k", transport=_transport(handler)) as adapter,
        pytest.raises(httpx.HTTPStatusError),
    ):
        adapter.complete(request)

    assert len(calls) == MAX_RETRIES + 1


def test_empty_api_key_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="api_key"):
        GeminiAdapter("")
