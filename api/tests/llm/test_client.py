"""Tests for the routing client: task resolution, dispatch and structured output."""

import pytest

from lexme.llm.client import STRUCTURED_ATTEMPTS, RoutingLlmClient
from lexme.llm.protocol import Message, StructuredOutputError, TaskNotConfiguredError
from lexme.llm.provider import ProviderRequest
from lexme.llm.registry import LlmConfigError, TaskModel, TaskRegistry
from tests.llm.conftest import Verdict


class RecordingAdapter:
    """A stub provider adapter that records requests and returns a fixed reply."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.requests: list[ProviderRequest] = []

    def complete(self, request: ProviderRequest) -> str:
        self.requests.append(request)
        return self._reply


class ReplayingAdapter:
    """A stub provider adapter that returns a different reply per call, in order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.requests: list[ProviderRequest] = []

    def complete(self, request: ProviderRequest) -> str:
        self.requests.append(request)
        return self._replies.pop(0)


def _registry() -> TaskRegistry:
    return TaskRegistry(
        {
            "synthesis": TaskModel("synthesis", "gemini", "gemini-2.5-flash", 0.0),
            "judge": TaskModel("judge", "openrouter", "some/model:free", 0.2),
        }
    )


def test_dispatches_each_task_to_its_own_provider_adapter() -> None:
    gemini = RecordingAdapter("from gemini")
    openrouter = RecordingAdapter("from openrouter")
    client = RoutingLlmClient(_registry(), {"gemini": gemini, "openrouter": openrouter})

    assert client.complete("synthesis", [Message("user", "hi")]) == "from gemini"
    assert client.complete("judge", [Message("user", "hi")]) == "from openrouter"
    assert gemini.requests[0].model == "gemini-2.5-flash"
    assert openrouter.requests[0].model == "some/model:free"


def test_passes_the_pinned_temperature_to_the_adapter() -> None:
    openrouter = RecordingAdapter('{"label": "ok", "score": 1}')
    adapters = {"gemini": RecordingAdapter(""), "openrouter": openrouter}
    client = RoutingLlmClient(_registry(), adapters)

    client.complete("judge", [Message("user", "hi")])

    assert openrouter.requests[0].temperature == 0.2


def test_structured_call_requests_json_mode_and_parses_the_reply() -> None:
    gemini = RecordingAdapter('{"label": "reflejado", "score": 2}')
    client = RoutingLlmClient(_registry(), {"gemini": gemini, "openrouter": RecordingAdapter("")})

    result = client.complete_structured("synthesis", [Message("user", "grade")], Verdict)

    assert result == Verdict(label="reflejado", score=2)
    sent = gemini.requests[0]
    assert sent.json_mode is True
    assert any("JSON" in message.content for message in sent.messages)


def test_the_schema_instruction_rides_the_system_message_before_the_user_turn() -> None:
    openrouter = RecordingAdapter('{"label": "ok", "score": 1}')
    adapters = {"gemini": RecordingAdapter(""), "openrouter": openrouter}
    client = RoutingLlmClient(_registry(), adapters)
    conversation = [Message("system", "eres un juez"), Message("user", "califica esto")]

    client.complete_structured("judge", conversation, Verdict)

    sent = openrouter.requests[0].messages
    assert [message.role for message in sent] == ["system", "user"]
    assert sent[0].content.startswith("eres un juez")
    assert "JSON" in sent[0].content
    assert sent[1].content == "califica esto"


def test_a_task_without_a_system_message_still_gets_the_schema_instruction() -> None:
    gemini = RecordingAdapter('{"label": "ok", "score": 1}')
    client = RoutingLlmClient(_registry(), {"gemini": gemini, "openrouter": RecordingAdapter("")})

    client.complete_structured("synthesis", [Message("user", "grade")], Verdict)

    sent = gemini.requests[0].messages
    assert [message.role for message in sent] == ["system", "user"]
    assert "JSON" in sent[0].content


def test_a_malformed_structured_reply_is_resampled_before_giving_up() -> None:
    gemini = ReplayingAdapter(["{not json at all", '{"label": "ok", "score": 1}'])
    client = RoutingLlmClient(_registry(), {"gemini": gemini, "openrouter": RecordingAdapter("")})

    result = client.complete_structured("synthesis", [Message("user", "grade")], Verdict)

    assert result == Verdict(label="ok", score=1)
    assert len(gemini.requests) == 2


def test_a_reply_that_never_parses_raises_structured_output_error() -> None:
    gemini = ReplayingAdapter(["nope"] * (STRUCTURED_ATTEMPTS + 1))
    client = RoutingLlmClient(_registry(), {"gemini": gemini, "openrouter": RecordingAdapter("")})

    with pytest.raises(StructuredOutputError):
        client.complete_structured("synthesis", [Message("user", "grade")], Verdict)

    assert len(gemini.requests) == STRUCTURED_ATTEMPTS


def test_unknown_task_raises_task_not_configured() -> None:
    adapters = {"gemini": RecordingAdapter(""), "openrouter": RecordingAdapter("")}
    client = RoutingLlmClient(_registry(), adapters)

    with pytest.raises(TaskNotConfiguredError, match="ghost"):
        client.complete("ghost", [Message("user", "hi")])


def test_construction_fails_when_a_referenced_provider_has_no_adapter() -> None:
    with pytest.raises(LlmConfigError, match="openrouter"):
        RoutingLlmClient(_registry(), {"gemini": RecordingAdapter("")})
