"""Tests for the deterministic fake LLM client."""

import pytest

from lexme.llm.fake import FakeLlmClient
from lexme.llm.protocol import LlmError, Message, StructuredOutputError
from tests.llm.conftest import Verdict


def test_drains_replies_per_task_in_order() -> None:
    fake = FakeLlmClient({"router": ["first", "second"]})

    assert fake.complete("router", [Message("user", "a")]) == "first"
    assert fake.complete("router", [Message("user", "b")]) == "second"


def test_records_each_call_with_its_task_and_messages() -> None:
    fake = FakeLlmClient({"router": ["ok"]})

    fake.complete("router", [Message("user", "hello")])

    assert fake.calls[0].task == "router"
    assert fake.calls[0].messages == (Message("user", "hello"),)


def test_structured_reply_from_a_model_instance() -> None:
    fake = FakeLlmClient()
    fake.queue("judge", Verdict(label="ok", score=1))

    result = fake.complete_structured("judge", [Message("user", "x")], Verdict)

    assert result == Verdict(label="ok", score=1)


def test_structured_reply_from_a_dict() -> None:
    fake = FakeLlmClient({"judge": [{"label": "ok", "score": 2}]})

    result = fake.complete_structured("judge", [Message("user", "x")], Verdict)

    assert result == Verdict(label="ok", score=2)


def test_structured_reply_from_a_raw_json_string() -> None:
    fake = FakeLlmClient({"judge": ['{"label": "ok", "score": 3}']})

    result = fake.complete_structured("judge", [Message("user", "x")], Verdict)

    assert result == Verdict(label="ok", score=3)


def test_malformed_structured_string_surfaces_structured_output_error() -> None:
    fake = FakeLlmClient({"judge": ["definitely not json"]})

    with pytest.raises(StructuredOutputError):
        fake.complete_structured("judge", [Message("user", "x")], Verdict)


def test_exhausted_queue_raises_a_clear_error() -> None:
    fake = FakeLlmClient({"router": ["only one"]})
    fake.complete("router", [Message("user", "a")])

    with pytest.raises(LlmError, match="no queued reply"):
        fake.complete("router", [Message("user", "b")])


def test_wrong_reply_type_for_plain_completion_is_rejected() -> None:
    fake = FakeLlmClient({"router": [{"not": "a string"}]})

    with pytest.raises(LlmError, match="not str"):
        fake.complete("router", [Message("user", "a")])
