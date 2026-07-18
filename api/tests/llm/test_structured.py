"""Tests for the structured-output helpers."""

import json

import pytest

from lexme.llm.protocol import StructuredOutputError
from lexme.llm.structured import json_schema_instruction, parse_model
from tests.llm.conftest import Verdict


def test_instruction_embeds_the_model_schema() -> None:
    instruction = json_schema_instruction(Verdict)

    assert "JSON" in instruction
    assert json.loads(instruction[instruction.index("{") :])["properties"].keys() >= {
        "label",
        "score",
    }


def test_parses_a_plain_json_object() -> None:
    result = parse_model('{"label": "ok", "score": 3}', Verdict)

    assert result == Verdict(label="ok", score=3)


def test_tolerates_a_json_code_fence() -> None:
    fenced = '```json\n{"label": "ok", "score": 3}\n```'

    assert parse_model(fenced, Verdict) == Verdict(label="ok", score=3)


def test_rejects_non_json_text() -> None:
    with pytest.raises(StructuredOutputError):
        parse_model("not json at all", Verdict)


def test_rejects_json_that_violates_the_schema() -> None:
    with pytest.raises(StructuredOutputError, match="Verdict"):
        parse_model('{"label": "ok", "score": "high"}', Verdict)
