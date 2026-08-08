"""Tests for wiring settings and the registry into a live client."""

from pathlib import Path

import pytest

from lexme.config import Settings
from lexme.llm.factory import build_client, build_llm_client
from lexme.llm.protocol import LlmClient
from lexme.llm.registry import load_task_registry


def _settings(**overrides: str) -> Settings:
    base = {
        "database_url": "postgresql://localhost/x",
        "gemini_api_key": "g-key",
        "openrouter_api_key": "o-key",
    }
    base.update(overrides)
    return Settings(**base)


def test_builds_a_client_for_a_two_provider_registry(tasks_file: Path) -> None:
    client = build_llm_client(_settings(), tasks_path=tasks_file)

    assert isinstance(client, LlmClient)


def test_only_referenced_providers_need_a_key(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(
        '{"tasks": {"t": {"provider": "gemini", "model": "m", "temperature": 0.0}}}',
        encoding="utf-8",
    )

    client = build_llm_client(_settings(openrouter_api_key=""), tasks_path=path)

    assert isinstance(client, LlmClient)


def test_missing_key_for_a_referenced_provider_fails_loudly(tasks_file: Path) -> None:
    with pytest.raises(ValueError, match="api_key"):
        build_llm_client(_settings(openrouter_api_key=""), tasks_path=tasks_file)


def test_a_registry_without_the_paid_task_needs_no_key_for_its_provider(tasks_file: Path) -> None:
    registry = load_task_registry(tasks_file).without("judge")

    client = build_client(registry, _settings(openrouter_api_key=""))

    assert isinstance(client, LlmClient)
