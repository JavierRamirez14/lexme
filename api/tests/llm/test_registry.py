"""Tests for loading and validating the task registry."""

from pathlib import Path

import pytest

from lexme.llm.registry import LlmConfigError, TaskModel, TaskRegistry, load_task_registry


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "tasks.json"
    path.write_text(body, encoding="utf-8")
    return path


def test_resolves_task_to_pinned_model_and_temperature(tasks_file: Path) -> None:
    registry = load_task_registry(tasks_file)

    synthesis = registry.resolve("synthesis")
    assert synthesis.provider == "gemini"
    assert synthesis.model == "gemini-2.5-flash"
    assert synthesis.temperature == 0.0


def test_reports_the_distinct_providers_in_use(tasks_file: Path) -> None:
    registry = load_task_registry(tasks_file)

    assert registry.providers() == frozenset({"gemini", "openrouter"})


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(LlmConfigError, match="not found"):
        load_task_registry(tmp_path / "absent.json")


def test_unsupported_provider_raises_config_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        '{"tasks": {"t": {"provider": "acme", "model": "m", "temperature": 0.0}}}',
    )
    with pytest.raises(LlmConfigError, match="unsupported provider"):
        load_task_registry(path)


def test_empty_model_raises_config_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        '{"tasks": {"t": {"provider": "gemini", "model": "", "temperature": 0.0}}}',
    )
    with pytest.raises(LlmConfigError, match="'model'"):
        load_task_registry(path)


def test_temperature_out_of_range_raises_config_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        '{"tasks": {"t": {"provider": "gemini", "model": "m", "temperature": 5}}}',
    )
    with pytest.raises(LlmConfigError, match="outside"):
        load_task_registry(path)


def test_boolean_temperature_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        '{"tasks": {"t": {"provider": "gemini", "model": "m", "temperature": true}}}',
    )
    with pytest.raises(LlmConfigError, match="numeric 'temperature'"):
        load_task_registry(path)


def test_empty_tasks_object_raises_config_error(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"tasks": {}}')
    with pytest.raises(LlmConfigError, match="non-empty 'tasks'"):
        load_task_registry(path)


def test_dropping_a_task_leaves_the_rest_and_its_provider_unreferenced() -> None:
    registry = TaskRegistry(
        _by_task={
            "gen": TaskModel(task="gen", provider="gemini", model="g", temperature=0.0),
            "judge": TaskModel(task="judge", provider="openrouter", model="d", temperature=0.0),
        }
    )

    without_judge = registry.without("judge")

    assert without_judge.tasks() == ("gen",)
    assert without_judge.providers() == frozenset({"gemini"})
    assert registry.tasks() == ("gen", "judge")


def test_dropping_a_task_the_registry_never_had_changes_nothing() -> None:
    registry = TaskRegistry(
        _by_task={"gen": TaskModel(task="gen", provider="gemini", model="g", temperature=0.0)}
    )

    assert registry.without("judge").tasks() == ("gen",)
