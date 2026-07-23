"""Tests pinning the shipped tasks.json to the blueprint's provider split."""

from lexme.llm.registry import DEFAULT_TASKS_PATH, load_task_registry


def test_shipped_registry_loads() -> None:
    registry = load_task_registry()

    assert registry.has("mode1_synthesis")
    assert registry.has("judge")


def test_judge_is_a_different_provider_family_than_the_generator() -> None:
    registry = load_task_registry()

    generator = registry.resolve("mode1_synthesis")
    judge = registry.resolve("judge")
    assert generator.provider == "openrouter"
    assert judge.provider == "gemini"
    assert judge.provider != generator.provider


def test_default_path_points_at_the_packaged_file() -> None:
    assert DEFAULT_TASKS_PATH.name == "tasks.json"
    assert DEFAULT_TASKS_PATH.is_file()
