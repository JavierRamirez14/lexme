"""Load the task-to-model registry that routes every LLM call.

``tasks.json`` is the single file that names a provider, a pinned model version
and a temperature per task. Nothing else in the codebase encodes that mapping, so
moving a task to another provider or model is an edit to this one file.
"""

import json
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_PROVIDERS = ("gemini", "openrouter")

DEFAULT_TASKS_PATH = Path(__file__).parent / "tasks.json"

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0


class LlmConfigError(ValueError):
    """Raised when the task registry file is missing or malformed."""


@dataclass(frozen=True)
class TaskModel:
    """The provider, pinned model and temperature bound to one task."""

    task: str
    provider: str
    model: str
    temperature: float


@dataclass(frozen=True)
class TaskRegistry:
    """An immutable task-name to :class:`TaskModel` mapping."""

    _by_task: dict[str, TaskModel]

    def resolve(self, task: str) -> TaskModel:
        """Return the :class:`TaskModel` for ``task``.

        Raises :class:`KeyError` if the task is unknown; callers that face user
        input translate this into a :class:`~lexme.llm.protocol.TaskNotConfiguredError`.
        """
        return self._by_task[task]

    def has(self, task: str) -> bool:
        """Return whether ``task`` is configured."""
        return task in self._by_task

    def providers(self) -> frozenset[str]:
        """Return the set of providers referenced by any task."""
        return frozenset(model.provider for model in self._by_task.values())

    def tasks(self) -> tuple[str, ...]:
        """Return the configured task names in file order."""
        return tuple(self._by_task)

    def without(self, task: str) -> "TaskRegistry":
        """Return a copy of this registry with ``task`` dropped, absent or not.

        Dropping the only task of a provider drops the provider too, so a run that
        does not use it needs no key for it.
        """
        return TaskRegistry(
            _by_task={name: model for name, model in self._by_task.items() if name != task}
        )


def load_task_registry(path: Path = DEFAULT_TASKS_PATH) -> TaskRegistry:
    """Read and validate ``tasks.json`` into a :class:`TaskRegistry`.

    The file must hold a non-empty ``tasks`` object whose values each carry a
    supported ``provider``, a non-empty ``model`` and a ``temperature`` in
    ``[0, 2]``. Raises :class:`LlmConfigError` on any structural problem.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise LlmConfigError(f"task registry not found: {path}") from error
    except json.JSONDecodeError as error:
        raise LlmConfigError(f"task registry {path} is not valid JSON: {error}") from error

    tasks = raw.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise LlmConfigError(f"task registry {path} missing a non-empty 'tasks' object")

    by_task = {name: _read_task(name, entry, path) for name, entry in tasks.items()}
    return TaskRegistry(_by_task=by_task)


def _read_task(name: str, entry: object, path: Path) -> TaskModel:
    """Validate one task entry into a :class:`TaskModel`."""
    if not isinstance(entry, dict):
        raise LlmConfigError(f"task registry {path} entry '{name}' is not an object")

    provider = entry.get("provider")
    if provider not in SUPPORTED_PROVIDERS:
        raise LlmConfigError(
            f"task registry {path} entry '{name}' has unsupported provider {provider!r}; "
            f"expected one of {SUPPORTED_PROVIDERS}"
        )

    model = entry.get("model")
    if not isinstance(model, str) or not model:
        raise LlmConfigError(f"task registry {path} entry '{name}' missing a non-empty 'model'")

    temperature = entry.get("temperature")
    if not isinstance(temperature, int | float) or isinstance(temperature, bool):
        raise LlmConfigError(f"task registry {path} entry '{name}' missing a numeric 'temperature'")
    if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
        raise LlmConfigError(
            f"task registry {path} entry '{name}' temperature {temperature} "
            f"outside [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}]"
        )

    return TaskModel(task=name, provider=provider, model=model, temperature=float(temperature))
