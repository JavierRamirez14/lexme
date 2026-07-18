"""Shared fixtures for LLM interface tests."""

from pathlib import Path

import pytest
from pydantic import BaseModel


class Verdict(BaseModel):
    """A tiny structured-output schema used across the LLM tests."""

    label: str
    score: int


@pytest.fixture
def verdict_model() -> type[Verdict]:
    """The sample response model."""
    return Verdict


@pytest.fixture
def tasks_file(tmp_path: Path) -> Path:
    """Write a minimal two-provider task registry and return its path."""
    path = tmp_path / "tasks.json"
    path.write_text(
        """
        {
          "tasks": {
            "synthesis": {"provider": "gemini", "model": "gemini-2.5-flash", "temperature": 0.0},
            "judge": {"provider": "openrouter", "model": "some/model:free", "temperature": 0.2}
          }
        }
        """,
        encoding="utf-8",
    )
    return path
