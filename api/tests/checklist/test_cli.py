"""The ``validate-checklist`` command loads by vertical config and gates on the corpus."""

import pytest

from lexme.checklist import cli
from lexme.config import Settings
from tests.checklist.conftest import REPO_ROOT, TARGET_DATE, build_corpus
from tests.checklist.fixtures.lau_excerpts import LAU_EXCERPTS, LAU_NORM_ID


@pytest.fixture(autouse=True)
def _settings_pointing_at_the_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the CLI resolve checklists from the repo's ``verticales`` directory."""
    settings = Settings(
        database_url="postgresql://unused", verticals_dir=str(REPO_ROOT / "verticales")
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)


def test_returns_zero_when_the_checklist_validates() -> None:
    corpus = build_corpus(LAU_NORM_ID, LAU_EXCERPTS)

    exit_code = cli.main(["--vertical", "vivienda"], corpus=corpus, today=TARGET_DATE)

    assert exit_code == 0


def test_returns_one_when_an_anchor_does_not_resolve() -> None:
    incomplete = {"a9": LAU_EXCERPTS["a9"]}
    corpus = build_corpus(LAU_NORM_ID, incomplete)

    exit_code = cli.main(["--vertical", "vivienda"], corpus=corpus, today=TARGET_DATE)

    assert exit_code == 1
