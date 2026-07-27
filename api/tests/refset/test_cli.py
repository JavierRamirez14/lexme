"""Integration tests for the ``refset`` CLI, wired to a temporary vertical."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from lexme.config import Settings
from lexme.eval.cases import load_cases
from lexme.llm import FakeLlmClient
from lexme.refset import cli
from lexme.refset.query_generator import QUERY_GENERATION_TASK, GeneratedQuery
from tests.refset.conftest import FakeCorpus

_NORM = "BOE-A-1994-26003"
_REF_A9 = f"{_NORM}:a9"
_NOW = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
_DATE = date(2024, 1, 1)


def _checklist(*item_ids: str) -> dict:
    def item(item_id: str, silence: str) -> dict:
        return {
            "id": item_id,
            "right": f"right {item_id}",
            "anchors": ["a9"],
            "character": "imperativo",
            "silence_tone": silence,
            "absence_template": "...",
            "citation": {"block_id": "a9", "text": "t"},
        }

    items = [item(item_id, "favorable") for item_id in item_ids]
    return {"vertical": "vivienda", "norm_id": _NORM, "items": items}


def _full_bank() -> dict:
    def clause(clause_id: str, level: str, chk_ids: list[str]) -> dict:
        return {
            "id": clause_id,
            "heading": "Cláusula",
            "text": f"cuerpo de {clause_id}",
            "expected_level": level,
            "chk_ids": chk_ids,
            "source": "consumo",
        }

    return {
        "vertical": "vivienda",
        "norm_id": _NORM,
        "clauses": [
            clause("b-illegal", "ilegal", ["CHK-01"]),
            clause("b-worse", "peor_que_default", ["CHK-02"]),
            clause("b-ok", "correcto", ["CHK-01"]),
            clause("b-neg", "negociable", []),
        ],
    }


@pytest.fixture
def vertical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a temporary 'vivienda' vertical and point the CLI's settings at it."""
    root = tmp_path / "verticales"
    vertical_dir = root / "vivienda"
    refset_dir = vertical_dir / "refset"
    refset_dir.mkdir(parents=True)
    (vertical_dir / "checklist.json").write_text(
        json.dumps(_checklist("CHK-01", "CHK-02")), encoding="utf-8"
    )
    (refset_dir / "clause_bank.json").write_text(json.dumps(_full_bank()), encoding="utf-8")

    settings = Settings(database_url="postgresql://unused", verticals_dir=str(root))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return vertical_dir


def test_validate_bank_passes_on_a_covering_bank(vertical: Path) -> None:
    assert cli.main(["validate-bank", "--vertical", "vivienda"]) == 0


def test_validate_bank_fails_when_a_level_is_missing(vertical: Path) -> None:
    bank = _full_bank()
    bank["clauses"] = [c for c in bank["clauses"] if c["expected_level"] != "ilegal"]
    (vertical / "refset" / "clause_bank.json").write_text(json.dumps(bank), encoding="utf-8")

    assert cli.main(["validate-bank", "--vertical", "vivienda"]) == 1


def test_assemble_then_accept_materializes_a_contract(vertical: Path) -> None:
    recipes = {"contracts": [{"id": "contrato-01", "clause_ids": ["b-illegal", "b-ok"]}]}
    (vertical / "refset" / "recipes.json").write_text(json.dumps(recipes), encoding="utf-8")

    assert cli.main(["assemble", "--vertical", "vivienda"], now=_NOW) == 0
    assert cli.main(["review", "list", "--vertical", "vivienda"]) == 0

    materialized = vertical / "refset" / "modo2" / "contrato-01.json"
    assert not materialized.exists()

    exit_code = cli.main(
        ["review", "accept", "--vertical", "vivienda", "--id", "contrato-01", "--by", "javier"],
        now=_NOW,
    )

    assert exit_code == 0
    data = json.loads(materialized.read_text(encoding="utf-8"))
    assert data["provenance"]["reviewed_by"] == "javier"
    assert len(data["clauses"]) == 2


def test_generate_queries_then_accept_feeds_eval(vertical: Path) -> None:
    seeds = {
        "seeds": [
            {"id": "plazo", "block_refs": [_REF_A9], "expected_outcome": "respuesta"},
        ]
    }
    (vertical / "refset" / "seeds.json").write_text(json.dumps(seeds), encoding="utf-8")
    corpus = FakeCorpus({"a9": "duración mínima de cinco años"})
    llm = FakeLlmClient(
        {
            QUERY_GENERATION_TASK: [
                GeneratedQuery(
                    question="¿cuánto puedo quedarme?",
                    key_points=[{"claim": "cinco años", "block_ref": _REF_A9}],
                )
            ]
        }
    )

    generated = cli.main(
        ["generate-queries", "--vertical", "vivienda"],
        corpus=corpus,
        llm=llm,
        now=_NOW,
        today=_DATE,
        model="pin-1",
    )

    assert generated == 0
    accepted = cli.main(
        ["review", "accept", "--vertical", "vivienda", "--id", "plazo", "--by", "javier"],
        now=_NOW,
    )
    assert accepted == 0

    eval_case = vertical / "eval" / "modo1" / "plazo.json"
    cases = load_cases(eval_case)
    assert cases[0].question == "¿cuánto puedo quedarme?"
    assert cases[0].gold_block_refs == (_REF_A9,)


def test_review_reject_keeps_the_case_out_of_the_set(vertical: Path) -> None:
    recipes = {"contracts": [{"id": "contrato-01", "clause_ids": ["b-ok"]}]}
    (vertical / "refset" / "recipes.json").write_text(json.dumps(recipes), encoding="utf-8")
    cli.main(["assemble", "--vertical", "vivienda"], now=_NOW)

    exit_code = cli.main(
        [
            "review",
            "reject",
            "--vertical",
            "vivienda",
            "--id",
            "contrato-01",
            "--by",
            "javier",
            "--reason",
            "mala",
        ],
        now=_NOW,
    )

    assert exit_code == 0
    assert not (vertical / "refset" / "modo2" / "contrato-01.json").exists()
