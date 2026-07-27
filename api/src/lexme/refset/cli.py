"""``refset`` command: build the constructed reference set behind the review filter.

Four jobs, one per subcommand. ``generate-queries`` walks the corpus backwards to
produce pending Mode 1 candidates from a seed file; ``assemble`` composes pending
Mode 2 contracts from the clause bank and a recipe file; ``validate-bank`` fails if
the clause bank has stopped covering the checklist; and ``review`` lists, accepts or
rejects pending candidates. Only ``review accept`` writes into the versioned set, so
the human filter is the single door into it.
"""

import argparse
import json
import logging
import sys
from contextlib import ExitStack
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypeVar

import psycopg
from pgvector.psycopg import register_vector
from pydantic import BaseModel, ValidationError

from lexme.checklist import load_checklist
from lexme.checklist.models import checklist_path
from lexme.config import Settings, get_settings
from lexme.llm import LlmClient, build_llm_client, load_task_registry
from lexme.refset.assembler import assemble_contract
from lexme.refset.clause_bank import (
    BankClause,
    ClauseBank,
    clause_bank_path,
    load_clause_bank,
    validate_bank_coverage,
)
from lexme.refset.models import CaseKind
from lexme.refset.query_generator import (
    QUERY_GENERATION_TASK,
    QuerySeed,
    generate_query_case,
)
from lexme.refset.store import CandidateStore
from lexme.verification import CorpusReader, PsycopgCorpusReader

logger = logging.getLogger(__name__)

SEEDS_FILENAME = "seeds.json"
RECIPES_FILENAME = "recipes.json"

_InputT = TypeVar("_InputT", bound=BaseModel)


class RefsetInputError(ValueError):
    """Raised when a seeds or recipes input file is missing or malformed."""


class ContractRecipe(BaseModel):
    """A recipe for one synthetic contract: which bank clauses, in which order."""

    id: str
    clause_ids: list[str]
    header: str = ""


class _SeedsFile(BaseModel):
    """The parsed seeds input: the Mode 1 query seeds to generate from."""

    seeds: list[QuerySeed]


class _RecipesFile(BaseModel):
    """The parsed recipes input: the Mode 2 contracts to assemble."""

    contracts: list[ContractRecipe]


def main(
    argv: list[str] | None = None,
    *,
    corpus: CorpusReader | None = None,
    llm: LlmClient | None = None,
    now: datetime | None = None,
    today: date | None = None,
    model: str | None = None,
) -> int:
    """Run the ``refset`` CLI and return its exit code.

    ``corpus`` and ``llm`` default to the real system built from settings; tests
    inject both to run ``generate-queries`` without a database or the network.
    ``now`` stamps provenance, ``today`` fixes the corpus-resolution date, and
    ``model`` overrides the model recorded in provenance.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    settings = get_settings()
    if args.command == "generate-queries":
        return _do_generate_queries(args, settings, corpus, llm, now, today, model)
    if args.command == "assemble":
        return _do_assemble(args, settings, now)
    if args.command == "validate-bank":
        return _do_validate_bank(args, settings)
    return _do_review(args, settings, now)


def _do_generate_queries(
    args: argparse.Namespace,
    settings: Settings,
    corpus: CorpusReader | None,
    llm: LlmClient | None,
    now: datetime | None,
    today: date | None,
    model: str | None,
) -> int:
    """Generate pending Mode 1 candidates from the seed file."""
    created_at = now or datetime.now(UTC)
    resolve_date = today or date.today()
    seeds = _load_seeds(_seeds_path(args, settings, args.vertical))
    store = _store(settings, args.vertical)

    with ExitStack() as stack:
        if corpus is None or llm is None:
            corpus, llm, model = _build_generation_deps(stack, settings, model)
        written = 0
        for seed in seeds:
            candidate = generate_query_case(
                seed,
                corpus=corpus,
                llm=llm,
                now=created_at,
                resolve_date=resolve_date,
                model=model,
            )
            store.write(candidate)
            written += 1
    logger.info("generated %d pending Mode 1 candidate(s) for '%s'", written, args.vertical)
    return 0


def _do_assemble(args: argparse.Namespace, settings: Settings, now: datetime | None) -> int:
    """Assemble pending Mode 2 contracts from the clause bank and recipe file."""
    created_at = now or datetime.now(UTC)
    checklist = load_checklist(checklist_path(settings.verticals_dir, args.vertical))
    bank = load_clause_bank(clause_bank_path(settings.verticals_dir, args.vertical))
    recipes = _load_recipes(_recipes_path(args, settings, args.vertical))
    store = _store(settings, args.vertical)

    written = 0
    for recipe in recipes:
        clauses = _resolve_recipe_clauses(recipe, bank)
        candidate = assemble_contract(
            recipe.id, clauses, checklist, now=created_at, header=recipe.header
        )
        store.write(candidate)
        written += 1
    logger.info("assembled %d pending Mode 2 contract(s) for '%s'", written, args.vertical)
    return 0


def _do_validate_bank(args: argparse.Namespace, settings: Settings) -> int:
    """Validate the clause bank against the checklist; non-zero on any gap."""
    checklist = load_checklist(checklist_path(settings.verticals_dir, args.vertical))
    bank = load_clause_bank(clause_bank_path(settings.verticals_dir, args.vertical))
    report = validate_bank_coverage(bank, checklist)
    if report.is_valid:
        logger.info("clause bank '%s' covers the checklist", args.vertical)
        return 0
    for finding in report.findings:
        logger.error("clause bank gap: %s", finding.message)
    logger.error("clause bank '%s' failed coverage: %d gap(s)", args.vertical, len(report.findings))
    return 1


def _do_review(args: argparse.Namespace, settings: Settings, now: datetime | None) -> int:
    """Run a review action: list pending candidates, or accept/reject one."""
    reviewed_at = now or datetime.now(UTC)
    store = _store(settings, args.vertical)
    if args.action == "list":
        return _report_pending(store, args.kind)
    if args.action == "accept":
        path = store.accept(args.id, reviewed_by=args.by, now=reviewed_at, note=args.note)
        logger.info("accepted '%s' -> %s", args.id, path)
        return 0
    store.reject(args.id, reviewed_by=args.by, reason=args.reason, now=reviewed_at)
    logger.info("rejected '%s': %s", args.id, args.reason)
    return 0


def _report_pending(store: CandidateStore, kind: CaseKind | None) -> int:
    """List the pending candidates, one per line, and how many there are."""
    pending = store.list_pending(kind)
    for candidate in pending:
        logger.info("pending %s: %s", candidate.kind.value, candidate.id)
    logger.info("%d pending candidate(s)", len(pending))
    return 0


def _build_generation_deps(
    stack: ExitStack, settings: Settings, model: str | None
) -> tuple[CorpusReader, LlmClient, str | None]:
    """Wire the real corpus and model for a live generation run."""
    connection = stack.enter_context(psycopg.connect(settings.database_url))
    register_vector(connection)
    corpus = PsycopgCorpusReader(connection)
    llm = build_llm_client(settings)
    resolved_model = model or load_task_registry().resolve(QUERY_GENERATION_TASK).model
    return corpus, llm, resolved_model


def _resolve_recipe_clauses(recipe: ContractRecipe, bank: ClauseBank) -> list[BankClause]:
    """Resolve a recipe's clause ids against the bank, failing on an unknown id."""
    clauses = []
    for clause_id in recipe.clause_ids:
        try:
            clauses.append(bank.by_id(clause_id))
        except KeyError as error:
            raise RefsetInputError(
                f"recipe '{recipe.id}' references unknown clause '{clause_id}'"
            ) from error
    return clauses


def _store(settings: Settings, vertical: str) -> CandidateStore:
    """Build the candidate store for ``vertical`` from settings."""
    vertical_dir = Path(settings.verticals_dir) / vertical
    return CandidateStore(refset_dir=vertical_dir / "refset", eval_dir=vertical_dir / "eval")


def _seeds_path(args: argparse.Namespace, settings: Settings, vertical: str) -> Path:
    """The seeds file: ``--seeds`` or the vertical's default location."""
    if args.seeds is not None:
        return args.seeds
    return Path(settings.verticals_dir) / vertical / "refset" / SEEDS_FILENAME


def _recipes_path(args: argparse.Namespace, settings: Settings, vertical: str) -> Path:
    """The recipes file: ``--recipes`` or the vertical's default location."""
    if args.recipes is not None:
        return args.recipes
    return Path(settings.verticals_dir) / vertical / "refset" / RECIPES_FILENAME


def _load_seeds(path: Path) -> list[QuerySeed]:
    """Load and validate the Mode 1 seeds file."""
    return _load_input(path, _SeedsFile).seeds


def _load_recipes(path: Path) -> list[ContractRecipe]:
    """Load and validate the Mode 2 recipes file."""
    return _load_input(path, _RecipesFile).contracts


def _load_input(path: Path, model: type[_InputT]) -> _InputT:
    """Read and validate an input file into ``model``, or raise a clear error."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RefsetInputError(f"input file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise RefsetInputError(f"input file {path} is not valid JSON: {error}") from error
    try:
        return model.model_validate(raw)
    except ValidationError as error:
        raise RefsetInputError(f"input file {path} is invalid: {error}") from error


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the ``refset`` subcommands and their options."""
    parser = argparse.ArgumentParser(description="Build the Lexme reference set.")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-queries", help="generate pending Mode 1 candidates")
    generate.add_argument("--vertical", required=True, help="the vertical to generate for")
    generate.add_argument("--seeds", type=Path, help="path to the seeds file")

    assemble = sub.add_parser("assemble", help="assemble pending Mode 2 contracts")
    assemble.add_argument("--vertical", required=True, help="the vertical to assemble for")
    assemble.add_argument("--recipes", type=Path, help="path to the recipes file")

    validate = sub.add_parser("validate-bank", help="check the clause bank covers the checklist")
    validate.add_argument("--vertical", required=True, help="the vertical whose bank to validate")

    review = sub.add_parser("review", help="list, accept or reject pending candidates")
    _add_review_actions(review)

    return parser.parse_args(argv)


def _add_review_actions(review: argparse.ArgumentParser) -> None:
    """Add the ``review`` sub-actions: list, accept, reject."""
    actions = review.add_subparsers(dest="action", required=True)

    listing = actions.add_parser("list", help="list pending candidates")
    listing.add_argument("--vertical", required=True)
    listing.add_argument("--kind", type=CaseKind, choices=list(CaseKind), help="filter by kind")

    accept = actions.add_parser("accept", help="accept a pending candidate into the set")
    accept.add_argument("--vertical", required=True)
    accept.add_argument("--id", required=True, help="the candidate id to accept")
    accept.add_argument("--by", required=True, help="the reviewer's name")
    accept.add_argument("--note", default="", help="an optional acceptance note")

    reject = actions.add_parser("reject", help="reject a pending candidate")
    reject.add_argument("--vertical", required=True)
    reject.add_argument("--id", required=True, help="the candidate id to reject")
    reject.add_argument("--by", required=True, help="the reviewer's name")
    reject.add_argument("--reason", required=True, help="why the candidate was rejected")


if __name__ == "__main__":
    sys.exit(main())
