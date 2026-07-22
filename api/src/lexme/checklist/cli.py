"""``validate-checklist`` command: fail if a vertical's checklist drifts from its corpus.

Loads the checklist from the vertical's data directory (by configuration, with no
"vivienda" in the code), validates it against the ingested corpus at today's date,
and exits non-zero on any finding so CI can gate on it.
"""

import argparse
import logging
import sys
from contextlib import ExitStack
from datetime import date

import psycopg
from pgvector.psycopg import register_vector

from lexme.checklist.models import checklist_path, load_checklist
from lexme.checklist.validator import ChecklistReport, validate_checklist
from lexme.config import get_settings
from lexme.verification import CorpusReader, PsycopgCorpusReader

logger = logging.getLogger(__name__)


def main(
    argv: list[str] | None = None,
    *,
    corpus: CorpusReader | None = None,
    today: date | None = None,
) -> int:
    """Validate a vertical's checklist and return 0 when valid, 1 otherwise.

    ``corpus`` and ``today`` default to a corpus reader over a real database
    connection built from settings and the system date; tests inject an in-memory
    corpus and a fixed date through these keyword arguments.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    settings = get_settings()

    checklist = load_checklist(checklist_path(settings.verticals_dir, args.vertical))
    target_date = today or date.today()

    with ExitStack() as stack:
        active_corpus = corpus or PsycopgCorpusReader(
            _open_connection(stack, settings.database_url)
        )
        report = validate_checklist(checklist, active_corpus, target_date)

    _report(args.vertical, report)
    return 0 if report.is_valid else 1


def _open_connection(stack: ExitStack, database_url: str) -> psycopg.Connection:
    """Open a database connection with the pgvector adapter registered."""
    conn = stack.enter_context(psycopg.connect(database_url))
    register_vector(conn)
    return conn


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate a vertical's checklist against its ingested corpus."
    )
    parser.add_argument(
        "--vertical",
        required=True,
        help="the vertical whose checklist.json to validate (e.g. vivienda)",
    )
    return parser.parse_args(argv)


def _report(vertical: str, report: ChecklistReport) -> None:
    """Log the validation outcome: a clean pass, or every finding."""
    if report.is_valid:
        logger.info("checklist '%s' is valid against the corpus", vertical)
        return
    for finding in report.findings:
        logger.error("%s [%s]: %s", finding.item_id, finding.block_id, finding.message)
    logger.error("checklist '%s' failed validation: %d finding(s)", vertical, len(report.findings))


if __name__ == "__main__":
    sys.exit(main())
