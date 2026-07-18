"""``ingest`` command: build or refresh a vertical's corpus from the BOE."""

import argparse
import logging
import sys
from contextlib import ExitStack
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from lexme.config import get_settings
from lexme.ingestion.boe_client import BoeClient
from lexme.ingestion.embeddings import TeiEmbedder
from lexme.ingestion.manifest import load_manifest
from lexme.ingestion.pipeline import Embedder, IngestResult, NormFetcher, run_ingest

logger = logging.getLogger(__name__)


def main(
    argv: list[str] | None = None,
    *,
    fetcher: NormFetcher | None = None,
    embedder: Embedder | None = None,
    connection: psycopg.Connection | None = None,
) -> int:
    """Run the ingestion pipeline for the manifest given on the command line.

    ``fetcher``, ``embedder`` and ``connection`` default to real BOE/TEI clients
    and a real database connection built from settings; tests inject doubles or a
    test-owned connection through these keyword arguments.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    manifest = load_manifest(args.manifest)
    settings = get_settings()

    with ExitStack() as stack:
        active_fetcher = fetcher or stack.enter_context(BoeClient())
        active_embedder = embedder or stack.enter_context(TeiEmbedder(settings.tei_url))
        conn = connection or _open_connection(stack, settings.database_url)
        result = run_ingest(conn, manifest, active_fetcher, active_embedder)

    _report(result)
    return 0


def _open_connection(stack: ExitStack, database_url: str) -> psycopg.Connection:
    """Open a database connection with the pgvector adapter registered."""
    conn = stack.enter_context(psycopg.connect(database_url))
    register_vector(conn)
    return conn


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build or refresh a vertical's BOE corpus.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="path to the vertical's manifest.json",
    )
    return parser.parse_args(argv)


def _report(result: IngestResult) -> None:
    """Log a one-line summary of the run."""
    logger.info(
        "ingest complete: %d updated, %d unchanged",
        len(result.updated),
        len(result.unchanged),
    )


if __name__ == "__main__":
    sys.exit(main())
