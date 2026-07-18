# Lexme API

FastAPI service for Lexme, an agentic RAG system over Spanish legislation.

This package hosts both HTTP endpoints and the `ingest`/`eval` CLIs behind clean
internal boundaries: `ingestion`, `retrieval`, `agents`, `eval`.

## Local development

```bash
uv sync
export DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/lexme
uv run uvicorn lexme.main:app --reload
uv run pytest
uv run ruff check .
```

Configuration is read from environment variables (see `lexme/config.py`):
`DATABASE_URL` (required), `TEI_URL` and `FRONTEND_ORIGIN`. The test suite sets a
throwaway `DATABASE_URL` itself, so `pytest` needs no database running.
