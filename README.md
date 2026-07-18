# Lexme

Agentic RAG over Spanish legislation: answers legal questions citing real BOE
articles, and analyses rental contracts flagging risky clauses.

This is the reproducible skeleton — no product features yet. Its promise: clone
the repo and `docker compose up` brings the whole system up from nothing.

## Stack

Four containers on one Compose network:

| Service    | What it is                                              | Port          |
| ---------- | ------------------------------------------------------- | ------------- |
| `frontend` | React + Vite SPA stub                                   | `5173`        |
| `api`      | FastAPI monolith (also hosts the `ingest`/`eval` CLIs)  | `8000`        |
| `tei`      | Text Embeddings Inference serving BGE-M3                | `8080` → `80` |
| `db`       | PostgreSQL + pgvector                                   | `5432`        |

## Quickstart

```bash
cp .env.example .env   # then set your database credentials
docker compose up
```

The first boot downloads the BGE-M3 weights (~2 GB) into a persistent volume, so
restarts do not re-download. Once up:

- Frontend: <http://localhost:5173>
- API health: <http://localhost:8000/health>
- TEI: <http://localhost:8080/health>

`GET /health` verifies real connectivity to both `db` and `tei`, returning `200`
with `{"status": "ok", ...}` only when both are reachable, otherwise `503` with
`status: degraded` and the failing dependency marked `error`.

Common tasks are wrapped in the `Makefile` (`make help`).

## Development

The API uses `uv` (dependencies), `ruff` (lint/format) and `pytest` (tests):

```bash
cd api
uv sync
uv run pytest
uv run ruff check .
```

The test suite includes an integration test that hits the real API over HTTP
(a live uvicorn server on a loopback socket), with the database and TEI probes
overridden — so it is deterministic and needs neither service running.

Internal package boundaries live under `api/src/lexme/`: `ingestion`,
`retrieval`, `agents`, `eval`.

## Configuration

The API reads its settings from environment variables (see
`api/src/lexme/config.py`): `DATABASE_URL` (required), `TEI_URL` and
`FRONTEND_ORIGIN`. The database credentials live only in `.env` (gitignored);
copy `.env.example` to `.env` and set them before starting the stack. Compose
composes `DATABASE_URL` from them and injects it into the `api` container.
