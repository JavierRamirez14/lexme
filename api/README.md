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
`DATABASE_URL` (required), `TEI_URL`, `FRONTEND_ORIGIN`, and the LLM provider keys
`GEMINI_API_KEY` and `OPENROUTER_API_KEY`. The test suite sets a throwaway
`DATABASE_URL` itself and never calls a provider, so `pytest` needs no database
and no API keys.

## LLM interface

Every component that needs a language model calls `lexme.llm`, never a provider
SDK. A call names a *task*; `lexme/llm/tasks.json` is the single file that maps a
task to a provider, a **pinnable model id** and a temperature. The model id *is*
the pin point: a stable/GA id (`gemini-2.5-flash`) holds a fixed release, while a
rolling `:free` slug can be re-pointed — so the exact id in use is recorded in
each harness run's config footprint (blueprint caveat 2). Swapping a provider or
model is an edit to that file — no graph or node changes. The
interface owns structured output (it derives the JSON Schema from a Pydantic model
and validates the reply); the Gemini and OpenRouter adapters only speak HTTP and
flip a JSON-mode flag. In tests, `FakeLlmClient` is the single substitution point
for the whole suite.

### Free-tier limits (verified 2026-07-19)

The blueprint splits work by *volume*, not preference, so the free tiers hold. Rate
limits move often — re-check before a large harness run.

| Provider | Free-tier ceiling | Role | Fit |
|---|---|---|---|
| **Gemini** (AI Studio) | ~1,500 requests/day, 10–15 RPM on 2.5 Flash | Generator (Modo 1 + Modo 2): a harness run is hundreds of calls | Comfortable |
| **OpenRouter** (`:free` models) | 20 RPM; **50 requests/day** under $10 lifetime spend, **1,000/day** once $10+ has ever been added | Judge: a different model family, temperature 0, ~70 structured calls per harness run | Exceeds the 50/day floor |

**Caveat (from the blueprint):** one harness run makes ~70 judge calls, above
OpenRouter's default free floor of 50/day. Two fixes, neither touching code: add
the one-time $10 (raises the floor to 1,000/day, never expires), or move the
`judge` task to another provider in `tasks.json`. The generator stays entirely on
Gemini's generous free tier.

Sources: [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits),
[OpenRouter limits](https://openrouter.ai/docs/api_reference/limits).
