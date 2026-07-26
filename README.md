# Lexme

**Agentic RAG over Spanish tenancy law that refuses to reassure you when the law says otherwise.**

Lexme answers rental-law questions and analyses rental contracts against the
official consolidated text of the *Ley de Arrendamientos Urbanos* (LAU), published
by Spain's *Boletín Oficial del Estado* (BOE). Every legal claim it shows is a
literal, re-verified quote of a specific article — or it abstains. It is a
portfolio piece about **measurable** AI engineering: agentic retrieval, tool
pipelines, and an evaluation harness that puts numbers on the one error that
matters most.

> The grave error a contract checker can make is telling a tenant that a genuinely
> illegal clause is fine. Lexme's headline metric is built on exactly that error —
> and on this run it was **zero**.

<!--
DEMO GIF — record and drop here (Mode 2): quick contract intake → spectrum risk
map → the ⚪ climax (a protection the law grants that the contract stays silent on).
Capture it against the running stack with:  make contract F=path/to/contrato.pdf
Then reference it as:  ![Mode 2 demo](docs/demo-modo2.gif)
-->
> **▶ Demo (Mode 2) — recording pending.** The flow to capture: quick contract
> intake → a clause-by-clause **spectrum risk map** → the ⚪ climax, where the map
> surfaces a right the LAU grants by default that the contract simply omits.

---

## 1. The problem

Legal language is a real barrier: most people don't understand the contracts they
sign or the rules that protect them, and a lawyer is slow or expensive. A tenant
signs clauses that quietly strip rights the law grants by default — and never knows.
Generic AI assistants make it worse: they answer with no verifiable basis,
hallucinate article numbers, and, faced with a contract, drift toward the false
comfort of "looks fine."

## 2. What Lexme does

Two complementary modes over the **housing / state-law** vertical (LAU):

- **Mode 1 — Agentic Q&A.** You ask in natural language ("can my landlord raise the
  rent 10%?"). An agentic graph routes the query, decomposes it into sub-queries,
  retrieves iteratively with a self-critique loop, and answers in three layers:
  the grounded citation (literal quote + BOE ELI link + point-in-time date), a
  plain-language explanation, and what you can do next. If there's no basis, it
  abstains — and **code decides that, not the model.**
- **Mode 2 — Contract risk map.** You upload a rental contract. Lexme segments it by
  clause and contrasts each against the LAU's default, returning a **spectrum risk
  map** — 🔴 void · 🟠 worse than the default · 🟡 lawful but negotiable · ⚪ a
  protection the contract omits · 🟢 conforming. It never gives a binary
  "contract OK" verdict. The ⚪ — what the law gives you and the contract stays
  silent about — is the differentiator.

## 3. Measured quality  (zoom 1 → 2 → 3)

Every number below is produced by the evaluation harness against the real system
and written to a versioned run artifact under [`eval-runs/`](eval-runs/), stamped
with a configuration fingerprint. Nothing here is hand-entered.

### The headline

| Result | Value | Condition |
| --- | --- | --- |
| **Literal-citation invariant** | **held — 0 discarded** | Every displayed citation re-verifies char-for-char against the point-in-time corpus. Mode 1: 5/5 shown citations verified (3 direct, 2 snap-repaired, 0 discarded). Mode 2: guardrail passed. A single non-verifiable citation fails the whole run. |
| **False-tranquility rate** | **0.0  (0 / 2)** | Share of genuinely 🔴/🟠 clauses the system let pass as reassuring (🟢 or silence), over the red/orange clauses it correctly delimited. |
| **Recall of 🔴/🟠 clauses** | **1.0  (2 / 2)** | Of the correctly delimited problematic clauses, how many it also flagged as problematic. |

### The full table (zoom 2 — precision always shown next to abstention)

| Metric | Value | Denominator / condition |
| --- | --- | --- |
| Citation literality (both modes) | 0 discarded | invariant; run fails on any discard |
| Retrieval recall, first pass (Mode 1, hand-authored cases) | 1.00 | gold block found before the agentic loop |
| Agentic recall delta | **+0.00** | 1.00 → 1.00; see the honesty note below |
| Recall 🔴/🟠 (Mode 2) | 1.00 (2/2) | over correctly delimited problematic clauses |
| False-tranquility rate (Mode 2) | 0.00 (0/2) | real 🔴/🟠 passed off as reassuring |
| Flag precision **·** abstention (Mode 2) | 0.50 (2/4) **·** 0.00 (0/10) | precision never read without its abstention rate |
| Segmentation (Mode 2) | 0.625 (10/16) | clauses delimited at IoU ≥ 0.80 |
| Absence ⚪ recall **·** precision (Mode 2) | 0.68 (30/44) **·** 1.00 | omitted rights the cross-check surfaced |
| Human–judge agreement (Mode 1) | *not yet calibrated* | requires the one-time human calibration pass |

### The versioned reports (zoom 3)

- Mode 1 — [`eval-runs/modo1-20260726T170157Z.json`](eval-runs/modo1-20260726T170157Z.json) · fingerprint `c089dbb30f299632`
- Mode 2 — [`eval-runs/modo2-20260726T184248Z.json`](eval-runs/modo2-20260726T184248Z.json) · fingerprint `73c795dbb7848f25`

Regenerate with `make eval` / `make eval-modo2`; diff a new run against a baseline
with `make eval-compare BASE=… RUN=…`. A changed fingerprint means the numbers are
an experiment, not a regression.

### Honesty notes (this is v1, and the harness reports its own limits)

- **Reference set is small and deliberately reviewable:** 17 Mode 1 cases and 3
  synthetic Mode 2 contracts, generated by construction with human review as the
  filter. The denominators above are real and intentionally shown — including that
  one of the three Mode 2 contracts was (correctly) rejected as out-of-scope, so the
  clause-level numbers come from the two analysed contracts.
- **Agentic delta is ~0 *on this corpus*, and that's an honest result:** the vertical
  corpus is one norm (64 blocks), so first-pass hybrid retrieval already recovers
  the gold article and the self-critique loop has nothing to add. The mechanism is
  built and measured; showing a positive delta needs a larger corpus where
  first-pass retrieval misses.
- **The generator runs on a free-tier model** (`gemini-3.5-flash-lite`, pinned in
  the run fingerprint). It over-triggers Mode 1's disambiguation gate on the
  generated cases, so the answer path is exercised mainly by the hand-authored
  cases (which reach the expected outcome with recall 1.00). The judge did not run
  this pass because those cases carry no reference key points yet.

## 4. How it works

Four containers on one Compose network:

| Service    | What it is                                              | Port          |
| ---------- | ------------------------------------------------------- | ------------- |
| `frontend` | React + Vite SPA                                        | `5173`        |
| `api`      | FastAPI monolith (also hosts the `ingest`/`eval` CLIs)  | `8000`        |
| `tei`      | Text Embeddings Inference serving BGE-M3                | `8080` → `80` |
| `db`       | PostgreSQL + pgvector                                   | `5432`        |

- **Ingestion** builds the corpus from the BOE Consolidated Legislation API (XML),
  chunked one *precept-block* (= article) per unit, keeping every version with its
  validity dates so Mode 1 can resolve the law *as it stood on a given date* with a
  SQL substitution.
- **Retrieval** is hybrid: pgvector (BGE-M3) + Postgres full-text `spanish`, fused
  with a hand-rolled Reciprocal Rank Fusion.
- **Orchestration** is a LangGraph agentic graph with a Pydantic state: router →
  decomposition → iterative retrieval with a self-critique loop → three-layer
  synthesis that may only cite blocks it actually retrieved.
- **The citation verifier** is shared by both modes: strict equality after
  conservative typographic normalisation (no fuzzy matching), deterministic snap
  repair only, and a hard discard otherwise. Its telemetry is the sole source of
  the literality numbers.

## 5. Design decisions

- **Own LLM interface, task → model in config.** The graphs never call a provider
  directly; a routing client maps each task to a model. Swapping providers or
  free-tier models is a config edit, and tests inject a deterministic fake. This
  layer also absorbs provider quirks — it disables the native JSON mode that
  truncates replies on these flash models, and backs off on `429` rate limits.
- **Code decides the things that must be trustworthy.** Abstention, the temporal
  gate, scope (LAU art. 4.2), the checklist × contract cross-check, and citation
  discard are all code, not model judgement — so they're testable and can't be
  talked out of a refusal.
- **A spectrum risk map, never a binary verdict.** Five discrete levels with full
  coverage, and the ⚪ absence findings anchored to checklist items — so a "clean"
  contract can't quietly reassure you about rights it omits.
- **Evaluation is layered and adversarial to its own precision.** Retrieval recall,
  agentic delta, the citation guardrail as a hard invariant, and for Mode 2 the
  recall of problematic clauses plus the **false-tranquility rate**, with abstention
  always published next to precision. The judge is a different model family from the
  generator, version-pinned, and calibrated once against human judgement.
- **The eval harness is not the test suite.** The unit/integration suite (fake LLM,
  deterministic, CI) protects contracts and code logic; the `eval` harness (real
  model, manual/versioned) measures AI quality. They never mix.

## 6. Reproduce it  (no live deployment)

```bash
git clone <this-repo> && cd lexme_v2
cp .env.example .env          # set DB credentials; add GEMINI_API_KEY / OPENROUTER_API_KEY
docker compose up -d          # or: make up-d
make ingest                   # build the LAU corpus from the BOE API (one command)
make ask Q="¿puede subirme el alquiler un 10%?"     # Mode 1 smoke test
make contract F=path/to/contrato.pdf                # Mode 2 smoke test
```

First boot downloads the BGE-M3 weights (~2 GB) into a persistent volume.
`GET /health` returns `200` only when both `db` and `tei` are reachable. Common
tasks are wrapped in the `Makefile` (`make help`). The LLM-backed modes and the
`eval` harness need a provider key; the generator is pinned to a **free-tier**
model, so eval runs are paced by the provider's rate limits (the harness backs off
and retries).

Reproduce the numbers:

```bash
make eval            # Mode 1 harness → a new artifact in eval-runs/
make eval-modo2      # Mode 2 harness → a new artifact in eval-runs/
make validate-checklist   # check the vertical's citations still match the corpus
```

## 7. Scope (v1)

Housing, **state law only** (LAU), a single vertical shipped as a data/config
package (corpus + checklist + prompts + reference set) with no plugin system in the
core. Out of scope for v1: a second vertical, regional law, contracts under earlier
LAU redactions in Mode 2 (rejected honestly by the temporal gate), OCR of scanned
documents, user accounts, and any live hosting — the project reproduces with
`docker compose up`.

## 8. License & data

- **Code:** [MIT](LICENSE).
- **Legal data:** not distributed with the repo — it is built at run time by
  `ingest` from the **BOE Consolidated Legislation** open-data API. Those texts are
  of a *"carácter meramente informativo"* (merely informative nature) and come from
  the **Agencia Estatal Boletín Oficial del Estado**, <https://www.boe.es>. The BOE
  reuse licence requires this attribution and this notice of informative character.
- **Not legal advice.** Lexme provides general information only; for a specific case,
  consult a qualified professional.

*Agent/author conversation artifacts (`.scratch/`, `docs/`, `CLAUDE.md`, …) are
excluded from the published repo via `.gitignore`.*
