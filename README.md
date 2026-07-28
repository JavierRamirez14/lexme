# Lexme

**Agentic RAG over Spanish tenancy law that refuses to reassure you when the law doesn't.**

Lexme answers rental-law questions and analyses rental contracts against the
official consolidated text of the Spanish state law on renting a home — the *Ley de
Arrendamientos Urbanos* (LAU), the *Código Civil*'s lease title, the eviction
procedure in the *Ley de Enjuiciamiento Civil* and the *Ley por el derecho a la
vivienda* — as published by Spain's *Boletín Oficial del Estado* (BOE). Every legal
claim it shows is a literal, re-verified quote of a specific article of a specific
law, or it abstains. I built it to see how far you can push a RAG system when
"sounds plausible" is not good enough and every answer has to be checkable against
the source.

`Python` · `FastAPI` · `LangGraph` · `PostgreSQL + pgvector` · `BGE-M3 / TEI` · `Docker Compose` · `React + Vite`

> A contract checker's worst failure is telling a tenant that a genuinely illegal
> clause is fine. That failure has a name here — the **false-tranquility rate** —
> and it is the number the whole design is built to drive down. On the latest run
> it was **zero**.

<!-- Mode 2 demo GIF goes here: contract intake → spectrum risk map → the ⚪ finding.
     Capture against the running stack: make contract F=path/to/contrato.pdf -->

---

## The problem

Legal language keeps people out. Most of us don't fully understand the contracts we
sign or the rules that are supposed to protect us, and a lawyer is slow or expensive.
A tenant signs a clause that quietly waives a right the law grants by default and
never finds out. Generic AI assistants make this worse, not better: they answer with
no verifiable basis, invent article numbers, and when you hand them a contract they
drift toward the comfortable "looks fine."

Lexme takes the opposite stance. It cites, or it says it can't.

## What it does

**Mode 1 — Agentic Q&A.** Ask in plain language ("can my landlord raise the rent
10%?"). An agentic graph routes the question, breaks it into sub-queries, retrieves
iteratively with a self-critique loop, and answers in three layers: the grounded
citation (literal quote + BOE ELI link + the date the law applied), a plain-language
explanation, and a concrete next step. When there's no basis in the corpus it
abstains, and that decision is made by code, not by the model.

**Mode 2 — Contract risk map.** Upload a rental contract. Lexme segments it clause by
clause and contrasts each one against the LAU default, returning a **spectrum risk
map** instead of a verdict:

| | | |
|---|---|---|
| 🔴 void | 🟠 worse than the default | 🟡 lawful but negotiable |
| ⚪ a protection the contract omits | 🟢 conforming | |

There is no aggregate "contract OK." The interesting level is ⚪: a right the law
gives you that the contract stays silent about. A checker that only flags bad clauses
would reassure you about a contract whose real problem is everything it leaves out.

## Measured quality

The numbers below come from the evaluation harness run against the real system.
Each run writes a versioned artifact under [`eval-runs/`](eval-runs/) stamped with a
configuration fingerprint, so results are reproducible and comparable across changes.

### Headline

| Result | Value | What it means |
| --- | --- | --- |
| **False-tranquility rate** | **0.0 (0 / 2)** | Genuinely 🔴/🟠 clauses the system let pass as reassuring (🟢 or silence), over the problematic clauses it correctly delimited. |
| **Recall of 🔴/🟠 clauses** | **1.0 (2 / 2)** | Of the correctly delimited problematic clauses, how many it also flagged as problematic. |
| **Citation literality** | **0 of 30 failed** | Every displayed citation is re-resolved from the point-in-time corpus and re-checked character-for-character, independently of the verdict the runtime path gave it. One displayed citation that does not re-verify fails the whole run. |
| **Judge–reviewer agreement** | **0.94 (47 / 50)** | How often the LLM judge that grades Mode 1 answers agrees with an independent reviewer re-reading its rulings against the article. The reviewer here was a stronger *model*, not a human — see the caveat under the judge's numbers. |

### Full results

Precision is always reported next to its abstention rate, so you can't inflate one by
hiding behind the other.

| Metric | Value | Denominator |
| --- | --- | --- |
| Citation literality (both modes) | 0 failed of 30 shown | invariant; a displayed citation that does not re-verify fails the run. A quote the verifier discards *before* display is that mechanism working, not a failure: this run discarded none |
| Retrieval recall, first pass (Mode 1) | 0.92 | gold blocks recovered before the agentic loop, over a four-norm corpus |
| Multi-hop recall (Mode 1) | 0.75 | 6 / 8 gold blocks across the 4 cases that need more than one norm |
| Outcome match rate (Mode 1) | 0.90 | 19 / 21 cases reached the outcome they were written for |
| Disambiguation (Mode 1) | 0.52 | 11 / 21 cases the gate stopped; all 11 resumed, 0 left stranded |
| Judge completeness · unsupported claims (Mode 1) | 0.75 · 0.06 | 13 judged cases, against human-reviewed key points |
| Judge–reviewer agreement (Mode 1) | 0.94 | 47 / 50 rulings, sampled with seed 20 from the run's 92, reviewed by a model |
| Recall 🔴/🟠 (Mode 2) | 1.00 | 2 / 2 correctly delimited problematic clauses |
| False-tranquility rate (Mode 2) | 0.00 | 0 / 2 real 🔴/🟠 passed off as reassuring |
| Flag precision · abstention (Mode 2) | 0.67 · 0.00 | 2 / 3 flags correct · 0 / 10 clauses abstained |
| Segmentation IoU ≥ 0.80 (Mode 2) | 0.625 | 10 / 16 clauses delimited |
| Absence ⚪ recall · precision (Mode 2) | 0.68 · 1.00 | 30 / 44 omitted rights surfaced · 0 false ones |

Mode 1 stops to ask one question when a case turns on a fact that changes the applicable
regime. Each case therefore carries the answers to those branches as reviewed data, and
the run reports how many cases answered straight through, how many were resumed with
that answer, and how many stopped at a pause with nothing to answer it — so the number
of cases actually measured end to end is never left ambiguous.

Reports: Mode 1 →
[`modo1-20260728T074601Z.json`](eval-runs/modo1-20260728T074601Z.json) (`d870d388…`),
Mode 2 →
[`modo2-20260728T083753Z.json`](eval-runs/modo2-20260728T083753Z.json) (`ee478eef…`).
Both were measured against the same four-norm corpus the repo builds today; every
number on this page comes from one of those two artifacts.
Regenerate with `make eval` / `make eval-modo2`; diff against a baseline with
`make eval-compare`. A changed fingerprint marks a run as an experiment rather than a
regression.

The reference set is deliberately small and fully human-reviewed (21 Mode 1 cases, of
which 4 are multi-hop — their gold blocks live in more than one norm — and 3 synthetic
Mode 2 contracts, one of which the temporal gate correctly rejects as out-of-scope),
which is why every denominator is shown rather than rounded away.

**The agentic self-critique loop's recall delta is 0.00, and I am publishing it flat.**
The loop only earns something when the first retrieval pass misses; on this corpus it
barely does — first-pass recall is 0.92, and quadrupling the corpus to four norms did
not change that. So the honest reading is not "the loop works", it is "retrieval
saturates before the loop gets a turn", and the number that would move it is a harder
reference set, not more law. The corpus expansion did buy what the single-norm set
could not show: two of the four multi-hop cases recover one of their two gold blocks,
and the two misses fail differently. `mh-01` (tensioned-zone extension) retrieves the
LAU article but not the definition it depends on in the Ley por el derecho a la
vivienda, and the gate **abstains** rather than answering half-grounded — a recall gap
surfacing as an abstention instead of a confident half-answer is the behaviour the whole
design is for. `mh-02` (stopping an eviction by paying) retrieves the procedural article
in the LEC but not LAU art. 27, so it answers correctly about *how* to stop the eviction
and never mentions *why* the landlord could terminate; nothing it says is wrong, and the
judge's completeness is what catches the half-answer. Those are the two shapes a
retrieval gap can take, and only one of them is safe.

**The judge's agreement figure was produced by a model reviewer, and that is a weaker
claim than the one I set out to make.** Calibration is the one-time pass that gives the
judge's numbers a unit: a sample of its own rulings is re-read against the article, and
the fraction the reviewer confirms is published beside every judged metric. The run
artifact records who did that reading in a `reviewer_kind` field the loader will not
accept as missing, because who reviewed decides what the number means. Here it is
`model`: a stronger model re-read 50 of the run's 92 rulings (sampled with seed 20) and
disagreed with 3. Two language models share blind spots a human would not, so **0.94 is
an upper bound on what a human pass would find, not a substitute for one** — the human
pass is still open, and running it replaces this record with `reviewer_kind: human`.

What the three disagreements say is more useful than the score. All three are the judge
being *too harsh*: twice it marked a reference key point uncovered while its own claim
list showed the answer stating it, and once it flagged a claim as unsupported that the
cited article plainly backs. So on this sample the judge does not rubber-stamp — the
failure mode to watch is the opposite one, and completeness 0.75 is more likely an
understatement than an inflation.

## Architecture

Four containers on one Compose network:

| Service    | Role                                                    | Port          |
| ---------- | ------------------------------------------------------- | ------------- |
| `frontend` | React + Vite SPA                                        | `5173`        |
| `api`      | FastAPI monolith (also hosts the `ingest`/`eval` CLIs)  | `8000`        |
| `tei`      | Text Embeddings Inference serving BGE-M3                | `8080` → `80` |
| `db`       | PostgreSQL + pgvector                                   | `5432`        |

- **Ingestion** builds the corpus from the BOE Consolidated Legislation API (XML),
  one *precept-block* (article) per chunk, keeping every historical version with its
  validity dates. That's what lets Mode 1 answer "as of" a past date with a plain SQL
  substitution rather than a second index. The corpus spans four norms, and the
  vertical's manifest declares how much of each one enters — the whole LAU, but only
  the lease title of the Código Civil and the eviction articles of the LEC, because a
  corpus should hold what a tenant asks about, not every article of every law it
  touches. A block is identified corpus-wide by `<norm_id>:<block_id>`; a bare
  article number is ambiguous once four laws each have an "article 9", and every
  citation, gold block and notice carries the qualified form end to end.
- **Retrieval** is hybrid: pgvector (BGE-M3 dense) plus Postgres full-text `spanish`,
  fused with a Reciprocal Rank Fusion I wrote by hand rather than pulling in a
  reranker for v1.
- **Orchestration** is a LangGraph graph over a typed Pydantic state: router →
  decomposition → iterative retrieval with a self-critique loop → three-layer
  synthesis that may only cite blocks it actually retrieved.
- **The citation verifier**, shared by both modes, is strict string equality after
  conservative typographic normalisation (no fuzzy matching), with deterministic snap
  repair as the only allowed fix and a hard discard otherwise. Its telemetry is the
  single source of the literality numbers.

## Design decisions

A few choices I'd call out to another engineer:

**One LLM interface, task → model in config.** The graphs never touch a provider SDK;
a routing client maps each task to a model, so swapping providers or free-tier models
is a one-line config change and tests inject a deterministic fake. That seam earned
its keep in practice: it's where I disabled the native JSON mode that silently
truncates replies on these models, and where I added `429` back-off so a rate-limited
run recovers instead of dying.

**Code owns the trustworthy decisions.** Abstention, the temporal gate, LAU art. 4.2
scope, the checklist × contract cross-check, and citation discard are all plain code,
not model judgement. They're unit-testable, and no prompt can talk the system out of a
refusal.

**A spectrum, never a binary verdict.** Five discrete levels with total coverage, and
the ⚪ absences anchored to checklist items. Collapsing that into "OK / not OK" would
reintroduce exactly the false calm the product exists to prevent.

**Evaluation that argues against its own precision.** Layered recall, the citation
guardrail as a hard invariant, and for Mode 2 the recall of problematic clauses plus
the false-tranquility rate, with abstention always beside precision. The LLM-judge is
a different model family from the generator, version-pinned, temperature 0, and
calibrated against an independent reviewer whose *kind* — human or model — the record
has to declare, because that is what decides how much the agreement is worth.

**The eval harness is not the test suite.** The unit/integration suite (fake LLM,
deterministic, in CI) protects contracts and code logic; the `eval` harness (real
model, run and versioned by hand) measures AI quality. Keeping them apart is what lets
the harness use a real model without ever making CI flaky.

## Run it locally

No hosted demo by design — the whole thing comes up from nothing with Compose.

```bash
git clone <this-repo> && cd lexme_v2
cp .env.example .env          # DB credentials; GEMINI_API_KEY / OPENROUTER_API_KEY for the LLM modes
docker compose up -d          # or: make up-d
make ingest                   # build the whole corpus from the BOE API (one command)
make ask Q="¿puede subirme el alquiler un 10%?"     # Mode 1
make contract F=path/to/contrato.pdf                # Mode 2
```

First boot pulls the BGE-M3 weights (~2 GB) into a persistent volume. `GET /health`
returns `200` only when both `db` and `tei` are reachable. `make help` lists the rest.
The API package uses `uv`, `ruff` and `pytest`:

```bash
cd api && uv sync && uv run pytest && uv run ruff check .
make eval          # reproduce the Mode 1 numbers
make eval-modo2    # reproduce the Mode 2 numbers
```

## Scope

Housing, **state law only**, shipped as a single data/config vertical (corpus +
checklist + prompts + reference set) with no plugin machinery in the core — adding a
second vertical shouldn't touch it. The corpus is four norms, chosen because the LAU
itself sends the reader outside it:

| Norm | What it contributes | Why it is in |
| --- | --- | --- |
| Ley 29/1994 (LAU) | whole law | the regime the vertical is about |
| Código Civil, arts. 1542–1582 | lease title | LAU art. 4.2 makes it the supletory regime, and arts. 21, 25 and 27 point at it by name |
| LEC, arts. 22, 250, 437–447, 549, 703–704 | eviction | "can they throw me out?" is the question the LAU deliberately does not answer; the procedure lives here |
| Ley 12/2023 (vivienda), arts. 3, 6, 18, 31 and DT 4.ª | definitions | LAU arts. 10 and 17 condition rights on "tensioned market area" and "large holder", both defined only here |

Mode 2 still contrasts contracts against the LAU checklist alone. Out for v1: a
second vertical, regional law, contracts under earlier LAU redactions in Mode 2 (the
temporal gate rejects them honestly), OCR of scanned documents, accounts, and any
live hosting.

## License & data

- **Code:** [MIT](LICENSE).
- **Legal data** is not shipped with the repo; `ingest` builds it at run time from the
  **BOE Consolidated Legislation** open-data API. Those consolidated texts are of a
  *"carácter meramente informativo"* (merely informative nature) and come from the
  **Agencia Estatal Boletín Oficial del Estado**, <https://www.boe.es>. The BOE reuse
  licence requires this attribution and this notice.
- **Not legal advice.** Lexme is general information only; for a specific case, consult
  a qualified professional.
