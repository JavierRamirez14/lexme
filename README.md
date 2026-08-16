# Lexme

**Agentic RAG over Spanish tenancy law that refuses to reassure you when the law doesn't.**

Lexme answers rental-law questions and analyses rental contracts against the official
consolidated text of the Spanish state law on renting a home — the *Ley de Arrendamientos
Urbanos* (LAU), the *Código Civil*'s lease title, the eviction procedure in the *Ley de
Enjuiciamiento Civil* and the *Ley por el derecho a la vivienda* — as published by Spain's
*Boletín Oficial del Estado* (BOE). Every legal claim it shows is a literal, re-verified
quote of a specific article of a specific law, or it abstains. I built it to see how far
you can push a RAG system when "sounds plausible" is not good enough and every answer has
to be checkable against the source.

`Python` · `FastAPI` · `LangGraph` · `PostgreSQL + pgvector` · `BGE-M3 / TEI` · `Docker` · `React + Vite`

> A contract checker's worst failure is telling a tenant that a genuinely illegal clause is
> fine. That failure has a name here — the **false-tranquility rate** — and it is the number
> the whole design is built to drive down. On the latest run it was **zero**.

![Mode 2: a rental contract goes in, and a spectrum risk map comes back — five void
clauses, one negotiable, and thirteen protections the law grants that the contract never
mentions, each with its literal article](assets/modo2-demo.gif)

*Recorded against the stack this repo builds — `docker compose up`, corpus ingested, one of
the three reference contracts uploaded through the real UI.*

---

## The problem

Legal language keeps people out. Most of us don't fully understand the contracts we sign,
and a lawyer is slow or expensive. A tenant signs a clause that quietly waives a right the
law grants by default and never finds out. Generic AI assistants make this worse, not
better: they answer with no verifiable basis, invent article numbers, and when you hand
them a contract they drift toward the comfortable "looks fine."

Lexme takes the opposite stance. It cites, or it says it can't.

## What it does

**Mode 1 — Agentic Q&A.** Ask in plain language ("can my landlord raise the rent 10%?"). An
agentic graph routes the question, breaks it into sub-queries, retrieves iteratively with a
self-critique loop, and answers in three layers: the grounded citation (literal quote + BOE
ELI link + the date the law applied), a plain-language explanation, and a concrete next
step. When there's no basis in the corpus it abstains, and that decision is made by code,
not by the model.

**Mode 2 — Contract risk map.** Upload a rental contract. Lexme segments it clause by clause
and contrasts each one against the LAU default, returning a **spectrum risk map** instead of
a verdict:

| | | |
|---|---|---|
| 🔴 void | 🟠 worse than the default | 🟡 lawful but negotiable |
| ⚪ a protection the contract omits | 🟢 conforming | |

There is no aggregate "contract OK." The interesting level is ⚪: a right the law gives you
that the contract stays silent about. A checker that only flags bad clauses would reassure
you about a contract whose real problem is everything it leaves out.

## Measured quality

| Result | Value | What it means |
| --- | --- | --- |
| **Recall of 🔴/🟠 clauses (Mode 2)** | **1.00** (7 / 7) | Every genuinely problematic clause in the reference set reached the reader. |
| **False-tranquility rate (Mode 2)** | **0.00** (0 / 7) | None of them was called reassuring. The failure the design exists to prevent. |
| **Citation literality (both modes)** | **0 failures / 84 shown** | An invariant, not a score: one displayed quote that doesn't re-verify fails the whole run. |
| **Retrieval recall, fused layer (Mode 1)** | **0.97** [0.94–0.97] | Gold blocks the candidate pool holds, over 18 cases with declared gold. |
| **Judge completeness (Mode 1)** | **0.85** [0.72–0.86] | 13 cases graded by an LLM judge against human-written key points. |
| **Judge–reviewer agreement** | **0.96** (48 / 50) | How often the judge agrees with an independent re-read of its rulings. |

Every number comes from the evaluation harness run against the real system, which writes a
versioned artifact under [`eval-runs/`](eval-runs/) stamped with a configuration
fingerprint. Reproduce with `make eval` and `make eval-modo2`.

Three things worth knowing about how to read those numbers:

- **Mode 1 figures are the median of 12 repetitions** (four runs of three under one
  byte-identical fingerprint) with the observed range in brackets. A bare number over 21
  cases can't tell a real regression from a re-roll, so `eval compare` only calls something
  a regression when it lands outside the band.
- **A band is not a guarantee.** Its own width is unstable: the same metric, same
  fingerprint, spanned `0.000` one day and `0.079` the next. Three repetitions are too few
  to measure a width you can lean on — which is why the published figure is the envelope
  over all twelve, not any one session's.
- **The judge is a model, and so was its reviewer.** Two language models share blind spots a
  human wouldn't, so `0.96` is an upper bound on what a human pass would find. Its two
  disagreements are both the judge being too lenient, so completeness is more likely a
  slight overstatement than an understatement.

Precision is always reported next to its abstention rate, so neither can be inflated by
hiding behind the other.

## Architecture

Four containers on one Compose network:

| Service    | Role                                                    | Port          |
| ---------- | ------------------------------------------------------- | ------------- |
| `frontend` | React + Vite SPA                                        | `5173`        |
| `api`      | FastAPI monolith (also hosts the `ingest`/`eval` CLIs)  | `8000`        |
| `tei`      | Text Embeddings Inference serving BGE-M3                | `8080` → `80` |
| `db`       | PostgreSQL + pgvector                                   | `5432`        |

- **Ingestion** builds the corpus from the BOE Consolidated Legislation API (XML), one
  *precept-block* (article) per chunk, keeping every historical version with its validity
  dates. That's what lets Mode 1 answer "as of" a past date with a plain SQL substitution
  rather than a second index. A block is identified corpus-wide by `<norm_id>:<block_id>` —
  a bare article number is ambiguous once four laws each have an "article 9" — and every
  citation, gold block and notice carries the qualified form end to end.
- **Retrieval** is hybrid: pgvector (BGE-M3 dense) plus Postgres full-text `spanish`, fused
  with a Reciprocal Rank Fusion I wrote by hand rather than pulling in a reranker for v1.
- **Orchestration** is a LangGraph graph over a typed Pydantic state: router → decomposition
  → iterative retrieval with a self-critique loop → three-layer synthesis that may only cite
  blocks it actually retrieved.
- **The citation verifier**, shared by both modes, is strict string equality after
  conservative typographic normalisation (no fuzzy matching), with deterministic snap repair
  as the only allowed fix and a hard discard otherwise.

## Design decisions

A few choices I'd call out to another engineer:

**One LLM interface, task → model in config.** The graphs never touch a provider SDK; a
routing client maps each task to a model, so swapping providers is a one-line config change
and tests inject a deterministic fake. That seam earned its keep: it's where I disabled the
native JSON mode that silently truncates replies on these models, and where I added `429`
back-off so a rate-limited run recovers instead of dying.

**Code owns the trustworthy decisions.** Abstention, the temporal gate, LAU art. 4.2 scope,
the checklist × contract cross-check and citation discard are all plain code, not model
judgement. They're unit-testable, and no prompt can talk the system out of a refusal — or
into one: the art. 4.2 gate rejects a contract only when the document declares an excluded
destination in a span code finds literally in the text, so "the model said temporada" is not
a reason to leave a lease unanalyzed.

**A spectrum, never a binary verdict.** Five discrete levels with total coverage, and the ⚪
absences anchored to checklist items. Collapsing that into "OK / not OK" would reintroduce
exactly the false calm the product exists to prevent.

**Evaluation that argues against its own precision.** Layered recall, the citation guardrail
as a hard invariant, and for Mode 2 the recall of problematic clauses beside the
false-tranquility rate. The LLM judge is a different model family from the generator,
version-pinned, temperature 0, and calibrated against an independent reviewer whose *kind* —
human or model — the record has to declare. Changing the judge invalidates that record by
construction, so the harness refuses to publish an agreement measured on a grader the run
did not use.

**The eval harness is not the test suite.** The unit/integration suite (fake LLM,
deterministic, in CI) protects contracts and code logic; the `eval` harness (real model, run
and versioned by hand) measures AI quality. Keeping them apart is what lets the harness use
a real model without ever making CI flaky.

## Honest limits

Three things I cut on purpose rather than ran out of time for. They're here because a limit
you can read is worth more than one you have to discover.

**The Mode 2 denominator is tiny.** Recall `1.00` and false tranquility `0.00` rest on **7
problematic clauses across 3 contracts**, and the `ilegal` row of the confusion matrix is
empty — the mode has never been measured on a clause that is outright void. Those numbers
are true and fragile: a denominator that small cannot tell a tool that works from one that
got lucky seven times. The fix isn't more labelling but a different construction — assemble
each clause *starting from the article it violates*, the way the Mode 1 reference set already
is, so the label comes from the norm rather than a model's opinion.

**The fingerprint does not cover the harness.** It pins the models, prompts, corpus and cases
— everything the system *under* test is made of — and nothing of the code doing the testing.
So a fix to the citation checker or the judge changes how a metric is measured while the
fingerprint keeps claiming two runs are comparable. That happened twice, and it made the
historical literality record unreadable. A digest of `lexme.eval` inside `ConfigFingerprint`
closes it, and it's the first thing I'd build.

**The agentic loop's recall delta is 0.00, and I publish it flat.** The self-critique loop
only earns something when the first retrieval pass misses, and on this corpus it barely does
— first-pass recall medians `1.00`, and quadrupling the corpus to four norms didn't change
that. The honest reading isn't "the loop works", it's "retrieval saturates before the loop
gets a turn". What would move it is a harder reference set, not more law.

## Run it locally

No hosted demo by design — the whole thing comes up from nothing with Compose.

```bash
git clone https://github.com/JavierRamirez14/lexme.git && cd lexme
cp .env.example .env          # DB credentials; GEMINI_API_KEY / OPENROUTER_API_KEY for the LLM modes
docker compose up -d          # or: make up-d
make ingest                   # build the whole corpus from the BOE API (one command)
make ask Q="¿puede subirme el alquiler un 10%?"     # Mode 1
make contract F=path/to/contrato.pdf                # Mode 2
```

First boot pulls the BGE-M3 weights (~2 GB) into a persistent volume. `GET /health` returns
`200` only when both `db` and `tei` are reachable. `make help` lists the rest.

```bash
cd api && uv sync && uv run pytest && uv run ruff check .
make eval          # reproduce the Mode 1 numbers
make eval-modo2    # reproduce the Mode 2 numbers
```

## Scope

Housing, **state law only**, shipped as a single data/config vertical (corpus + checklist +
prompts + reference set) with no plugin machinery in the core — adding a second vertical
shouldn't touch it. The corpus is four norms, chosen because the LAU itself sends the reader
outside it:

| Norm | What it contributes | Why it is in |
| --- | --- | --- |
| Ley 29/1994 (LAU) | whole law | the regime the vertical is about |
| Código Civil, arts. 1542–1582 | lease title | LAU art. 4.2 makes it the supletory regime, and arts. 21, 25 and 27 point at it by name |
| LEC, arts. 22, 250, 437–447, 549, 703–704 | eviction | "can they throw me out?" is the question the LAU deliberately does not answer |
| Ley 12/2023 (vivienda), arts. 3, 6, 18, 31, DT 4.ª | definitions | LAU arts. 10 and 17 condition rights on "tensioned market area" and "large holder", defined only here |

Mode 2 still contrasts contracts against the LAU checklist alone. Out for v1: a second
vertical, regional law, contracts under earlier LAU redactions in Mode 2 (the temporal gate
rejects them honestly), OCR of scanned documents, accounts, and any live hosting.

## License & data

- **Code:** [MIT](LICENSE).
- **Legal data** is not shipped with the repo; `ingest` builds it at run time from the **BOE
  Consolidated Legislation** open-data API. Those consolidated texts are of a *"carácter
  meramente informativo"* (merely informative nature) and come from the **Agencia Estatal
  Boletín Oficial del Estado**, <https://www.boe.es>. The BOE reuse licence requires this
  attribution and this notice.
- **Not legal advice.** Lexme is general information only; for a specific case, consult a
  qualified professional.
