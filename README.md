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
| **Recall of 🔴/🟠 clauses, end to end** | **1.0 (7 / 7)** | Of every genuinely problematic clause in the reference set — delimited or not — how many the system actually surfaced to a reader. This is the number a tenant experiences; the conditioned figure below is a diagnostic, not the headline. |
| **False-tranquility rate, end to end** | **0.0 (0 / 7)** | Genuinely 🔴/🟠 clauses the system called reassuring (🟢), over every problematic clause that exists. With `not_reported` at 0, nothing sits outside this denominator either. |
| **Recall of 🔴/🟠 clauses, conditioned on segmentation** | **1.0 (7 / 7)** | Of only the problematic clauses the segmentation layer correctly delimited, how many it also flagged as problematic — isolates a classification failure from a segmentation one. It equals the end-to-end figure only because segmentation now delimits every reference clause; the two are kept apart precisely so that stops being an assumption. |
| **Citation literality** | **0 of 84 failed** | Every displayed citation is re-resolved from the point-in-time corpus and re-checked character-for-character, independently of the verdict the runtime path gave it. One displayed citation that does not re-verify fails the whole run. |
| **Judge–reviewer agreement** | **0.94 (47 / 50)** | How often the LLM judge that grades Mode 1 answers agrees with an independent reviewer re-reading its rulings against the article. The reviewer here was a stronger *model*, not a human — see the caveat under the judge's numbers. |

### Full results

Precision is always reported next to its abstention rate, so you can't inflate one by
hiding behind the other.

Every figure is the **median of 3 repetitions** under one fingerprint, with the observed
range in brackets. A bare number over 21 cases was not comparable: see the note below.

| Metric | Value | Denominator |
| --- | --- | --- |
| Citation literality (both modes) | 0 failed, every repetition | invariant; a displayed citation that does not re-verify fails the run. Mode 1 showed 30 citations [27–31], Mode 2 showed 56. A quote the verifier discards *before* display is that mechanism working, not a failure: Mode 1 discarded 1 [0–1] |
| Retrieval recall by layer (Mode 1) | 0.94 · 0.83 · 0.97 · 0.94 | dense · lexical · fused · evidence, over 36 gold blocks. The gap that matters is the last two: what the candidate pool holds versus what survives the cut into synthesis. **Single draw, not banded** — the layer recalls sat outside the banded projection when this run was written; they are inside it now, so the next run bands them |
| Retrieval recall, first pass (Mode 1) | 1.00 [0.94–1.00] | gold blocks recovered before the agentic loop, over a four-norm corpus |
| Retrieval recall, final (Mode 1) | 0.95 [0.89–0.95] | over the evidence accumulated across passes. It reads *below* first-pass because the two average over different case sets — a case with no recorded first pass is scored in one and not the other — not because the loop loses ground |
| Multi-hop recall (Mode 1) | 6 / 8 | gold blocks across the 4 cases that need more than one norm. First repetition only, not banded |
| Outcome match rate (Mode 1) | 0.95 [0.86–0.95] | cases that reached the outcome they were written for |
| Disambiguation (Mode 1) | 0.33 [0.24–0.38] | fraction of cases the gate stopped; all resumed, 0 left stranded |
| Judge completeness · unsupported claims (Mode 1) | 0.74 [0.74–0.80] · 0.03 [0.00–0.06] | 11 judged cases, against human-reviewed key points |
| Judge–reviewer agreement (Mode 1) | 0.94 | 47 / 50 rulings, sampled with seed 20, reviewed by a model — calibrated on the July baseline and carried forward, not re-drawn on this run |
| Outcome match rate (Mode 2) | 1.00 [1.00–1.00] | 3 / 3 contracts reached the outcome their case declared |
| Recall 🔴/🟠, end to end (Mode 2) | 1.00 [1.00–1.00] | 7 / 7 problematic clauses in the reference set, delimited or not — the headline |
| False-tranquility rate, end to end (Mode 2) | 0.00 [0.00–0.00] | 0 / 7 real 🔴/🟠 called reassuring |
| Recall 🔴/🟠, conditioned on segmentation (Mode 2) | 1.00 [1.00–1.00] | 7 / 7 correctly delimited problematic clauses — diagnostic only |
| False-tranquility rate, conditioned on segmentation (Mode 2) | 0.00 [0.00–0.00] | 0 / 7 real 🔴/🟠 passed off as reassuring, over the delimited ones |
| Not reported (Mode 2) | 0 | problematic clauses the segmentation layer never delimited, so never shown to a reader as anything |
| Flag precision · abstention (Mode 2) | 0.875 [0.875–1.00] · 0.06 [0.00–0.06] | 7 / 8 flags correct · 1 / 16 clauses abstained |
| Segmentation IoU ≥ 0.80 (Mode 2) | 1.00 [1.00–1.00] | 16 / 16 clauses delimited |
| Absence ⚪ recall · precision (Mode 2) | 0.98 [0.98–0.98] · 1.00 [1.00–1.00] | 43 / 44 omitted rights surfaced · 0 false ones |

Mode 1 stops to ask one question when a case turns on a fact that changes the applicable
regime. Each case therefore carries the answers to those branches as reviewed data, and
the run reports how many cases answered straight through, how many were resumed with
that answer, and how many stopped at a pause with nothing to answer it — so the number
of cases actually measured end to end is never left ambiguous.

Recall is reported by layer because a single number hides where a case is lost. Evidence
recall sits `0.03` under the fused layer — one gold block, in one case, that the candidate
pool held and the cut into synthesis dropped. That is the honest reading of the last two
columns, and it is why they are published side by side rather than collapsed into one
"retrieval recall". Note that the fused layer is the *union of the dense and lexical
candidates*, not a ranked shortlist, so the two can only be equal if nothing is ever cut.

**Why the brackets.** Two runs of this suite under an identical fingerprint — same models,
same prompts, same corpus, same cases — disagree. Across the three repetitions here,
`outcome_match_rate` spans `0.86–0.95` and `unsupported_claim_rate` spans `0.00–0.06`;
a fourth draw at the same fingerprint put completeness at `0.70`, below the `0.74–0.80`
the three measured. A single number over 21 cases could not tell a real regression from
a re-roll, so the run now reports the median and the range it was drawn from, and a
comparison only calls something a regression when it lands outside the band.

The case-level spread is wider than the aggregate suggests, which is the part worth
internalising: mean recall moved `0.89 → 0.95` across repetitions while `mh-02` alone
swung the full `0.50 → 1.00`. A steady headline number is not evidence that nothing moved
underneath it.

Reports: Mode 1 →
[`modo1-20260805T152307Z.json`](eval-runs/modo1-20260805T152307Z.json) (`d1029d9c…`, 3
repetitions — every banded figure above comes from it), with
[`modo1-20260805T171438Z.json`](eval-runs/modo1-20260805T171438Z.json) a fourth draw at the
same fingerprint; Mode 2 →
[`modo2-20260808T125407Z.json`](eval-runs/modo2-20260808T125407Z.json) (`9443e189…`, 3
repetitions, same fingerprint as the July run it replaces).
Both were measured against the same four-norm corpus the repo builds today; every
number on this page comes from one of those two artifacts.
Regenerate with `make eval` / `make eval-modo2`; diff against a baseline with
`make eval-compare`. A changed fingerprint marks a run as an experiment rather than a
regression — this Mode 2 run carries the *same* fingerprint as the July one it replaces,
so the two are directly comparable, and the only figure that moved is the abstention rate
(`0.00 → 0.06`, one clause of sixteen, inside its own measured band).

The Mode 1 fingerprint moved, and for two reasons at once: the planner's prompt
changed, and so did the vertical's scope package. So the aggregate deltas against the
July baseline are **not attributable to either change alone**, and the fingerprint is
doing exactly the job it exists for by refusing to call this a clean comparison. The one
result here that *is* attributable is `mh-02`, because its retrieval path was measured
directly across repetitions rather than inferred from the aggregate.

The reference set is deliberately small and fully human-reviewed (21 Mode 1 cases, of
which 4 are multi-hop — their gold blocks live in more than one norm — and 3 synthetic
Mode 2 contracts, each declaring in its own `expected_outcome` the result it was built
to reach, checked against it rather than assumed). Every denominator is shown rather
than rounded away — including the one that used to disappear. `contrato-abusivo-01` is
an ordinary permanent lease by construction, and the **scope** gate — art. 4.2, not the
temporal one — used to reject it as an out-of-scope use: the model read the contract's
own illegal fixed 11-month, no-renewal clause as a seasonal let, and the gate rejected
on that reading alone. **That rejection was wrong**, and expensively so: the LAU
distinguishes by the property's destination, not by the term, which is exactly why that
clause is labelled illegal in the reference set. A tenant was told "I can't analyze
this" instead of "four of these clauses are illegal", and until the cases declared an
expected outcome the miss appeared in no metric at all.

The gate now closes only on evidence. The triage classifies the use by the destination
the document declares and returns the span that declares it; code checks that span is
literally in the document *and* carries one of the vertical's markers for that use, so
a term clause — a real span that says nothing about destination — cannot close it.
Without such a span the reading degrades to `indeterminado`, the contract is analyzed,
and the assumption is stated to the reader, the same treatment an absent signing date
gets. The direction is the asymmetry the whole mode is built on: wrongly refusing an
abusive lease costs a tenant four illegal clauses they never hear about, while wrongly
analyzing a genuine seasonal let costs some less pertinent findings. A contract that
*does* declare a local, an office or a season is still refused, and there are tests
pinning both directions.

That single change moves four numbers at once: `outcome_match_rate` to `1.00 (3/3)`,
segmentation from `0.625` to `1.00` (the 6 undelimited clauses were all in the rejected
contract), absence recall from `0.68` to `0.98`, and the problematic denominator from 2
to 7. What it does not fix is visible in the confusion matrix: 3 clauses that are
🟠 worse-than-default are called 🔴 illegal, and one 🟢 correct clause is called 🟠 —
the latter is the whole of the `0.875` flag precision. Erring toward severity is the
safe direction here, but it is over-severity, not accuracy, and calling it out is
cheaper than pretending the band is clean.

**The agentic self-critique loop's recall delta is 0.00, and I am publishing it flat.**
The loop only earns something when the first retrieval pass misses; on this corpus it
barely does — first-pass recall medians 1.00, and quadrupling the corpus to four norms did
not change that. So the honest reading is not "the loop works", it is "retrieval
saturates before the loop gets a turn", and the number that would move it is a harder
reference set, not more law. The corpus expansion did buy what the single-norm set
could not show: in the first repetition two of the four multi-hop cases recover both gold blocks,
and the way the fourth fails is the useful part. `mh-01` (tensioned-zone extension) retrieves the
LAU article but not the definition it depends on in the Ley por el derecho a la
vivienda, and the gate **abstains** rather than answering half-grounded — a recall gap
surfacing as an abstention instead of a confident half-answer is the behaviour the whole
design is for.

`mh-02` (stopping an eviction by paying) is worth reading at length, because it is the
clearest thing this eval has taught me and none of the lesson is flattering. It used to
retrieve the procedural article in the LEC but not LAU art. 27, so it explained *how* to
stop the eviction and never *why* the landlord could terminate. The cause was not a
ranking problem: art. 27 was a dense-only hit at rank 7, which reciprocal rank fusion
sinks below every block the two retrievers agreed on, and no re-ordering of an 8-block
cut recovers it — I simulated the alternatives against the real rankings before changing
anything. It was a planning problem: the plan asked only about the remedy, never the
cause. Wording a sub-query for the cause puts art. 27 at fused rank 1.

That fix is real and it is not sufficient, in two separate ways.

First, it is not reliable. Across four draws under an **identical fingerprint** — same
code, same prompts — `mh-02` lands at `1.00` three times and `0.50` once. The planner is
nominally at temperature 0 and is not deterministic in practice: the failing draw
decomposed the question into three sub-queries and none surfaced art. 27 into the cut,
while another used two and one did. Asking for the cause raises the odds that the plan
covers it; it does not guarantee it. (Only the first repetition stores per-case detail, so
the three successes are read off the first-pass recall band reaching `1.00`, which no
draw could reach with `mh-02` short.)

Second, even in the run that retrieves it, the answer still does not mention the cause.
The block reaches evidence and synthesis declines to use it. So the case that once
demonstrated a retrieval gap now demonstrates a synthesis one, and completeness — not
recall — is still the metric that catches it.

The layered metrics are what make both of those statements sayable instead of guessable,
and the honest summary is that a failure moved from one measured layer to another and
became less frequent, which is worth having and is not a fix.

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
failure mode to watch is the opposite one, and completeness 0.74 is more likely an
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
refusal — or into one: the art. 4.2 gate rejects a contract only when the document
declares an excluded destination in a span code finds literally in the text and matches
against the vertical's own markers, so "the model said temporada" is not a reason to
leave a lease unanalyzed.

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
