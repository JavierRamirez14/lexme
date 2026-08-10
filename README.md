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
| **Judge–reviewer agreement** | **0.96 (48 / 50)** | How often the LLM judge that grades Mode 1 answers agrees with an independent reviewer re-reading its rulings against the article. The reviewer here was a *model*, not a human — see the caveat under the judge's numbers. |

### Full results

Precision is always reported next to its abstention rate, so you can't inflate one by
hiding behind the other.

Every figure is the **median of 3 repetitions** under one fingerprint, with the observed
range in brackets. A bare number over 21 cases was not comparable: see the note below.

| Metric | Value | Denominator |
| --- | --- | --- |
| Citation literality (both modes) | 0 failed, every repetition | invariant; a displayed citation that does not re-verify fails the run. Mode 1 showed 29 citations [25–30], Mode 2 showed 56. A quote the verifier discards *before* display is that mechanism working, not a failure: Mode 1 discarded 0 in every repetition this time |
| Retrieval recall by layer (Mode 1) | 0.94 [0.92–0.94] · 0.78 [0.75–0.81] · 0.94 [0.94–0.97] · 0.92 [0.92–0.92] | dense · lexical · fused · evidence — banded for the first time. Each is the mean of the per-case recalls over the 18 cases that declare gold blocks and reached retrieval (24 gold blocks between them), not a pooled count. The gap that matters is the last two: what the candidate pool holds versus what survives the cut into synthesis |
| Retrieval recall, first pass (Mode 1) | 0.92 [0.92–0.92] | gold blocks recovered before the agentic loop, over a four-norm corpus |
| Retrieval recall, final (Mode 1) | 0.87 [0.87–0.87] | over the evidence accumulated across passes. It reads *below* first-pass because the two average over different case sets — a case with no recorded first pass is scored in one and not the other — not because the loop loses ground |
| Multi-hop recall (Mode 1) | 6 / 8 | gold blocks across the 4 cases that need more than one norm. First repetition only, not banded |
| Outcome match rate (Mode 1) | 0.90 [0.90–0.90] | cases that reached the outcome they were written for |
| Disambiguation (Mode 1) | 0.52 [0.38–0.52] | fraction of cases the gate stopped. In the repetition the artifact keeps per-case detail for, all 8 were resumed with the reply their case pins and 0 were left stranded |
| Judge completeness · unsupported claims (Mode 1) | 0.83 [0.79–0.84] · 0.05 [0.04–0.05] | 13 judged cases, against human-written key points, graded by the new judge |
| Judge–reviewer agreement (Mode 1) | 0.96 | 48 / 50 rulings, sampled with seed 25, reviewed by a model — re-done on this run for the new judge; the old judge's record was refused rather than carried forward |
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

Neither the case nor the loss itself is fixed. On 5 August the block dropped at the cut was
`mh-02`'s `LAU:a27`; on 9 August `mh-02` came back clean at `1.00` and the loss landed on
`mh-03`, the *Código Civil*'s `art1554`; on 10 August the first repetition lost nothing at
all, and across that run's three repetitions evidence recall spanned `0.889–0.972` — between
zero and two gold blocks cut, in one session, under one fingerprint. So there is no case to
name and no per-run count to quote: what the cut costs is drawn afresh every repetition.
Reading one repetition's `covered=false` as a defect in a particular case, which is what the
first pass at this did, was reading a draw as a finding.

**Why the brackets.** Two runs of this suite under an identical fingerprint — same models,
same prompts, same corpus, same cases — disagree. Across the three repetitions here,
`disambiguation_rate` spans `0.38–0.52` and completeness `0.79–0.84`; on the July baseline
a fourth draw put completeness at `0.70`, below the `0.74–0.80` its three repetitions
measured. A single number over 21 cases could not tell a real regression from a re-roll,
so the run reports the median and the range it was drawn from, and a comparison only
calls something a regression when it lands outside the band.

**What a band measures, and what it does not.** Three repetitions inside one session share
whatever the provider is serving that hour, so a band is the spread *within a session*. It
is not the noise between two sessions, and I have now measured how far apart those two are
rather than asserting it. Grouping the archive under [`eval-runs/`](eval-runs/) by
fingerprint gives sets of runs that differ in nothing but when they were launched; the
spread across each set is drift, and it is versioned per metric in
[`drift-modo1.json`](eval-runs/drift-modo1.json) and
[`drift-modo2.json`](eval-runs/drift-modo2.json).

Both columns are measured the same way — the gap between two runs' *observed ranges*, never
between their medians, so the within-session noise is not counted twice and then handed to a
rule that adds both bands back.

The answer is not what I expected, and it is not the one this section said when the drift
record was first built. **The band does not systematically underestimate the noise. Its own
width is unstable.** Here is `mean_recall`, same suite, same fingerprint, three sessions:

| Session | The three repetitions | Band width |
| --- | --- | --- |
| 5 August | 0.895 · 0.947 · 0.947 | 0.053 |
| 9 August | 0.868 · 0.868 · 0.868 | **0.000** |
| 10 August | 0.921 · 0.842 · 0.895 | **0.079** |

On 9 August three repetitions returned the identical number and the band said the noise was
zero. The next day, with nothing changed — the fingerprint is byte-identical `4906958e`, the
container image predates the change — the same metric spanned 0.079. So the 9 August band was
not evidence that the metric is stable; it was one draw that happened to come up flat. Three
repetitions are too few to measure a width you can lean on, which means a band read as a floor
is wrong just as often as a band read as a ceiling.

That reframes the drift comparison rather than cancelling it. Across the archive:

| Metric | Widest within one session | Widest between sessions |
| --- | --- | --- |
| Mean recall (Mode 1) | 0.079 | 0.053 |
| First-pass recall (Mode 1) | 0.083 | 0.056 |
| Evidence layer recall (Mode 1) | 0.083 | 0.056 |
| Lexical layer recall (Mode 1) | 0.111 | 0.111 |
| Judge completeness (Mode 1) | 0.060 | 0.079 |
| Disambiguation rate (Mode 1) | 0.143 | 0.095 |
| Every headline Mode 2 metric | 0.000 | 0.000 |

The two columns are the same order of magnitude, and which one is larger depends on the
metric. What is unsafe is not "the band is smaller than the drift" — it is trusting *any*
single run's band as the noise. The envelope over sessions is the thing to publish against,
and that is what the drift record is.

**The out-of-sample test passed.** The 10 August run is the first evidence the allowance had
never seen. Comparing it against 9 August at identical fingerprint, all twenty metrics come
back `variance` or `drift` — not one regression. The only metric outside both runs' bands is
lexical recall (`[0.750, 0.806]` against `[0.833, 0.944]`, a gap of 0.028 against an allowance
of 0.111), and that is precisely the move the old rule would have published as a result.

The `0.95 → 0.87` mean-recall drop between 5 and 9 August was never a regression either: the
gap between those runs' ranges is 0.026 against a 0.053 allowance, and `eval compare` now
reports it as `drift`. There is no retriever defect to chase.

Two limits worth stating. Of the between-session figures, only `mean_recall`, first-pass and
the layer recalls rest on pairs a **day** apart; completeness comes from two sessions three
hours apart on 27 July, and disambiguation from two on 5 August. What the archive establishes
is that the boundary is the **session**, not the day. And the Mode 2 row is one fingerprint
with **two** runs behind it — enough to say those numbers did not move across 11 days, not
enough to say they cannot; its `precision_problematic` shows 0.000 between sessions against
0.125 within one, which is what a thin sample looks like.

`make eval-drift` rebuilds the records as runs accumulate; `eval compare` reads the one
lying beside the baseline, and `--no-drift` puts it back to classifying against the
repetition bands alone.

The case-level spread is wider than the aggregate suggests too: on the July baseline mean
recall moved `0.89 → 0.95` across repetitions while `mh-02` alone swung the full
`0.50 → 1.00`. A steady headline number is not evidence that nothing moved underneath it.

Reports: Mode 1 →
[`modo1-20260809T164728Z.json`](eval-runs/modo1-20260809T164728Z.json) (`4906958e…`, 3
repetitions — every banded Mode 1 figure above comes from it), replicated the next day at the
identical fingerprint by
[`modo1-20260810T193500Z.json`](eval-runs/modo1-20260810T193500Z.json), which moves no metric
outside the measured noise. Read the two together rather than the first alone: the brackets in
the table above are one session's draw, and the section on drift says what that is and is not
worth. Both are an **experiment against**, not a regression on,
[`modo1-20260805T152307Z.json`](eval-runs/modo1-20260805T152307Z.json) (`d1029d9c…`): the
judge model and the synthesis prompt both changed, so the two are different
configurations and their judged numbers are not one series. Mode 2 →
[`modo2-20260808T125407Z.json`](eval-runs/modo2-20260808T125407Z.json) (`9443e189…`, 3
repetitions, same fingerprint as the July run it replaces).
All were measured against the same four-norm corpus the repo builds today; every
number on this page comes from one of those artifacts.
Regenerate with `make eval` / `make eval-modo2`; remeasure the drift with `make eval-drift`;
diff against a baseline with
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

**The judge was the weakest link in this whole table, so I replaced it — and the
completeness numbers before and after are not a series.** Until this run the grader was
`openai/gpt-oss-20b:free`: a 20B model on a free tier deciding whether an answer covers
the points a human marked as essential. It is now `deepseek/deepseek-v3.2`, the one task
in the project on a paid provider, at roughly cents per run.

Choosing it was itself a measurement, and the method is the part worth keeping. Grading
untouched answers did not discriminate at all — six candidates returned the same
verdicts. What separated them was planting faults with a known answer into a real reply.
Delete a key point's sentence and append an invented deadline: `qwen3-235b` and `kimi-k2`
both read the deleted key point as **covered**, which inflates completeness, the wrong
direction to fail in. Then take a complete answer whose third point rests on a Código
Civil article it never cites: `gpt-oss-20b` called every claim supported, while DeepSeek
flagged the three the cited article does not back. That is the unsupported-claim rate
firing instead of reading zero by default, and it is the real reason for the change.
Latency settled the free tier separately — 64.5 s per ruling is ~40 minutes of judging
per three-repetition run, and one free candidate never returned at all. The full
comparison is in [`verticales/vivienda/eval/README.md`](verticales/vivienda/eval/README.md).

Swapping the grader invalidated the old calibration by construction, and the harness now
refuses to carry it: a run whose pinned judge is not the record's `judge_model` publishes
no agreement rather than the previous grader's. That fired on the first real run under
DeepSeek, which came out honestly uncalibrated until the pass was re-done.

**The new agreement figure was again produced by a model reviewer, which is still weaker
than the claim I set out to make.** The artifact records who read in a `reviewer_kind`
field the loader will not accept as missing. Here it is `model`: a model re-read 50 of
this run's 124 rulings (seed 25) and disagreed with 2. Two language models share blind
spots a human would not, so **0.96 is an upper bound on what a human pass would find**;
that pass is still open.

What the two disagreements say is more useful than the score, and the direction has
flipped. Both are the new judge being *too lenient*: it marked a two-limb key point
covered when the answer supplied only one limb, and it accepted a claim that attaches
art. 36.2's one-or-two-month cap to art. 36.3's period, which instead defers to whatever
the parties agreed. The old judge erred by being too harsh; this one errs by waving
things through, so **completeness 0.83 is more likely a slight overstatement than an
understatement** — the opposite caveat to the one this section used to carry.

**Read the completeness jump as an experiment, not a result.** Against the July baseline
the median moves `0.74 → 0.83`, but the fingerprint changed for *two* reasons at once —
the judge model and the synthesis prompt — so the delta cannot be attributed to either
one, and `eval compare` classifies it as variance rather than improvement because the
bands still overlap by a hair (`[0.74–0.80]` against `[0.79–0.84]`). Separating the two
causes would need a third run, new judge with the old prompt, which costs a full day of
the generator's free-tier quota. `unsupported_claim_rate` moved `0.03 → 0.05` over the
same change: inside the baseline band, classified as variance, and partly just DeepSeek
enumerating claims more finely than a 20B did. It did not worsen in any sense the
measurement can distinguish from noise, and I would not claim more than that.

**One number moved that nothing I changed can explain.** Retrieval recall fell
`0.95 → 0.87`, outside its band, and the lexical layer with it. The synthesis prompt runs
*after* retrieval and the judge never touches it; the corpus and dataset hashes are
byte-identical between the two runs. So this is either provider drift between 5 and 9
August or, more likely, evidence that three repetitions inside one session understate the
real spread — they share whatever the provider is doing that hour. That is a caution
about how much the bands are worth, and it applies to the completeness reading above just
as much as to this one.

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
has to declare, because that is what decides how much the agreement is worth. Changing
the judge invalidates that record by construction, so the harness refuses to publish an
agreement measured on a grader the run did not use: swapping the model leaves the numbers
honestly uncalibrated until the pass is re-done.

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
