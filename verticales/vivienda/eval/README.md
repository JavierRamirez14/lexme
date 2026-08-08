# Mode 1 evaluation (vivienda)

Run the harness against the real system with `make eval V=vivienda`. It measures
Mode 1 in layers and end-to-end over the reference set and writes one JSON run
artifact stamped with its configuration fingerprint.

## What a run reports

- **Retrieval recall by layer** — recall over the accumulated evidence, broken out
  per layer (`dense`, `lexical`, `fused`, `evidence`) and attributed per sub-query,
  so a miss is attributable (the fused layer should recall at least as much as
  either retriever alone).
- **Agentic recall delta** — the first-pass-to-final recall pair, quantifying what
  the self-critique loop earned (`mean_first_pass_recall → mean_recall`).
- **With-reference judge** — completeness against the case's `key_points`, the
  unsupported-claim rate (hallucination judged claim by claim, not answer by
  answer), and a clarity rubric.
- **Abstention** — the abstention rate and how many of the cases meant to abstain
  did, published next to the outcome match rate so abstention is never read alone.
- **Disambiguation** — how many cases answered straight through (`directo`), were
  resumed with the answer they pin (`reanudado`) and stopped at a pause they bring
  no answer for (`sin_respuesta`), plus the rate of cases the gate fired on, so how
  many cases a run really measured end to end is never ambiguous.
- **Citation guardrail** — a hard invariant: every displayed citation must
  re-verify literally against the corpus at the case's date, or the run fails.

## Cases that pause

Mode 1 stops to ask one question when the case turns on a critical branch (the
contract's signing date, the kind of use). A case that never gets past that pause
is never measured on retrieval, citations or the judge, so each case carries the
answers to those branches as reviewed data:

```json
"clarification_answers": [
  { "branch_id": "fecha_firma", "answer": "01/09/2025" },
  { "branch_id": "uso_vivienda", "answer": "Es mi vivienda habitual." }
]
```

The harness resumes the paused run on its own thread with the answer written for
the branch actually asked; the guardrail and the judge then rule on the answer the
case ended with, not on the pause. A branch with no written answer leaves the case
at outcome `desambiguacion` — counted as such, never as a hit. The answers are part
of the dataset digest, so editing one changes the run's fingerprint.

Because the signing date re-anchors the run's point-in-time clock, an answer to
`fecha_firma` must land in the same redaction the case's `key_points` were
extracted from, or the case would be graded against a reference that does not apply
to it.

## The judge

The judge task (`judge` in `api/src/lexme/llm/tasks.json`) runs at temperature 0
and is a different model family than the Mode 1 generator, so it cannot prefer its
own family's answers. The harness asserts this at startup
(`assert_judge_distinct_from_generator`) and records the judge model in the run's
configuration fingerprint.

Following the blueprint's volume split, the generator (which concentrates the
hundreds of calls a full run makes) sits on Gemini's generous free tier, and the
judge (~1 structured call per answered case) sits on an OpenRouter model of another
family. The shipped judge is `deepseek/deepseek-v3.2` — paid, at roughly cents per
run. Re-allocating a task to another provider is a one-line edit to `tasks.json`,
no code change.

Check the pin against `https://openrouter.ai/api/v1/models` before a real run: an
id that has left the catalog fails the run with a 404 rather than degrading
quietly, and has to be replaced before the run, not after it fails.

### Why this judge and not a free one

The judge used to be `openai/gpt-oss-20b:free`. It was the weakest link in the
whole Mode 1 quality measurement — a 20B model on a free tier deciding whether an
answer covers the points a human marked as essential — so the pin was chosen again
from scratch, and this is what the candidates were measured on. Each was given the
harness's own judge prompt over real Mode 1 answers.

| Candidate | Stable at temp. 0 | Planted-fault probe | Latency | Price /M in·out |
| --- | --- | --- | --- | --- |
| `openai/gpt-oss-20b:free` (incumbent) | yes | found the omission, **missed the uncited claims** | 64.5 s | free |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | — | — | 210 s | free |
| `nvidia/nemotron-3-super-120b-a12b:free` | — | hung >20 min on the free queue | — | free |
| `qwen/qwen3-235b-a22b-2507` | **no** | **read a removed key point as covered**; 1 unparseable reply | 12.7 s | $0.09 · $0.55 |
| `moonshotai/kimi-k2-0905` | no | read a removed key point as covered | 16.1 s | $0.60 · $2.50 |
| **`deepseek/deepseek-v3.2`** | **yes** | **found the omission and all the uncited claims** | **6.7 s** | $0.26 · $0.38 |

Two probes did the discriminating, because agreeing on easy rulings did not:
every candidate graded the unmodified answers the same way.

- *Planted omission and planted claims.* A real answer with one key point's
  sentence deleted and an invented deadline appended. The correct reading is that
  key point uncovered and the invention unsupported. Qwen and Kimi both marked the
  deleted key point **covered** — inflating completeness, the failure direction that
  matters most, since it makes the metric read better than the system is.
- *A complete, paraphrased answer.* All three key points covered in the tenant's
  own words, including one resting on the Código Civil the answer never cited.
  `gpt-oss-20b` called every claim supported; DeepSeek flagged the three the cited
  article does not back, including both Código Civil assertions. That is the
  unsupported-claim rate doing its job rather than reading zero by default.

The free tier is also not a service a repeated measurement can be built on: with
issue 23 multiplying repetitions, the incumbent's 64.5 s per ruling is ~40 minutes
of judging per three-repetition run, and one free candidate never returned at all.
DeepSeek at 6.7 s is around four minutes and a few cents.

DeepSeek is also a third family, distinct from both the Gemini generator it grades
and the Claude reviewer that calibrates it — so neither the self-preference bias
nor the shared-blind-spot problem is made worse by the swap.

## Judge calibration (one-time)

The judged numbers are only trustworthy read next to a reviewer agreement.
Calibration is a one-time pass, versioned as `judge-calibration.json` in this
directory; when present the harness recomputes the agreement from its reviewed
items and publishes it on the run artifact, and when absent the run reports the
judge as "not calibrated" rather than inventing a number.

**Who reviewed decides what the number means**, so the record declares it in
`reviewer_kind` and the harness refuses to load a record that does not:

- `human` — the intended pass. A human with the article in front of them is
  independent of the models being measured, so the agreement is evidence about the
  judge.
- `model` — a stronger model reviewing the judge's rulings. Cheap and repeatable,
  and it does catch the judge rubber-stamping, but two language models share blind
  spots, so the agreement is an upper bound on what a human pass would find, not a
  substitute for one. Publish it labelled as such, never as human agreement.

Workflow:

1. Run the harness. Each run artifact keeps the verdicts the judge returned, not
   just the averages they produced, so the sample is drawn from the same rulings the
   published numbers came from.
2. Draw the sample:

   ```bash
   make eval-calibrate-export RUN=modo1-<stamp>.json SIZE=50 SEED=0
   ```

   This writes `eval-runs/modo1-<stamp>-judge-review.json` (the sample, with its
   population and seed) and `…-judge-review.md` (the sheet the reviewer reads). Each
   ruling is numbered within the run's whole population and rendered with the article
   it hangs on, resolved at the case's own point-in-time date.
3. The reviewer reads every ruling in the sheet with that article in front of them.
   The review is by exception: only the numbers they disagree with are recorded, and
   everything else counts as confirmed.
4. Record the review:

   ```bash
   make eval-calibrate-build SAMPLE=modo1-<stamp>-judge-review.json \
     RUN=modo1-<stamp>.json BY="<name>" KIND=human DISAGREE="3,7,12"  # "none" if all confirmed
   ```

   The agreement is derived from the labels, never taken on trust, and a number that
   is not in the sample fails loudly instead of quietly inflating it. `RUN` publishes
   the agreement on the very run whose rulings were reviewed — the labels can only
   exist after that run, so stamping it there is what keeps the number attached to
   the rulings it was measured on.
5. Every later run reads `judge-calibration.json` and publishes the same agreement
   next to its judged metrics, until a change to the judge calls for a new pass.

Changing the judge model invalidates the record by construction — it grades a
grader that is no longer running — so the harness will not carry it forward: a run
whose pinned judge is not the record's `judge_model` logs the mismatch and
publishes no agreement at all, which is what "not calibrated yet" honestly looks
like. Redo the pass on the first run under the new judge.

The record it writes:

```json
{
  "judge_model": "deepseek/deepseek-v3.2",
  "reviewed_by": "<name>",
  "reviewer_kind": "human",
  "reviewed_at": "2026-07-25T00:00:00+00:00",
  "items": [
    { "case_id": "gen-fianza", "kind": "key_point", "ref": "BOE-A-1994-26003:a36", "judge_label": true, "reviewer_label": true },
    { "case_id": "gen-fianza", "kind": "claim", "ref": "…", "judge_label": false, "reviewer_label": false }
  ]
}
```

- `kind` — `key_point` or `claim`.
- `ref` — the key point's `block_ref`, or the claim text.
- `reviewer_kind` — `human` or `model`; a record without it does not load.
- The agreement is derived from `items` at load time, never trusted from the file.

Disagreements are inspected → the judge prompt is adjusted → the pass is repeated.
