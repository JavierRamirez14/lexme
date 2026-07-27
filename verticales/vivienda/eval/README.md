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
judge (~1 structured call per answered case, ~70 per run) sits on an OpenRouter
free open model of another family. The shipped judge is
`openai/gpt-oss-20b:free`. A `:free` id is not a stable contract:
OpenRouter retires them, and a retired id fails the run with a 404 rather than
degrading quietly, so confirm the pin is still in the catalog before a real run and
re-check both providers' live daily limits, which change often (blueprint asset 04,
caveats 1 and 2). Re-allocating a task to another provider is a one-line edit to
`tasks.json`, no code change.

## Judge calibration (one-time, human)

The judged numbers are only trustworthy read next to a human-judge agreement.
Calibration is a one-time pass, versioned as `judge-calibration.json` in this
directory; when present the harness recomputes the agreement from its reviewed
items and publishes it on the run artifact, and when absent the run reports the
judge as "not calibrated" rather than inventing a number.

Workflow:

1. Run the harness once and collect a sample (~50) of the judge's per-item rulings
   (key-point coverage and per-claim support).
2. A human reviews each ruling with the article in front of them and records the
   human label beside the judge label.
3. Save the reviewed sample as `judge-calibration.json`:

```json
{
  "judge_model": "openai/gpt-oss-20b:free",
  "reviewed_by": "<name>",
  "reviewed_at": "2026-07-25T00:00:00+00:00",
  "items": [
    { "case_id": "gen-fianza", "kind": "key_point", "ref": "a36", "judge_label": true, "human_label": true },
    { "case_id": "gen-fianza", "kind": "claim", "ref": "…", "judge_label": false, "human_label": false }
  ]
}
```

- `kind` — `key_point` or `claim`.
- `ref` — the key point's `block_id`, or the claim text.
- The agreement is derived from `items` at load time, never trusted from the file.

Disagreements are inspected → the judge prompt is adjusted → the pass is repeated.
