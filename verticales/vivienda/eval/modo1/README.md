# Mode 1 evaluation cases (vivienda)

One JSON file per case, versioned and reviewable in a pull request. Each case is a
Spanish user question plus the corpus blocks a correct answer must ground on.

```json
{
  "id": "02-fianza",
  "question": "…",
  "gold_block_ids": ["a36"],
  "expected_outcome": "respuesta",
  "key_points": [
    { "claim": "la fianza obligatoria es de una mensualidad de renta", "block_id": "a36" }
  ],
  "target_date": "2020-01-01"
}
```

- `id` — unique, sorts the run order.
- `question` — the user's natural-language question.
- `gold_block_ids` — corpus block ids the answer should cite; the reference recall
  is measured against. Empty for an out-of-scope case.
- `expected_outcome` — optional, one of the Mode 1 outcomes
  (`respuesta`, `respuesta_parcial`, `abstencion`, `rechazo_router`).
- `key_points` — optional; the 2–5 legal claims a good answer must contain, each
  tied to one of the case's gold blocks. They are the reference the with-reference
  judge grades completeness and per-claim support against. A `block_id` outside
  `gold_block_ids` is rejected at load.
- `target_date` — optional ISO date pinning the point-in-time clock for a
  time-sensitive case; the run answers and re-verifies its citations at that date.
  Absent (or `null`) means the run's default date.

The generated-by-construction set (`gen-*.json`) walks the corpus backwards to
synthesise questions whose gold blocks and key points are known by construction,
and produces this same file shape.
