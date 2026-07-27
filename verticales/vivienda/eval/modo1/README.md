# Mode 1 evaluation cases (vivienda)

One JSON file per case, versioned and reviewable in a pull request. Each case is a
Spanish user question plus the corpus blocks a correct answer must ground on.

```json
{
  "id": "02-fianza",
  "question": "…",
  "gold_block_refs": ["BOE-A-1994-26003:a36"],
  "expected_outcome": "respuesta",
  "key_points": [
    {
      "claim": "la fianza obligatoria es de una mensualidad de renta",
      "block_ref": "BOE-A-1994-26003:a36"
    }
  ],
  "target_date": "2020-01-01"
}
```

- `id` — unique, sorts the run order.
- `question` — the user's natural-language question.
- `gold_block_refs` — the corpus blocks the answer should cite, each a
  norm-qualified `<norm_id>:<block_id>` reference; the reference recall is measured
  against them. Empty for an out-of-scope case. A bare block id is rejected at
  load: the corpus holds several norms and each of them has an `a1`.
- `expected_outcome` — optional, one of the Mode 1 outcomes
  (`respuesta`, `respuesta_parcial`, `abstencion`, `rechazo_router`).
- `key_points` — optional; the 2–5 legal claims a good answer must contain, each
  tied to one of the case's gold blocks. They are the reference the with-reference
  judge grades completeness and per-claim support against. A `block_ref` outside
  `gold_block_refs` is rejected at load.
- `target_date` — optional ISO date pinning the point-in-time clock for a
  time-sensitive case; the run answers and re-verifies its citations at that date.
  Absent (or `null`) means the run's default date.

The generated-by-construction set (`gen-*.json`) walks the corpus backwards to
synthesise questions whose gold blocks and key points are known by construction,
and produces this same file shape.

The multi-hop set (`mh-*.json`) is hand-authored and hand-reviewed: each case names
gold blocks in **more than one norm**, so a correct answer cannot be reached by
recovering a single article. They are what the corpus expansion is measured with —
the LAU sends the reader outside itself (art. 4 and 21 to the Código Civil, arts. 10
and 17 to the Ley por el derecho a la vivienda, art. 27 to the eviction process in
the LEC), and these cases follow those references.
