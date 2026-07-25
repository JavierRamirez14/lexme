# Mode 1 evaluation cases (vivienda)

One JSON file per case, versioned and reviewable in a pull request. Each case is a
Spanish user question plus the corpus blocks a correct answer must ground on.

```json
{
  "id": "01-plazo-minimo",
  "question": "…",
  "gold_block_ids": ["a9"],
  "expected_outcome": "respuesta"
}
```

- `id` — unique, sorts the run order.
- `question` — the user's natural-language question.
- `gold_block_ids` — corpus block ids the answer should cite; the reference recall
  is measured against. Empty for an out-of-scope case.
- `expected_outcome` — optional, one of the Mode 1 outcomes
  (`respuesta`, `respuesta_parcial`, `abstencion`, `rechazo_router`).

This is the minimal hand-authored starter set. The generated-by-construction set,
which walks the corpus backwards to synthesise questions with gold blocks known by
construction, arrives with its own ticket and produces the same file shape.
