# Constructed reference set (vivienda)

No public dataset covers Spanish tenancy law under this project's risk taxonomy, so
the evaluation reference set is generated **by construction** and filtered by a human
reviewer. The label is correct by construction; the human accepts, rejects or edits
rather than authors. Every accepted case carries its own provenance, so the set is
versioned together with where each case came from.

## Inputs (curated, versioned)

- `clause_bank.json` — labelled Mode 2 clauses, the ground truth the assembler samples
  from. Illegal and worse-than-default clauses are seeded from the official Consumo
  report on abusive tenancy clauses; conforming clauses from public templates;
  negotiable clauses for burdens the LAU is silent on. Every clause carries its
  expected level and the checklist items it reflects. `refset validate-bank` fails if
  the bank stops covering the checklist.
- `seeds.json` — Mode 1 query seeds. Each names the corpus blocks that are the gold
  blocks *by construction* and the outcome the case should reach; the generator writes
  the user-language question and extracts the key points from the blocks' own text.
- `recipes.json` — Mode 2 contract recipes. Each lists the bank clauses to assemble
  into one synthetic contract, in order.

## The review flow

```
refset generate-queries --vertical vivienda   # seeds  -> pending Mode 1 candidates
refset assemble         --vertical vivienda   # recipes -> pending Mode 2 candidates
refset review list      --vertical vivienda   # what is awaiting review
refset review accept    --vertical vivienda --id <id> --by <name> [--note ...]
refset review reject    --vertical vivienda --id <id> --by <name> --reason ...
```

Generation writes **pending** candidates under `candidates/`. Nothing reaches the set
until `review accept` materializes it, stamping the reviewer and time onto its
provenance. Accepted Mode 1 cases land in `../eval/modo1/` (consumed directly by
`eval`); accepted Mode 2 contracts land in `modo2/` here.

## Outputs

- `candidates/` — every generated candidate with its review status (the audit trail).
- `modo2/` — accepted synthetic contracts. Each carries per-clause ground truth with
  the exact character span of every clause (the segmentation reference comes free from
  construction), the expected level per clause, and the deliberately omitted checklist
  protections as expected whites (⚪).
