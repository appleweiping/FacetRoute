# Counterfactual trace schema

FacetRoute calibration and benchmark inputs use newline-delimited JSON. One
non-empty line is one independent request. The reader is streaming and applies
a 1 MiB line limit and one-million-record limit by default.

## Top-level fields

| Field | Required | Meaning |
|---|---:|---|
| `request_id` | yes | Stable, unique ID; must agree with `request.request_id`. |
| `request` | yes | Native `RouteRequest` JSON, including local evaluation text. |
| `outcomes` | yes | Object keyed by declared model ID; never empty. |
| `preferred_model` | no | Human or task-metric label; must have an outcome. |
| `route_score` | pair calibration | Finite score in `[0, 1)`; larger favors strong. |
| `strong_model` | pair calibration | Selected when `route_score >= threshold`. |
| `weak_model` | pair calibration | Selected otherwise. |

Each outcome has exactly four fields: `quality` in `[0, 1]`, non-negative
`cost_usd`, non-negative `latency_ms`, and Boolean `success`. Numbers must be
finite. Unknown fields are rejected instead of silently ignored.

## Integrity rules

- Duplicate keys, `NaN`, infinity, invalid UTF-8, and non-object rows are
  rejected with source path and line number.
- Request IDs are stable and unique across the file.
- Strong and weak IDs are supplied together, differ, and both have outcomes.
- Every row in one calibration run uses the same ordered pair.
- CLI reports hash exact file bytes. Library reports hash canonical parsed
  traces when no external digest is supplied.
- Calibration and held-out files must have disjoint values for the declared
  `held_out_group_by` leakage key and the same ordered strong/weak model pair.
  The default leakage key is `request_id`; use `user_id` or
  `metadata:<field>` when requests share a person or task family.

Trace files may contain evaluation text. Keep private prompts and personal
data out of public fixtures; the evaluation operator owns access-control and
retention decisions.

## Canonical partition files

`split-traces` writes canonical UTF-8 JSONL with sorted object keys. Its
`manifest.json` records SHA-256 of the exact bytes parsed from the input and,
for each partition, the byte SHA-256, canonical parsed SHA-256, record count,
and filename. It also records the canonical source hash, requested and actual
fractions, absolute fraction error, and per-partition group counts. The writer
rejects an input path that aliases any output and refuses partitions that do
not exactly match the source and declared deterministic algorithm. The split
is deterministic for the same records, group keys, fractions, and seed. It
preserves source row order inside each partition.

Grouping by `user_id` or `metadata:<field>` requires a non-empty value on every
row. At least three distinct groups are required so train, calibration, and
test cannot be empty.
