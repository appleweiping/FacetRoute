# Offline benchmark methodology

## Public benchmark format normalization

`facetroute normalize-benchmark` is a schema adapter, not a model evaluator. It
accepts the record shapes used by MMLU (`question`, `choices`, `answer`), GSM8K
(`question`, `answer`), and MT-Bench (`question_id`, `turns`, optional
`category`) in JSON or JSONL form. It rejects duplicate IDs, malformed choices,
missing answers, mixed formats, oversized files, and excessive record counts.
The canonical JSONL output keeps the reference answer outside the generated
`RouteRequest.query`, so a later evaluator can score a provider response without
answer leakage. The resulting request ID is stable (`format:example-id`) and
can be joined to a separately hashed counterfactual trace.

FacetRoute performs counterfactual replay: a policy selects a model for each
request, and the runner retrieves that model's outcome from the same trace
row. It never estimates a missing outcome or calls a provider.

## Comparability

Rule, Pareto, online LinUCB/Thompson, and optional loaded similarity and
factorization arms
receive the identical ordered request sequence and shared catalog. The online
arms start fresh in the CLI and update only after the selected outcome is
revealed. Similarity and factorization are never refit by `benchmark`; the
exact artifact digests are part of the input manifest. Fixed-model baselines intentionally ignore
constraints; violations are measured instead of hidden.

Quality regret uses the best observed quality among candidates passing the
shared hard-constraint engine. An ineligible fixed selection has no regret for
that row and increments the separate violation metric.

## Confidence intervals

For each policy and metric, rows are sampled with replacement using a stable
policy-specific random stream derived from the manifest seed. Reports use
percentile intervals. Averages are conditional on routed rows; failure and
violation rates use all rows. P95 uses the nearest-rank definition.

These intervals quantify resampling variation inside the supplied trace. They
do not account for biased judges, missing counterfactuals, correlated users,
temporal drift, or repeated tuning on the same holdout.

Use `split-traces` before examining results. Fit an upstream score model or
similarity prototype or factorization weights only on `train`, choose a
threshold or model hyperparameters only on `calibration`,
and report the untouched `test` result. If rows from one user, task family,
model pair, or source can be correlated, select that stable key with
`--group-by` rather than accepting the request-level default.

## Reproducibility manifest

Every CLI report records `dataset_file_sha256` for the exact trace byte stream
and `dataset_canonical_sha256` for the canonical parsed records in execution
order. It also records the canonical catalog SHA-256, input configuration
digests, seed, bootstrap count, confidence level, record count, policy names,
and installed FacetRoute version. Version reports alongside the collection
protocol and exact model revisions. Provenance inputs are parsed and hashed in
one bounded read/stream, so replacing a path between parsing and a second hash
cannot produce a manifest for bytes the run did not consume. Output paths are
checked against every input before any benchmark artifact is written. The
canonical trace hash preserves row order because online policies consume that
order; similarity training separately documents its request-ID canonical sort.
