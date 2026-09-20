# Offline threshold sweep and cache

`facetroute benchmark-sweep` evaluates a strong/weak routing score against
*observed counterfactual* outcomes in strict local traces. It does not call a
provider, download a public benchmark, or claim parity with an official MMLU,
GSM8K, or MT-Bench score. Each trace needs the same `strong_model`,
`weak_model`, both observed outcomes, and a `route_score` in `[0, 1)`.

```powershell
facetroute benchmark-sweep --models examples/models.json `
  --traces examples/traces.jsonl --bins 10 `
  --cache-dir .facetroute-sweep-cache --output sweep.json
```

The first invocation computes the sweep and stores an aggregate-only cache;
the second reports a cache hit and writes the same report bytes. Use
`--refresh-cache` to replace an invalid or outdated cache entry after
investigation. For exact cutoffs, repeat `--threshold 0.25 --threshold 0.75`;
the always-strong `0` and always-weak `1` endpoints are included automatically.
Otherwise `--bins` chooses a deterministic score-quantile grid, with duplicate
cutoffs removed. At a cutoff the strong model is selected when
`route_score >= threshold`.

Every point reports strong-call fraction, average observed quality, cost,
latency, success, gains relative to the always-weak baseline, and cost/quality
Pareto membership. `optimal_quality_at_same_strong_calls` is an *oracle upper
bound* that assigns the strong model to the rows with the largest positive
observed quality improvements, up to that point's strong-call count. It is not
an achievable router and it is not a dollar-budget optimum; the actual costs
may differ across requests. Its gap to the routed quality is reported as
`quality_regret_to_call_budget_oracle`.

The cache key includes schema version, canonical and exact-file SHA-256 of the
traces and model catalog, and a SHA-256 of all sweep configuration values. A
different dataset, catalog, or cutoff configuration creates a different entry.
Cache JSON has a strict schema and checksum; corruption fails closed instead
of silently changing a result. A checksum is not an authenticity signature:
only use cache directories you trust. Cache files contain aggregate metrics
and hashes, never queries, raw responses, or access tokens. Reads and writes
are bounded (32 MiB trace input, 64 KiB trace line, 10,000 records, 101 sweep
points, 4 MiB cache document); writes are staged and atomically replaced.

For a leakage-safe claim, use `split-traces` first: train any route-score model
on training data, choose cutoffs only on calibration data, and report one
preselected cutoff on untouched test data. Repeatedly choosing the best test
point would invalidate an unbiased held-out claim. Public response scoring and
contamination filtering are separate, still-unimplemented workflows.
