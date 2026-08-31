# Offline benchmark methodology

FacetRoute performs counterfactual replay: a policy selects a model for each
request, and the runner retrieves that model's outcome from the same trace
row. It never estimates a missing outcome or calls a provider.

## Comparability

Rule, Pareto, and online LinUCB arms receive the identical ordered request
sequence and shared catalog. LinUCB starts fresh in the CLI and updates only
after the selected outcome is revealed. Fixed-model baselines intentionally
ignore constraints; violations are measured instead of hidden.

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

## Reproducibility manifest

Every CLI report records the ordered trace byte SHA-256, canonical catalog
SHA-256, input configuration digests, seed, bootstrap count, confidence level,
record count, policy names, and installed FacetRoute version. Version reports
alongside the collection protocol and exact model revisions.
