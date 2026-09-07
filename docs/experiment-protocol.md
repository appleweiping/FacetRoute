# Leakage-resistant route evaluation

FacetRoute can make a routing experiment auditable; it cannot make a biased
outcome log representative. A defensible study declares the population,
candidate model revisions, judge or task metric, prompt handling, collection
window, cost source, latency environment, failure policy, and missing-outcome
policy before examining the test result.

## 1. Freeze and validate the trace

Keep the original trace immutable. Record its source URI, license or access
terms, exact byte SHA-256, collection version, and any conversion script. Every
row needs a stable request ID and an observed outcome for each candidate a
policy may select. A preferred-model label and a route score answer different
questions and should not be substituted for one another.

## 2. Split by the unit that can leak

`split-traces` assigns a whole declared group to one partition, considering
larger groups first and using a seeded SHA-256 order to break equal-size ties.
`user_id` prevents one person's prompts from crossing folds;
`metadata:domain` can isolate task families; request-level grouping is only
valid when rows are independent. The manifest records requested and actual
fractions, absolute deviations, and group counts because unequal group sizes
make exact ratios impossible. It is emitted only after the source snapshot and
the deterministic partition outputs agree exactly; output paths may not alias
the source file.

The three roles are intentionally distinct:

- `train` fits an upstream score model or adaptive-policy configuration;
- `calibration` selects a threshold, cost bound, and other operating point;
- `test` is opened once for the final comparison.

FacetRoute does not train the supplied `route_score`. A study that imports
scores must separately prove they were produced without test labels.

## 3. Evaluate once

`calibrate --held-out-traces` first chooses the threshold from the calibration
curve, then applies exactly that threshold to held-out rows. Pass the same
leakage unit used to split through `--held-out-group-by`; it defaults to
`request_id`, while `user_id` and `metadata:<field>` audit broader units. Any
overlap at that unit and model-pair drift are rejected. The report retains both
dataset hashes, group key and counts, record counts, the selected threshold,
its complete calibration curve, and the held-out quality, cost, latency,
success, routing fraction, and optional label accuracy.

Run all benchmark policies against the same test trace, catalog, constraints,
seed, bootstrap count, and confidence level. Fixed baselines remain subject to
the same observed outcomes; their constraint violations are shown rather than
silently filtered.

## 4. Interpret the result

Seeded bootstrap intervals describe only resampling variation in the supplied
rows. They do not cover correlated groups unless the sampling unit reflects
those groups, judge error, provider drift, prompt contamination, or multiple
comparisons. Report every tried policy and operating point, not only the best
row. Runtime measured by counterfactual replay is evaluator runtime, not live
provider latency.
