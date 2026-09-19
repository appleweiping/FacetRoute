# Similarity router contract

`SimilarityRouter` is a supervised route classifier built only from strict
`RouteTrace` records. It is intended for operators who have labelled examples
but do not want an embedding service, a model download, or a new runtime
dependency. Every training, calibration, evaluation, and routing operation is
local.

## Data boundary

Every row used by the learner must have a `preferred_model`, and that identifier
must have an observed outcome in the trace. The CLI consumes separate files:

- `--train-traces` fits the vocabulary, IDF weights, and route prototypes;
- `--calibration-traces` chooses one similarity threshold;
- `--held-out-traces`, when supplied, is evaluated once at that fixed threshold.

`--group-by` accepts `request_id`, `user_id`, or `metadata:<field>`. A group may
occur in exactly one partition. Request IDs must additionally be globally
disjoint across all partitions, even when a broader grouping key is selected.
Training order is canonicalized by request ID, so row order does not change the
model; duplicate request IDs within a partition are also rejected.
Labels absent from the fitted routes are rejected as route drift rather than
mapped to a different model. The artifact records canonical training and
calibration SHA-256 digests, while the report also records the held-out digest
and group counts.

The threshold sweep contains zero, one, and every observed top-cosine value in
the calibration partition. A point covers a row when `cosine >= threshold`.
Only points meeting `--minimum-coverage` are eligible. Selection maximizes
preference-label accuracy, then coverage, then uses the lower threshold as a
stable final tie-break. Accuracy is conditional on covered rows; the complete
curve is retained so coverage is never hidden.

## Feature schema 1

The extractor case-folds Unicode word tokens and counts at most the configured
number of tokens. A term count `n` receives sublinear weight `1 + log(n)`.
Its difficulty calculation's `long_query_characters` scale is part of the
persisted feature configuration. Similarity APIs accept only the built-in
`QueryFeatureExtractor` with that exact scale; subclasses and mismatched
instances are rejected rather than silently changing the feature space.
Versioned structured features represent:

- inferred or supplied task;
- required capabilities;
- request sensitivity and tool/JSON requirements;
- difficulty, code fraction, math fraction, bounded character/token length,
  question count, and multi-step language;
- a constant bias that guarantees a nonzero cold-start vector.

Neither user ID, request ID, metadata values, region, outcome, cost, latency,
quality, nor preferred label enters the request vector. Vocabulary selection is
deterministic: features below `min_document_frequency` are removed, remaining
features are ordered by document frequency and name for truncation, then stored
in lexical order. For `N` training rows and document frequency `df_j`:

```text
idf_j = log((1 + N) / (1 + df_j)) + 1
q_j   = raw_j * idf_j
q     = q / ||q||_2
```

For route `r`, its prototype is the L2-normalized arithmetic mean of the
normalized request vectors carrying label `r`. Inference is the ordinary dot
product of two unit vectors:

```text
similarity(r, q) = sum_j prototype[r, j] * q[j]
```

All current feature values are nonnegative, so scores are clamped to `[0, 1]`
only to absorb floating-point round-off. Ranking uses descending score and then
ascending model ID. An explanation lists at most the requested 20 largest
individual products. The cosine always uses the full vector; the displayed
products are intentionally a bounded, potentially partial explanation and need
not sum to the score.

## Constraint and fallback order

The online order is fixed:

1. reject a query outside the persisted resource envelope;
2. extract the existing routing features needed by hard constraints;
3. filter disabled/blocked models and enforce capability, tool/JSON, context,
   region, restricted-locality, cost, latency, and quality bounds;
4. compute similarities only for prototype IDs still eligible;
5. use the best prototype when it reaches the persisted threshold;
6. otherwise use the existing rule-aware multi-objective scorer over the
   eligible catalog models.

A prototype never bypasses a hard constraint. A catalog model added after
training can participate in fallback but has no invented prototype. A state
file containing a route absent from the current catalog is rejected, making
catalog/model drift explicit.
When a prototype wins, `RouteDecision.score` and `breakdown.total` contain its
cosine and alternative scores use the same scale; the individual quality/cost/
latency fields remain operational audit data. Soft preference and rule bonuses
are zeroed because they did not determine that selection. On fallback, the
ordinary objective totals and bonuses retain their existing meaning.

## Persistence and limits

State schema 1 contains the format and feature-schema identifiers, complete
feature configuration, sorted vocabulary, IDF vector, route counts and
normalized prototypes, threshold, training/calibration digests, and
`state_sha256`. The latter hashes canonical JSON for every other field.
`save()` encodes and checks the complete UTF-8 payload against the same default
512 MiB envelope used by `load()` before touching the destination. This finite
limit is derived from schema 1's maxima: 100,000 route identifiers and 8,192
feature names of at most 512 Unicode scalars, plus 250,000 numeric prototype
cells. Counting six bytes for every string scalar, 256 bytes of syntax per
route, 64 bytes per numeric/feature entry, and 1 MiB of fixed overhead gives a
conservative ceiling of 376,062,976 bytes. The 512 MiB default therefore leaves
160,807,936 bytes of margin for every state accepted by the public schema.
Construction first snapshots nested strings, integers, floats, tuples, mappings,
and feature configuration into exact built-in immutable values, so subclass
hooks or later changes to caller-owned containers cannot expand or alter the
accepted state. The writer uses same-directory staging files, flushes and syncs
them, validates the exact staged model, then atomically replaces the destination.
When a training report is requested, model and report form a recoverable bundle:
a write or replacement failure restores every prior destination. The CLI rejects
input/output identity collisions before reading or writing.

Loading rejects duplicate JSON keys, every non-finite value spelling, unpaired
Unicode surrogates, nesting beyond 256 containers, missing or unknown fields,
unsupported schema versions, checksum mismatch, unnormalized/wrong-size
vectors, invalid IDF or counts, and files above the read limit. Training and
inference also bound records, query characters, tokens, categories, feature
name length, vocabulary size, aggregate document-feature occurrences, and total
prototype-index cells. Fitting keeps documents sparse and accumulates only the
bounded route-by-feature prototype state; it never constructs a
record-by-vocabulary matrix. Calibration ranks each row once, groups equal
scores, and computes every `score >= threshold` operating point with cumulative
counts. Arithmetic checks fail closed on zero or non-finite norms.

The payload checksum is corruption and tamper evidence, not authentication: a
party able to replace the artifact can recompute it. Use a signed release,
authenticated artifact store, and ordinary filesystem access controls when the
model's origin is security-sensitive. The model also stores vocabulary terms,
which can reveal words present in training data; review, minimize, and protect
the artifact under the same policy as the trace.

## Minimal API

```python
from facetroute import (
    SimilarityModel,
    SimilarityRouter,
    fit_calibrate_evaluate,
)

model, report = fit_calibrate_evaluate(
    training_traces,
    calibration_traces,
    held_out_traces,
    group_by="user_id",
    minimum_coverage=0.5,
)
model.save("similarity-model.json")

router = SimilarityRouter(catalog, SimilarityModel.load("similarity-model.json"))
decision = router.route(request)
```

The counterfactual benchmark accepts this same artifact through
`--similarity-model`. Its file digest is included in the benchmark input
manifest, making a result reproducible against the exact learned index. Every
configuration, trace, and similarity digest is computed from the same bounded
byte stream that was parsed for execution, rather than by reopening a mutable
path after the benchmark. Benchmark manifests distinguish
`dataset_file_sha256`, the exact trace bytes, from
`dataset_canonical_sha256`, the canonical parsed records in execution order.
Similarity train/calibration/held-out digests are canonical semantic hashes
after the documented request-ID ordering and are labelled accordingly in the
held-out report.
