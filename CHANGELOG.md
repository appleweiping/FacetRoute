# Changelog

Notable changes are recorded here. Versions follow semantic versioning.

## [Unreleased]

_No changes yet._

## [0.6.0] - 2026-09-19

### Added

- A dependency-free, trainable similarity router using bounded lexical and
  request features, TF-IDF weighting, and normalized per-route prototypes.
- Leakage-audited fit/calibration/held-out evaluation with deterministic
  threshold sweeps, minimum coverage, group-disjoint enforcement, and dataset
  digests.
- Versioned, atomic similarity state containing the feature vocabulary, IDF,
  prototype index, route counts, configuration, training/calibration digests,
  and a canonical payload integrity checksum.
- `train-similarity`, `--policy similarity`, and optional similarity arms in
  the counterfactual benchmark, with end-to-end CLI and benchmark coverage.
- Sparse two-pass prototype fitting and a grouped cumulative threshold sweep,
  avoiding record-by-vocabulary matrices and repeated calibration ranking.
- A branch-only coverage gate in CI and release workflows, distinct from
  pytest-cov's combined statement/branch percentage.

### Security

- Similarity inference preserves the hard-constraint boundary: capability,
  context, cost, latency, region, block-list, and restricted-data locality
  filtering completes before the learned index sees an eligible route set.
- Model/query/feature/record/state sizes and all persisted numeric values are
  bounded; malformed, duplicate-key, non-finite, corrupt, or schema-mismatched
  state fails closed.
- Similarity extractor settings are fully persisted and enforced. State writes
  are size-checked before atomic replacement, training and benchmark inputs may
  not alias outputs, and benchmark digests come from the same bounded snapshots
  that were parsed for execution.
- The finite default similarity-state envelope is derived from the complete
  public schema and closes default save/load for every accepted state. Model
  state is detached into exact built-in immutable snapshots, including inputs
  supplied through container/scalar/configuration subclasses. Extreme numeric
  and recursive JSON failures are normalized to domain errors. Strict inputs
  reject exponent-overflow non-finite numbers, unpaired Unicode surrogates, and
  nesting beyond 256 containers consistently across supported Python/platform
  combinations. Interrupted atomic writes clean their temporary files and
  preserve the prior target before the replacement boundary. Benchmark manifests
  separate raw-file from canonical trace digests.
- Public benchmark-format input is bounded by bytes actually read, including
  a file that grows after an earlier metadata check; invalid UTF-8 and
  non-integer resource limits fail with explicit errors.

## [0.5.0] - 2026-09-07

### Added

- Optional OpenAI-compatible `POST /v1/chat/completions` execution after the
  selected catalog model is resolved through a fixed provider registry.
- A typed, injectable provider protocol and zero-dependency HTTP executor with
  non-streaming JSON and bounded SSE support.
- Environment-only provider credentials, fixed upstream model mappings,
  loopback-only plain HTTP by default, response/event byte limits, redacted
  upstream errors, and routed model/policy response headers.
- End-to-end tests through both injected fake providers and a loopback
  OpenAI-compatible fixture server; no API key or external network is required.

### Changed

- `facetroute serve` can load provider bindings with `--providers`; without the
  option it remains the previous decision-only service.

## [0.4.0] - 2026-09-07

### Added

- Public benchmark format normalization for MMLU, GSM8K, and MT-Bench with
  stable `format:example-id` request keys and answer-leakage protection.

## [0.3.2] - 2026-09-07

### Fixed

- Synchronized the citation metadata with the released package version and
  added a regression gate that keeps `pyproject.toml`, the source fallback,
  `CITATION.cff`, and release tags consistent.

## [0.3.1] - 2026-09-07

### Security

- Added a final CR/LF-removal guard at the HTTP response boundary for
  `X-Request-ID`. The request parser already rejected control characters; the
  sink-side guard also protects future internal callers and makes the
  response-splitting invariant explicit to static analysis.

## [0.3.0] - 2026-09-07

### Added

- Added a tag-gated release pipeline with locked builds, clean wheel and sdist
  installation checks, CycloneDX SBOM, SHA-256 manifest, and GitHub provenance.
- Deterministic, size-aware, group-disjoint trace partitioning with explicit
  dataset name, source URI, license, seed, requested/actual fractions and
  deviations, group counts, and input/partition SHA-256 provenance. The writer
  rejects source/output aliases and partitions inconsistent with their source.
  Groups can be request IDs, users, or a declared metadata field.
- Held-out threshold evaluation: calibration selects the operating point, then
  applies it once to a disjoint trace. Overlap at a declared request, user, or
  metadata leakage unit and model-pair drift are rejected. Both dataset hashes,
  group audit, and complete held-out metrics are retained in report schema 2;
  calibration without a held-out set remains wire-compatible schema 1.
- Canonical trace writing/fingerprinting, a documented no-leakage experiment
  protocol, CLI end-to-end coverage, and group-invariance tests.

- `--policy thompson`: posterior sampling over the same per-arm state LinUCB already
  maintains. LinUCB explores by optimism, scoring every arm at the top of its confidence
  interval; Thompson draws from each arm's posterior, so an arm is tried roughly in
  proportion to the probability that it is best. `alpha` means the same thing to both and a
  saved state loads under either, so a deployment can switch without discarding what it has
  learned.
- The draw is a function of the seed, the arm and the context rather than of a mutable
  generator. The same request against the same state always samples the same value, so a
  routing log replays to the routes it recorded -- which a generator advanced per call could
  not do -- while a different request or seed still explores elsewhere.
- `benchmark` accepts `--policy thompson` alongside the others, which is the point: on one
  synthetic four-arm world over 1,200 rounds averaged across five worlds, neither policy
  dominates and the ordering flips with the exploration scale. LinUCB is ahead at its best
  `alpha` (32.1 against 37.2) and Thompson is ahead at `alpha=0.2` (38.7 against 41.2). The
  README publishes the whole sweep rather than the favourable row, and notes that the
  shipped default of 0.35 is best for neither.

## [0.2.0] - 2026-08-31

- Added strict counterfactual traces and strong/weak threshold calibration with
  cost-quality Pareto curves.
- Added reproducible multi-policy benchmarks with fixed baselines, online
  LinUCB replay, bootstrap intervals, constraint metrics, SHA-256 manifests,
  and JSON/CSV/standalone HTML reports.
- Added a bounded decision-only HTTP service with native and text-only
  chat-shaped parsing, optional bearer authentication, and structured errors.
- Hardened JSON against duplicate keys and non-finite constants and tightened
  request numeric types.

## [0.1.0] - 2026-08-31

- Added rule, Pareto, and LinUCB policies behind a shared hard-constraint engine.
- Added strict catalogs and profiles, explainable scoring, atomic bandit state, and append-only feedback.
- Added batch routing, offline simulation, CLI workflows, examples, and cross-platform CI.
