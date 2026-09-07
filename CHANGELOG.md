# Changelog

Notable changes are recorded here. Versions follow semantic versioning.

## [Unreleased]

### Added

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
