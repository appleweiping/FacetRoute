# Changelog

Notable changes are recorded here. Versions follow semantic versioning.

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
