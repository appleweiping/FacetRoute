# FacetRoute

[![CI](https://github.com/appleweiping/FacetRoute/actions/workflows/ci.yml/badge.svg)](https://github.com/appleweiping/FacetRoute/actions/workflows/ci.yml)
[![CodeQL](https://github.com/appleweiping/FacetRoute/actions/workflows/codeql.yml/badge.svg)](https://github.com/appleweiping/FacetRoute/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](https://www.python.org/)
[![MIT](https://img.shields.io/badge/license-MIT-2ea44f.svg)](LICENSE)

FacetRoute is an original, offline-first Python library for choosing among a
declared set of large-language-model candidates. It treats routing as a
transparent decision problem: reject models that cannot satisfy the request,
score the remaining trade-offs, explain the result, and optionally learn from
local feedback.

It does **not** call an LLM provider, download a model, require an API key, or
send telemetry. The core package has no runtime dependencies beyond Python
3.11 or newer.

## Why this project exists

Model routing often becomes an opaque collection of price tables, provider
conditionals, and learned scores. FacetRoute separates those concerns:

- a model catalog describes capabilities and operating limits;
- a request declares non-negotiable constraints;
- a user profile declares quality, cost, and latency preferences;
- deterministic policies produce auditable decisions;
- a contextual bandit learns only after explicit feedback is recorded;
- counterfactual benchmarks and calibration measure behavior before a policy
  is used in an application.

FacetRoute selects a model identifier. Your application remains responsible
for authentication, prompts, provider calls, retries, and output validation.

## Architecture

```mermaid
flowchart LR
    Q[RouteRequest] --> F[QueryFeatureExtractor]
    C[Model catalog] --> H[Hard constraints]
    P[User profile] --> H
    F --> H
    H -->|eligible| S{Policy}
    H -->|rejected + reasons| X[Decision audit]
    R[Declarative rules] --> S
    S --> A[Rule score]
    S --> B[Pareto frontier]
    S --> U[LinUCB]
    A --> D[RouteDecision]
    B --> D
    U --> D
    D --> E[Offline simulation]
    D --> APP[Caller/provider adapter]
    APP --> FB[FeedbackEvent JSONL]
    E --> FB
    FB --> U
    FB --> REP[Report]
    T[Strict counterfactual trace] --> CAL[Threshold calibration]
    T --> BENCH[Policy benchmark + bootstrap CI]
    CAL --> REP
    BENCH --> REP
    D --> HTTP[Bounded local HTTP decision service]
```

The important invariant is that every policy receives only models that passed
the same hard-constraint engine. A rule or Bandit confidence interval can
change preferences; neither can make an ineligible model selectable.

## Features

- **Model candidate specification**: capabilities, per-task quality, input and
  output price, p50/p95 latency, context window, regions, tool/JSON support,
  enabled state, and inspectable metadata.
- **Deterministic query features**: task category, token approximation,
  difficulty, code/math signals, question count, multi-step language, and
  required capabilities. No embeddings or network calls are required.
- **Personalized objectives**: quality/cost/latency weights, task-specific
  overrides, preferred and blocked models, cost and latency budgets, region,
  minimum quality, and exploration strength.
- **Hard constraints**: context capacity, required capabilities, tools,
  structured output, region, restricted-data locality, budgets, model state,
  and profile blocks.
- **Explainable scoring**: normalized cost and latency utility alongside raw
  task quality, explicit bonuses, ranked alternatives, rejected candidates,
  and human-readable reasons.
- **Three policies**:
  - `RuleRouter` applies serializable rules as bounded score bonuses;
  - `ParetoRouter` removes quality/cost/latency-dominated candidates;
  - `LinUCBRouter` learns per-model reward estimates and uncertainty online.
- **Local state**: atomic versioned JSON for Bandit/profile state and
  append-only JSONL feedback suitable for inspection and replay.
- **Strict counterfactual traces**: duplicate-key and non-finite-number
  rejection, stable request IDs, bounded line/record sizes, complete observed
  outcome validation, and SHA-256 provenance.
- **Calibration and policy benchmarks**: strong/weak score thresholds,
  cost-quality Pareto curves, fixed-model baselines, rule/Pareto/LinUCB
  comparisons, quality regret, constraint violations, and seeded bootstrap
  confidence intervals.
- **Portable reports**: deterministic JSON, analysis-ready CSV, and a
  standalone HTML table with an embedded reproducibility manifest.
- **Decision service**: standard-library `/health`, `/v1/models`, and
  `/v1/route` endpoints with bounded request bodies and concurrency, socket
  timeouts, optional bearer authentication, and structured error semantics.
- **CLI**: `route`, `simulate`, `feedback`, `report`, `calibrate`, `benchmark`,
  and `serve`.

## Install

```bash
python -m pip install -e .
```

Development tools are optional:

```bash
python -m pip install -e ".[dev]"
```

## Five-minute offline walkthrough

Route one request with the deterministic rule policy:

```bash
facetroute route \
  --models examples/models.json \
  --preferences examples/preferences.json \
  --rules examples/rules.json \
  --policy rule \
  --user analyst \
  --task math \
  --query "Derive the beta-binomial posterior step by step"
```

On PowerShell, use backticks instead of backslashes, or place the command on a
single line. The result is JSON containing the chosen identifier, the complete
score breakdown, alternatives, exclusions, matched rules, extracted features,
and an explanation.

Run a seeded simulation and learn after each synthetic observation:

```bash
facetroute simulate \
  --models examples/models.json \
  --preferences examples/preferences.json \
  --rules examples/rules.json \
  --queries examples/queries.jsonl \
  --policy linucb \
  --state bandit-state.json \
  --feedback-log feedback.jsonl \
  --learn \
  --seed 17 \
  --output simulation-report.json
```

Summarize recorded feedback:

```bash
facetroute report --log feedback.jsonl
```

Everything above is local. Names in `examples/models.json` are fictional and
the simulator does not contact them.

Calibrate a pairwise router and compare all policies against the included
counterfactual fixture:

```bash
facetroute calibrate \
  --traces examples/traces.jsonl \
  --max-average-cost 0.0025 \
  --output artifacts/calibration.json \
  --csv artifacts/cost-quality.csv

facetroute benchmark \
  --models examples/models.json \
  --preferences examples/preferences.json \
  --rules examples/rules.json \
  --traces examples/traces.jsonl \
  --bootstrap-samples 1000 \
  --seed 17 \
  --output-dir artifacts/benchmark
```

The benchmark writes `benchmark.json`, `benchmark.csv`, and a standalone
`benchmark.html`. The included trace is a fictional format demonstration, not
a published performance claim.

## Python API

```python
from facetroute import (
    ModelCandidate,
    ParetoRouter,
    RouteRequest,
    UserPreferences,
)

models = (
    ModelCandidate(
        model_id="small-local",
        display_name="Small Local",
        capabilities=frozenset({"text", "code"}),
        input_cost_per_million=0,
        output_cost_per_million=0,
        latency_ms_p50=70,
        latency_ms_p95=140,
        context_window=8_192,
        quality_by_task={"default": 0.55, "code": 0.68},
        regions=frozenset({"local"}),
        metadata={"local": True},
    ),
    ModelCandidate(
        model_id="deep-remote",
        display_name="Deep Remote",
        capabilities=frozenset({"text", "code", "reasoning"}),
        input_cost_per_million=2,
        output_cost_per_million=6,
        latency_ms_p50=500,
        latency_ms_p95=1_100,
        context_window=65_536,
        quality_by_task={"default": 0.86, "code": 0.92},
        regions=frozenset({"us", "eu"}),
    ),
)

profiles = {
    "sam": UserPreferences(
        user_id="sam",
        quality_weight=0.45,
        cost_weight=0.4,
        latency_weight=0.15,
    )
}

router = ParetoRouter(models, profiles)
decision = router.route(
    RouteRequest(
        query="Write a Python parser for this line format",
        user_id="sam",
        max_cost_usd=0.01,
    )
)

print(decision.selected_model)
print(decision.breakdown.to_dict())
print(decision.explanation)
```

### Batch routing

Every built-in router supports `route_many`; `BatchRouter` adapts any object
with `route(request)`:

```python
from facetroute import BatchRouter

result = BatchRouter(router).route(requests, fail_fast=False)
for decision in result.decisions:
    print(decision.request_id, decision.selected_model)
for index, error in result.errors.items():
    print("failed input", index, error)
```

The decision tuple retains input order. When `fail_fast=False`, failures are
keyed by original input index and successful requests continue.

## Catalog and request formats

The model catalog is a JSON list or an object containing `models`. Costs use
USD per million tokens, latencies use milliseconds, and quality is normalized
to `[0, 1]`:

```json
{
  "models": [{
    "model_id": "model-a",
    "display_name": "Model A",
    "capabilities": ["text", "reasoning", "json"],
    "input_cost_per_million": 0.5,
    "output_cost_per_million": 1.5,
    "latency_ms_p50": 250,
    "latency_ms_p95": 600,
    "context_window": 32768,
    "quality_by_task": {"default": 0.72, "reasoning": 0.81},
    "regions": ["us", "eu"],
    "supports_tools": false,
    "supports_json": true,
    "enabled": true,
    "metadata": {"local": false}
  }]
}
```

Simulation requests are one JSON object per line. Important optional fields:

- `user_id`, `request_id`, `task_hint`;
- `expected_output_tokens` and an explicit `context_tokens` override;
- `required_capabilities`, `needs_tools`, `needs_json`;
- `max_cost_usd`, `max_latency_ms`, and `region`;
- `sensitivity`: `normal`, `sensitive`, or `restricted`.

`restricted` is deliberately strict: only a candidate with
`metadata.local=true` passes. `sensitive` is a label applications may use in
their own rules; FacetRoute does not silently infer legal or privacy policy.
A request cannot override a different `required_region` in its user profile;
that conflict rejects every candidate instead of weakening the profile.

## Routing policies

FacetRoute ships exactly three policies, and they form one class hierarchy:
`ParetoRouter` and `LinUCBRouter` subclass `RuleRouter` and override only how
the eligible set is narrowed or re-scored. Every policy therefore runs the same
`ConstraintEngine` and the same `MultiObjectiveScorer` before it chooses, and
each decision records the policy that produced it. Select one with
`--policy rule` (the default), `--policy pareto`, or `--policy linucb` on
`route`, `simulate`, and `serve`; `benchmark` accepts the same three names plus
`fixed` for single-model baselines.

Only `linucb` holds state or changes with feedback. `rule` and `pareto` are
functions of the catalog, the profile, the rules, and the request alone, so the
same inputs always yield the same selection and the same score.

The same request under all three:

```bash
show='import json, sys
decision = json.load(sys.stdin)
print(decision["policy"], decision["selected_model"], round(decision["score"], 4))'

for policy in rule pareto linucb; do
  facetroute route \
    --models examples/models.json \
    --preferences examples/preferences.json \
    --rules examples/rules.json \
    --policy "$policy" \
    --user analyst \
    --task math \
    --query "Derive the beta-binomial posterior step by step" \
  | python -c "$show"
done
```

```text
rule marble-reasoner 0.829
pareto marble-reasoner 0.829
linucb marble-reasoner 0.8534
```

The projection only keeps the example short; `route` still prints the complete
decision JSON described above. All three pick `marble-reasoner` here. The
LinUCB total is higher because an untrained arm predicts reward `0.0000` and
adds its exploration bonus on top of the weighted deterministic prior, which is
visible in that decision's own `explanation` field.

### Rule policy

The rule policy performs four steps:

1. extract deterministic request features;
2. remove candidates that violate hard constraints;
3. normalize eligible cost and latency, then combine them with task quality;
4. apply matched user/model preference bonuses and choose by score, breaking
   exact ties by `model_id`.

Rules are data, not Python callbacks. A rule matches tasks, capabilities, and a
difficulty interval, then adds a declared bonus to listed eligible models.
This keeps configuration serializable and explanations repeatable.

### Pareto policy

For each eligible candidate, Pareto routing uses three objectives:

- maximize task quality;
- minimize estimated request cost;
- minimize p95 latency.

A model is dominated only when another model is at least as good on every
objective and strictly better on one. Multi-objective scoring chooses within
the resulting frontier. Equal points remain on the frontier.

### LinUCB policy

LinUCB maintains an independent linear reward model for every model identifier.
Its context is a bounded 16-value vector containing query difficulty, estimated
length, code/math/multi-step signals, task one-hot values, and normalized user
objective weights.

The selection value is:

```text
predicted reward + alpha × profile exploration × uncertainty
                 + prior_weight × deterministic score
```

The deterministic prior makes cold-start choices operationally sensible while
confidence encourages exploration. Updates use a Sherman–Morrison inverse
covariance update implemented with the Python standard library. Rewards must be
explicit finite values in `[0, 1]`.

State JSON stores inverse covariance matrices, reward vectors, update counts,
dimension, `alpha`, ridge strength, and a schema version. Save operations write
and fsync a temporary file in the same directory before `os.replace`.

## Feedback and replay

A `FeedbackEvent` records:

- stable event/request/user/model identifiers and timestamp;
- normalized reward and success flag;
- policy and the exact Bandit context vector;
- optional observed latency, cost, and string tags.

`FeedbackLog` rejects duplicate event IDs and reports malformed line numbers.
Because JSONL is append-only and provider-independent, teams can inspect,
filter, redact, or replay observations using ordinary tools.

Append feedback through the CLI:

```bash
facetroute feedback \
  --log feedback.jsonl \
  --request-id request-42 \
  --user sam \
  --model model-a \
  --reward 0.9 \
  --policy rule \
  --latency-ms 410 \
  --cost-usd 0.0012
```

LinUCB feedback always requires the decision's JSON `context_vector` through
`--context`. Pass an existing `--state` as well to update it immediately.

## Calibration traces

Each strict JSONL trace contains one provider-independent request and observed
counterfactual outcomes keyed by model ID. Pairwise calibration additionally
declares `strong_model`, `weak_model`, a `route_score` in `[0, 1)`, and an
optional human or task-metric `preferred_model` label. Threshold `t` chooses
the strong model when `route_score >= t`; threshold `1` always chooses weak.

```json
{
  "request_id": "eval-001",
  "request": {"query": "Local evaluation input", "request_id": "eval-001"},
  "outcomes": {
    "small": {"quality": 0.71, "cost_usd": 0.001, "latency_ms": 90, "success": true},
    "strong": {"quality": 0.89, "cost_usd": 0.012, "latency_ms": 510, "success": true}
  },
  "preferred_model": "strong",
  "route_score": 0.82,
  "strong_model": "strong",
  "weak_model": "small"
}
```

Input rejects duplicate JSON keys, `NaN`/infinity, unknown trace/outcome
fields, duplicate or unstable request IDs, inconsistent pairs, missing
outcomes, and oversized records. See [the trace schema](docs/trace-schema.md).

## Offline benchmark methodology

`benchmark` replays the same ordered traces through rule, Pareto, fresh online
LinUCB, and fixed-candidate policies by default. For each selection it looks up
the already observed outcome; it never makes a model call. Reports contain:

- quality, observed cost, latency, p95 latency, and success;
- quality regret against the best observed eligible candidate;
- routing failure and hard-constraint violation rates;
- selection counts and index-keyed, non-traceback errors;
- seeded percentile-bootstrap intervals;
- exact trace/configuration and canonical catalog SHA-256 digests.

Fixed baselines deliberately remain selectable when ineligible so their
violation rate is visible. Confidence intervals describe sampling uncertainty
inside the supplied trace, not biased labels or distribution shift. See
[the benchmark methodology](docs/benchmark-methodology.md).

## Local routing service

Start a decision-only endpoint:

```bash
facetroute serve \
  --models examples/models.json \
  --preferences examples/preferences.json \
  --rules examples/rules.json \
  --policy pareto \
  --host 127.0.0.1 \
  --port 8080

curl -s http://127.0.0.1:8080/v1/route \
  -H 'Content-Type: application/json' \
  -d '{"query":"Explain this proof","user_id":"analyst"}'
```

Authentication is optional on loopback. For a non-loopback bind, set a token
without placing it in process arguments:

```bash
export FACETROUTE_BEARER_TOKEN='replace-with-a-secret'
facetroute serve --models examples/models.json --host 0.0.0.0
```

The service accepts the native request or a text-only subset of an OpenAI
chat-shaped request. It returns a **routing decision**, never `choices`; it
does not forward the prompt, execute the model, or claim to be an OpenAI API
proxy. See [the HTTP contract](docs/http-api.md).

## Simulation metrics

The included simulator is designed for policy plumbing and regression tests,
not as evidence that a real model has a given quality. With a fixed seed it:

- derives a synthetic reward from declared task quality, difficulty, an
  explicit preferred-model bonus, and bounded noise;
- samples latency between declared p50 and p95;
- computes success, estimated cost, selection counts, quality regret, and p95;
- optionally appends each observation and updates LinUCB online.

For research evaluation, replace synthetic rewards with held-out human or task
metrics while preserving the same `FeedbackEvent` contract.

## Design choices and limitations

- Token counts are a deterministic character-based approximation unless the
  request supplies `context_tokens`. Provider billing should use observed token
  counts after execution.
- Declared quality is configuration, not a claim about a real model. Keep it
  versioned with the benchmark and population that produced it.
- Counterfactual comparison requires an outcome for the selected model.
  Missing outcomes are failures; FacetRoute does not impute observations.
- Bootstrap intervals assume the supplied rows form a useful empirical
  population. They cannot repair judge bias, temporal leakage, or repeated
  tuning against the same holdout.
- Linear contextual Bandits cannot represent every interaction. Their value
  here is inspectability, fast online updates, and a small dependency surface.
- JSONL appends are protected inside one process, not coordinated across a
  distributed fleet. Use a transactional event store when multiple processes
  write the same stream.
- User identifiers are opaque strings. FacetRoute does not collect attributes
  or decide which personalization is legally or ethically appropriate.
- A selected model is not executed. Network behavior stays in the caller.

## Privacy and safety defaults

- no outbound provider calls, downloads, telemetry, or prompt forwarding;
- the optional inbound service binds to loopback by default, reads only its
  named bearer-token environment variable, and logs no headers or bodies;
- no model/provider names embedded in core routing logic;
- no automatic logging—callers must explicitly create a `FeedbackLog`;
- no raw query text in `FeedbackEvent` by default;
- restricted requests require a catalog entry explicitly marked local;
- every rejected candidate and numeric decision component is returned.

These defaults reduce accidental disclosure, but applications still need
access control, retention limits, encryption, redaction, and jurisdictional
review appropriate to their context.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check src tests examples
mypy -p facetroute
pytest --cov=facetroute --cov-branch
python -m build
```

The test suite covers validation, deterministic feature extraction, every
constraint, score normalization, rules, Pareto dominance, batch errors,
LinUCB learning and persistence, feedback integrity, strict traces,
calibration, bootstrap benchmarking, reports, HTTP security/error boundaries,
simulation, configuration, and all seven CLI commands. Tests are offline and
use temporary directories.

See [CONTRIBUTING.md](CONTRIBUTING.md) for change and disclosure expectations.

## Algorithm reference

FacetRoute's code and interfaces are independently implemented. Its contextual
bandit uses the LinUCB update described by Li, Chu, Langford, and Schapire in
“A Contextual-Bandit Approach to Personalized News Article Recommendation”
(WWW 2010, DOI `10.1145/1772690.1772758`). The paper is cited for the algorithm;
no external project code is included.

## License

MIT. See [LICENSE](LICENSE).
Project policies and release history are in [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [CHANGELOG.md](CHANGELOG.md).
