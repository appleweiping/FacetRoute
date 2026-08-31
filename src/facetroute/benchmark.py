"""Counterfactual offline benchmarks for routing policies and fixed baselines."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .constraints import ConstraintEngine
from .errors import ConfigurationError
from .features import QueryFeatureExtractor
from .feedback import FeedbackEvent
from .routers import Router
from .traces import RouteTrace, TraceOutcome
from .types import ModelCandidate, RouteRequest, UserPreferences

Aggregate = Callable[[Sequence[float]], float]


def _facetroute_version() -> str:
    try:
        return version("facetroute")
    except PackageNotFoundError:  # pragma: no cover - unpacked source tree
        return "uninstalled"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


@dataclass(frozen=True, slots=True)
class IntervalEstimate:
    estimate: float | None
    lower: float | None
    upper: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {"estimate": self.estimate, "lower": self.lower, "upper": self.upper}


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    requests: int
    routed: int
    average_quality: IntervalEstimate
    average_cost_usd: IntervalEstimate
    average_latency_ms: IntervalEstimate
    p95_latency_ms: IntervalEstimate
    success_rate: IntervalEstimate
    average_quality_regret: IntervalEstimate
    failure_rate: IntervalEstimate
    constraint_violation_rate: IntervalEstimate
    selection_counts: Mapping[str, int]
    errors: Mapping[int, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "routed": self.routed,
            "average_quality": self.average_quality.to_dict(),
            "average_cost_usd": self.average_cost_usd.to_dict(),
            "average_latency_ms": self.average_latency_ms.to_dict(),
            "p95_latency_ms": self.p95_latency_ms.to_dict(),
            "success_rate": self.success_rate.to_dict(),
            "average_quality_regret": self.average_quality_regret.to_dict(),
            "failure_rate": self.failure_rate.to_dict(),
            "constraint_violation_rate": self.constraint_violation_rate.to_dict(),
            "selection_counts": dict(self.selection_counts),
            "errors": {str(key): value for key, value in self.errors.items()},
        }


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    dataset_sha256: str
    catalog_sha256: str
    records: int
    seed: int
    bootstrap_samples: int
    confidence_level: float
    policy_names: tuple[str, ...]
    input_sha256: Mapping[str, str]
    facetroute_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset_sha256": self.dataset_sha256,
            "catalog_sha256": self.catalog_sha256,
            "records": self.records,
            "seed": self.seed,
            "bootstrap_samples": self.bootstrap_samples,
            "confidence_level": self.confidence_level,
            "policy_names": list(self.policy_names),
            "input_sha256": dict(sorted(self.input_sha256.items())),
            "facetroute_version": self.facetroute_version,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    manifest: BenchmarkManifest
    policies: Mapping[str, BenchmarkMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "manifest": self.manifest.to_dict(),
            "policies": {
                name: metrics.to_dict() for name, metrics in sorted(self.policies.items())
            },
        }


@dataclass(slots=True)
class PolicySpec:
    """One benchmark arm: a router or a deliberately fixed candidate."""

    name: str
    router: Router | None = None
    fixed_model: str | None = None
    learn_online: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("benchmark policy name cannot be empty")
        if (self.router is None) == (self.fixed_model is None):
            raise ConfigurationError("policy must define exactly one of router or fixed_model")
        if self.fixed_model is not None and not self.fixed_model.strip():
            raise ConfigurationError("fixed_model cannot be empty")
        if self.learn_online and self.router is None:
            raise ConfigurationError("online learning requires a router")


@dataclass(slots=True)
class _Row:
    outcome: TraceOutcome | None
    regret: float | None
    failed: float
    violation: float


class BenchmarkRunner:
    """Replay policies against per-request counterfactual outcomes.

    This runner never calls a provider.  Reported quality is only as reliable
    as the supplied trace and is not presented as a live-system measurement.
    """

    def __init__(
        self,
        models: Iterable[ModelCandidate],
        preferences: Mapping[str, UserPreferences] | None = None,
        *,
        seed: int = 17,
        bootstrap_samples: int = 1_000,
        confidence_level: float = 0.95,
        extractor: QueryFeatureExtractor | None = None,
        constraints: ConstraintEngine | None = None,
    ) -> None:
        model_tuple = tuple(models)
        if not model_tuple:
            raise ConfigurationError("benchmark requires at least one model")
        identifiers = [model.model_id for model in model_tuple]
        if len(identifiers) != len(set(identifiers)):
            raise ConfigurationError("benchmark model identifiers must be unique")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigurationError("seed must be an integer")
        if (
            isinstance(bootstrap_samples, bool)
            or not isinstance(bootstrap_samples, int)
            or bootstrap_samples < 100
        ):
            raise ConfigurationError("bootstrap_samples must be an integer >= 100")
        if (
            isinstance(confidence_level, bool)
            or not isinstance(confidence_level, (int, float))
            or not math.isfinite(confidence_level)
            or not 0 < confidence_level < 1
        ):
            raise ConfigurationError("confidence_level must be between 0 and 1")
        self.models = model_tuple
        self.model_ids = frozenset(identifiers)
        self.preferences = dict(preferences or {})
        self.seed = seed
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.extractor = extractor or QueryFeatureExtractor()
        self.constraints = constraints or ConstraintEngine()

    def run(
        self,
        traces: Iterable[RouteTrace],
        policies: Iterable[PolicySpec],
        *,
        dataset_sha256: str | None = None,
        input_sha256: Mapping[str, str] | None = None,
    ) -> BenchmarkReport:
        trace_tuple = tuple(traces)
        if not trace_tuple:
            raise ConfigurationError("benchmark trace set cannot be empty")
        specs = tuple(policies)
        if not specs:
            raise ConfigurationError("benchmark requires at least one policy")
        names = [spec.name for spec in specs]
        if len(names) != len(set(names)):
            raise ConfigurationError("benchmark policy names must be unique")
        router_ids = [id(spec.router) for spec in specs if spec.router is not None]
        if len(router_ids) != len(set(router_ids)):
            raise ConfigurationError(
                "benchmark policies must use distinct router instances to avoid state leakage"
            )
        for spec in specs:
            if spec.fixed_model is not None and spec.fixed_model not in self.model_ids:
                raise ConfigurationError(
                    f"fixed policy {spec.name!r} references unknown model {spec.fixed_model!r}"
                )
        for trace in trace_tuple:
            unknown = set(trace.outcomes) - self.model_ids
            if unknown:
                raise ConfigurationError(
                    f"trace {trace.request.request_id} contains unknown models: {sorted(unknown)}"
                )
        metrics = {spec.name: self._run_policy(trace_tuple, spec) for spec in specs}
        dataset_digest = dataset_sha256 or self._trace_digest(trace_tuple)
        if (
            not isinstance(dataset_digest, str)
            or len(dataset_digest) != 64
            or any(char not in "0123456789abcdef" for char in dataset_digest)
        ):
            raise ConfigurationError("dataset_sha256 must be a lowercase SHA-256 digest")
        input_digests = dict(input_sha256 or {})
        for name, digest in input_digests.items():
            if not isinstance(name, str) or not name.strip():
                raise ConfigurationError("input digest names cannot be empty")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ConfigurationError(f"input digest for {name!r} is not lowercase SHA-256")
        return BenchmarkReport(
            manifest=BenchmarkManifest(
                dataset_sha256=dataset_digest,
                catalog_sha256=self._catalog_digest(),
                records=len(trace_tuple),
                seed=self.seed,
                bootstrap_samples=self.bootstrap_samples,
                confidence_level=self.confidence_level,
                policy_names=tuple(names),
                input_sha256=input_digests,
                facetroute_version=_facetroute_version(),
            ),
            policies=metrics,
        )

    def _preference(self, request: RouteRequest) -> UserPreferences:
        return self.preferences.get(request.user_id, UserPreferences(user_id=request.user_id))

    def _eligible_ids(self, request: RouteRequest) -> frozenset[str]:
        features = self.extractor.extract(request)
        result = self.constraints.filter(self.models, request, features, self._preference(request))
        return frozenset(model.model_id for model in result.eligible)

    def _run_policy(self, traces: tuple[RouteTrace, ...], spec: PolicySpec) -> BenchmarkMetrics:
        rows: list[_Row] = []
        errors: dict[int, str] = {}
        selections: dict[str, int] = {}
        for index, trace in enumerate(traces):
            eligible = self._eligible_ids(trace.request)
            oracle_values = [
                trace.outcomes[model_id].quality
                for model_id in eligible
                if model_id in trace.outcomes
            ]
            try:
                if spec.router is not None:
                    decision = spec.router.route(trace.request)
                    selected = decision.selected_model
                else:
                    assert spec.fixed_model is not None
                    selected = spec.fixed_model
                if selected not in self.model_ids:
                    raise ConfigurationError(f"policy selected unknown model {selected!r}")
                if selected not in trace.outcomes:
                    raise ConfigurationError(
                        f"trace has no outcome for selected model {selected!r}"
                    )
                outcome = trace.outcomes[selected]
                violation = float(selected not in eligible)
                regret = (
                    max(oracle_values) - outcome.quality
                    if oracle_values and selected in eligible
                    else None
                )
                if spec.learn_online and spec.router is not None:
                    update = getattr(spec.router, "update_feedback", None)
                    if not callable(update):
                        raise ConfigurationError(
                            f"policy {spec.name!r} does not support online feedback"
                        )
                    update(
                        FeedbackEvent(
                            request_id=trace.request.request_id,
                            user_id=trace.request.user_id,
                            model_id=selected,
                            reward=outcome.quality,
                            policy=decision.policy,
                            context_vector=decision.context_vector,
                            success=outcome.success,
                            latency_ms=outcome.latency_ms,
                            cost_usd=outcome.cost_usd,
                            tags={"source": "counterfactual-benchmark"},
                        )
                    )
                rows.append(
                    _Row(
                        outcome,
                        max(0.0, regret) if regret is not None else None,
                        0.0,
                        violation,
                    )
                )
                selections[selected] = selections.get(selected, 0) + 1
            except Exception as exc:
                errors[index] = f"{type(exc).__name__}: policy evaluation failed"
                rows.append(_Row(None, None, 1.0, 0.0))
        return self._metrics(spec.name, rows, selections, errors)

    def _metrics(
        self,
        policy_name: str,
        rows: list[_Row],
        selections: Mapping[str, int],
        errors: Mapping[int, str],
    ) -> BenchmarkMetrics:
        seed = self.seed ^ int.from_bytes(
            hashlib.sha256(policy_name.encode("utf-8")).digest()[:8], "big"
        )
        return BenchmarkMetrics(
            requests=len(rows),
            routed=sum(row.outcome is not None for row in rows),
            average_quality=self._interval(
                rows, lambda row: row.outcome.quality if row.outcome else None, _mean, seed
            ),
            average_cost_usd=self._interval(
                rows, lambda row: row.outcome.cost_usd if row.outcome else None, _mean, seed + 1
            ),
            average_latency_ms=self._interval(
                rows, lambda row: row.outcome.latency_ms if row.outcome else None, _mean, seed + 2
            ),
            p95_latency_ms=self._interval(
                rows,
                lambda row: row.outcome.latency_ms if row.outcome else None,
                lambda values: _percentile(values, 0.95),
                seed + 3,
            ),
            success_rate=self._interval(
                rows,
                lambda row: float(row.outcome.success) if row.outcome else None,
                _mean,
                seed + 4,
            ),
            average_quality_regret=self._interval(rows, lambda row: row.regret, _mean, seed + 5),
            failure_rate=self._interval(rows, lambda row: row.failed, _mean, seed + 6),
            constraint_violation_rate=self._interval(
                rows, lambda row: row.violation, _mean, seed + 7
            ),
            selection_counts=dict(sorted(selections.items())),
            errors=dict(errors),
        )

    def _interval(
        self,
        rows: list[_Row],
        value: Callable[[_Row], float | None],
        aggregate: Aggregate,
        seed: int,
    ) -> IntervalEstimate:
        observed = [item for row in rows if (item := value(row)) is not None]
        if not observed:
            return IntervalEstimate(None, None, None)
        estimate = aggregate(observed)
        randomizer = random.Random(seed)
        samples: list[float] = []
        for _ in range(self.bootstrap_samples):
            sampled_rows = [rows[randomizer.randrange(len(rows))] for _ in rows]
            sampled_values = [item for row in sampled_rows if (item := value(row)) is not None]
            if sampled_values:
                samples.append(aggregate(sampled_values))
        if not samples:
            return IntervalEstimate(estimate, estimate, estimate)
        alpha = (1.0 - self.confidence_level) / 2.0
        return IntervalEstimate(
            estimate=estimate,
            lower=_percentile(samples, alpha),
            upper=_percentile(samples, 1.0 - alpha),
        )

    def _catalog_digest(self) -> str:
        canonical = json.dumps(
            [model.to_dict() for model in sorted(self.models, key=lambda item: item.model_id)],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _trace_digest(traces: tuple[RouteTrace, ...]) -> str:
        digest = hashlib.sha256()
        for trace in traces:
            canonical = json.dumps(
                trace.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            digest.update(canonical)
            digest.update(b"\n")
        return digest.hexdigest()
