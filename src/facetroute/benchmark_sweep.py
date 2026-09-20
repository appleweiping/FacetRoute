"""Deterministic, bounded counterfactual threshold sweeps with an integrity-checked cache.

The inputs are *observed* strong/weak outcomes.  Nothing here calls a model or
turns a public benchmark answer into a prompt.  A cache stores only aggregates,
not queries or raw responses.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ._json import loads_strict
from .config import _load_models_with_sha256
from .errors import ConfigurationError, PersistenceError
from .persistence import _atomic_write_bytes_bundle
from .traces import RouteTrace, _iter_trace_stream, traces_sha256
from .types import ModelCandidate

SWEEP_SCHEMA_VERSION = 1
MAX_SWEEP_RECORDS = 10_000
MAX_SWEEP_INPUT_BYTES = 32 * 1024 * 1024
MAX_SWEEP_LINE_BYTES = 64 * 1024
MAX_SWEEP_CACHE_BYTES = 4 * 1024 * 1024
MAX_SWEEP_POINTS = 101


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(value: str | None, label: str) -> None:
    if value is not None and (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ConfigurationError(f"{label} must be a lowercase SHA-256 digest")


def _paths_alias(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return left.resolve(strict=False) == right.resolve(strict=False)


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Quantile-grid or explicit cutoffs, with finite local input limits."""

    bins: int = 10
    thresholds: tuple[float, ...] | None = None
    max_records: int = MAX_SWEEP_RECORDS
    max_input_bytes: int = MAX_SWEEP_INPUT_BYTES
    max_line_bytes: int = MAX_SWEEP_LINE_BYTES

    def __post_init__(self) -> None:
        for label, value, ceiling in (
            ("bins", self.bins, MAX_SWEEP_POINTS - 1),
            ("max_records", self.max_records, MAX_SWEEP_RECORDS),
            ("max_input_bytes", self.max_input_bytes, MAX_SWEEP_INPUT_BYTES),
            ("max_line_bytes", self.max_line_bytes, MAX_SWEEP_LINE_BYTES),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
                raise ConfigurationError(f"{label} must be an integer in [1, {ceiling}]")
        if self.thresholds is not None:
            if (
                not isinstance(self.thresholds, tuple)
                or len(self.thresholds) > MAX_SWEEP_POINTS - 2
            ):
                raise ConfigurationError("thresholds must be a bounded tuple")
            values: list[float] = []
            for threshold in self.thresholds:
                if (
                    isinstance(threshold, bool)
                    or not isinstance(threshold, (int, float))
                    or not math.isfinite(threshold)
                    or not 0 <= threshold <= 1
                ):
                    raise ConfigurationError("thresholds must be finite numbers in [0, 1]")
                values.append(float(threshold))
            if len(values) != len(set(values)):
                raise ConfigurationError("thresholds must be unique")
            object.__setattr__(self, "thresholds", tuple(sorted(values)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "bins": self.bins,
            "thresholds": list(self.thresholds) if self.thresholds is not None else None,
            "max_records": self.max_records,
            "max_input_bytes": self.max_input_bytes,
            "max_line_bytes": self.max_line_bytes,
        }


@dataclass(frozen=True, slots=True)
class SweepPoint:
    threshold: float
    strong_calls: int
    strong_fraction: float
    average_quality: float
    average_cost_usd: float
    average_latency_ms: float
    success_rate: float
    quality_gain_over_weak: float
    incremental_cost_over_weak_usd: float
    optimal_quality_at_same_strong_calls: float
    quality_regret_to_call_budget_oracle: float
    pareto_optimal: bool = False

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "threshold": self.threshold,
            "strong_calls": self.strong_calls,
            "strong_fraction": self.strong_fraction,
            "average_quality": self.average_quality,
            "average_cost_usd": self.average_cost_usd,
            "average_latency_ms": self.average_latency_ms,
            "success_rate": self.success_rate,
            "quality_gain_over_weak": self.quality_gain_over_weak,
            "incremental_cost_over_weak_usd": self.incremental_cost_over_weak_usd,
            "optimal_quality_at_same_strong_calls": self.optimal_quality_at_same_strong_calls,
            "quality_regret_to_call_budget_oracle": self.quality_regret_to_call_budget_oracle,
            "pareto_optimal": self.pareto_optimal,
        }


@dataclass(frozen=True, slots=True)
class SweepReport:
    manifest: dict[str, Any]
    weak_baseline: dict[str, float]
    strong_baseline: dict[str, float]
    points: tuple[SweepPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SWEEP_SCHEMA_VERSION,
            "manifest": dict(self.manifest),
            "weak_baseline": dict(self.weak_baseline),
            "strong_baseline": dict(self.strong_baseline),
            "points": [point.to_dict() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class CachedSweep:
    report: SweepReport
    cache_hit: bool
    cache_path: Path


def load_sweep_traces(
    path: str | Path, *, config: SweepConfig | None = None
) -> tuple[tuple[RouteTrace, ...], str]:
    """Read a bounded byte stream once, parsing and hashing those exact bytes."""

    active = config or SweepConfig()
    source = Path(path)
    try:
        with source.open("rb") as stream:
            raw = stream.read(active.max_input_bytes + 1)
    except OSError as error:
        raise ConfigurationError(f"cannot read sweep traces {source}: {error}") from error
    if len(raw) > active.max_input_bytes:
        raise ConfigurationError("sweep trace file exceeds max_input_bytes")
    traces = tuple(
        _iter_trace_stream(
            source,
            io.BytesIO(raw),
            max_line_bytes=active.max_line_bytes,
            max_records=active.max_records,
        )
    )
    if not traces:
        raise ConfigurationError("sweep trace file contains no records")
    return traces, hashlib.sha256(raw).hexdigest()


def _model_digest(models: tuple[ModelCandidate, ...]) -> str:
    return _digest([model.to_dict() for model in sorted(models, key=lambda item: item.model_id)])


def _thresholds(traces: tuple[RouteTrace, ...], config: SweepConfig) -> tuple[float, ...]:
    if config.thresholds is not None:
        return tuple(sorted({0.0, 1.0, *config.thresholds}))
    scores = sorted(trace.route_score for trace in traces if trace.route_score is not None)
    count = len(scores)
    return tuple(
        sorted(
            {
                0.0,
                1.0,
                *(
                    scores[min(count - 1, index * count // config.bins)]
                    for index in range(1, config.bins)
                ),
            }
        )
    )


def _dominates(left: SweepPoint, right: SweepPoint) -> bool:
    return (
        left.average_quality >= right.average_quality
        and left.average_cost_usd <= right.average_cost_usd
        and (
            left.average_quality > right.average_quality
            or left.average_cost_usd < right.average_cost_usd
        )
    )


def _nonnegative_mean(values: list[float]) -> float:
    """Average finite nonnegative values without overflowing their sum."""

    scale = max(values)
    if scale == 0.0:
        return 0.0
    mean = math.fsum(value / scale for value in values) / len(values) * scale
    if not math.isfinite(mean):
        raise ConfigurationError("sweep aggregate must be finite")
    return mean


def _prepare_manifest(
    traces: tuple[RouteTrace, ...],
    models: tuple[ModelCandidate, ...],
    *,
    config: SweepConfig,
    trace_file_sha256: str | None = None,
    model_file_sha256: str | None = None,
) -> tuple[dict[str, Any], tuple[str, str]]:
    if not traces or len(traces) > config.max_records:
        raise ConfigurationError("sweep requires 1 to max_records traces")
    if not models or any(not isinstance(model, ModelCandidate) for model in models):
        raise ConfigurationError("sweep requires a non-empty ModelCandidate catalog")
    model_ids = [model.model_id for model in models]
    if len(model_ids) != len(set(model_ids)):
        raise ConfigurationError("sweep catalog model IDs must be unique")
    if any(not isinstance(trace, RouteTrace) for trace in traces):
        raise ConfigurationError("sweep traces must contain RouteTrace values")
    _sha256(trace_file_sha256, "trace_file_sha256")
    _sha256(model_file_sha256, "model_file_sha256")
    pair = (traces[0].strong_model, traces[0].weak_model)
    if pair[0] is None or pair[1] is None:
        raise ConfigurationError("sweep traces require a strong/weak pair and route_score")
    if pair[0] not in model_ids or pair[1] not in model_ids:
        raise ConfigurationError("sweep model pair must exist in the catalog")
    for trace in traces:
        if (trace.strong_model, trace.weak_model) != pair or trace.route_score is None:
            raise ConfigurationError("all sweep traces require the same scored strong/weak pair")
    manifest = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "records": len(traces),
        "strong_model": pair[0],
        "weak_model": pair[1],
        "trace_canonical_sha256": traces_sha256(traces),
        "trace_file_sha256": trace_file_sha256,
        "model_catalog_canonical_sha256": _model_digest(models),
        "model_catalog_file_sha256": model_file_sha256,
        "config_sha256": _digest(config.to_dict()),
        "config": config.to_dict(),
        "evaluation_universe": "observed_strong_weak_outcomes_only",
    }
    return manifest, (pair[0], pair[1])


def run_threshold_sweep(
    traces: tuple[RouteTrace, ...],
    models: tuple[ModelCandidate, ...],
    *,
    config: SweepConfig | None = None,
    trace_file_sha256: str | None = None,
    model_file_sha256: str | None = None,
) -> SweepReport:
    """Compare observed quality/cost for every cutoff and call-budget oracle."""

    active = config or SweepConfig()
    manifest, (strong_id, weak_id) = _prepare_manifest(
        traces,
        models,
        config=active,
        trace_file_sha256=trace_file_sha256,
        model_file_sha256=model_file_sha256,
    )
    count = len(traces)
    strong = [trace.outcomes[strong_id] for trace in traces]
    weak = [trace.outcomes[weak_id] for trace in traces]
    weak_quality_sum = sum(item.quality for item in weak)
    improving_deltas = sorted(
        (
            max(0.0, strong_item.quality - weak_item.quality)
            for strong_item, weak_item in zip(strong, weak, strict=True)
        ),
        reverse=True,
    )
    oracle_prefix = [0.0]
    for delta in improving_deltas:
        oracle_prefix.append(oracle_prefix[-1] + delta)
    weak_baseline = {
        "average_quality": weak_quality_sum / count,
        "average_cost_usd": _nonnegative_mean([item.cost_usd for item in weak]),
        "average_latency_ms": _nonnegative_mean([item.latency_ms for item in weak]),
        "success_rate": sum(item.success for item in weak) / count,
    }
    strong_baseline = {
        "average_quality": sum(item.quality for item in strong) / count,
        "average_cost_usd": _nonnegative_mean([item.cost_usd for item in strong]),
        "average_latency_ms": _nonnegative_mean([item.latency_ms for item in strong]),
        "success_rate": sum(item.success for item in strong) / count,
    }
    raw_points: list[SweepPoint] = []
    for threshold in _thresholds(traces, active):
        selected = [
            strong_item
            if trace.route_score is not None and trace.route_score >= threshold
            else weak_item
            for trace, strong_item, weak_item in zip(traces, strong, weak, strict=True)
        ]
        strong_calls = sum(
            trace.route_score is not None and trace.route_score >= threshold for trace in traces
        )
        quality = sum(item.quality for item in selected) / count
        cost = _nonnegative_mean([item.cost_usd for item in selected])
        oracle = (weak_quality_sum + oracle_prefix[strong_calls]) / count
        raw_points.append(
            SweepPoint(
                threshold=threshold,
                strong_calls=strong_calls,
                strong_fraction=strong_calls / count,
                average_quality=quality,
                average_cost_usd=cost,
                average_latency_ms=_nonnegative_mean([item.latency_ms for item in selected]),
                success_rate=sum(item.success for item in selected) / count,
                quality_gain_over_weak=quality - weak_baseline["average_quality"],
                incremental_cost_over_weak_usd=cost - weak_baseline["average_cost_usd"],
                optimal_quality_at_same_strong_calls=oracle,
                quality_regret_to_call_budget_oracle=max(0.0, oracle - quality),
            )
        )
    points = tuple(
        replace(
            point,
            pareto_optimal=not any(
                _dominates(other, point) for other in raw_points if other is not point
            ),
        )
        for point in raw_points
    )
    return SweepReport(manifest, weak_baseline, strong_baseline, points)


def _cache_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "trace_canonical_sha256": manifest["trace_canonical_sha256"],
        "trace_file_sha256": manifest["trace_file_sha256"],
        "model_catalog_canonical_sha256": manifest["model_catalog_canonical_sha256"],
        "model_catalog_file_sha256": manifest["model_catalog_file_sha256"],
        "config_sha256": manifest["config_sha256"],
    }


def _checked_cached_report(
    payload: Any, manifest: dict[str, Any], key: str, thresholds: tuple[float, ...]
) -> SweepReport:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "key",
        "result_sha256",
        "result",
    }:
        raise PersistenceError("sweep cache has an invalid envelope")
    if payload["schema_version"] != SWEEP_SCHEMA_VERSION or payload["key"] != key:
        raise PersistenceError("sweep cache schema or key mismatch")
    result = payload["result"]
    if not isinstance(result, dict) or result.get("manifest") != manifest:
        raise PersistenceError("sweep cache identity mismatch")
    try:
        if payload["result_sha256"] != _digest(result):
            raise PersistenceError("sweep cache result checksum mismatch")
    except (TypeError, ValueError) as error:
        raise PersistenceError(f"sweep cache result is invalid: {error}") from error
    # A checksum detects accidental corruption, not malicious replacement.
    try:
        if set(result) != {
            "schema_version",
            "manifest",
            "weak_baseline",
            "strong_baseline",
            "points",
        }:
            raise ValueError("unexpected result fields")
        if result["schema_version"] != SWEEP_SCHEMA_VERSION:
            raise ValueError("unexpected result schema")
        if not isinstance(result["points"], list) or len(result["points"]) != len(thresholds):
            raise ValueError("unexpected point count")
        baselines = (result["weak_baseline"], result["strong_baseline"])
        baseline_fields = {
            "average_quality",
            "average_cost_usd",
            "average_latency_ms",
            "success_rate",
        }
        if any(not isinstance(item, dict) or set(item) != baseline_fields for item in baselines):
            raise ValueError("invalid baseline shape")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for item in baselines
            for value in item.values()
        ):
            raise ValueError("invalid baseline value")
        fields = set(SweepPoint.__dataclass_fields__)
        points: list[SweepPoint] = []
        for raw, threshold in zip(result["points"], thresholds, strict=True):
            if not isinstance(raw, dict) or set(raw) != fields:
                raise ValueError("invalid point shape")
            if raw["threshold"] != threshold:
                raise ValueError("threshold mismatch")
            if isinstance(raw["strong_calls"], bool) or not isinstance(raw["strong_calls"], int):
                raise ValueError("invalid strong call count")
            if not 0 <= raw["strong_calls"] <= manifest["records"]:
                raise ValueError("strong call count outside records")
            if not isinstance(raw["pareto_optimal"], bool):
                raise ValueError("invalid Pareto flag")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for name, value in raw.items()
                if name not in {"strong_calls", "pareto_optimal"}
            ):
                raise ValueError("invalid point value")
            points.append(SweepPoint(**raw))
    except (TypeError, ValueError, KeyError) as error:
        raise PersistenceError(f"sweep cache result validation failed: {error}") from error
    return SweepReport(manifest, baselines[0], baselines[1], tuple(points))


def cached_threshold_sweep(
    traces: tuple[RouteTrace, ...],
    models: tuple[ModelCandidate, ...],
    cache_dir: str | Path,
    *,
    config: SweepConfig | None = None,
    trace_file_sha256: str | None = None,
    model_file_sha256: str | None = None,
    refresh: bool = False,
    protected_paths: tuple[str | Path, ...] = (),
) -> CachedSweep:
    """Use a versioned content-addressed aggregate cache; fail closed on corruption."""

    if not isinstance(refresh, bool):
        raise ConfigurationError("refresh must be a boolean")
    active = config or SweepConfig()
    manifest, _pair = _prepare_manifest(
        traces,
        models,
        config=active,
        trace_file_sha256=trace_file_sha256,
        model_file_sha256=model_file_sha256,
    )
    key = _digest(_cache_identity(manifest))
    path = Path(cache_dir) / f"{key}.json"
    if any(_paths_alias(path, Path(source)) for source in protected_paths):
        raise ConfigurationError("sweep cache file must not alias an input or output")
    if path.exists() and not refresh:
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_SWEEP_CACHE_BYTES + 1)
        except OSError as error:
            raise PersistenceError(f"cannot read sweep cache: {error}") from error
        if len(raw) > MAX_SWEEP_CACHE_BYTES:
            raise PersistenceError("sweep cache exceeds byte limit")
        try:
            payload = loads_strict(raw)
        except (UnicodeError, ValueError, RecursionError) as error:
            raise PersistenceError(f"sweep cache is invalid JSON: {error}") from error
        return CachedSweep(
            _checked_cached_report(payload, manifest, key, _thresholds(traces, active)), True, path
        )
    expected = run_threshold_sweep(
        traces,
        models,
        config=active,
        trace_file_sha256=trace_file_sha256,
        model_file_sha256=model_file_sha256,
    )
    result = expected.to_dict()
    envelope = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "key": key,
        "result_sha256": _digest(result),
        "result": result,
    }
    encoded = _canonical(envelope)
    if len(encoded) > MAX_SWEEP_CACHE_BYTES:
        raise PersistenceError("sweep cache exceeds byte limit")
    try:
        _atomic_write_bytes_bundle({path: encoded})
    except OSError as error:
        raise PersistenceError(f"cannot write sweep cache: {error}") from error
    return CachedSweep(expected, False, path)


def load_sweep_catalog(path: str | Path) -> tuple[tuple[ModelCandidate, ...], str]:
    """Load the existing strict catalog and hash the exact consumed bytes."""

    return _load_models_with_sha256(path)
