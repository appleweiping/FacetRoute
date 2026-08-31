"""Strict, local routing traces for calibration and offline evaluation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._json import loads_strict
from .config import request_from_dict
from .errors import ConfigurationError
from .types import RouteRequest

_TRACE_FIELDS = {
    "request_id",
    "request",
    "outcomes",
    "preferred_model",
    "route_score",
    "strong_model",
    "weak_model",
}
_OUTCOME_FIELDS = {"quality", "cost_usd", "latency_ms", "success"}
_REQUEST_FIELDS = {
    "query",
    "user_id",
    "expected_output_tokens",
    "required_capabilities",
    "max_cost_usd",
    "max_latency_ms",
    "region",
    "needs_tools",
    "needs_json",
    "sensitivity",
    "task_hint",
    "context_tokens",
    "request_id",
    "metadata",
}


def _finite(name: str, value: Any, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ConfigurationError(f"{name} must be finite and >= {minimum}")
    return number


def _optional_identifier(data: Mapping[str, Any], name: str) -> str | None:
    value = data.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class TraceOutcome:
    """Observed outcome for one candidate on one request."""

    quality: float
    cost_usd: float
    latency_ms: float
    success: bool

    def __post_init__(self) -> None:
        quality = _finite("quality", self.quality)
        if quality > 1.0:
            raise ConfigurationError("quality must be between 0 and 1")
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "cost_usd", _finite("cost_usd", self.cost_usd))
        object.__setattr__(self, "latency_ms", _finite("latency_ms", self.latency_ms))
        if not isinstance(self.success, bool):
            raise ConfigurationError("success must be a boolean")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TraceOutcome:
        unknown = set(data) - _OUTCOME_FIELDS
        if unknown:
            raise ConfigurationError(f"unknown outcome fields: {sorted(unknown)}")
        try:
            success = data["success"]
            if not isinstance(success, bool):
                raise ConfigurationError("success must be a boolean")
            return cls(
                quality=_finite("quality", data["quality"]),
                cost_usd=_finite("cost_usd", data["cost_usd"]),
                latency_ms=_finite("latency_ms", data["latency_ms"]),
                success=success,
            )
        except KeyError as exc:
            raise ConfigurationError(f"missing outcome field: {exc.args[0]}") from exc

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "quality": self.quality,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "success": self.success,
        }


@dataclass(frozen=True, slots=True)
class RouteTrace:
    """One request and counterfactual outcomes for declared candidates.

    ``route_score`` is a calibrated-router input in ``[0, 1)`` where larger
    means stronger evidence for ``strong_model``.  The upper endpoint is
    excluded so threshold ``1`` has the unambiguous meaning "always weak".
    """

    request: RouteRequest
    outcomes: Mapping[str, TraceOutcome]
    preferred_model: str | None = None
    route_score: float | None = None
    strong_model: str | None = None
    weak_model: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, RouteRequest):
            raise ConfigurationError("trace request must be a RouteRequest")
        outcomes = dict(self.outcomes)
        if not outcomes:
            raise ConfigurationError("trace outcomes cannot be empty")
        if any(
            not isinstance(key, str) or not key.strip() or key != key.strip() for key in outcomes
        ):
            raise ConfigurationError("outcome model identifiers must be trimmed non-empty strings")
        if any(not isinstance(outcome, TraceOutcome) for outcome in outcomes.values()):
            raise ConfigurationError("trace outcomes must contain TraceOutcome values")
        if self.preferred_model is not None and self.preferred_model not in outcomes:
            raise ConfigurationError("preferred_model must have an observed outcome")
        pair = (self.strong_model, self.weak_model)
        if (pair[0] is None) != (pair[1] is None):
            raise ConfigurationError("strong_model and weak_model must be provided together")
        if pair[0] is not None:
            if pair[0] == pair[1]:
                raise ConfigurationError("strong_model and weak_model must differ")
            if pair[0] not in outcomes or pair[1] not in outcomes:
                raise ConfigurationError("strong_model and weak_model must have outcomes")
            if self.route_score is None:
                raise ConfigurationError("route_score is required for a strong/weak pair")
        if self.route_score is not None:
            score = _finite("route_score", self.route_score)
            if score >= 1.0:
                raise ConfigurationError("route_score must be in [0, 1)")
            object.__setattr__(self, "route_score", score)
        object.__setattr__(self, "outcomes", outcomes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RouteTrace:
        unknown = set(data) - _TRACE_FIELDS
        if unknown:
            raise ConfigurationError(f"unknown trace fields: {sorted(unknown)}")
        try:
            request_payload = data["request"]
            if not isinstance(request_payload, dict):
                raise ConfigurationError("request must be a JSON object")
            request_data = dict(request_payload)
            unknown_request_fields = set(request_data) - _REQUEST_FIELDS
            if unknown_request_fields:
                raise ConfigurationError(
                    f"unknown trace request fields: {sorted(unknown_request_fields)}"
                )
            outer_id = _optional_identifier(data, "request_id")
            inner_id = request_data.get("request_id")
            if outer_id is None and inner_id is None:
                raise ConfigurationError("trace request_id is required for reproducibility")
            if outer_id is not None and inner_id is not None and outer_id != inner_id:
                raise ConfigurationError("request_id disagrees with request.request_id")
            if outer_id is not None:
                request_data["request_id"] = outer_id
            outcomes_payload = data["outcomes"]
            if not isinstance(outcomes_payload, dict):
                raise ConfigurationError("outcomes must be a JSON object")
            outcomes: dict[str, TraceOutcome] = {}
            for model_id, raw in outcomes_payload.items():
                if not isinstance(model_id, str) or not model_id.strip():
                    raise ConfigurationError("outcome model identifiers must be non-empty strings")
                if not isinstance(raw, dict):
                    raise ConfigurationError(f"outcome for {model_id!r} must be an object")
                outcomes[model_id] = TraceOutcome.from_dict(raw)
            route_score_raw = data.get("route_score")
            return cls(
                request=request_from_dict(request_data),
                outcomes=outcomes,
                preferred_model=_optional_identifier(data, "preferred_model"),
                route_score=(
                    _finite("route_score", route_score_raw) if route_score_raw is not None else None
                ),
                strong_model=_optional_identifier(data, "strong_model"),
                weak_model=_optional_identifier(data, "weak_model"),
            )
        except KeyError as exc:
            raise ConfigurationError(f"missing trace field: {exc.args[0]}") from exc

    def to_dict(self, *, include_query: bool = True) -> dict[str, Any]:
        request: dict[str, Any] = {
            "request_id": self.request.request_id,
            "user_id": self.request.user_id,
            "expected_output_tokens": self.request.expected_output_tokens,
            "required_capabilities": sorted(self.request.required_capabilities),
            "max_cost_usd": self.request.max_cost_usd,
            "max_latency_ms": self.request.max_latency_ms,
            "region": self.request.region,
            "needs_tools": self.request.needs_tools,
            "needs_json": self.request.needs_json,
            "sensitivity": self.request.sensitivity,
            "task_hint": self.request.task_hint,
            "context_tokens": self.request.context_tokens,
            "metadata": dict(self.request.metadata),
        }
        if include_query:
            request["query"] = self.request.query
        return {
            "request_id": self.request.request_id,
            "request": request,
            "outcomes": {
                model_id: outcome.to_dict() for model_id, outcome in sorted(self.outcomes.items())
            },
            "preferred_model": self.preferred_model,
            "route_score": self.route_score,
            "strong_model": self.strong_model,
            "weak_model": self.weak_model,
        }


def iter_traces(
    path: str | Path,
    *,
    max_line_bytes: int = 1_048_576,
    max_records: int = 1_000_000,
) -> Iterator[RouteTrace]:
    """Stream strict JSONL traces with bounded records and line sizes."""

    if max_line_bytes <= 0 or max_records <= 0:
        raise ValueError("trace limits must be positive")
    source = Path(path)
    seen_ids: set[str] = set()
    count = 0
    try:
        with source.open("rb") as handle:
            line_number = 0
            while raw := handle.readline(max_line_bytes + 1):
                line_number += 1
                if len(raw) > max_line_bytes:
                    raise ConfigurationError(
                        f"trace line exceeds {max_line_bytes} bytes at {source}:{line_number}"
                    )
                if not raw.strip():
                    continue
                count += 1
                if count > max_records:
                    raise ConfigurationError(f"trace file exceeds {max_records} records")
                try:
                    payload = loads_strict(raw)
                    if not isinstance(payload, dict):
                        raise ConfigurationError("trace line must be a JSON object")
                    trace = RouteTrace.from_dict(payload)
                except (UnicodeDecodeError, ValueError, ConfigurationError) as exc:
                    raise ConfigurationError(
                        f"invalid trace at {source}:{line_number}: {exc}"
                    ) from exc
                if trace.request.request_id in seen_ids:
                    raise ConfigurationError(
                        f"duplicate request_id at {source}:{line_number}: "
                        f"{trace.request.request_id}"
                    )
                seen_ids.add(trace.request.request_id)
                yield trace
    except OSError as exc:
        raise ConfigurationError(f"cannot read trace file {source}: {exc}") from exc


def load_traces(path: str | Path, **limits: int) -> tuple[RouteTrace, ...]:
    traces = tuple(iter_traces(path, **limits))
    if not traces:
        raise ConfigurationError("trace file contains no records")
    return traces


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
