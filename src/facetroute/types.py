"""Typed domain objects shared by routing policies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any
from uuid import uuid4

from .errors import ConfigurationError


def _non_negative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ConfigurationError(f"{name} must be a finite non-negative number")


def _probability(name: str, value: float) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{name} must be between 0 and 1")


def _names(values: Iterable[str], field_name: str = "values") -> frozenset[str]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise ConfigurationError(f"{field_name} must be a collection of strings")
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ConfigurationError(f"{field_name} must contain only strings")
        if value.strip():
            normalized.add(value.strip().lower())
    return frozenset(normalized)


def _identifiers(values: Iterable[str], field_name: str) -> frozenset[str]:
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        raise ConfigurationError(f"{field_name} must be a collection of strings")
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ConfigurationError(f"{field_name} must contain only strings")
        if value.strip():
            normalized.add(value.strip())
    return frozenset(normalized)


def _boolean_field(data: Mapping[str, Any], name: str, default: bool) -> bool:
    value = data.get(name, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """A locally declared model option and its operational envelope.

    Costs are expressed in US dollars per one million tokens. Latency values are
    milliseconds. Quality values are normalized to the inclusive ``[0, 1]``
    interval and may contain a ``default`` fallback.
    """

    model_id: str
    display_name: str
    capabilities: frozenset[str]
    input_cost_per_million: float
    output_cost_per_million: float
    latency_ms_p50: float
    latency_ms_p95: float
    context_window: int
    quality_by_task: Mapping[str, float]
    regions: frozenset[str] = field(default_factory=frozenset)
    supports_tools: bool = False
    supports_json: bool = False
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model_id = self.model_id.strip()
        if not model_id:
            raise ConfigurationError("model_id cannot be empty")
        if not self.display_name.strip():
            raise ConfigurationError("display_name cannot be empty")
        for name, value in (
            ("input_cost_per_million", self.input_cost_per_million),
            ("output_cost_per_million", self.output_cost_per_million),
            ("latency_ms_p50", self.latency_ms_p50),
            ("latency_ms_p95", self.latency_ms_p95),
        ):
            _non_negative(name, value)
        if self.latency_ms_p95 < self.latency_ms_p50:
            raise ConfigurationError("latency_ms_p95 cannot be below latency_ms_p50")
        if self.context_window <= 0:
            raise ConfigurationError("context_window must be positive")
        for name, value in (
            ("supports_tools", self.supports_tools),
            ("supports_json", self.supports_json),
            ("enabled", self.enabled),
        ):
            if not isinstance(value, bool):
                raise ConfigurationError(f"{name} must be a boolean")
        if not self.quality_by_task:
            raise ConfigurationError("quality_by_task must contain at least one value")
        normalized_quality: dict[str, float] = {}
        for task, quality in self.quality_by_task.items():
            key = task.strip().lower()
            if not key:
                raise ConfigurationError("quality task names cannot be empty")
            _probability(f"quality_by_task[{key}]", quality)
            normalized_quality[key] = float(quality)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "capabilities", _names(self.capabilities, "capabilities"))
        object.__setattr__(self, "regions", _names(self.regions, "regions"))
        object.__setattr__(self, "quality_by_task", normalized_quality)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def quality_for(self, task: str) -> float:
        """Return task quality, falling back to ``default`` then the mean."""

        task_key = task.strip().lower()
        if task_key in self.quality_by_task:
            return self.quality_by_task[task_key]
        if "default" in self.quality_by_task:
            return self.quality_by_task["default"]
        return sum(self.quality_by_task.values()) / len(self.quality_by_task)

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate request cost without making a provider API call."""

        if prompt_tokens < 0 or completion_tokens < 0:
            raise ConfigurationError("token counts cannot be negative")
        return (
            prompt_tokens * self.input_cost_per_million
            + completion_tokens * self.output_cost_per_million
        ) / 1_000_000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "capabilities": sorted(self.capabilities),
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_p95": self.latency_ms_p95,
            "context_window": self.context_window,
            "quality_by_task": dict(self.quality_by_task),
            "regions": sorted(self.regions),
            "supports_tools": self.supports_tools,
            "supports_json": self.supports_json,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelCandidate:
        try:
            return cls(
                model_id=str(data["model_id"]),
                display_name=str(data.get("display_name", data["model_id"])),
                capabilities=_names(data.get("capabilities", []), "capabilities"),
                input_cost_per_million=float(data.get("input_cost_per_million", 0.0)),
                output_cost_per_million=float(data.get("output_cost_per_million", 0.0)),
                latency_ms_p50=float(data.get("latency_ms_p50", 0.0)),
                latency_ms_p95=float(data.get("latency_ms_p95", 0.0)),
                context_window=int(data["context_window"]),
                quality_by_task={
                    str(key): float(value)
                    for key, value in dict(data["quality_by_task"]).items()
                },
                regions=_names(data.get("regions", []), "regions"),
                supports_tools=_boolean_field(data, "supports_tools", False),
                supports_json=_boolean_field(data, "supports_json", False),
                enabled=_boolean_field(data, "enabled", True),
                metadata=dict(data.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid model candidate: {exc}") from exc


@dataclass(frozen=True, slots=True)
class UserPreferences:
    """A transparent user-level objective and policy profile."""

    user_id: str
    quality_weight: float = 0.6
    cost_weight: float = 0.25
    latency_weight: float = 0.15
    exploration_weight: float = 1.0
    preferred_models: frozenset[str] = field(default_factory=frozenset)
    blocked_models: frozenset[str] = field(default_factory=frozenset)
    required_region: str | None = None
    max_cost_usd: float | None = None
    max_latency_ms: float | None = None
    minimum_quality: float | None = None
    task_weight_overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise ConfigurationError("user_id cannot be empty")
        for name, value in (
            ("quality_weight", self.quality_weight),
            ("cost_weight", self.cost_weight),
            ("latency_weight", self.latency_weight),
            ("exploration_weight", self.exploration_weight),
        ):
            _non_negative(name, value)
        if self.quality_weight + self.cost_weight + self.latency_weight <= 0:
            raise ConfigurationError("at least one objective weight must be positive")
        if self.max_cost_usd is not None:
            _non_negative("max_cost_usd", self.max_cost_usd)
        if self.max_latency_ms is not None:
            _non_negative("max_latency_ms", self.max_latency_ms)
        if self.minimum_quality is not None:
            _probability("minimum_quality", self.minimum_quality)
        preferred = _identifiers(self.preferred_models, "preferred_models")
        blocked = _identifiers(self.blocked_models, "blocked_models")
        overlap = preferred & blocked
        if overlap:
            raise ConfigurationError(f"models cannot be both preferred and blocked: {sorted(overlap)}")
        region = self.required_region.strip().lower() if self.required_region else None
        normalized_overrides: dict[str, dict[str, float]] = {}
        for task, weights in self.task_weight_overrides.items():
            values = {str(key): float(value) for key, value in dict(weights).items()}
            for key, value in values.items():
                if key not in {"quality", "cost", "latency"}:
                    raise ConfigurationError(f"unsupported task weight '{key}'")
                _non_negative(f"task_weight_overrides[{task}][{key}]", value)
            normalized_overrides[str(task).strip().lower()] = values
        object.__setattr__(self, "preferred_models", preferred)
        object.__setattr__(self, "blocked_models", blocked)
        object.__setattr__(self, "required_region", region)
        object.__setattr__(self, "task_weight_overrides", normalized_overrides)

    def objective_weights(self, task: str) -> tuple[float, float, float]:
        override = self.task_weight_overrides.get(task.strip().lower(), {})
        values = (
            override.get("quality", self.quality_weight),
            override.get("cost", self.cost_weight),
            override.get("latency", self.latency_weight),
        )
        total = sum(values)
        if total <= 0:
            raise ConfigurationError(f"objective weights for task '{task}' sum to zero")
        quality, cost, latency = values
        return quality / total, cost / total, latency / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "quality_weight": self.quality_weight,
            "cost_weight": self.cost_weight,
            "latency_weight": self.latency_weight,
            "exploration_weight": self.exploration_weight,
            "preferred_models": sorted(self.preferred_models),
            "blocked_models": sorted(self.blocked_models),
            "required_region": self.required_region,
            "max_cost_usd": self.max_cost_usd,
            "max_latency_ms": self.max_latency_ms,
            "minimum_quality": self.minimum_quality,
            "task_weight_overrides": {
                task: dict(weights) for task, weights in self.task_weight_overrides.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UserPreferences:
        try:
            return cls(
                user_id=str(data["user_id"]),
                quality_weight=float(data.get("quality_weight", 0.6)),
                cost_weight=float(data.get("cost_weight", 0.25)),
                latency_weight=float(data.get("latency_weight", 0.15)),
                exploration_weight=float(data.get("exploration_weight", 1.0)),
                preferred_models=_identifiers(data.get("preferred_models", []), "preferred_models"),
                blocked_models=_identifiers(data.get("blocked_models", []), "blocked_models"),
                required_region=(
                    str(data["required_region"]) if data.get("required_region") else None
                ),
                max_cost_usd=(float(data["max_cost_usd"]) if data.get("max_cost_usd") is not None else None),
                max_latency_ms=(
                    float(data["max_latency_ms"])
                    if data.get("max_latency_ms") is not None
                    else None
                ),
                minimum_quality=(
                    float(data["minimum_quality"])
                    if data.get("minimum_quality") is not None
                    else None
                ),
                task_weight_overrides=dict(data.get("task_weight_overrides", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid user preferences: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RouteRequest:
    """A provider-independent routing request."""

    query: str
    user_id: str = "default"
    expected_output_tokens: int = 256
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    max_cost_usd: float | None = None
    max_latency_ms: float | None = None
    region: str | None = None
    needs_tools: bool = False
    needs_json: bool = False
    sensitivity: str = "normal"
    task_hint: str | None = None
    context_tokens: int | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ConfigurationError("query cannot be empty")
        if not self.user_id.strip():
            raise ConfigurationError("user_id cannot be empty")
        if not self.request_id.strip():
            raise ConfigurationError("request_id cannot be empty")
        if self.expected_output_tokens < 0:
            raise ConfigurationError("expected_output_tokens cannot be negative")
        if self.context_tokens is not None and self.context_tokens < 0:
            raise ConfigurationError("context_tokens cannot be negative")
        if self.max_cost_usd is not None:
            _non_negative("max_cost_usd", self.max_cost_usd)
        if self.max_latency_ms is not None:
            _non_negative("max_latency_ms", self.max_latency_ms)
        if not isinstance(self.needs_tools, bool) or not isinstance(self.needs_json, bool):
            raise ConfigurationError("needs_tools and needs_json must be booleans")
        sensitivity = self.sensitivity.strip().lower()
        if sensitivity not in {"normal", "sensitive", "restricted"}:
            raise ConfigurationError("sensitivity must be normal, sensitive, or restricted")
        object.__setattr__(
            self,
            "required_capabilities",
            _names(self.required_capabilities, "required_capabilities"),
        )
        object.__setattr__(self, "region", self.region.strip().lower() if self.region else None)
        object.__setattr__(self, "sensitivity", sensitivity)
        object.__setattr__(self, "task_hint", self.task_hint.strip().lower() if self.task_hint else None)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class QueryFeatures:
    """Deterministic, provider-independent features extracted from a request."""

    task: str
    token_estimate: int
    character_count: int
    difficulty: float
    code_fraction: float
    math_fraction: float
    question_count: int
    has_multistep_language: bool
    required_capabilities: frozenset[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "token_estimate": self.token_estimate,
            "character_count": self.character_count,
            "difficulty": self.difficulty,
            "code_fraction": self.code_fraction,
            "math_fraction": self.math_fraction,
            "question_count": self.question_count,
            "has_multistep_language": self.has_multistep_language,
            "required_capabilities": sorted(self.required_capabilities),
        }


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Auditable objective components for a single model."""

    quality_utility: float
    cost_utility: float
    latency_utility: float
    preference_bonus: float
    rule_bonus: float
    total: float
    estimated_cost_usd: float
    estimated_latency_ms: float
    raw_quality: float

    def to_dict(self) -> dict[str, float]:
        return {
            "quality_utility": self.quality_utility,
            "cost_utility": self.cost_utility,
            "latency_utility": self.latency_utility,
            "preference_bonus": self.preference_bonus,
            "rule_bonus": self.rule_bonus,
            "total": self.total,
            "estimated_cost_usd": self.estimated_cost_usd,
            "estimated_latency_ms": self.estimated_latency_ms,
            "raw_quality": self.raw_quality,
        }


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """A complete routing result, including rejected alternatives."""

    request_id: str
    user_id: str
    selected_model: str
    policy: str
    score: float
    breakdown: ScoreBreakdown
    alternatives: tuple[tuple[str, float], ...]
    excluded: Mapping[str, tuple[str, ...]]
    matched_rules: tuple[str, ...]
    feature_summary: Mapping[str, Any]
    explanation: tuple[str, ...]
    context_vector: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "selected_model": self.selected_model,
            "policy": self.policy,
            "score": self.score,
            "breakdown": self.breakdown.to_dict(),
            "alternatives": [
                {"model_id": model_id, "score": score}
                for model_id, score in self.alternatives
            ],
            "excluded": {key: list(value) for key, value in self.excluded.items()},
            "matched_rules": list(self.matched_rules),
            "feature_summary": dict(self.feature_summary),
            "explanation": list(self.explanation),
            "context_vector": list(self.context_vector),
        }
