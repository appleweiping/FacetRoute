"""Declarative routing rules used only as transparent score bonuses."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import ConfigurationError
from .types import QueryFeatures


def _string_items(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set, frozenset)):
        raise ConfigurationError(f"{field_name} must be a collection of strings")
    if any(not isinstance(item, str) for item in value):
        raise ConfigurationError(f"{field_name} must contain only strings")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class RoutingRule:
    """A bounded, serializable preference rule.

    Rules never bypass hard constraints. A match only adds ``bonus`` to the
    listed models before deterministic selection.
    """

    name: str
    prefer_models: tuple[str, ...]
    tasks: frozenset[str] = field(default_factory=frozenset)
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    minimum_difficulty: float = 0.0
    maximum_difficulty: float = 1.0
    bonus: float = 0.08

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("rule name cannot be empty")
        models = tuple(
            dict.fromkeys(
                item.strip()
                for item in _string_items(self.prefer_models, "prefer_models")
                if item.strip()
            )
        )
        if not models:
            raise ConfigurationError("a routing rule must prefer at least one model")
        if not 0 <= self.minimum_difficulty <= self.maximum_difficulty <= 1:
            raise ConfigurationError("rule difficulty bounds must satisfy 0 <= min <= max <= 1")
        if self.bonus < 0 or not math.isfinite(self.bonus):
            raise ConfigurationError("rule bonus must be finite and non-negative")
        object.__setattr__(self, "prefer_models", models)
        object.__setattr__(
            self,
            "tasks",
            frozenset(
                item.strip().lower()
                for item in _string_items(self.tasks, "tasks")
                if item.strip()
            ),
        )
        object.__setattr__(
            self,
            "required_capabilities",
            frozenset(
                item.strip().lower()
                for item in _string_items(
                    self.required_capabilities, "required_capabilities"
                )
                if item.strip()
            ),
        )

    def matches(self, features: QueryFeatures) -> bool:
        if self.tasks and features.task not in self.tasks:
            return False
        if not self.required_capabilities.issubset(features.required_capabilities):
            return False
        return self.minimum_difficulty <= features.difficulty <= self.maximum_difficulty

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoutingRule:
        try:
            return cls(
                name=str(data["name"]),
                prefer_models=_string_items(data["prefer_models"], "prefer_models"),
                tasks=frozenset(_string_items(data.get("tasks", []), "tasks")),
                required_capabilities=frozenset(
                    _string_items(
                        data.get("required_capabilities", []), "required_capabilities"
                    )
                ),
                minimum_difficulty=float(data.get("minimum_difficulty", 0.0)),
                maximum_difficulty=float(data.get("maximum_difficulty", 1.0)),
                bonus=float(data.get("bonus", 0.08)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid routing rule: {exc}") from exc


def match_rules(
    rules: tuple[RoutingRule, ...], features: QueryFeatures, eligible_model_ids: set[str]
) -> tuple[dict[str, float], tuple[str, ...]]:
    bonuses: dict[str, float] = {}
    matched: list[str] = []
    for rule in rules:
        if not rule.matches(features):
            continue
        applied = False
        for model_id in rule.prefer_models:
            if model_id in eligible_model_ids:
                bonuses[model_id] = bonuses.get(model_id, 0.0) + rule.bonus
                applied = True
        if applied:
            matched.append(rule.name)
    return bonuses, tuple(matched)
