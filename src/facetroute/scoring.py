"""Explainable multi-objective scoring."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .types import ModelCandidate, QueryFeatures, RouteRequest, ScoreBreakdown, UserPreferences


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    candidate: ModelCandidate
    breakdown: ScoreBreakdown


class MultiObjectiveScorer:
    """Score quality, cost, and latency on comparable utility scales."""

    def __init__(self, preferred_model_bonus: float = 0.05) -> None:
        if preferred_model_bonus < 0 or not math.isfinite(preferred_model_bonus):
            raise ValueError("preferred_model_bonus must be finite and non-negative")
        self.preferred_model_bonus = preferred_model_bonus

    def score(
        self,
        candidates: Iterable[ModelCandidate],
        request: RouteRequest,
        features: QueryFeatures,
        preferences: UserPreferences,
        estimated_costs: Mapping[str, float],
        rule_bonuses: Mapping[str, float] | None = None,
    ) -> tuple[ScoredCandidate, ...]:
        options = tuple(candidates)
        if not options:
            return ()
        costs = [estimated_costs[item.model_id] for item in options]
        latencies = [item.latency_ms_p95 for item in options]
        quality_weight, cost_weight, latency_weight = preferences.objective_weights(features.task)
        bonuses = rule_bonuses or {}
        scored: list[ScoredCandidate] = []
        for candidate in options:
            raw_quality = candidate.quality_for(features.task)
            cost_utility = self._inverse_minmax(estimated_costs[candidate.model_id], costs)
            latency_utility = self._inverse_minmax(candidate.latency_ms_p95, latencies)
            preference_bonus = (
                self.preferred_model_bonus
                if candidate.model_id in preferences.preferred_models
                else 0.0
            )
            rule_bonus = float(bonuses.get(candidate.model_id, 0.0))
            total = (
                quality_weight * raw_quality
                + cost_weight * cost_utility
                + latency_weight * latency_utility
                + preference_bonus
                + rule_bonus
            )
            scored.append(
                ScoredCandidate(
                    candidate=candidate,
                    breakdown=ScoreBreakdown(
                        quality_utility=raw_quality,
                        cost_utility=cost_utility,
                        latency_utility=latency_utility,
                        preference_bonus=preference_bonus,
                        rule_bonus=rule_bonus,
                        total=total,
                        estimated_cost_usd=estimated_costs[candidate.model_id],
                        estimated_latency_ms=candidate.latency_ms_p95,
                        raw_quality=raw_quality,
                    ),
                )
            )
        return tuple(sorted(scored, key=lambda item: (-item.breakdown.total, item.candidate.model_id)))

    @staticmethod
    def _inverse_minmax(value: float, values: list[float]) -> float:
        low = min(values)
        high = max(values)
        if high - low <= 1e-15:
            return 1.0
        return 1.0 - (value - low) / (high - low)


def explain_score(
    selected: ScoredCandidate,
    features: QueryFeatures,
    preferences: UserPreferences,
) -> tuple[str, ...]:
    """Create stable human-readable reasons from a numeric breakdown."""

    quality_weight, cost_weight, latency_weight = preferences.objective_weights(features.task)
    reasons = [
        f"task={features.task}, estimated difficulty={features.difficulty:.3f}",
        (
            "objective weights: "
            f"quality={quality_weight:.3f}, cost={cost_weight:.3f}, "
            f"latency={latency_weight:.3f}"
        ),
        (
            f"{selected.candidate.model_id} utilities: "
            f"quality={selected.breakdown.quality_utility:.3f}, "
            f"cost={selected.breakdown.cost_utility:.3f}, "
            f"latency={selected.breakdown.latency_utility:.3f}"
        ),
        (
            f"estimated cost=${selected.breakdown.estimated_cost_usd:.8f}, "
            f"p95 latency={selected.breakdown.estimated_latency_ms:.1f}ms"
        ),
    ]
    if selected.breakdown.preference_bonus:
        reasons.append("the selected model is explicitly preferred by the user profile")
    if selected.breakdown.rule_bonus:
        reasons.append(f"matching rules added {selected.breakdown.rule_bonus:.3f} to its score")
    return tuple(reasons)
