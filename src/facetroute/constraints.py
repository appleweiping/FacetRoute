"""Hard eligibility constraints applied before any model is scored."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .types import ModelCandidate, QueryFeatures, RouteRequest, UserPreferences


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    eligible: tuple[ModelCandidate, ...]
    rejected: dict[str, tuple[str, ...]]
    estimated_costs: dict[str, float]


class ConstraintEngine:
    """Evaluate explainable, non-negotiable routing requirements."""

    def filter(
        self,
        candidates: Iterable[ModelCandidate],
        request: RouteRequest,
        features: QueryFeatures,
        preferences: UserPreferences,
    ) -> ConstraintResult:
        eligible: list[ModelCandidate] = []
        rejected: dict[str, tuple[str, ...]] = {}
        costs: dict[str, float] = {}
        cost_limit = self._minimum_optional(request.max_cost_usd, preferences.max_cost_usd)
        latency_limit = self._minimum_optional(request.max_latency_ms, preferences.max_latency_ms)
        region_conflict = bool(
            request.region
            and preferences.required_region
            and request.region != preferences.required_region
        )
        region = request.region or preferences.required_region
        required_context = features.token_estimate + request.expected_output_tokens

        for candidate in candidates:
            reasons: list[str] = []
            cost = candidate.estimate_cost(features.token_estimate, request.expected_output_tokens)
            costs[candidate.model_id] = cost
            if not candidate.enabled:
                reasons.append("model is disabled")
            if candidate.model_id in preferences.blocked_models:
                reasons.append("model is blocked by the user profile")
            general_capabilities = set(features.required_capabilities)
            if request.needs_tools:
                general_capabilities.discard("tools")
            if request.needs_json:
                general_capabilities.discard("json")
            missing = general_capabilities - candidate.capabilities
            if missing:
                reasons.append(f"missing capabilities: {', '.join(sorted(missing))}")
            if request.needs_tools and not candidate.supports_tools:
                reasons.append("tool calling is required")
            if request.needs_json and not candidate.supports_json:
                reasons.append("structured JSON output is required")
            if candidate.context_window < required_context:
                reasons.append(
                    f"context limit {candidate.context_window} is below required {required_context}"
                )
            if region_conflict:
                reasons.append(
                    f"request region '{request.region}' conflicts with profile-required region "
                    f"'{preferences.required_region}'"
                )
            elif region and candidate.regions and region not in candidate.regions:
                reasons.append(f"region '{region}' is unavailable")
            if request.sensitivity == "restricted" and candidate.metadata.get("local") is not True:
                reasons.append("restricted data requires a model marked local")
            if cost_limit is not None and cost > cost_limit + 1e-12:
                reasons.append(f"estimated cost {cost:.8f} exceeds limit {cost_limit:.8f}")
            if latency_limit is not None and candidate.latency_ms_p95 > latency_limit:
                reasons.append(
                    f"p95 latency {candidate.latency_ms_p95:.1f} exceeds limit {latency_limit:.1f}"
                )
            if (
                preferences.minimum_quality is not None
                and candidate.quality_for(features.task) < preferences.minimum_quality
            ):
                reasons.append(
                    f"quality {candidate.quality_for(features.task):.3f} is below minimum "
                    f"{preferences.minimum_quality:.3f}"
                )
            if reasons:
                rejected[candidate.model_id] = tuple(reasons)
            else:
                eligible.append(candidate)
        return ConstraintResult(tuple(eligible), rejected, costs)

    @staticmethod
    def _minimum_optional(first: float | None, second: float | None) -> float | None:
        values = [value for value in (first, second) if value is not None]
        return min(values) if values else None
