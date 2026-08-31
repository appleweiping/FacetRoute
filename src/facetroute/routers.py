"""Deterministic rule and Pareto routing policies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from .constraints import ConstraintEngine
from .errors import NoEligibleModelError
from .features import QueryFeatureExtractor
from .pareto import pareto_front
from .rules import RoutingRule, match_rules
from .scoring import MultiObjectiveScorer, ScoredCandidate, explain_score
from .types import (
    ModelCandidate,
    QueryFeatures,
    RouteDecision,
    RouteRequest,
    UserPreferences,
)


class Router(Protocol):
    def route(self, request: RouteRequest) -> RouteDecision: ...


@dataclass(frozen=True, slots=True)
class BatchRouteResult:
    decisions: tuple[RouteDecision, ...]
    errors: Mapping[int, str]


class RuleRouter:
    """Route by hard constraints, declarative bonuses, and objective score."""

    policy_name = "rule"

    def __init__(
        self,
        candidates: Iterable[ModelCandidate],
        preferences: Mapping[str, UserPreferences] | None = None,
        rules: Iterable[RoutingRule] = (),
        *,
        extractor: QueryFeatureExtractor | None = None,
        constraints: ConstraintEngine | None = None,
        scorer: MultiObjectiveScorer | None = None,
    ) -> None:
        candidate_tuple = tuple(candidates)
        identifiers = [item.model_id for item in candidate_tuple]
        if not candidate_tuple:
            raise ValueError("at least one model candidate is required")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("model_id values must be unique")
        self.candidates = candidate_tuple
        self.preferences = dict(preferences or {})
        self.rules = tuple(rules)
        self.extractor = extractor or QueryFeatureExtractor()
        self.constraints = constraints or ConstraintEngine()
        self.scorer = scorer or MultiObjectiveScorer()

    def preference_for(self, user_id: str) -> UserPreferences:
        return self.preferences.get(user_id, UserPreferences(user_id=user_id))

    def _candidate_pool(
        self,
        eligible: tuple[ModelCandidate, ...],
        features: QueryFeatures,
        estimated_costs: Mapping[str, float],
    ) -> tuple[ModelCandidate, ...]:
        del features, estimated_costs
        return eligible

    def route(self, request: RouteRequest) -> RouteDecision:
        preferences = self.preference_for(request.user_id)
        features = self.extractor.extract(request)
        constraint_result = self.constraints.filter(
            self.candidates, request, features, preferences
        )
        if not constraint_result.eligible:
            raise NoEligibleModelError(constraint_result.rejected)
        pool = self._candidate_pool(
            constraint_result.eligible, features, constraint_result.estimated_costs
        )
        bonuses, matched = match_rules(
            self.rules, features, {candidate.model_id for candidate in pool}
        )
        scored = self.scorer.score(
            pool,
            request,
            features,
            preferences,
            constraint_result.estimated_costs,
            bonuses,
        )
        selected = scored[0]
        return self._decision(
            request,
            preferences,
            features.to_dict(),
            selected,
            scored,
            constraint_result.rejected,
            matched,
        )

    def _decision(
        self,
        request: RouteRequest,
        preferences: UserPreferences,
        feature_summary: dict[str, object],
        selected: ScoredCandidate,
        scored: tuple[ScoredCandidate, ...],
        rejected: Mapping[str, tuple[str, ...]],
        matched: tuple[str, ...],
        *,
        context_vector: tuple[float, ...] = (),
        explanation_prefix: tuple[str, ...] = (),
    ) -> RouteDecision:
        features = self.extractor.extract(request)
        return RouteDecision(
            request_id=request.request_id,
            user_id=request.user_id,
            selected_model=selected.candidate.model_id,
            policy=self.policy_name,
            score=selected.breakdown.total,
            breakdown=selected.breakdown,
            alternatives=tuple(
                (item.candidate.model_id, item.breakdown.total) for item in scored[1:]
            ),
            excluded=dict(rejected),
            matched_rules=matched,
            feature_summary=feature_summary,
            explanation=explanation_prefix + explain_score(selected, features, preferences),
            context_vector=context_vector,
        )

    def route_many(self, requests: Iterable[RouteRequest], *, fail_fast: bool = True) -> BatchRouteResult:
        decisions: list[RouteDecision] = []
        errors: dict[int, str] = {}
        for index, request in enumerate(requests):
            try:
                decisions.append(self.route(request))
            except Exception as exc:
                if fail_fast:
                    raise
                errors[index] = str(exc)
        return BatchRouteResult(tuple(decisions), errors)


class ParetoRouter(RuleRouter):
    """Score only models on the quality/cost/latency Pareto frontier."""

    policy_name = "pareto"

    def _candidate_pool(
        self,
        eligible: tuple[ModelCandidate, ...],
        features: QueryFeatures,
        estimated_costs: Mapping[str, float],
    ) -> tuple[ModelCandidate, ...]:
        return pareto_front(eligible, features, estimated_costs)

    def route(self, request: RouteRequest) -> RouteDecision:
        decision = super().route(request)
        front_ids = [decision.selected_model, *(model_id for model_id, _ in decision.alternatives)]
        eligible_ids = {
            candidate.model_id
            for candidate in self.candidates
            if candidate.model_id not in decision.excluded
        }
        dominated_ids = sorted(eligible_ids - set(front_ids))
        summary = dict(decision.feature_summary)
        summary["pareto_front"] = front_ids
        summary["pareto_dominated"] = dominated_ids
        return RouteDecision(
            request_id=decision.request_id,
            user_id=decision.user_id,
            selected_model=decision.selected_model,
            policy=self.policy_name,
            score=decision.score,
            breakdown=decision.breakdown,
            alternatives=decision.alternatives,
            excluded=decision.excluded,
            matched_rules=decision.matched_rules,
            feature_summary=summary,
            explanation=(
                (
                    f"selected from a {len(front_ids)}-model non-dominated frontier; "
                    f"{len(dominated_ids)} eligible model(s) were dominated"
                ),
                *decision.explanation,
            ),
            context_vector=decision.context_vector,
        )


class BatchRouter:
    """A small adapter for code that prefers a dedicated batch object."""

    def __init__(self, router: Router) -> None:
        self.router = router

    def route(self, requests: Iterable[RouteRequest], *, fail_fast: bool = True) -> BatchRouteResult:
        decisions: list[RouteDecision] = []
        errors: dict[int, str] = {}
        for index, request in enumerate(requests):
            try:
                decisions.append(self.router.route(request))
            except Exception as exc:
                if fail_fast:
                    raise
                errors[index] = str(exc)
        return BatchRouteResult(tuple(decisions), errors)
