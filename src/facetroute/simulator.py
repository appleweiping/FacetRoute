"""Offline policy simulation and aggregate evaluation."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .features import QueryFeatureExtractor
from .feedback import FeedbackEvent, FeedbackLog
from .routers import Router
from .types import ModelCandidate, RouteDecision, RouteRequest


@dataclass(frozen=True, slots=True)
class SimulationObservation:
    decision: RouteDecision
    feedback: FeedbackEvent
    oracle_quality: float
    regret: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    total_requests: int
    routed_requests: int
    failed_requests: int
    average_reward: float
    average_cost_usd: float
    p95_latency_ms: float | None
    success_rate: float
    average_quality_regret: float
    selection_counts: Mapping[str, int]
    failures: Mapping[int, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "routed_requests": self.routed_requests,
            "failed_requests": self.failed_requests,
            "average_reward": self.average_reward,
            "average_cost_usd": self.average_cost_usd,
            "p95_latency_ms": self.p95_latency_ms,
            "success_rate": self.success_rate,
            "average_quality_regret": self.average_quality_regret,
            "selection_counts": dict(self.selection_counts),
            "failures": {str(key): value for key, value in self.failures.items()},
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")


class OfflineSimulator:
    """Exercise a routing policy using a seeded, inspectable reward model."""

    def __init__(
        self,
        router: Router,
        candidates: Iterable[ModelCandidate],
        *,
        seed: int = 7,
        feedback_log: FeedbackLog | None = None,
        extractor: QueryFeatureExtractor | None = None,
    ) -> None:
        self.router = router
        self.candidates = {item.model_id: item for item in candidates}
        if not self.candidates:
            raise ValueError("simulator requires at least one candidate")
        self.random = random.Random(seed)
        self.feedback_log = feedback_log
        self.extractor = extractor or QueryFeatureExtractor()

    def run(
        self, requests: Iterable[RouteRequest], *, learn: bool = False
    ) -> tuple[tuple[SimulationObservation, ...], EvaluationReport]:
        request_list = tuple(requests)
        update = getattr(self.router, "update_feedback", None) if learn else None
        if learn and not callable(update):
            raise ValueError("selected router does not support online updates")
        observations: list[SimulationObservation] = []
        failures: dict[int, str] = {}
        for index, request in enumerate(request_list):
            try:
                decision = self.router.route(request)
                observation = self._observe(request, decision)
                if self.feedback_log is not None:
                    self.feedback_log.append(observation.feedback)
                if update is not None:
                    update(observation.feedback)
                observations.append(observation)
            except Exception as exc:
                failures[index] = str(exc)
        report = self._report(len(request_list), observations, failures)
        return tuple(observations), report

    def _observe(self, request: RouteRequest, decision: RouteDecision) -> SimulationObservation:
        candidate = self.candidates[decision.selected_model]
        features = self.extractor.extract(request)
        selected_quality = candidate.quality_for(features.task)
        eligible_candidates = (
            item
            for item in self.candidates.values()
            if item.model_id not in decision.excluded
        )
        oracle_quality = max(item.quality_for(features.task) for item in eligible_candidates)
        difficulty_penalty = 0.18 * features.difficulty
        preference_bonus = 0.03 if candidate.model_id in getattr(
            getattr(self.router, "preference_for", lambda _user: None)(request.user_id),
            "preferred_models",
            (),
        ) else 0.0
        noise = self.random.uniform(-0.025, 0.025)
        reward = min(1.0, max(0.0, selected_quality - difficulty_penalty + preference_bonus + noise))
        latency = self.random.uniform(candidate.latency_ms_p50, candidate.latency_ms_p95)
        success = self.random.random() < reward
        feedback = FeedbackEvent(
            request_id=request.request_id,
            user_id=request.user_id,
            model_id=candidate.model_id,
            reward=reward,
            policy=decision.policy,
            context_vector=decision.context_vector,
            success=success,
            latency_ms=latency,
            cost_usd=decision.breakdown.estimated_cost_usd,
            tags={"task": features.task, "source": "offline-simulation"},
        )
        return SimulationObservation(
            decision=decision,
            feedback=feedback,
            oracle_quality=oracle_quality,
            regret=max(0.0, oracle_quality - selected_quality),
        )

    @staticmethod
    def _report(
        total: int,
        observations: list[SimulationObservation],
        failures: Mapping[int, str],
    ) -> EvaluationReport:
        count = len(observations)
        rewards = [item.feedback.reward for item in observations]
        costs = [item.feedback.cost_usd or 0.0 for item in observations]
        latencies = sorted(
            item.feedback.latency_ms
            for item in observations
            if item.feedback.latency_ms is not None
        )
        selections: dict[str, int] = {}
        for item in observations:
            selections[item.decision.selected_model] = selections.get(item.decision.selected_model, 0) + 1
        p95: float | None = None
        if latencies:
            index = min(len(latencies) - 1, max(0, math.ceil(0.95 * len(latencies)) - 1))
            p95 = latencies[index]
        return EvaluationReport(
            total_requests=total,
            routed_requests=count,
            failed_requests=len(failures),
            average_reward=(sum(rewards) / count if count else 0.0),
            average_cost_usd=(sum(costs) / count if count else 0.0),
            p95_latency_ms=p95,
            success_rate=(
                sum(1 for item in observations if item.feedback.success) / count if count else 0.0
            ),
            average_quality_regret=(
                sum(item.regret for item in observations) / count if count else 0.0
            ),
            selection_counts=dict(sorted(selections.items())),
            failures=dict(failures),
        )
