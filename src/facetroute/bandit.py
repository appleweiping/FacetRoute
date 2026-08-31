"""A standard-library LinUCB contextual bandit and routing adapter."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, NoEligibleModelError, PersistenceError
from .features import CONTEXT_DIMENSION, QueryFeatureExtractor
from .feedback import FeedbackEvent
from .persistence import AtomicJsonStore
from .routers import RuleRouter
from .rules import RoutingRule, match_rules
from .scoring import ScoredCandidate
from .types import ModelCandidate, RouteDecision, RouteRequest, UserPreferences


def _identity(size: int, diagonal: float) -> list[list[float]]:
    return [[diagonal if row == column else 0.0 for column in range(size)] for row in range(size)]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _validate_inverse_covariance(matrix: list[list[float]], model_id: str) -> None:
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise PersistenceError(f"Non-finite inverse covariance for arm {model_id}")
    dimension = len(matrix)
    for row in range(dimension):
        for column in range(row):
            if not math.isclose(
                matrix[row][column], matrix[column][row], rel_tol=1e-9, abs_tol=1e-12
            ):
                raise PersistenceError(f"Non-symmetric inverse covariance for arm {model_id}")

    # A valid inverse covariance is symmetric positive definite. Cholesky
    # validation prevents corrupted state from producing negative uncertainty
    # or a zero Sherman-Morrison denominator on a later update.
    lower = [[0.0] * dimension for _ in range(dimension)]
    for row in range(dimension):
        for column in range(row + 1):
            residual = matrix[row][column] - sum(
                lower[row][index] * lower[column][index] for index in range(column)
            )
            if row == column:
                if residual <= 0.0 or not math.isfinite(residual):
                    raise PersistenceError(
                        f"Inverse covariance is not positive definite for arm {model_id}"
                    )
                lower[row][column] = math.sqrt(residual)
            else:
                lower[row][column] = residual / lower[column][column]


@dataclass(slots=True)
class LinUCBArm:
    """Per-model inverse covariance, reward vector, and update count."""

    inverse_covariance: list[list[float]]
    reward_vector: list[float]
    updates: int = 0

    @classmethod
    def fresh(cls, dimension: int, ridge: float) -> LinUCBArm:
        return cls(_identity(dimension, 1.0 / ridge), [0.0] * dimension, 0)

    def estimate(self, context: Sequence[float]) -> tuple[float, float]:
        projected_context = _mat_vec(self.inverse_covariance, context)
        theta = _mat_vec(self.inverse_covariance, self.reward_vector)
        prediction = _dot(theta, context)
        uncertainty = math.sqrt(max(0.0, _dot(context, projected_context)))
        return prediction, uncertainty

    def update(self, context: Sequence[float], reward: float) -> None:
        projected = _mat_vec(self.inverse_covariance, context)
        denominator = 1.0 + _dot(context, projected)
        size = len(context)
        for row in range(size):
            for column in range(size):
                self.inverse_covariance[row][column] -= (
                    projected[row] * projected[column] / denominator
                )
        for index, value in enumerate(context):
            self.reward_vector[index] += reward * value
        self.updates += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "inverse_covariance": self.inverse_covariance,
            "reward_vector": self.reward_vector,
            "updates": self.updates,
        }


class LinUCBPolicy:
    """Multi-arm LinUCB state with JSON round-trip support."""

    schema_version = 1

    def __init__(
        self,
        model_ids: Iterable[str],
        *,
        dimension: int = CONTEXT_DIMENSION,
        alpha: float = 0.35,
        ridge: float = 1.0,
    ) -> None:
        raw_identifiers = tuple(model_ids)
        if not raw_identifiers:
            raise ConfigurationError("LinUCB requires at least one model arm")
        if any(not isinstance(identifier, str) or not identifier.strip() for identifier in raw_identifiers):
            raise ConfigurationError("LinUCB model arm identifiers must be non-empty strings")
        normalized_identifiers = tuple(identifier.strip() for identifier in raw_identifiers)
        if len(normalized_identifiers) != len(set(normalized_identifiers)):
            raise ConfigurationError("LinUCB model arm identifiers must be unique")
        identifiers = tuple(sorted(normalized_identifiers))
        if dimension <= 0:
            raise ConfigurationError("LinUCB dimension must be positive")
        if alpha < 0 or not math.isfinite(alpha):
            raise ConfigurationError("LinUCB alpha must be finite and non-negative")
        if ridge <= 0 or not math.isfinite(ridge):
            raise ConfigurationError("LinUCB ridge must be finite and positive")
        self.dimension = dimension
        self.alpha = alpha
        self.ridge = ridge
        self.arms = {identifier: LinUCBArm.fresh(dimension, ridge) for identifier in identifiers}

    def score(
        self,
        model_id: str,
        context: Sequence[float],
        exploration_scale: float = 1.0,
    ) -> tuple[float, float, float]:
        self._validate_context(context)
        if model_id not in self.arms:
            raise ConfigurationError(f"unknown LinUCB arm: {model_id}")
        if exploration_scale < 0 or not math.isfinite(exploration_scale):
            raise ConfigurationError("exploration_scale must be finite and non-negative")
        prediction, uncertainty = self.arms[model_id].estimate(context)
        upper_confidence = prediction + self.alpha * exploration_scale * uncertainty
        return upper_confidence, prediction, uncertainty

    def update(self, model_id: str, context: Sequence[float], reward: float) -> None:
        self._validate_context(context)
        if model_id not in self.arms:
            raise ConfigurationError(f"unknown LinUCB arm: {model_id}")
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ConfigurationError("LinUCB reward must be between 0 and 1")
        self.arms[model_id].update(context, reward)

    def _validate_context(self, context: Sequence[float]) -> None:
        if len(context) != self.dimension:
            raise ConfigurationError(
                f"context dimension {len(context)} does not match {self.dimension}"
            )
        if any(not math.isfinite(value) for value in context):
            raise ConfigurationError("context values must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dimension": self.dimension,
            "alpha": self.alpha,
            "ridge": self.ridge,
            "arms": {model_id: arm.to_dict() for model_id, arm in sorted(self.arms.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LinUCBPolicy:
        try:
            if int(data["schema_version"]) != cls.schema_version:
                raise PersistenceError("Unsupported LinUCB schema")
            arm_payload = dict(data["arms"])
            if any(
                not isinstance(model_id, str)
                or not model_id.strip()
                or model_id != model_id.strip()
                for model_id in arm_payload
            ):
                raise PersistenceError("LinUCB arm identifiers must be trimmed non-empty strings")
            policy = cls(
                arm_payload.keys(),
                dimension=int(data["dimension"]),
                alpha=float(data["alpha"]),
                ridge=float(data["ridge"]),
            )
            for model_id, raw in arm_payload.items():
                arm_data = dict(raw)
                matrix = [[float(value) for value in row] for row in arm_data["inverse_covariance"]]
                vector = [float(value) for value in arm_data["reward_vector"]]
                if len(matrix) != policy.dimension or any(
                    len(row) != policy.dimension for row in matrix
                ) or len(vector) != policy.dimension:
                    raise PersistenceError(f"Invalid matrix shape for arm {model_id}")
                _validate_inverse_covariance(matrix, str(model_id))
                if any(not math.isfinite(value) for value in vector):
                    raise PersistenceError(f"Non-finite reward vector for arm {model_id}")
                updates_raw = arm_data.get("updates", 0)
                if (
                    isinstance(updates_raw, bool)
                    or not isinstance(updates_raw, int)
                    or updates_raw < 0
                ):
                    raise PersistenceError(f"Invalid update count for arm {model_id}")
                policy.arms[str(model_id)] = LinUCBArm(
                    inverse_covariance=matrix,
                    reward_vector=vector,
                    updates=updates_raw,
                )
            return policy
        except PersistenceError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise PersistenceError(f"Invalid LinUCB state: {exc}") from exc

    def save(self, path: str | Path) -> None:
        AtomicJsonStore(path).save(self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> LinUCBPolicy:
        payload = AtomicJsonStore(path).load()
        if payload is None:
            raise PersistenceError(f"LinUCB state does not exist: {path}")
        if not isinstance(payload, dict):
            raise PersistenceError("LinUCB state root must be an object")
        return cls.from_dict(payload)


class LinUCBRouter(RuleRouter):
    """Filter candidates, then combine LinUCB confidence with an explicit prior."""

    policy_name = "linucb"

    def __init__(
        self,
        candidates: Iterable[ModelCandidate],
        preferences: Mapping[str, UserPreferences] | None = None,
        rules: Iterable[RoutingRule] = (),
        *,
        policy: LinUCBPolicy | None = None,
        prior_weight: float = 0.2,
        extractor: QueryFeatureExtractor | None = None,
    ) -> None:
        super().__init__(candidates, preferences, rules, extractor=extractor)
        if prior_weight < 0 or not math.isfinite(prior_weight):
            raise ConfigurationError("prior_weight must be finite and non-negative")
        self.policy = policy or LinUCBPolicy(item.model_id for item in self.candidates)
        if self.policy.dimension != CONTEXT_DIMENSION:
            raise ConfigurationError(
                f"LinUCB state dimension must be {CONTEXT_DIMENSION}, got {self.policy.dimension}"
            )
        missing = {item.model_id for item in self.candidates} - set(self.policy.arms)
        if missing:
            raise ConfigurationError(f"LinUCB state is missing model arms: {sorted(missing)}")
        self.prior_weight = prior_weight

    def route(self, request: RouteRequest) -> RouteDecision:
        preferences = self.preference_for(request.user_id)
        features = self.extractor.extract(request)
        filtered = self.constraints.filter(self.candidates, request, features, preferences)
        if not filtered.eligible:
            raise NoEligibleModelError(filtered.rejected)
        bonuses, matched = match_rules(
            self.rules, features, {candidate.model_id for candidate in filtered.eligible}
        )
        priors = self.scorer.score(
            filtered.eligible,
            request,
            features,
            preferences,
            filtered.estimated_costs,
            bonuses,
        )
        context = self.extractor.context_vector(features, preferences)
        rescored: list[ScoredCandidate] = []
        diagnostics: dict[str, tuple[float, float]] = {}
        for prior in priors:
            upper, prediction, uncertainty = self.policy.score(
                prior.candidate.model_id,
                context,
                preferences.exploration_weight,
            )
            total = upper + self.prior_weight * prior.breakdown.total
            diagnostics[prior.candidate.model_id] = (prediction, uncertainty)
            rescored.append(
                ScoredCandidate(prior.candidate, replace(prior.breakdown, total=total))
            )
        scored = tuple(
            sorted(rescored, key=lambda item: (-item.breakdown.total, item.candidate.model_id))
        )
        selected = scored[0]
        prediction, uncertainty = diagnostics[selected.candidate.model_id]
        return self._decision(
            request,
            preferences,
            features.to_dict(),
            selected,
            scored,
            filtered.rejected,
            matched,
            context_vector=context,
            explanation_prefix=(
                f"LinUCB predicted reward={prediction:.4f}, uncertainty={uncertainty:.4f}",
                f"deterministic prior weight={self.prior_weight:.3f}",
            ),
        )

    def update_feedback(self, event: FeedbackEvent) -> None:
        if event.policy != self.policy_name:
            raise ConfigurationError(
                f"feedback policy '{event.policy}' does not match '{self.policy_name}'"
            )
        self.policy.update(event.model_id, event.context_vector, event.reward)

    def save_state(self, path: str | Path) -> None:
        self.policy.save(path)
