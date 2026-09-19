"""Deterministic low-rank pairwise routing over training-only lexical features."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ._json import loads_strict
from .constraints import ConstraintEngine
from .errors import ConfigurationError, NoEligibleModelError, PersistenceError
from .features import QueryFeatureExtractor
from .persistence import AtomicJsonStore, _json_bytes
from .routers import RuleRouter
from .rules import RoutingRule, match_rules
from .scoring import MultiObjectiveScorer, ScoredCandidate
from .similarity import (
    SimilarityFeatureConfig,
    SimilarityModel,
    _bounded_integer,
    _bounded_number,
    _configured_extractor,
    _ordered_traces,
    _plain_utf8_string,
    _tuple_snapshot,
)
from .splitting import _group_key
from .traces import RouteTrace, traces_sha256
from .types import ModelCandidate, RouteDecision, RouteRequest, UserPreferences

FACTOR_FORMAT = "facetroute-pairwise-factorization"
FACTOR_SCHEMA_VERSION = 1
_MAX_STATE_BYTES = 32 * 1024 * 1024
_MAX_FACTORS = 250_000
_MAX_TRAINING_UPDATES = 20_000_000


def _integer(name: str, value: object, minimum: int, maximum: int) -> int:
    return _bounded_integer(name, value, minimum, maximum)


def _number(name: str, value: object, minimum: float, maximum: float) -> float:
    return _bounded_number(name, value, minimum, maximum)


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _weight(seed: int, label: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{label}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return (2.0 * fraction - 1.0) * 0.05


def _logistic_derivative(delta: float) -> float:
    """Return sigmoid(-delta) without overflow."""

    if delta >= 0:
        value = math.exp(-delta)
        return value / (1.0 + value)
    value = math.exp(delta)
    return 1.0 / (1.0 + value)


def _logistic_loss(delta: float) -> float:
    if delta >= 0:
        return math.log1p(math.exp(-delta))
    return -delta + math.log1p(math.exp(delta))


@dataclass(frozen=True, slots=True)
class FactorizationConfig:
    """Bounded optimization settings; all stochastic-looking state is hash seeded."""

    dimension: int = 8
    epochs: int = 30
    learning_rate: float = 0.05
    regularization: float = 0.0001
    seed: int = 17
    max_records: int = 10_000
    max_pairs: int = 100_000
    max_features: int = 512

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("dimension", 1, 64),
            ("epochs", 1, 200),
            ("max_records", 1, 100_000),
            ("max_pairs", 1, 1_000_000),
            ("max_features", 16, 8_192),
        ):
            object.__setattr__(self, name, _integer(name, getattr(self, name), minimum, maximum))
        object.__setattr__(self, "seed", _integer("seed", self.seed, -(2**63), 2**63 - 1))
        object.__setattr__(
            self,
            "learning_rate",
            _number("learning_rate", self.learning_rate, 0.000001, 0.5),
        )
        object.__setattr__(
            self,
            "regularization",
            _number("regularization", self.regularization, 0.0, 0.1),
        )
        if self.dimension * self.max_features > _MAX_FACTORS:
            raise ConfigurationError("factorization matrix exceeds the supported factor limit")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "dimension": self.dimension,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "seed": self.seed,
            "max_records": self.max_records,
            "max_pairs": self.max_pairs,
            "max_features": self.max_features,
        }

    @classmethod
    def from_dict(cls, payload: object) -> FactorizationConfig:
        if not isinstance(payload, dict) or set(payload) != set(cls().to_dict()):
            raise ConfigurationError("factorization config fields are invalid")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PairwiseFactorModel:
    """A low-rank model-feature interaction matrix trained on route preferences."""

    config: FactorizationConfig
    encoder: SimilarityModel
    route_ids: tuple[str, ...]
    route_factors: tuple[tuple[float, ...], ...]
    projection: tuple[tuple[float, ...], ...]
    training_pairs: int
    final_loss: float

    def __post_init__(self) -> None:
        if type(self.config) is not FactorizationConfig:
            raise ConfigurationError("factorization config must be exact built-in settings")
        if type(self.encoder) is not SimilarityModel:
            raise ConfigurationError("factorization encoder must be a SimilarityModel")
        object.__setattr__(self, "config", FactorizationConfig.from_dict(self.config.to_dict()))
        object.__setattr__(self, "encoder", SimilarityModel.from_dict(self.encoder.to_dict()))
        routes_raw = _tuple_snapshot(self.route_ids)
        if routes_raw is None:
            raise ConfigurationError("factorization routes must be a tuple")
        normalized_route_ids: list[str] = []
        for raw_route in routes_raw:
            route = _plain_utf8_string(raw_route)
            if route is None:
                raise ConfigurationError("factorization routes must be UTF-8 identifiers")
            normalized_route_ids.append(route)
        routes = tuple(normalized_route_ids)
        if (
            len(routes) < 2
            or len(routes) > self.config.max_records
            or routes != tuple(sorted(set(routes)))
            or any(
                not route.strip() or route != route.strip() or len(route) > 512 for route in routes
            )
        ):
            raise ConfigurationError("factorization routes must be sorted unique identifiers")
        object.__setattr__(self, "route_ids", routes)
        if not set(self.encoder.prototypes).issubset(routes):
            raise ConfigurationError("factorization encoder references routes outside factor state")
        expected_encoder_config = SimilarityFeatureConfig(
            max_features=self.config.max_features,
            max_records=self.config.max_records,
            max_prototype_values=_MAX_FACTORS,
        )
        if self.encoder.config != expected_encoder_config:
            raise ConfigurationError(
                "factorization encoder configuration disagrees with factor state"
            )
        dimension = self.config.dimension
        features = len(self.encoder.feature_names)
        if (
            features > self.config.max_features
            or dimension * features + dimension * len(routes) > _MAX_FACTORS
        ):
            raise ConfigurationError("factorization feature matrix exceeds configured bounds")
        route_rows = _tuple_snapshot(self.route_factors)
        projection_rows = _tuple_snapshot(self.projection)
        if route_rows is None or projection_rows is None:
            raise ConfigurationError("factorization matrices must be tuples")
        if len(route_rows) != len(routes) or len(projection_rows) != dimension:
            raise ConfigurationError("factorization matrix dimensions are invalid")
        normalized_routes: list[tuple[float, ...]] = []
        for row_raw in route_rows:
            row = _tuple_snapshot(row_raw)
            if row is None or len(row) != dimension:
                raise ConfigurationError("factorization route matrix dimensions are invalid")
            normalized_routes.append(
                tuple(_number("route factor", value, -1_000_000.0, 1_000_000.0) for value in row)
            )
        normalized_projection: list[tuple[float, ...]] = []
        for row_raw in projection_rows:
            row = _tuple_snapshot(row_raw)
            if row is None or len(row) != features:
                raise ConfigurationError("factorization projection dimensions are invalid")
            normalized_projection.append(
                tuple(
                    _number("projection factor", value, -1_000_000.0, 1_000_000.0) for value in row
                )
            )
        object.__setattr__(self, "route_factors", tuple(normalized_routes))
        object.__setattr__(self, "projection", tuple(normalized_projection))
        object.__setattr__(
            self,
            "training_pairs",
            _integer("training_pairs", self.training_pairs, 1, self.config.max_pairs),
        )
        object.__setattr__(
            self, "final_loss", _number("final_loss", self.final_loss, 0.0, 1_000_000.0)
        )
        try:
            self._state_bytes()
        except ValueError as exc:
            raise ConfigurationError(
                "factorization state exceeds the configured byte limit"
            ) from exc

    @classmethod
    def fit(
        cls,
        traces: Iterable[RouteTrace],
        *,
        config: FactorizationConfig | None = None,
    ) -> PairwiseFactorModel:
        """Fit a bilinear preference model without looking at held-out traces."""

        settings = FactorizationConfig() if config is None else config
        if type(settings) is not FactorizationConfig:
            raise ConfigurationError("factorization config must be FactorizationConfig")
        feature_settings = SimilarityFeatureConfig(
            max_features=settings.max_features,
            max_records=settings.max_records,
            max_prototype_values=_MAX_FACTORS,
        )
        ordered = _ordered_traces(traces, feature_settings, purpose="factorization training")
        observed_routes: set[str] = set()
        proposed_pairs = 0
        for trace in ordered:
            for raw_route in trace.outcomes:
                route = _plain_utf8_string(raw_route)
                if route is None or not route.strip() or route != route.strip() or len(route) > 512:
                    raise ConfigurationError("factorization outcome route ID must be bounded UTF-8")
                observed_routes.add(route)
            proposed_pairs += max(0, len(trace.outcomes) - 1)
            if len(observed_routes) > settings.max_records:
                raise ConfigurationError("factorization exceeds the configured route limit")
            if proposed_pairs > settings.max_pairs:
                raise ConfigurationError("factorization exceeds the configured pair limit")
        route_ids = tuple(sorted(observed_routes))
        if len(route_ids) < 2:
            raise ConfigurationError("factorization requires at least two observed routes")
        try:
            encoder = SimilarityModel.fit(ordered, config=feature_settings)
        except ConfigurationError:
            raise
        except (UnicodeError, TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ConfigurationError(
                "factorization traces cannot be encoded as strict JSON"
            ) from exc
        indices = {route: index for index, route in enumerate(route_ids)}
        documents: list[tuple[tuple[int, float], ...]] = []
        pairs: list[tuple[int, int, int]] = []
        for trace in ordered:
            sparse = tuple(
                (index, value)
                for index, value in enumerate(encoder.vectorize(trace.request))
                if value != 0.0
            )
            documents.append(sparse)
            winner = trace.preferred_model
            if winner is None:  # _ordered_traces rejects missing labels.
                raise ConfigurationError("factorization requires preferred_model labels")
            for loser in sorted(trace.outcomes):
                if loser == winner:
                    continue
                if len(pairs) >= settings.max_pairs:
                    raise ConfigurationError("factorization exceeds the configured pair limit")
                pairs.append((len(documents) - 1, indices[winner], indices[loser]))
        if not pairs:
            raise ConfigurationError("factorization needs at least one labelled comparison")
        dimension = settings.dimension
        if dimension * (len(route_ids) + len(encoder.feature_names)) > _MAX_FACTORS:
            raise ConfigurationError("factorization matrices exceed the supported factor limit")
        estimated_updates = settings.epochs * sum(
            dimension * (len(documents[document_index]) + 3)
            for document_index, _winner, _loser in pairs
        )
        if estimated_updates > _MAX_TRAINING_UPDATES:
            raise ConfigurationError("factorization optimization exceeds the training work limit")
        projection = [
            [
                _weight(settings.seed, f"projection:{axis}:{feature}")
                for feature in encoder.feature_names
            ]
            for axis in range(dimension)
        ]
        route_factors = [
            [_weight(settings.seed, f"route:{route}:{axis}") for axis in range(dimension)]
            for route in route_ids
        ]
        for _epoch in range(settings.epochs):
            for document_index, winner_index, loser_index in pairs:
                sparse = documents[document_index]
                embedded = [
                    math.fsum(row[index] * value for index, value in sparse) for row in projection
                ]
                old_difference = [
                    route_factors[winner_index][axis] - route_factors[loser_index][axis]
                    for axis in range(dimension)
                ]
                delta = math.fsum(
                    difference * value
                    for difference, value in zip(old_difference, embedded, strict=True)
                )
                if not math.isfinite(delta):
                    raise ConfigurationError(
                        "factorization optimization produced a non-finite score"
                    )
                gradient = _logistic_derivative(delta)
                for axis in range(dimension):
                    win_old = route_factors[winner_index][axis]
                    lose_old = route_factors[loser_index][axis]
                    route_factors[winner_index][axis] += settings.learning_rate * (
                        gradient * embedded[axis] - settings.regularization * win_old
                    )
                    route_factors[loser_index][axis] += settings.learning_rate * (
                        -gradient * embedded[axis] - settings.regularization * lose_old
                    )
                    row = projection[axis]
                    for index, value in sparse:
                        old = row[index]
                        row[index] += settings.learning_rate * (
                            gradient * old_difference[axis] * value - settings.regularization * old
                        )
            if any(
                not math.isfinite(value) for row in (*route_factors, *projection) for value in row
            ):
                raise ConfigurationError("factorization optimization produced non-finite factors")
        final_losses: list[float] = []
        for document_index, winner_index, loser_index in pairs:
            sparse = documents[document_index]
            final_embedding = tuple(
                math.fsum(row[index] * value for index, value in sparse) for row in projection
            )
            margin = math.fsum(
                (route_factors[winner_index][axis] - route_factors[loser_index][axis])
                * final_embedding[axis]
                for axis in range(dimension)
            )
            if not math.isfinite(margin):
                raise ConfigurationError("factorization produced a non-finite final score")
            final_losses.append(_logistic_loss(margin))
        final_loss = math.fsum(final_losses) / len(final_losses)
        return cls(
            config=settings,
            encoder=encoder,
            route_ids=route_ids,
            route_factors=tuple(tuple(row) for row in route_factors),
            projection=tuple(tuple(row) for row in projection),
            training_pairs=len(pairs),
            final_loss=final_loss,
        )

    def scores(self, request: RouteRequest) -> tuple[tuple[str, float], ...]:
        sparse = tuple(
            (index, value)
            for index, value in enumerate(self.encoder.vectorize(request))
            if value != 0.0
        )
        embedded = tuple(
            math.fsum(row[index] * value for index, value in sparse) for row in self.projection
        )
        ranked = []
        for route, factors in zip(self.route_ids, self.route_factors, strict=True):
            score = math.fsum(left * right for left, right in zip(factors, embedded, strict=True))
            if not math.isfinite(score):
                raise ConfigurationError("factorization produced a non-finite score")
            ranked.append((route, score))
        return tuple(sorted(ranked, key=lambda item: (-item[1], item[0])))

    def _payload(self) -> dict[str, Any]:
        return {
            "format": FACTOR_FORMAT,
            "schema_version": FACTOR_SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "encoder": self.encoder.to_dict(),
            "route_ids": list(self.route_ids),
            "route_factors": [list(row) for row in self.route_factors],
            "projection": [list(row) for row in self.projection],
            "training_pairs": self.training_pairs,
            "final_loss": self.final_loss,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        return {**payload, "state_sha256": _digest(payload)}

    @classmethod
    def from_dict(cls, payload: object) -> PairwiseFactorModel:
        try:
            if type(payload) is not dict or set(payload) != {
                "format",
                "schema_version",
                "config",
                "encoder",
                "route_ids",
                "route_factors",
                "projection",
                "training_pairs",
                "final_loss",
                "state_sha256",
            }:
                raise ConfigurationError("factorization state has invalid fields")
            state_hash = payload["state_sha256"]
            if not isinstance(state_hash, str) or not hmac.compare_digest(
                state_hash,
                _digest({key: value for key, value in payload.items() if key != "state_sha256"}),
            ):
                raise ConfigurationError("factorization state integrity check failed")
            if (
                type(payload["format"]) is not str
                or payload["format"] != FACTOR_FORMAT
                or type(payload["schema_version"]) is not int
                or payload["schema_version"] != FACTOR_SCHEMA_VERSION
            ):
                raise ConfigurationError("unsupported factorization state schema")
            return cls(
                config=FactorizationConfig.from_dict(payload["config"]),
                encoder=SimilarityModel.from_dict(payload["encoder"]),
                route_ids=tuple(payload["route_ids"]),
                route_factors=tuple(tuple(row) for row in payload["route_factors"]),
                projection=tuple(tuple(row) for row in payload["projection"]),
                training_pairs=payload["training_pairs"],
                final_loss=payload["final_loss"],
            )
        except (
            ConfigurationError,
            PersistenceError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
        ) as exc:
            raise PersistenceError(f"Invalid factorization state: {exc}") from exc

    def save(self, path: str | Path) -> None:
        AtomicJsonStore(path).save(self.to_dict(), max_bytes=_MAX_STATE_BYTES)

    @classmethod
    def load(cls, path: str | Path) -> PairwiseFactorModel:
        model, _digest_hex = cls.load_with_sha256(path)
        return model

    @classmethod
    def load_with_sha256(cls, path: str | Path) -> tuple[PairwiseFactorModel, str]:
        source = Path(path)
        try:
            with source.open("rb") as handle:
                raw = handle.read(_MAX_STATE_BYTES + 1)
        except OSError as exc:
            raise PersistenceError(f"Cannot read factorization state {source}: {exc}") from exc
        if len(raw) > _MAX_STATE_BYTES:
            raise PersistenceError("factorization state exceeds the configured byte limit")
        try:
            return cls.from_dict(loads_strict(raw)), hashlib.sha256(raw).hexdigest()
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise PersistenceError(f"Cannot decode factorization state: {exc}") from exc

    def _state_bytes(self) -> bytes:
        return _json_bytes(self.to_dict(), max_bytes=_MAX_STATE_BYTES)


def evaluate_held_out(
    model: PairwiseFactorModel,
    training: Iterable[RouteTrace],
    held_out: Iterable[RouteTrace],
    *,
    group_by: str = "request_id",
) -> dict[str, object]:
    """Evaluate once on disjoint labelled traces; never change the fitted model."""

    if type(model) is not PairwiseFactorModel:
        raise ConfigurationError("evaluation requires a PairwiseFactorModel")
    feature_config = model.encoder.config
    train = _ordered_traces(training, feature_config, purpose="factorization training audit")
    test = _ordered_traces(held_out, feature_config, purpose="factorization held-out audit")
    train_groups = {_group_key(trace, group_by) for trace in train}
    test_groups = {_group_key(trace, group_by) for trace in test}
    if train_groups & test_groups:
        raise ConfigurationError("factorization training and held-out groups overlap")
    if {trace.request.request_id for trace in train} & {trace.request.request_id for trace in test}:
        raise ConfigurationError("factorization training and held-out request ids overlap")
    if _trace_digest(train, "training") != model.encoder.training_sha256:
        raise ConfigurationError("training audit traces do not match the fitted encoder")
    test_digest = _trace_digest(test, "held-out")
    correct = 0
    top1_evaluable = 0
    pairwise_wins = 0
    pairwise_total = 0
    losses: list[float] = []
    for trace in test:
        if trace.preferred_model not in model.route_ids:
            raise ConfigurationError("held-out label was unseen in factorization training")
        if set(trace.outcomes) - set(model.route_ids):
            raise ConfigurationError("held-out outcomes include unseen factorization routes")
        scores = dict(model.scores(trace.request))
        preferred = trace.preferred_model
        if preferred is None:  # _ordered_traces rejects missing labels.
            raise ConfigurationError("held-out trace requires preferred_model")
        if len(trace.outcomes) > 1:
            top1_evaluable += 1
            observed_winner = next(route for route in scores if route in trace.outcomes)
            if observed_winner == preferred:
                correct += 1
        for loser in sorted(trace.outcomes):
            if loser == preferred or loser not in scores:
                continue
            margin = scores[preferred] - scores[loser]
            losses.append(_logistic_loss(margin))
            pairwise_wins += margin > 0.0
            pairwise_total += 1
    return {
        "training_sha256": model.encoder.training_sha256,
        "held_out_sha256": test_digest,
        "group_by": group_by,
        "train_records": len(train),
        "held_out_records": len(test),
        "top1_accuracy": correct / top1_evaluable if top1_evaluable else None,
        "top1_evaluable_records": top1_evaluable,
        "pairwise_accuracy": pairwise_wins / pairwise_total if pairwise_total else None,
        "pairwise_log_loss": math.fsum(losses) / pairwise_total if pairwise_total else None,
        "training_pairs": model.training_pairs,
        "training_loss": model.final_loss,
    }


def _trace_digest(traces: tuple[RouteTrace, ...], purpose: str) -> str:
    try:
        return traces_sha256(traces)
    except ConfigurationError:
        raise
    except (UnicodeError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ConfigurationError(f"factorization {purpose} traces are not strict JSON") from exc


class FactorizationRouter(RuleRouter):
    """Use learned pairwise logits only after all hard catalog constraints pass."""

    policy_name = "factorization"

    def __init__(
        self,
        candidates: Iterable[ModelCandidate],
        model: PairwiseFactorModel,
        preferences: Mapping[str, UserPreferences] | None = None,
        rules: Iterable[RoutingRule] = (),
        *,
        extractor: QueryFeatureExtractor | None = None,
        constraints: ConstraintEngine | None = None,
        scorer: MultiObjectiveScorer | None = None,
    ) -> None:
        if type(model) is not PairwiseFactorModel:
            raise ConfigurationError("model must be PairwiseFactorModel")
        super().__init__(
            candidates,
            preferences,
            rules,
            extractor=_configured_extractor(model.encoder.config, extractor),
            constraints=constraints,
            scorer=scorer,
        )
        unknown = set(model.route_ids) - {candidate.model_id for candidate in self.candidates}
        if unknown:
            raise ConfigurationError(
                f"factorization state references models outside catalog: {sorted(unknown)}"
            )
        self.model = model

    def route(self, request: RouteRequest) -> RouteDecision:
        preferences = self.preference_for(request.user_id)
        features = self.extractor.extract(request)
        result = self.constraints.filter(self.candidates, request, features, preferences)
        if not result.eligible:
            raise NoEligibleModelError(result.rejected)
        eligible_ids = {candidate.model_id for candidate in result.eligible}
        trained_scores = tuple(
            (route, score) for route, score in self.model.scores(request) if route in eligible_ids
        )
        bonuses, matched = match_rules(self.rules, features, eligible_ids)
        objective = self.scorer.score(
            result.eligible,
            request,
            features,
            preferences,
            result.estimated_costs,
            bonuses,
        )
        explanation: tuple[str, ...]
        if trained_scores:
            score_by_id = dict(trained_scores)
            ranked = tuple(
                sorted(
                    (
                        ScoredCandidate(
                            item.candidate,
                            replace(
                                item.breakdown,
                                preference_bonus=0.0,
                                rule_bonus=0.0,
                                total=score_by_id[item.candidate.model_id],
                            ),
                        )
                        for item in objective
                        if item.candidate.model_id in score_by_id
                    ),
                    key=lambda item: (-item.breakdown.total, item.candidate.model_id),
                )
            )
            explanation = (
                "trainable pairwise factorization selected the highest eligible logit",
                "soft preference and rule bonuses are audit metadata and do not affect learned scores",
            )
        else:
            ranked = objective
            explanation = ("no trained route remained eligible; used objective fallback",)
        summary = features.to_dict()
        summary["factorization"] = {
            "used": bool(trained_scores),
            "training_sha256": self.model.encoder.training_sha256,
            "training_pairs": self.model.training_pairs,
            "ranking": [{"model_id": route, "logit": score} for route, score in trained_scores],
        }
        return self._decision(
            request,
            preferences,
            summary,
            ranked[0],
            ranked,
            result.rejected,
            matched,
            explanation_prefix=explanation,
        )
