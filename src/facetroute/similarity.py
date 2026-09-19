"""Leakage-safe, dependency-free similarity routing from labelled traces."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._json import loads_strict
from .constraints import ConstraintEngine
from .errors import ConfigurationError, NoEligibleModelError, PersistenceError
from .features import QueryFeatureExtractor
from .persistence import AtomicJsonStore, _json_bytes
from .routers import RuleRouter
from .rules import RoutingRule, match_rules
from .scoring import MultiObjectiveScorer, ScoredCandidate
from .splitting import _group_key
from .traces import RouteTrace, traces_sha256
from .types import ModelCandidate, RouteDecision, RouteRequest, UserPreferences

SIMILARITY_FORMAT = "facetroute-spherical-prototypes"
SIMILARITY_SCHEMA_VERSION = 1
SIMILARITY_FEATURE_SCHEMA_VERSION = 1
_SCHEMA_MAX_ROUTES = 100_000
_SCHEMA_MAX_FEATURES = 8_192
_SCHEMA_MAX_STRING_CHARACTERS = 512
_SCHEMA_MAX_PROTOTYPE_VALUES = 250_000
# A valid UTF-8 string scalar uses at most four bytes; an escaped JSON control
# character uses six. The remaining constants deliberately over-count syntax,
# indentation, finite float/count representations, hashes, and configuration.
_MAX_JSON_BYTES_PER_STRING_CHARACTER = 6
_MAX_ROUTE_JSON_OVERHEAD = 256
_MAX_NUMERIC_JSON_ENTRY_BYTES = 64
_MAX_FIXED_STATE_JSON_OVERHEAD = 1_048_576
_STATE_SCHEMA_UPPER_BOUND_BYTES = (
    _SCHEMA_MAX_ROUTES * _SCHEMA_MAX_STRING_CHARACTERS * _MAX_JSON_BYTES_PER_STRING_CHARACTER
    + _SCHEMA_MAX_FEATURES * _SCHEMA_MAX_STRING_CHARACTERS * _MAX_JSON_BYTES_PER_STRING_CHARACTER
    + _SCHEMA_MAX_ROUTES * _MAX_ROUTE_JSON_OVERHEAD
    + _SCHEMA_MAX_PROTOTYPE_VALUES * _MAX_NUMERIC_JSON_ENTRY_BYTES
    + 2 * _SCHEMA_MAX_FEATURES * _MAX_NUMERIC_JSON_ENTRY_BYTES
    + _MAX_FIXED_STATE_JSON_OVERHEAD
)
# The public schema's conservative upper bound is 376,062,976 bytes, leaving
# 160,807,936 bytes of margin for every state accepted by schema 1.
_MAX_STATE_BYTES = 512 * 1024 * 1024
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_HEX = frozenset("0123456789abcdef")
_FIXED_FEATURE_NAMES = frozenset(
    {
        "bias",
        "numeric:difficulty",
        "numeric:code_fraction",
        "numeric:math_fraction",
        "numeric:length",
        "numeric:tokens",
        "numeric:questions",
        "numeric:multistep",
        "request:tools",
        "request:json",
    }
)
_CATEGORY_PREFIXES = ("task:", "capability:")
_SENSITIVITY_FEATURES = frozenset(
    {"sensitivity:normal", "sensitivity:sensitive", "sensitivity:restricted"}
)


def _bounded_integer(name: str, value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer in [{minimum}, {maximum}]")
    # Call the built-in descriptor directly so an int subclass cannot forge
    # comparisons or conversion. int.__int__ returns an exact immutable int.
    number = int.__int__(value)
    if not minimum <= number <= maximum:
        raise ConfigurationError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return number


def _bounded_number(name: str, value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a finite number in [{minimum}, {maximum}]")
    try:
        raw = int.__int__(value) if isinstance(value, int) else float.__float__(value)
        number = float(raw)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{name} must be a finite number in [{minimum}, {maximum}]"
        ) from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ConfigurationError(f"{name} must be a finite number in [{minimum}, {maximum}]")
    return number


def _plain_utf8_string(value: object) -> str | None:
    """Snapshot the underlying value of a str/subclass as an exact built-in str."""

    if not isinstance(value, str):
        return None
    try:
        # Unbound built-in methods bypass subclass ``encode``/``__str__`` hooks.
        return str.encode(value, "utf-8", "strict").decode("utf-8", "strict")
    except UnicodeError:
        return None


def _tuple_snapshot(value: object) -> tuple[Any, ...] | None:
    """Copy the underlying cells of a tuple/subclass without subclass iteration hooks."""

    if not isinstance(value, tuple):
        return None
    return tuple(tuple.__iter__(value))


def _sha256(value: object) -> str:
    normalized = _plain_utf8_string(value)
    if (
        normalized is None
        or len(normalized) != 64
        or any(character not in _HEX for character in normalized)
    ):
        raise ConfigurationError("digest must be a lowercase SHA-256 value")
    return normalized


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_utf8(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class SimilarityFeatureConfig:
    """Resource limits and the complete versioned feature configuration."""

    max_features: int = 2_048
    min_document_frequency: int = 1
    max_records: int = 100_000
    max_query_characters: int = 100_000
    max_tokens_per_query: int = 2_048
    max_categories_per_request: int = 128
    max_feature_name_characters: int = 128
    max_feature_occurrences: int = 1_000_000
    max_prototype_values: int = 250_000
    long_query_characters: int = 1_200

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_features",
            _bounded_integer("max_features", self.max_features, 16, _SCHEMA_MAX_FEATURES),
        )
        object.__setattr__(
            self,
            "min_document_frequency",
            _bounded_integer("min_document_frequency", self.min_document_frequency, 1, 10_000),
        )
        object.__setattr__(
            self,
            "max_records",
            _bounded_integer("max_records", self.max_records, 1, _SCHEMA_MAX_ROUTES),
        )
        object.__setattr__(
            self,
            "max_query_characters",
            _bounded_integer("max_query_characters", self.max_query_characters, 1, 1_000_000),
        )
        object.__setattr__(
            self,
            "max_tokens_per_query",
            _bounded_integer("max_tokens_per_query", self.max_tokens_per_query, 1, 16_384),
        )
        object.__setattr__(
            self,
            "max_categories_per_request",
            _bounded_integer(
                "max_categories_per_request", self.max_categories_per_request, 1, 1_024
            ),
        )
        object.__setattr__(
            self,
            "max_feature_name_characters",
            _bounded_integer(
                "max_feature_name_characters",
                self.max_feature_name_characters,
                16,
                _SCHEMA_MAX_STRING_CHARACTERS,
            ),
        )
        object.__setattr__(
            self,
            "max_feature_occurrences",
            _bounded_integer("max_feature_occurrences", self.max_feature_occurrences, 1, 2_000_000),
        )
        object.__setattr__(
            self,
            "max_prototype_values",
            _bounded_integer(
                "max_prototype_values",
                self.max_prototype_values,
                1,
                _SCHEMA_MAX_PROTOTYPE_VALUES,
            ),
        )
        object.__setattr__(
            self,
            "long_query_characters",
            _bounded_integer("long_query_characters", self.long_query_characters, 1, 1_000_000),
        )
        if self.min_document_frequency > self.max_records:
            raise ConfigurationError("min_document_frequency cannot exceed max_records")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_features": self.max_features,
            "min_document_frequency": self.min_document_frequency,
            "max_records": self.max_records,
            "max_query_characters": self.max_query_characters,
            "max_tokens_per_query": self.max_tokens_per_query,
            "max_categories_per_request": self.max_categories_per_request,
            "max_feature_name_characters": self.max_feature_name_characters,
            "max_feature_occurrences": self.max_feature_occurrences,
            "max_prototype_values": self.max_prototype_values,
            "long_query_characters": self.long_query_characters,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimilarityFeatureConfig:
        expected = {
            "max_features",
            "min_document_frequency",
            "max_records",
            "max_query_characters",
            "max_tokens_per_query",
            "max_categories_per_request",
            "max_feature_name_characters",
            "max_feature_occurrences",
            "max_prototype_values",
            "long_query_characters",
        }
        if set(data) != expected:
            raise ConfigurationError(
                f"similarity config fields differ: expected {sorted(expected)}, got {sorted(data)}"
            )
        return cls(**{name: data[name] for name in expected})


_SIMILARITY_CONFIG_FIELDS = (
    "max_features",
    "min_document_frequency",
    "max_records",
    "max_query_characters",
    "max_tokens_per_query",
    "max_categories_per_request",
    "max_feature_name_characters",
    "max_feature_occurrences",
    "max_prototype_values",
    "long_query_characters",
)


def _snapshot_config(value: object) -> SimilarityFeatureConfig:
    """Detach configuration from subclasses and later ``object.__setattr__`` mutation."""

    if not isinstance(value, SimilarityFeatureConfig):
        raise ConfigurationError("config must be SimilarityFeatureConfig")
    try:
        values = {
            name: getattr(SimilarityFeatureConfig, name).__get__(value, SimilarityFeatureConfig)
            for name in _SIMILARITY_CONFIG_FIELDS
        }
        return SimilarityFeatureConfig(**values)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError("config must be a valid SimilarityFeatureConfig") from exc


def _feature_name_is_valid(name: str) -> bool:
    if name in _FIXED_FEATURE_NAMES or name in _SENSITIVITY_FEATURES:
        return True
    if name.startswith("term:"):
        token = name.removeprefix("term:")
        return bool(token) and token == token.casefold() and _TOKEN_RE.fullmatch(token) is not None
    for prefix in _CATEGORY_PREFIXES:
        value = name.removeprefix(prefix) if name.startswith(prefix) else ""
        if value and value == value.strip().casefold():
            return True
    return False


def _configured_extractor(
    config: SimilarityFeatureConfig,
    extractor: QueryFeatureExtractor | None,
) -> QueryFeatureExtractor:
    if extractor is not None:
        if type(extractor) is not QueryFeatureExtractor:
            raise ConfigurationError(
                "similarity extractor must be the built-in QueryFeatureExtractor"
            )
        if extractor.long_query_chars != config.long_query_characters:
            raise ConfigurationError(
                "similarity extractor long_query_chars differs from persisted configuration"
            )
    return QueryFeatureExtractor(long_query_chars=config.long_query_characters)


def _category_feature(prefix: str, value: str, config: SimilarityFeatureConfig) -> str:
    normalized = value.strip().casefold()
    name = f"{prefix}:{normalized}"
    if not normalized or len(name) > config.max_feature_name_characters or not _is_utf8(name):
        raise ConfigurationError(
            f"{prefix} feature must be non-empty and at most "
            f"{config.max_feature_name_characters} characters"
        )
    return name


def _raw_features(
    request: RouteRequest,
    extractor: QueryFeatureExtractor,
    config: SimilarityFeatureConfig,
) -> dict[str, float]:
    if len(request.query) > config.max_query_characters:
        raise ConfigurationError(
            f"query exceeds similarity limit of {config.max_query_characters} characters"
        )
    features = extractor.extract(request)
    category_count = (
        2 + len(features.required_capabilities) + int(request.needs_tools) + int(request.needs_json)
    )
    if category_count > config.max_categories_per_request:
        raise ConfigurationError(
            "request exceeds similarity categorical-feature limit of "
            f"{config.max_categories_per_request}"
        )
    categories = {
        _category_feature("task", features.task, config),
        _category_feature("sensitivity", request.sensitivity, config),
        *(
            _category_feature("capability", capability, config)
            for capability in features.required_capabilities
        ),
    }
    if request.needs_tools:
        categories.add("request:tools")
    if request.needs_json:
        categories.add("request:json")
    counts: Counter[str] = Counter()
    for token_count, match in enumerate(_TOKEN_RE.finditer(request.query.casefold()), start=1):
        if token_count > config.max_tokens_per_query:
            raise ConfigurationError(
                f"query exceeds similarity limit of {config.max_tokens_per_query} tokens"
            )
        token = match.group()
        if len(f"term:{token}") <= config.max_feature_name_characters:
            counts[token] += 1
    result: dict[str, float] = {
        "bias": 1.0,
        "numeric:difficulty": features.difficulty,
        "numeric:code_fraction": features.code_fraction,
        "numeric:math_fraction": features.math_fraction,
        "numeric:length": min(len(request.query), config.max_query_characters)
        / config.max_query_characters,
        "numeric:tokens": min(features.token_estimate, 32_768) / 32_768.0,
        "numeric:questions": min(features.question_count, 16) / 16.0,
        "numeric:multistep": float(features.has_multistep_language),
    }
    result.update({name: 1.0 for name in categories})
    result.update({f"term:{token}": 1.0 + math.log(count) for token, count in counts.items()})
    return result


def _normalize(values: Iterable[float]) -> tuple[float, ...]:
    vector = tuple(values)
    try:
        squared_norm = math.fsum(value * value for value in vector)
    except (OverflowError, ValueError) as exc:
        raise ConfigurationError("similarity vector has a non-finite or zero norm") from exc
    if not math.isfinite(squared_norm) or squared_norm <= 0:
        raise ConfigurationError("similarity vector has a non-finite or zero norm")
    try:
        norm = math.sqrt(squared_norm)
        normalized = tuple(value / norm for value in vector)
    except (OverflowError, ValueError, ZeroDivisionError) as exc:
        raise ConfigurationError("similarity vector normalization overflowed") from exc
    if any(not math.isfinite(value) for value in normalized):
        raise ConfigurationError("similarity vector normalization overflowed")
    return normalized


def _ordered_traces(
    traces: Iterable[RouteTrace], config: SimilarityFeatureConfig, *, purpose: str
) -> tuple[RouteTrace, ...]:
    result: list[RouteTrace] = []
    identifiers: set[str] = set()
    for trace in traces:
        if len(result) >= config.max_records:
            raise ConfigurationError(f"{purpose} exceeds {config.max_records} records")
        if not isinstance(trace, RouteTrace):
            raise ConfigurationError(f"{purpose} must contain RouteTrace records")
        if trace.request.request_id in identifiers:
            raise ConfigurationError(f"{purpose} contains duplicate request_id values")
        if trace.preferred_model is None:
            raise ConfigurationError(f"{purpose} traces require preferred_model labels")
        identifiers.add(trace.request.request_id)
        result.append(trace)
    if not result:
        raise ConfigurationError(f"{purpose} requires at least one trace")
    return tuple(sorted(result, key=lambda trace: trace.request.request_id))


def _ordered_digest(traces: tuple[RouteTrace, ...]) -> str:
    return traces_sha256(tuple(sorted(traces, key=lambda trace: trace.request.request_id)))


@dataclass(frozen=True, slots=True)
class SimilarityMatch:
    """One route and its cosine score with inspectable feature contributions."""

    model_id: str
    similarity: float
    contributions: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "similarity": self.similarity,
            "contributions": [
                {"feature": name, "contribution": contribution}
                for name, contribution in self.contributions
            ],
        }


@dataclass(frozen=True, slots=True)
class SimilarityModel:
    """A normalized TF-IDF vocabulary and one spherical centroid per route."""

    config: SimilarityFeatureConfig
    training_sha256: str
    feature_names: tuple[str, ...]
    inverse_document_frequency: tuple[float, ...]
    prototypes: Mapping[str, tuple[float, ...]]
    route_counts: Mapping[str, int]
    similarity_threshold: float = 0.0
    calibration_sha256: str | None = None

    def __post_init__(self) -> None:
        config = _snapshot_config(self.config)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "training_sha256", _sha256(self.training_sha256))
        if self.calibration_sha256 is not None:
            object.__setattr__(self, "calibration_sha256", _sha256(self.calibration_sha256))
        object.__setattr__(
            self,
            "similarity_threshold",
            _bounded_number("similarity_threshold", self.similarity_threshold, 0.0, 1.0),
        )
        raw_names = _tuple_snapshot(self.feature_names)
        if raw_names is None:
            raise ConfigurationError("similarity feature names must be a tuple")
        normalized_names: list[str] = []
        for raw_name in raw_names:
            name = _plain_utf8_string(raw_name)
            if name is None:
                raise ConfigurationError("similarity feature name is outside feature schema 1")
            normalized_names.append(name)
        names = tuple(normalized_names)
        if not names or len(names) > config.max_features:
            raise ConfigurationError("similarity vocabulary size is outside configured bounds")
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ConfigurationError("similarity feature names must be sorted and unique")
        if any(
            not name
            or len(name) > config.max_feature_name_characters
            or not _feature_name_is_valid(name)
            for name in names
        ):
            raise ConfigurationError("similarity feature name is outside feature schema 1")
        if "bias" not in names:
            raise ConfigurationError("similarity feature schema requires bias")
        raw_idf = _tuple_snapshot(self.inverse_document_frequency)
        if raw_idf is None or len(raw_idf) != len(names):
            raise ConfigurationError("similarity IDF vector is invalid")
        maximum_idf = math.log(config.max_records + 1) + 1.0 + 1e-12
        try:
            idf = tuple(
                _bounded_number("similarity IDF value", value, 1.0, maximum_idf)
                for value in raw_idf
            )
        except ConfigurationError as exc:
            raise ConfigurationError("similarity IDF vector is invalid") from exc
        bias_index = names.index("bias")
        if idf[bias_index] != 1.0:
            raise ConfigurationError("similarity bias IDF must equal 1")
        try:
            raw_prototypes = dict(self.prototypes)
            raw_counts = dict(self.route_counts)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError("similarity routes must be mappings") from exc
        prototypes: dict[str, tuple[float, ...]] = {}
        for raw_model_id, raw_vector in raw_prototypes.items():
            model_id = _plain_utf8_string(raw_model_id)
            if (
                model_id is None
                or not model_id.strip()
                or model_id != model_id.strip()
                or len(model_id) > config.max_feature_name_characters
            ):
                raise ConfigurationError("similarity route identifiers must be bounded and trimmed")
            if model_id in prototypes:
                raise ConfigurationError("similarity route identifiers must be unique")
            vector_values = _tuple_snapshot(raw_vector)
            if vector_values is None or len(vector_values) != len(names):
                raise ConfigurationError(f"prototype for {model_id!r} has the wrong dimension")
            try:
                vector = tuple(
                    _bounded_number(f"prototype for {model_id!r} value", value, 0.0, math.inf)
                    for value in vector_values
                )
            except ConfigurationError as exc:
                raise ConfigurationError(
                    f"prototype for {model_id!r} contains invalid values"
                ) from exc
            try:
                norm = math.sqrt(math.fsum(value * value for value in vector))
            except (OverflowError, ValueError) as exc:
                raise ConfigurationError(
                    f"prototype for {model_id!r} contains invalid values"
                ) from exc
            if not math.isfinite(norm):
                raise ConfigurationError(f"prototype for {model_id!r} contains invalid values")
            if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ConfigurationError(f"prototype for {model_id!r} is not normalized")
            prototypes[model_id] = vector
        counts: dict[str, int] = {}
        for raw_model_id, raw_count in raw_counts.items():
            model_id = _plain_utf8_string(raw_model_id)
            if model_id is None or model_id in counts:
                raise ConfigurationError("similarity route count identifiers are invalid")
            counts[model_id] = _bounded_integer(
                f"route_counts[{model_id}]", raw_count, 1, config.max_records
            )
        if not prototypes or set(prototypes) != set(counts):
            raise ConfigurationError(
                "similarity routes and route counts must match and be non-empty"
            )
        if len(prototypes) > config.max_records:
            raise ConfigurationError("similarity route count exceeds configured bounds")
        if len(prototypes) * len(names) > config.max_prototype_values:
            raise ConfigurationError("similarity prototype index exceeds configured bounds")
        if sum(counts.values()) > config.max_records:
            raise ConfigurationError("similarity training count exceeds configured bounds")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "inverse_document_frequency", idf)
        object.__setattr__(self, "prototypes", MappingProxyType(prototypes))
        object.__setattr__(self, "route_counts", MappingProxyType(counts))

    @classmethod
    def fit(
        cls,
        traces: Iterable[RouteTrace],
        *,
        config: SimilarityFeatureConfig | None = None,
        extractor: QueryFeatureExtractor | None = None,
    ) -> SimilarityModel:
        """Fit route prototypes using only the supplied labelled training partition."""

        if config is None and extractor is not None:
            if type(extractor) is not QueryFeatureExtractor:
                raise ConfigurationError(
                    "similarity extractor must be the built-in QueryFeatureExtractor"
                )
            settings = SimilarityFeatureConfig(long_query_characters=extractor.long_query_chars)
        elif config is None:
            settings = SimilarityFeatureConfig()
        else:
            settings = _snapshot_config(config)
        feature_extractor = _configured_extractor(settings, extractor)
        ordered = _ordered_traces(traces, settings, purpose="similarity training")
        feature_occurrences = 0
        document_frequency: Counter[str] = Counter()
        for trace in ordered:
            document = _raw_features(trace.request, feature_extractor, settings)
            feature_occurrences += len(document)
            if feature_occurrences > settings.max_feature_occurrences:
                raise ConfigurationError(
                    "similarity training exceeds aggregate feature-occurrence limit of "
                    f"{settings.max_feature_occurrences}"
                )
            document_frequency.update(name for name, value in document.items() if value > 0)
        eligible_names = [
            name
            for name, frequency in document_frequency.items()
            if frequency >= settings.min_document_frequency
        ]
        eligible_names.sort(key=lambda name: (-document_frequency[name], name))
        feature_names = tuple(sorted(eligible_names[: settings.max_features]))
        if "bias" not in feature_names:
            raise ConfigurationError(
                "feature selection removed the required bias; increase max_features or lower min_df"
            )
        route_ids = {trace.preferred_model for trace in ordered}
        if len(route_ids) * len(feature_names) > settings.max_prototype_values:
            raise ConfigurationError(
                "similarity prototype index exceeds configured limit of "
                f"{settings.max_prototype_values} values"
            )
        idf = tuple(
            math.log((1.0 + len(ordered)) / (1.0 + document_frequency[name])) + 1.0
            for name in feature_names
        )
        feature_indices = {name: index for index, name in enumerate(feature_names)}
        route_counts: Counter[str] = Counter()
        route_totals: dict[str, list[float]] = {}
        route_compensations: dict[str, list[float]] = {}
        for trace in ordered:
            document = _raw_features(trace.request, feature_extractor, settings)
            weighted = sorted(
                (
                    index,
                    document[name] * idf[index],
                )
                for name in document
                if (index := feature_indices.get(name)) is not None and document[name] > 0.0
            )
            squared_norm = math.fsum(value * value for _, value in weighted)
            if not math.isfinite(squared_norm) or squared_norm <= 0.0:
                raise ConfigurationError("similarity vector has a non-finite or zero norm")
            norm = math.sqrt(squared_norm)
            model_id = trace.preferred_model
            if model_id is None:  # Defensive: _ordered_traces rejects this before fitting.
                raise ConfigurationError("similarity training traces require labels")
            route_counts[model_id] += 1
            totals = route_totals.setdefault(model_id, [0.0] * len(feature_names))
            compensations = route_compensations.setdefault(model_id, [0.0] * len(feature_names))
            for index, value in weighted:
                normalized = value / norm
                combined = totals[index] + normalized
                if abs(totals[index]) >= abs(normalized):
                    compensations[index] += (totals[index] - combined) + normalized
                else:
                    compensations[index] += (normalized - combined) + totals[index]
                totals[index] = combined
        prototypes = {
            model_id: _normalize(
                (total + correction) / route_counts[model_id]
                for total, correction in zip(
                    route_totals[model_id], route_compensations[model_id], strict=True
                )
            )
            for model_id in sorted(route_totals)
        }
        return cls(
            config=settings,
            training_sha256=_ordered_digest(ordered),
            feature_names=feature_names,
            inverse_document_frequency=idf,
            prototypes=prototypes,
            route_counts=dict(sorted(route_counts.items())),
        )

    def vectorize(
        self, request: RouteRequest, *, extractor: QueryFeatureExtractor | None = None
    ) -> tuple[float, ...]:
        raw = _raw_features(request, _configured_extractor(self.config, extractor), self.config)
        return _normalize(
            raw.get(name, 0.0) * weight
            for name, weight in zip(
                self.feature_names, self.inverse_document_frequency, strict=True
            )
        )

    def rank(
        self,
        request: RouteRequest,
        *,
        allowed_model_ids: Iterable[str] | None = None,
        top_features: int = 5,
        extractor: QueryFeatureExtractor | None = None,
    ) -> tuple[SimilarityMatch, ...]:
        """Rank persisted routes by cosine similarity with deterministic ties."""

        _bounded_integer("top_features", top_features, 0, 20)
        if allowed_model_ids is None:
            allowed = set(self.prototypes)
        else:
            if isinstance(allowed_model_ids, (str, bytes, Mapping)):
                raise ConfigurationError("allowed_model_ids must be a collection of identifiers")
            allowed = set()
            for consumed, model_id in enumerate(allowed_model_ids, start=1):
                if consumed > self.config.max_records:
                    raise ConfigurationError("allowed model identifiers exceed configured bounds")
                if (
                    not isinstance(model_id, str)
                    or not model_id.strip()
                    or model_id != model_id.strip()
                    or len(model_id) > self.config.max_feature_name_characters
                    or not _is_utf8(model_id)
                ):
                    raise ConfigurationError(
                        "allowed model identifiers must be bounded and trimmed"
                    )
                allowed.add(model_id)
        query = self.vectorize(request, extractor=extractor)
        matches: list[SimilarityMatch] = []
        for model_id in sorted(set(self.prototypes) & allowed):
            prototype = self.prototypes[model_id]
            if top_features == 0:
                similarity = math.fsum(
                    left * right for left, right in zip(query, prototype, strict=True)
                )
                contributions: tuple[tuple[str, float], ...] = ()
            else:
                products = tuple(
                    (name, query[index] * prototype[index])
                    for index, name in enumerate(self.feature_names)
                    if query[index] and prototype[index]
                )
                similarity = math.fsum(value for _, value in products)
                contributions = tuple(
                    sorted(products, key=lambda item: (-item[1], item[0]))[:top_features]
                )
            similarity = min(1.0, max(0.0, similarity))
            matches.append(SimilarityMatch(model_id, similarity, contributions))
        return tuple(sorted(matches, key=lambda item: (-item.similarity, item.model_id)))

    def _payload(self) -> dict[str, Any]:
        return {
            "format": SIMILARITY_FORMAT,
            "schema_version": SIMILARITY_SCHEMA_VERSION,
            "feature_schema_version": SIMILARITY_FEATURE_SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "training_sha256": self.training_sha256,
            "calibration_sha256": self.calibration_sha256,
            "similarity_threshold": self.similarity_threshold,
            "feature_names": list(self.feature_names),
            "inverse_document_frequency": list(self.inverse_document_frequency),
            "routes": {
                model_id: {
                    "count": self.route_counts[model_id],
                    "prototype": list(self.prototypes[model_id]),
                }
                for model_id in sorted(self.prototypes)
            },
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        return {**payload, "state_sha256": _canonical_digest(payload)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimilarityModel:
        expected = {
            "format",
            "schema_version",
            "feature_schema_version",
            "config",
            "training_sha256",
            "calibration_sha256",
            "similarity_threshold",
            "feature_names",
            "inverse_document_frequency",
            "routes",
            "state_sha256",
        }
        try:
            if set(data) != expected:
                raise ConfigurationError("similarity state has unknown or missing fields")
            integrity = data["state_sha256"]
            if not isinstance(integrity, str):
                raise ConfigurationError("state_sha256 must be a string")
            payload = {name: data[name] for name in expected - {"state_sha256"}}
            if not hmac.compare_digest(integrity, _canonical_digest(payload)):
                raise ConfigurationError("similarity state integrity check failed")
            if data["format"] != SIMILARITY_FORMAT:
                raise ConfigurationError("unsupported similarity state format")
            if (
                isinstance(data["schema_version"], bool)
                or not isinstance(data["schema_version"], int)
                or data["schema_version"] != SIMILARITY_SCHEMA_VERSION
            ):
                raise ConfigurationError("unsupported similarity schema version")
            if (
                isinstance(data["feature_schema_version"], bool)
                or not isinstance(data["feature_schema_version"], int)
                or data["feature_schema_version"] != SIMILARITY_FEATURE_SCHEMA_VERSION
            ):
                raise ConfigurationError("unsupported similarity feature schema version")
            config_data = data["config"]
            if not isinstance(config_data, dict):
                raise ConfigurationError("similarity config must be an object")
            config = SimilarityFeatureConfig.from_dict(config_data)
            names_data = data["feature_names"]
            idf_data = data["inverse_document_frequency"]
            routes_data = data["routes"]
            if not isinstance(names_data, list) or not isinstance(idf_data, list):
                raise ConfigurationError("similarity features and IDF must be arrays")
            if not isinstance(routes_data, dict):
                raise ConfigurationError("similarity routes must be an object")
            prototypes: dict[str, tuple[float, ...]] = {}
            counts: dict[str, int] = {}
            for model_id, route in routes_data.items():
                if not isinstance(model_id, str) or not isinstance(route, dict):
                    raise ConfigurationError("similarity route entry is invalid")
                if set(route) != {"count", "prototype"}:
                    raise ConfigurationError("similarity route fields are invalid")
                vector = route["prototype"]
                if not isinstance(vector, list):
                    raise ConfigurationError("similarity prototype must be an array")
                prototypes[model_id] = tuple(vector)
                counts[model_id] = route["count"]
            calibration = data["calibration_sha256"]
            if calibration is not None and not isinstance(calibration, str):
                raise ConfigurationError("calibration_sha256 must be null or a string")
            training = data["training_sha256"]
            if not isinstance(training, str):
                raise ConfigurationError("training_sha256 must be a string")
            return cls(
                config=config,
                training_sha256=training,
                calibration_sha256=calibration,
                similarity_threshold=data["similarity_threshold"],
                feature_names=tuple(names_data),
                inverse_document_frequency=tuple(idf_data),
                prototypes=prototypes,
                route_counts=counts,
            )
        except (
            ConfigurationError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
        ) as exc:
            raise PersistenceError(f"Invalid similarity state: {exc}") from exc

    def save(self, path: str | Path, *, max_state_bytes: int = _MAX_STATE_BYTES) -> None:
        limit = _bounded_integer("max_state_bytes", max_state_bytes, 1, _MAX_STATE_BYTES)
        AtomicJsonStore(path).save(self.to_dict(), max_bytes=limit)

    def _state_bytes(self, *, max_state_bytes: int = _MAX_STATE_BYTES) -> bytes:
        """Encode a bounded artifact for a larger transactional output bundle."""

        limit = _bounded_integer("max_state_bytes", max_state_bytes, 1, _MAX_STATE_BYTES)
        try:
            return _json_bytes(self.to_dict(), max_bytes=limit)
        except (TypeError, ValueError, RecursionError) as exc:
            raise PersistenceError(f"Cannot encode similarity state: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path, *, max_state_bytes: int = _MAX_STATE_BYTES) -> SimilarityModel:
        return cls.load_with_sha256(path, max_state_bytes=max_state_bytes)[0]

    @classmethod
    def load_with_sha256(
        cls, path: str | Path, *, max_state_bytes: int = _MAX_STATE_BYTES
    ) -> tuple[SimilarityModel, str]:
        """Parse and hash one bounded byte snapshot of a persisted model."""

        limit = _bounded_integer("max_state_bytes", max_state_bytes, 1, _MAX_STATE_BYTES)
        source = Path(path)
        try:
            with source.open("rb") as handle:
                raw = handle.read(limit + 1)
        except OSError as exc:
            raise PersistenceError(f"Cannot read similarity state {source}: {exc}") from exc
        if len(raw) > limit:
            raise PersistenceError(f"Similarity state exceeds {limit} bytes")
        try:
            payload = loads_strict(raw)
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise PersistenceError(f"Cannot decode similarity state {source}: {exc}") from exc
        if not isinstance(payload, dict):
            raise PersistenceError("Invalid similarity state: root must be an object")
        return cls.from_dict(payload), hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class SimilarityEvaluation:
    records: int
    covered: int
    correct: int
    coverage: float
    accuracy: float | None
    mean_similarity: float

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "records": self.records,
            "covered": self.covered,
            "correct": self.correct,
            "coverage": self.coverage,
            "accuracy": self.accuracy,
            "mean_similarity": self.mean_similarity,
        }


@dataclass(frozen=True, slots=True)
class SimilarityCalibrationPoint:
    threshold: float
    metrics: SimilarityEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {"threshold": self.threshold, **self.metrics.to_dict()}


@dataclass(frozen=True, slots=True)
class SimilarityExperimentReport:
    training_sha256: str
    calibration_sha256: str
    held_out_sha256: str | None
    group_by: str
    training_groups: int
    calibration_groups: int
    held_out_groups: int | None
    points: tuple[SimilarityCalibrationPoint, ...]
    selected_threshold: float
    minimum_coverage: float
    held_out: SimilarityEvaluation | None
    model_state_sha256: str

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": 1,
            "training_sha256": self.training_sha256,
            "calibration_sha256": self.calibration_sha256,
            "group_by": self.group_by,
            "groups": {
                "training": self.training_groups,
                "calibration": self.calibration_groups,
                "held_out": self.held_out_groups,
            },
            "minimum_coverage": self.minimum_coverage,
            "selected_threshold": self.selected_threshold,
            "points": [point.to_dict() for point in self.points],
            "model_state_sha256": self.model_state_sha256,
        }
        if self.held_out is not None:
            result["held_out"] = {
                "dataset_canonical_sha256": self.held_out_sha256,
                "metrics": self.held_out.to_dict(),
            }
        return result


def _evaluate(
    model: SimilarityModel,
    traces: tuple[RouteTrace, ...],
    threshold: float,
) -> SimilarityEvaluation:
    covered = 0
    correct = 0
    similarities: list[float] = []
    for trace in traces:
        matches = model.rank(trace.request, top_features=0)
        match = matches[0]
        similarities.append(match.similarity)
        if match.similarity >= threshold:
            covered += 1
            correct += int(match.model_id == trace.preferred_model)
    return SimilarityEvaluation(
        records=len(traces),
        covered=covered,
        correct=correct,
        coverage=covered / len(traces),
        accuracy=correct / covered if covered else None,
        mean_similarity=math.fsum(similarities) / len(similarities),
    )


def _calibration_points(
    model: SimilarityModel,
    traces: tuple[RouteTrace, ...],
) -> tuple[SimilarityCalibrationPoint, ...]:
    observations: list[tuple[float, bool]] = []
    for trace in traces:
        match = model.rank(trace.request, top_features=0)[0]
        observations.append((match.similarity, match.model_id == trace.preferred_model))
    mean_similarity = math.fsum(score for score, _ in observations) / len(observations)
    grouped: dict[float, tuple[int, int]] = {}
    for score, is_correct in observations:
        count, correct_count = grouped.get(score, (0, 0))
        grouped[score] = count + 1, correct_count + int(is_correct)
    thresholds = sorted({0.0, 1.0, *grouped}, reverse=True)
    score_groups = sorted(grouped.items(), reverse=True)
    group_index = 0
    covered = 0
    cumulative_correct = 0
    by_threshold: dict[float, SimilarityCalibrationPoint] = {}
    for threshold in thresholds:
        while group_index < len(score_groups) and score_groups[group_index][0] >= threshold:
            _, (group_count, group_correct) = score_groups[group_index]
            covered += group_count
            cumulative_correct += group_correct
            group_index += 1
        by_threshold[threshold] = SimilarityCalibrationPoint(
            threshold,
            SimilarityEvaluation(
                records=len(observations),
                covered=covered,
                correct=cumulative_correct,
                coverage=covered / len(observations),
                accuracy=cumulative_correct / covered if covered else None,
                mean_similarity=mean_similarity,
            ),
        )
    return tuple(by_threshold[threshold] for threshold in sorted(by_threshold))


def _partition_groups(
    partitions: Mapping[str, tuple[RouteTrace, ...]], group_by: str
) -> dict[str, set[str]]:
    request_partitions: dict[str, str] = {}
    for name, traces in partitions.items():
        for trace in traces:
            request_id = trace.request.request_id
            previous = request_partitions.setdefault(request_id, name)
            if previous != name:
                raise ConfigurationError(
                    f"similarity {previous} and {name} partitions overlap by request_id"
                )
    groups = {
        name: {_group_key(trace, group_by) for trace in traces}
        for name, traces in partitions.items()
    }
    names = tuple(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ConfigurationError(
                    f"similarity {left} and {right} partitions overlap by {group_by}"
                )
    return groups


def fit_calibrate_evaluate(
    training_traces: Iterable[RouteTrace],
    calibration_traces: Iterable[RouteTrace],
    held_out_traces: Iterable[RouteTrace] | None = None,
    *,
    config: SimilarityFeatureConfig | None = None,
    group_by: str = "request_id",
    minimum_coverage: float = 0.5,
) -> tuple[SimilarityModel, SimilarityExperimentReport]:
    """Fit, tune on a disjoint partition, then evaluate a held-out partition once."""

    settings = SimilarityFeatureConfig() if config is None else _snapshot_config(config)
    coverage_floor = _bounded_number("minimum_coverage", minimum_coverage, 0.01, 1.0)
    training = _ordered_traces(training_traces, settings, purpose="similarity training")
    calibration = _ordered_traces(calibration_traces, settings, purpose="similarity calibration")
    held_out = (
        _ordered_traces(held_out_traces, settings, purpose="similarity held-out evaluation")
        if held_out_traces is not None
        else None
    )
    partitions = {"training": training, "calibration": calibration}
    if held_out is not None:
        partitions["held-out"] = held_out
    groups = _partition_groups(partitions, group_by)
    model = SimilarityModel.fit(training, config=settings)
    known_routes = set(model.prototypes)
    for name, traces in (("calibration", calibration), ("held-out", held_out)):
        if traces is None:
            continue
        unknown = sorted(
            {
                trace.preferred_model
                for trace in traces
                if trace.preferred_model is not None and trace.preferred_model not in known_routes
            }
        )
        if unknown:
            raise ConfigurationError(
                f"similarity {name} labels have no training prototype: {unknown}"
            )
    points = _calibration_points(model, calibration)
    feasible = [point for point in points if point.metrics.coverage >= coverage_floor]
    if not feasible:
        raise ConfigurationError("no similarity threshold satisfies minimum_coverage")
    selected = min(
        feasible,
        key=lambda point: (
            -(point.metrics.accuracy if point.metrics.accuracy is not None else -1.0),
            -point.metrics.coverage,
            point.threshold,
        ),
    )
    calibration_digest = _ordered_digest(calibration)
    calibrated = replace(
        model,
        similarity_threshold=selected.threshold,
        calibration_sha256=calibration_digest,
    )
    held_out_metrics = (
        _evaluate(calibrated, held_out, calibrated.similarity_threshold)
        if held_out is not None
        else None
    )
    state_digest = calibrated.to_dict()["state_sha256"]
    if not isinstance(state_digest, str):  # Defensive: to_dict creates this digest.
        raise ConfigurationError("similarity state digest is invalid")
    report = SimilarityExperimentReport(
        training_sha256=calibrated.training_sha256,
        calibration_sha256=calibration_digest,
        held_out_sha256=_ordered_digest(held_out) if held_out is not None else None,
        group_by=group_by,
        training_groups=len(groups["training"]),
        calibration_groups=len(groups["calibration"]),
        held_out_groups=len(groups["held-out"]) if held_out is not None else None,
        points=points,
        selected_threshold=selected.threshold,
        minimum_coverage=coverage_floor,
        held_out=held_out_metrics,
        model_state_sha256=state_digest,
    )
    return calibrated, report


class SimilarityRouter(RuleRouter):
    """Apply hard constraints first, then a trained route prototype or safe fallback."""

    policy_name = "similarity"

    def __init__(
        self,
        candidates: Iterable[ModelCandidate],
        model: SimilarityModel,
        preferences: Mapping[str, UserPreferences] | None = None,
        rules: Iterable[RoutingRule] = (),
        *,
        extractor: QueryFeatureExtractor | None = None,
        constraints: ConstraintEngine | None = None,
        scorer: MultiObjectiveScorer | None = None,
    ) -> None:
        if not isinstance(model, SimilarityModel):
            raise ConfigurationError("model must be SimilarityModel")
        feature_extractor = _configured_extractor(model.config, extractor)
        super().__init__(
            candidates,
            preferences,
            rules,
            extractor=feature_extractor,
            constraints=constraints,
            scorer=scorer,
        )
        unknown = set(model.prototypes) - {candidate.model_id for candidate in self.candidates}
        if unknown:
            raise ConfigurationError(
                f"similarity state references models outside the catalog: {sorted(unknown)}"
            )
        self.model = model

    def route(self, request: RouteRequest) -> RouteDecision:
        if len(request.query) > self.model.config.max_query_characters:
            raise ConfigurationError(
                f"query exceeds similarity limit of "
                f"{self.model.config.max_query_characters} characters"
            )
        preferences = self.preference_for(request.user_id)
        features = self.extractor.extract(request)
        constraint_result = self.constraints.filter(self.candidates, request, features, preferences)
        if not constraint_result.eligible:
            raise NoEligibleModelError(constraint_result.rejected)
        eligible_ids = {candidate.model_id for candidate in constraint_result.eligible}
        matches = self.model.rank(
            request,
            allowed_model_ids=eligible_ids,
            extractor=self.extractor,
        )
        bonuses, matched_rules = match_rules(self.rules, features, eligible_ids)
        objective_order = self.scorer.score(
            constraint_result.eligible,
            request,
            features,
            preferences,
            constraint_result.estimated_costs,
            bonuses,
        )
        uses_similarity = bool(matches and matches[0].similarity >= self.model.similarity_threshold)
        if uses_similarity:
            similarity_by_model = {match.model_id: match.similarity for match in matches}
            ordered = tuple(
                sorted(
                    (
                        ScoredCandidate(
                            item.candidate,
                            replace(
                                item.breakdown,
                                preference_bonus=0.0,
                                rule_bonus=0.0,
                                total=similarity_by_model[item.candidate.model_id],
                            ),
                        )
                        for item in objective_order
                        if item.candidate.model_id in similarity_by_model
                    ),
                    key=lambda item: (-item.breakdown.total, item.candidate.model_id),
                )
            )
        else:
            ordered = objective_order
        selected = ordered[0]
        selected_id = selected.candidate.model_id
        prefix: tuple[str, ...]
        if uses_similarity:
            contribution_text = ", ".join(
                f"{name}={value:.3f}" for name, value in matches[0].contributions
            )
            prefix = (
                f"trained prototype selected {selected_id} with cosine similarity "
                f"{matches[0].similarity:.6f} (threshold {self.model.similarity_threshold:.6f})",
                f"largest similarity contributions: {contribution_text or 'none'}",
                "soft preference and matched-rule bonuses remain audit metadata; they affect only fallback",
            )
        elif matches:
            prefix = (
                f"highest eligible cosine similarity {matches[0].similarity:.6f} was below "
                f"threshold {self.model.similarity_threshold:.6f}; used objective fallback",
            )
        else:
            prefix = ("no trained prototype remained eligible; used objective fallback",)
        summary = features.to_dict()
        summary["similarity"] = {
            "used": uses_similarity,
            "threshold": self.model.similarity_threshold,
            "training_sha256": self.model.training_sha256,
            "calibration_sha256": self.model.calibration_sha256,
            "ranking": [match.to_dict() for match in matches],
        }
        return self._decision(
            request,
            preferences,
            summary,
            selected,
            ordered,
            constraint_result.rejected,
            matched_rules,
            explanation_prefix=prefix,
        )
