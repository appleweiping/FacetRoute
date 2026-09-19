from __future__ import annotations

import builtins
import gc
import hashlib
import json
import math
import tracemalloc
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import facetroute.cli as cli_module
import facetroute.persistence as persistence_module
from facetroute import ModelCandidate, RouteRequest, UserPreferences
from facetroute.cli import main
from facetroute.errors import ConfigurationError, NoEligibleModelError, PersistenceError
from facetroute.similarity import (
    _MAX_STATE_BYTES,
    _STATE_SCHEMA_UPPER_BOUND_BYTES,
    SimilarityFeatureConfig,
    SimilarityMatch,
    SimilarityModel,
    SimilarityRouter,
    fit_calibrate_evaluate,
)
from facetroute.traces import (
    RouteTrace,
    TraceOutcome,
    file_sha256,
    load_traces,
    traces_sha256,
    write_traces,
)


def _trace(
    request_id: str,
    query: str,
    preferred: str,
    *,
    user_id: str | None = None,
    task: str | None = None,
) -> RouteTrace:
    outcomes = {
        "alpha": TraceOutcome(0.7, 0.001, 100, True),
        "beta": TraceOutcome(0.8, 0.002, 200, True),
    }
    if preferred not in outcomes:
        outcomes[preferred] = TraceOutcome(0.6, 0.003, 300, True)
    return RouteTrace(
        RouteRequest(
            query,
            request_id=request_id,
            user_id=user_id or request_id,
            task_hint=task,
        ),
        outcomes,
        preferred_model=preferred,
    )


def _training() -> tuple[RouteTrace, ...]:
    return (
        _trace("a1", "python api debug function", "alpha", task="code"),
        _trace("a2", "compile python class", "alpha", task="code"),
        _trace("b1", "prove algebra theorem", "beta", task="math"),
        _trace("b2", "matrix equation derivative", "beta", task="math"),
    )


def _resign(payload: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "state_sha256"}
    payload["state_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_similarity_matches_independent_cosine_oracle_and_exposes_terms() -> None:
    model = SimilarityModel.fit(_training())
    request = RouteRequest("debug python api", request_id="query", task_hint="code")
    query = model.vectorize(request)
    alpha = model.prototypes["alpha"]
    expected = math.fsum(left * right for left, right in zip(query, alpha, strict=True))

    matches = model.rank(request)

    assert matches[0].model_id == "alpha"
    assert matches[0].similarity == pytest.approx(expected, abs=1e-15)
    assert matches[0].similarity > matches[1].similarity
    assert any(name == "term:python" for name, _ in matches[0].contributions)
    assert math.fsum(value * value for value in query) == pytest.approx(1.0)


def test_two_axis_hand_calculated_cosine_and_selection_oracle() -> None:
    model = SimilarityModel(
        config=SimilarityFeatureConfig(),
        training_sha256="0" * 64,
        feature_names=("bias", "term:x"),
        inverse_document_frequency=(1.0, 1.0),
        prototypes={"alpha": (1.0, 0.0), "beta": (0.0, 1.0)},
        route_counts={"alpha": 1, "beta": 1},
    )

    matches = model.rank(RouteRequest("x"), top_features=2)

    expected = 1.0 / math.sqrt(2.0)
    assert [item.model_id for item in matches] == ["alpha", "beta"]
    assert [item.similarity for item in matches] == pytest.approx([expected, expected])
    assert matches[0].contributions[0][0] == "bias"
    assert matches[0].contributions[0][1] == pytest.approx(expected)
    assert matches[1].contributions[0][0] == "term:x"
    assert matches[1].contributions[0][1] == pytest.approx(expected)


def test_fit_is_invariant_to_input_order_and_ties_use_model_id() -> None:
    forward = SimilarityModel.fit(_training())
    reverse = SimilarityModel.fit(tuple(reversed(_training())))
    assert forward.to_dict() == reverse.to_dict()

    tied = SimilarityModel.fit(
        (
            _trace("one", "same request", "beta", task="general"),
            _trace("two", "same request", "alpha", task="general"),
        )
    )
    assert tied.rank(RouteRequest("same request", task_hint="general"))[0].model_id == "alpha"


def test_similarity_state_round_trip_and_detects_tampering(tmp_path) -> None:
    model = SimilarityModel.fit(_training())
    path = tmp_path / "similarity.json"
    model.save(path)
    assert SimilarityModel.load(path).to_dict() == model.to_dict()

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["routes"]["alpha"]["prototype"][0] = 0.25
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PersistenceError, match="integrity"):
        SimilarityModel.load(path)


def test_similarity_state_mappings_are_deeply_frozen_and_rank_is_stable() -> None:
    model = SimilarityModel.fit(_training())
    request = RouteRequest("debug python api", task_hint="code")
    before_rank = model.rank(request)
    before_state = model.to_dict()

    with pytest.raises(TypeError):
        model.prototypes["alpha"] = model.prototypes["beta"]  # type: ignore[index]
    with pytest.raises(TypeError):
        model.route_counts["alpha"] = 999  # type: ignore[index]

    exported = model.to_dict()
    exported["routes"]["alpha"]["prototype"][0] = 0.0
    exported["routes"]["alpha"]["count"] = 999
    _resign(exported)

    assert model.to_dict() == before_state
    assert model.rank(request) == before_rank


def test_similarity_constructor_snapshots_exact_builtin_nested_state() -> None:
    class HookedString(str):
        def __len__(self) -> int:
            return 1

        def __iter__(self):
            return iter("f" * 64)

        def encode(self, *args, **kwargs):
            return b"forged"

        def strip(self, *args, **kwargs):
            return ""

    class HookedInteger(int):
        def __int__(self) -> int:
            return 0

        def __le__(self, other: object) -> bool:
            return True

        def __ge__(self, other: object) -> bool:
            return True

    class HookedFloat(float):
        def __float__(self) -> float:
            return float("inf")

    class SwitchingTuple(tuple):
        forged = False

        def __iter__(self):
            return iter((0.0, 1.0) if self.forged else (1.0, 0.0))

        def __len__(self) -> int:
            return 99

    class ExtendedConfig(SimilarityFeatureConfig):
        def to_dict(self) -> dict[str, int]:
            return {**super().to_dict(), "unbounded_extension": 1}

    config = ExtendedConfig()
    prototypes = {HookedString("alpha"): SwitchingTuple((1.0, 0.0))}
    counts = {HookedString("alpha"): HookedInteger(1)}
    model = SimilarityModel(
        config=config,
        training_sha256=HookedString("0" * 64),
        feature_names=SwitchingTuple((HookedString("bias"), HookedString("term:x"))),
        inverse_document_frequency=SwitchingTuple((HookedInteger(1), HookedFloat(1.0))),
        prototypes=prototypes,
        route_counts=counts,
        similarity_threshold=HookedFloat(0.25),
    )
    request = RouteRequest("x")
    before_rank = model.rank(request)
    before_state = model.to_dict()

    SwitchingTuple.forged = True
    object.__setattr__(config, "max_records", 1)
    prototypes.clear()
    counts.clear()

    assert type(model.config) is SimilarityFeatureConfig
    assert model.config.max_records == 100_000
    assert all(type(value) is int for value in model.config.to_dict().values())
    assert type(model.training_sha256) is str
    assert type(model.feature_names) is tuple
    assert all(type(name) is str for name in model.feature_names)
    assert type(model.inverse_document_frequency) is tuple
    assert all(type(value) is float for value in model.inverse_document_frequency)
    assert all(type(model_id) is str for model_id in model.prototypes)
    assert all(type(vector) is tuple for vector in model.prototypes.values())
    assert all(type(value) is float for vector in model.prototypes.values() for value in vector)
    assert all(type(value) is int for value in model.route_counts.values())
    assert type(model.similarity_threshold) is float
    assert "unbounded_extension" not in model.to_dict()["config"]
    assert model.rank(request) == before_rank
    assert model.to_dict() == before_state


def test_similarity_constructor_validates_underlying_subclass_values() -> None:
    class HiddenLength(str):
        def __len__(self) -> int:
            return 1

    class ForgedDigest(str):
        def __len__(self) -> int:
            return 64

        def __iter__(self):
            return iter("0" * 64)

    class ForgedCount(int):
        def __le__(self, other: object) -> bool:
            return True

        def __ge__(self, other: object) -> bool:
            return True

    base = SimilarityModel.fit(_training())
    with pytest.raises(ConfigurationError, match="digest"):
        replace(base, training_sha256=ForgedDigest("bad"))
    with pytest.raises(ConfigurationError, match="route identifiers"):
        SimilarityModel(
            config=SimilarityFeatureConfig(max_feature_name_characters=512),
            training_sha256="0" * 64,
            feature_names=("bias",),
            inverse_document_frequency=(1.0,),
            prototypes={HiddenLength("x" * 513): (1.0,)},
            route_counts={HiddenLength("x" * 513): 1},
        )
    with pytest.raises(ConfigurationError, match="route_counts"):
        replace(
            base,
            route_counts={
                model_id: ForgedCount(0) if model_id == "alpha" else count
                for model_id, count in base.route_counts.items()
            },
        )


def test_similarity_state_validates_recomputed_malformed_payload() -> None:
    payload = SimilarityModel.fit(_training()).to_dict()
    payload["routes"]["alpha"]["prototype"] = [0.0] * len(payload["feature_names"])
    _resign(payload)
    with pytest.raises(PersistenceError, match="not normalized"):
        SimilarityModel.from_dict(payload)


def test_similarity_load_rejects_duplicate_keys_and_oversize_state(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"state_sha256":"a","state_sha256":"b"}', encoding="utf-8")
    with pytest.raises(PersistenceError, match="duplicate"):
        SimilarityModel.load(duplicate)

    large = tmp_path / "large.json"
    large.write_bytes(b"x" * 33)
    with pytest.raises(PersistenceError, match="exceeds 32 bytes"):
        SimilarityModel.load(large, max_state_bytes=32)


def test_deep_similarity_json_is_a_persistence_error_and_cli_exit_two(tmp_path, capsys) -> None:
    nested = tmp_path / "nested.json"
    nested.write_text("[" * 1_100 + "0" + "]" * 1_100, encoding="utf-8")
    with pytest.raises(PersistenceError, match="Cannot decode similarity state"):
        SimilarityModel.load(nested)

    code = main(
        [
            "route",
            "--models",
            str(Path(__file__).parents[1] / "examples" / "models.json"),
            "--policy",
            "similarity",
            "--similarity-model",
            str(nested),
            "--query",
            "hello",
        ]
    )
    assert code == 2
    error = capsys.readouterr().err
    assert "Cannot decode similarity state" in error
    assert "Traceback" not in error


def test_similarity_resource_and_finite_bounds_fail_closed() -> None:
    with pytest.raises(ConfigurationError, match="max_features"):
        SimilarityFeatureConfig(max_features=1)
    with pytest.raises(ConfigurationError, match="min_document_frequency"):
        SimilarityFeatureConfig(max_records=2, min_document_frequency=3)

    config = SimilarityFeatureConfig(max_query_characters=8, max_tokens_per_query=2)
    with pytest.raises(ConfigurationError, match="characters"):
        SimilarityModel.fit((_trace("x", "query too long", "alpha"),), config=config)
    model = SimilarityModel.fit(
        (_trace("x", "one two", "alpha"),),
        config=SimilarityFeatureConfig(max_query_characters=100, max_tokens_per_query=2),
    )
    with pytest.raises(ConfigurationError, match="2 tokens"):
        model.rank(RouteRequest("one two three"))
    with pytest.raises(ConfigurationError, match="similarity_threshold"):
        replace(model, similarity_threshold=float("inf"))
    with pytest.raises(ConfigurationError, match="finite number"):
        replace(model, similarity_threshold=True)
    with pytest.raises(ConfigurationError, match="top_features"):
        model.rank(RouteRequest("hello"), top_features=21)
    with pytest.raises(ConfigurationError, match="collection"):
        model.rank(RouteRequest("hello"), allowed_model_ids="alpha")
    with pytest.raises(ConfigurationError, match="bounded and trimmed"):
        model.rank(RouteRequest("hello"), allowed_model_ids={" alpha"})
    with pytest.raises(ConfigurationError, match="config must"):
        SimilarityModel.fit(_training(), config=object())  # type: ignore[arg-type]


def test_extreme_numbers_have_public_configuration_and_persistence_boundaries() -> None:
    model = SimilarityModel.fit(_training())
    huge = 10**10_000
    with pytest.raises(ConfigurationError, match="similarity_threshold"):
        replace(model, similarity_threshold=huge)
    with pytest.raises(ConfigurationError, match="IDF"):
        replace(
            model,
            inverse_document_frequency=(huge, *model.inverse_document_frequency[1:]),
        )
    with pytest.raises(ConfigurationError, match="invalid values"):
        replace(
            model,
            prototypes={
                "alpha": (huge, *model.prototypes["alpha"][1:]),
                "beta": model.prototypes["beta"],
            },
        )
    with pytest.raises(ConfigurationError, match="minimum_coverage"):
        fit_calibrate_evaluate(_training()[:2], _training()[2:], minimum_coverage=huge)

    persisted = model.to_dict()
    persisted["similarity_threshold"] = 10**1_000
    _resign(persisted)
    with pytest.raises(PersistenceError, match="similarity_threshold"):
        SimilarityModel.from_dict(persisted)


def test_schema_one_conservative_bound_closes_default_persistence_limit() -> None:
    assert _STATE_SCHEMA_UPPER_BOUND_BYTES == 376_062_976
    assert _MAX_STATE_BYTES == 536_870_912
    assert _MAX_STATE_BYTES - _STATE_SCHEMA_UPPER_BOUND_BYTES == 160_807_936
    maxima = SimilarityFeatureConfig(
        max_features=8_192,
        max_records=100_000,
        max_feature_name_characters=512,
        max_prototype_values=250_000,
    )
    assert maxima.max_features == 8_192
    assert maxima.max_records == 100_000
    assert maxima.max_feature_name_characters == 512
    assert maxima.max_prototype_values == 250_000


def test_feature_config_and_request_category_schema_are_strict() -> None:
    config = SimilarityFeatureConfig()
    bad = config.to_dict()
    bad["unknown"] = 1
    with pytest.raises(ConfigurationError, match="config fields differ"):
        SimilarityFeatureConfig.from_dict(bad)

    with pytest.raises(ConfigurationError, match="categorical-feature"):
        SimilarityModel.fit(
            (_trace("x", "one", "alpha"),),
            config=SimilarityFeatureConfig(max_categories_per_request=1),
        )
    with pytest.raises(ConfigurationError, match="at most"):
        SimilarityModel.fit(
            (_trace("x", "one", "alpha", task="x" * 200),),
            config=SimilarityFeatureConfig(max_feature_name_characters=32),
        )
    with pytest.raises(ConfigurationError, match="required bias"):
        SimilarityModel.fit(
            (_trace("x", "one", "alpha"),),
            config=SimilarityFeatureConfig(max_records=2, min_document_frequency=2),
        )
    with pytest.raises(ConfigurationError, match="feature-occurrence"):
        SimilarityModel.fit(
            (_trace("x", "one", "alpha"),),
            config=SimilarityFeatureConfig(max_feature_occurrences=1),
        )
    with pytest.raises(ConfigurationError, match="prototype index"):
        SimilarityModel.fit(
            (
                _trace("x", "one", "alpha"),
                _trace("y", "two", "beta"),
            ),
            config=SimilarityFeatureConfig(max_prototype_values=1),
        )

    special = RouteTrace(
        RouteRequest(
            "call api",
            request_id="special",
            needs_tools=True,
            needs_json=True,
        ),
        {"alpha": TraceOutcome(0.8, 0, 1, True)},
        preferred_model="alpha",
    )
    names = SimilarityModel.fit((special,)).feature_names
    assert "request:tools" in names
    assert "request:json" in names


def test_similarity_training_rejects_missing_labels_duplicates_and_record_overflow() -> None:
    unlabelled = RouteTrace(
        RouteRequest("hello", request_id="u"),
        {"alpha": TraceOutcome(0.5, 0, 1, True)},
    )
    with pytest.raises(ConfigurationError, match="preferred_model"):
        SimilarityModel.fit((unlabelled,))
    duplicate = _trace("same", "first", "alpha")
    with pytest.raises(ConfigurationError, match="duplicate request_id"):
        SimilarityModel.fit((duplicate, _trace("same", "second", "beta")))
    with pytest.raises(ConfigurationError, match="exceeds 1 records"):
        SimilarityModel.fit(
            (_trace("1", "one", "alpha"), _trace("2", "two", "beta")),
            config=SimilarityFeatureConfig(max_records=1),
        )
    with pytest.raises(ConfigurationError, match="RouteTrace"):
        SimilarityModel.fit((object(),))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="at least one"):
        SimilarityModel.fit(())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("format", "format"),
        ("schema", "schema version"),
        ("feature_schema", "feature schema"),
        ("config", "config must be an object"),
        ("arrays", "features and IDF"),
        ("routes", "routes must be an object"),
        ("route_entry", "route entry"),
        ("route_fields", "route fields"),
        ("prototype", "prototype must be an array"),
        ("calibration", "calibration_sha256"),
        ("training", "training_sha256"),
    ],
)
def test_similarity_state_rejects_schema_and_shape_corruption(mutation, message) -> None:
    payload = SimilarityModel.fit(_training()).to_dict()
    if mutation == "format":
        payload["format"] = "unknown"
    elif mutation == "schema":
        payload["schema_version"] = True
    elif mutation == "feature_schema":
        payload["feature_schema_version"] = 99
    elif mutation == "config":
        payload["config"] = []
    elif mutation == "arrays":
        payload["feature_names"] = "not-an-array"
    elif mutation == "routes":
        payload["routes"] = []
    elif mutation == "route_entry":
        payload["routes"]["alpha"] = []
    elif mutation == "route_fields":
        payload["routes"]["alpha"]["extra"] = 1
    elif mutation == "prototype":
        payload["routes"]["alpha"]["prototype"] = "vector"
    elif mutation == "calibration":
        payload["calibration_sha256"] = 7
    else:
        payload["training_sha256"] = 7
    _resign(payload)
    with pytest.raises(PersistenceError, match=message):
        SimilarityModel.from_dict(payload)


def test_similarity_state_rejects_constructor_invariants() -> None:
    model = SimilarityModel.fit(_training())
    with pytest.raises(ConfigurationError, match="config must"):
        replace(model, config=object())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="digest"):
        replace(model, training_sha256="bad")
    with pytest.raises(ConfigurationError, match="sorted and unique"):
        replace(model, feature_names=tuple(reversed(model.feature_names)))
    with pytest.raises(ConfigurationError, match="IDF"):
        replace(model, inverse_document_frequency=(0.5, *model.inverse_document_frequency[1:]))
    with pytest.raises(ConfigurationError, match="routes and route counts"):
        replace(model, route_counts={})
    with pytest.raises(ConfigurationError, match="wrong dimension"):
        replace(model, prototypes={"alpha": (1.0,), "beta": model.prototypes["beta"]})
    with pytest.raises(ConfigurationError, match="invalid values"):
        bad = (-1.0, *model.prototypes["alpha"][1:])
        replace(model, prototypes={"alpha": bad, "beta": model.prototypes["beta"]})


def test_similarity_load_rejects_missing_non_object_and_bad_limit(tmp_path) -> None:
    with pytest.raises(PersistenceError, match="Cannot read"):
        SimilarityModel.load(tmp_path / "missing.json")
    root = tmp_path / "root.json"
    root.write_text("[]", encoding="utf-8")
    with pytest.raises(PersistenceError, match="root must be an object"):
        SimilarityModel.load(root)
    with pytest.raises(ConfigurationError, match="max_state_bytes"):
        SimilarityModel.load(root, max_state_bytes=True)


def test_fit_calibrate_evaluate_is_group_disjoint_and_held_out_once() -> None:
    training = (
        _trace("t1", "python code", "alpha", user_id="train-a", task="code"),
        _trace("t2", "matrix math", "beta", user_id="train-b", task="math"),
    )
    calibration = (
        _trace("c1", "debug code", "alpha", user_id="cal-a", task="code"),
        _trace("c2", "algebra math", "beta", user_id="cal-b", task="math"),
    )
    held_out = (
        _trace("h1", "python function", "alpha", user_id="test-a", task="code"),
        _trace("h2", "prove equation", "beta", user_id="test-b", task="math"),
    )

    model, report = fit_calibrate_evaluate(
        training, calibration, held_out, group_by="user_id", minimum_coverage=1.0
    )

    assert model.training_sha256 == report.training_sha256
    assert model.calibration_sha256 == report.calibration_sha256
    assert report.held_out is not None
    assert report.held_out.records == 2
    assert report.held_out.accuracy == 1.0
    assert report.to_dict()["groups"] == {"training": 2, "calibration": 2, "held_out": 2}
    assert report.model_state_sha256 == model.to_dict()["state_sha256"]


def test_similarity_experiment_rejects_leakage_and_unseen_label() -> None:
    training = (_trace("t", "python", "alpha", user_id="same"),)
    calibration = (_trace("c", "algebra", "beta", user_id="same"),)
    with pytest.raises(ConfigurationError, match="overlap by user_id"):
        fit_calibrate_evaluate(training, calibration, group_by="user_id")

    with pytest.raises(ConfigurationError, match="no training prototype"):
        fit_calibrate_evaluate(
            training,
            (_trace("c", "unknown route", "gamma", user_id="different"),),
        )

    model, report = fit_calibrate_evaluate(
        (_trace("t1", "python", "alpha"), _trace("t2", "math", "beta")),
        (_trace("c1", "code", "alpha"), _trace("c2", "equation", "beta")),
    )
    assert model.calibration_sha256 is not None
    assert report.held_out is None
    assert "held_out" not in report.to_dict()


@pytest.mark.parametrize(
    ("beta_overrides", "request_overrides", "reason"),
    [
        (
            {"capabilities": frozenset({"text"})},
            {"required_capabilities": frozenset({"math"})},
            "missing capabilities",
        ),
        (
            {"input_cost_per_million": 100.0, "output_cost_per_million": 100.0},
            {"max_cost_usd": 0.001, "expected_output_tokens": 1_000},
            "exceeds limit",
        ),
        ({"latency_ms_p95": 900.0}, {"max_latency_ms": 500.0}, "p95 latency"),
        ({"metadata": {"local": False}}, {"sensitivity": "restricted"}, "restricted data"),
    ],
)
def test_router_filters_hard_constraints_before_similarity(
    make_model: Callable[..., ModelCandidate],
    beta_overrides,
    request_overrides,
    reason: str,
) -> None:
    training = (
        _trace("a", "ordinary local request", "alpha"),
        _trace("b", "private financial records", "beta"),
    )
    model = SimilarityModel.fit(training)
    alpha = make_model(
        "alpha",
        metadata={"local": True},
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
    )
    beta = make_model("beta", **beta_overrides)
    router = SimilarityRouter((alpha, beta), model)

    decision = router.route(RouteRequest("private financial records", **request_overrides))

    assert decision.selected_model == "alpha"
    assert any(reason in item for item in decision.excluded["beta"])
    assert [item["model_id"] for item in decision.feature_summary["similarity"]["ranking"]] == [
        "alpha"
    ]
    assert decision.score == decision.feature_summary["similarity"]["ranking"][0]["similarity"]


def test_router_falls_back_when_no_trained_route_is_eligible(
    make_model: Callable[..., ModelCandidate],
) -> None:
    model = SimilarityModel.fit((_trace("a", "python code", "alpha"),))
    preferences = {"u": UserPreferences("u", blocked_models=frozenset({"alpha"}))}
    router = SimilarityRouter(
        (make_model("alpha"), make_model("beta", quality_by_task={"default": 0.9})),
        model,
        preferences,
    )

    decision = router.route(RouteRequest("python code", user_id="u"))

    assert decision.selected_model == "beta"
    assert decision.feature_summary["similarity"]["used"] is False
    assert "no trained prototype" in decision.explanation[0]

    disabled = SimilarityRouter((make_model("alpha", enabled=False),), model)
    with pytest.raises(NoEligibleModelError, match="No model satisfies"):
        disabled.route(RouteRequest("python code"))


def test_similarity_hit_excludes_untrained_candidates_but_fallback_retains_them(
    make_model: Callable[..., ModelCandidate],
) -> None:
    fitted = SimilarityModel.fit(_training())
    catalog = (
        make_model("alpha", quality_by_task={"default": 0.6}),
        make_model("beta", quality_by_task={"default": 0.7}),
        make_model("gamma", quality_by_task={"default": 0.99}),
    )
    hit = SimilarityRouter(catalog, fitted).route(
        RouteRequest("debug python api", task_hint="code")
    )
    hit_scores = {hit.selected_model: hit.score, **dict(hit.alternatives)}
    assert hit.feature_summary["similarity"]["used"] is True
    assert set(hit_scores) == {"alpha", "beta"}
    assert all(0.0 <= score <= 1.0 for score in hit_scores.values())

    abstaining = replace(fitted, similarity_threshold=1.0)
    fallback = SimilarityRouter(catalog, abstaining).route(
        RouteRequest("completely unseen words", task_hint="general")
    )
    fallback_scores = {fallback.selected_model: fallback.score, **dict(fallback.alternatives)}
    assert fallback.feature_summary["similarity"]["used"] is False
    assert set(fallback_scores) == {"alpha", "beta", "gamma"}
    assert all(0.0 <= score <= 1.0 for score in fallback_scores.values())


def test_router_threshold_abstention_and_unknown_state_route(
    make_model: Callable[..., ModelCandidate],
) -> None:
    model = replace(SimilarityModel.fit(_training()), similarity_threshold=1.0)
    router = SimilarityRouter(
        (
            make_model("alpha", quality_by_task={"default": 0.5}),
            make_model("beta", quality_by_task={"default": 0.9}),
        ),
        model,
    )
    decision = router.route(RouteRequest("completely unseen words", task_hint="general"))
    assert decision.selected_model == "beta"
    assert "objective fallback" in decision.explanation[0]
    assert model.rank(RouteRequest("hello"), allowed_model_ids={"unknown"}) == ()

    unknown = SimilarityModel.fit((_trace("x", "text", "gamma"),))
    with pytest.raises(ConfigurationError, match="outside the catalog"):
        SimilarityRouter((make_model("alpha"),), unknown)
    with pytest.raises(ConfigurationError, match="model must"):
        SimilarityRouter((make_model("alpha"),), object())  # type: ignore[arg-type]

    bounded = SimilarityModel.fit(
        (_trace("b", "short", "alpha"),),
        config=SimilarityFeatureConfig(max_query_characters=8),
    )
    with pytest.raises(ConfigurationError, match="8 characters"):
        SimilarityRouter((make_model("alpha"),), bounded).route(RouteRequest("query is too long"))


def test_similarity_cli_trains_routes_and_runs_counterfactual_benchmark(
    tmp_path,
    capsys,
    make_model: Callable[..., ModelCandidate],
) -> None:
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps(
            {
                "models": [
                    make_model("alpha", quality_by_task={"default": 0.7}).to_dict(),
                    make_model("beta", quality_by_task={"default": 0.8}).to_dict(),
                ]
            }
        ),
        encoding="utf-8",
    )
    train = tmp_path / "train.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    test = tmp_path / "test.jsonl"
    write_traces(train, _training())
    write_traces(
        calibration,
        (
            _trace("c1", "python function", "alpha", task="code"),
            _trace("c2", "algebra equation", "beta", task="math"),
        ),
    )
    write_traces(
        test,
        (
            _trace("h1", "debug api", "alpha", task="code"),
            _trace("h2", "matrix theorem", "beta", task="math"),
        ),
    )
    state = tmp_path / "similarity.json"
    report = tmp_path / "training-report.json"

    assert (
        main(
            [
                "train-similarity",
                "--train-traces",
                str(train),
                "--calibration-traces",
                str(calibration),
                "--held-out-traces",
                str(test),
                "--output",
                str(state),
                "--report",
                str(report),
            ]
        )
        == 0
    )
    training_report = json.loads(capsys.readouterr().out)
    assert training_report == json.loads(report.read_text(encoding="utf-8"))
    assert training_report["held_out"]["metrics"]["accuracy"] == 1.0

    assert (
        main(
            [
                "route",
                "--models",
                str(models),
                "--policy",
                "similarity",
                "--similarity-model",
                str(state),
                "--query",
                "debug python function",
                "--task",
                "code",
            ]
        )
        == 0
    )
    decision = json.loads(capsys.readouterr().out)
    assert decision["policy"] == "similarity"
    assert decision["selected_model"] == "alpha"

    output = tmp_path / "benchmark"
    assert (
        main(
            [
                "benchmark",
                "--models",
                str(models),
                "--traces",
                str(test),
                "--policy",
                "similarity",
                "--similarity-model",
                str(state),
                "--bootstrap-samples",
                "100",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    benchmark = json.loads(capsys.readouterr().out)
    assert benchmark["policies"]["similarity"]["routed"] == 2
    assert "similarity_model" in benchmark["manifest"]["input_sha256"]
    assert (output / "benchmark.html").is_file()


def test_similarity_cli_requires_a_model_state(tmp_path, capsys) -> None:
    code = main(
        [
            "route",
            "--models",
            str(Path(__file__).parents[1] / "examples" / "models.json"),
            "--policy",
            "similarity",
            "--query",
            "hello",
        ]
    )
    assert code == 2
    assert "--similarity-model is required" in capsys.readouterr().err


def test_fit_matches_independent_tfidf_and_spherical_centroid_oracle() -> None:
    rows = (
        _trace("a1", "red red", "alpha", task="general"),
        _trace("a2", "red blue", "alpha", task="general"),
        _trace("b1", "blue blue", "beta", task="general"),
    )
    config = SimilarityFeatureConfig(
        max_query_characters=100,
        long_query_characters=10,
    )
    model = SimilarityModel.fit(rows, config=config)
    names = (
        "bias",
        "capability:text",
        "numeric:difficulty",
        "numeric:length",
        "numeric:tokens",
        "sensitivity:normal",
        "task:general",
        "term:blue",
        "term:red",
    )
    term_idf = math.log(4.0 / 3.0) + 1.0
    assert model.feature_names == names
    assert model.inverse_document_frequency == pytest.approx(
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, term_idf, term_idf),
        abs=1e-15,
    )

    def normalized(raw: tuple[float, ...]) -> tuple[float, ...]:
        norm = math.sqrt(sum(value * value for value in raw))
        return tuple(value / norm for value in raw)

    first = normalized((1, 1, 0.445, 0.07, 2 / 32768, 1, 1, 0, (1 + math.log(2)) * term_idf))
    second = normalized((1, 1, 0.48, 0.08, 2 / 32768, 1, 1, term_idf, term_idf))
    third = normalized((1, 1, 0.515, 0.09, 3 / 32768, 1, 1, (1 + math.log(2)) * term_idf, 0))
    alpha_mean = tuple((left + right) / 2 for left, right in zip(first, second, strict=True))
    expected_alpha = normalized(alpha_mean)
    assert model.prototypes["alpha"] == pytest.approx(expected_alpha, abs=1e-14)
    assert model.prototypes["beta"] == pytest.approx(third, abs=1e-14)


def test_threshold_curve_uses_one_rank_per_row_and_preserves_ties(monkeypatch) -> None:
    training = (
        _trace("ta", "train alpha", "alpha"),
        _trace("tb", "train beta", "beta"),
    )
    calibration = (
        _trace("c1", "calibration one", "alpha"),
        _trace("c2", "calibration two", "beta"),
        _trace("c3", "calibration three", "beta"),
        _trace("c4", "calibration four", "beta"),
    )
    held_out = (_trace("h1", "held out", "alpha"),)
    outcomes = {
        "c1": ("alpha", 0.8),
        "c2": ("alpha", 0.8),
        "c3": ("beta", 0.5),
        "c4": ("alpha", 0.2),
        "h1": ("alpha", 0.7),
    }
    calls: list[str] = []

    def rank(self, request, **_kwargs):
        del self
        calls.append(request.request_id)
        model_id, score = outcomes[request.request_id]
        return (SimilarityMatch(model_id, score, ()),)

    monkeypatch.setattr(SimilarityModel, "rank", rank)
    _, report = fit_calibrate_evaluate(
        training,
        calibration,
        held_out,
        minimum_coverage=0.5,
    )

    assert calls == ["c1", "c2", "c3", "c4", "h1"]
    assert report.selected_threshold == 0.5
    by_threshold = {point.threshold: point.metrics for point in report.points}
    assert tuple(sorted(by_threshold)) == (0.0, 0.2, 0.5, 0.8, 1.0)
    assert (by_threshold[0.8].covered, by_threshold[0.8].correct) == (2, 1)
    assert (by_threshold[0.5].covered, by_threshold[0.5].correct) == (3, 2)
    assert (by_threshold[0.2].covered, by_threshold[0.2].correct) == (4, 2)
    assert by_threshold[1.0].accuracy is None
    assert all(point.metrics.mean_similarity == pytest.approx(0.575) for point in report.points)


def test_fit_memory_tracks_sparse_occurrences_not_document_vocabulary_product() -> None:
    outcomes = {"alpha": TraceOutcome(0.7, 0.0, 1.0, True)}
    rows = tuple(
        RouteTrace(
            RouteRequest(f"unique{index}", request_id=f"memory-{index}", task_hint="general"),
            outcomes,
            preferred_model="alpha",
        )
        for index in range(1_200)
    )
    gc.collect()
    tracemalloc.start()
    try:
        model = SimilarityModel.fit(rows)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(model.feature_names) > 1_000
    assert peak < 8 * 1024 * 1024


def test_extractor_configuration_round_trips_and_mismatch_fails(
    tmp_path, make_model: Callable[..., ModelCandidate]
) -> None:
    from facetroute.features import QueryFeatureExtractor

    extractor = QueryFeatureExtractor(long_query_chars=17)
    model = SimilarityModel.fit(_training(), extractor=extractor)
    path = tmp_path / "custom-extractor.json"
    model.save(path)
    loaded = SimilarityModel.load(path)
    request = RouteRequest("a request whose difficulty depends on scale", task_hint="general")
    assert loaded.config.long_query_characters == 17
    assert loaded.vectorize(request) == pytest.approx(model.vectorize(request), abs=1e-15)

    with pytest.raises(ConfigurationError, match="differs from persisted"):
        loaded.rank(request, extractor=QueryFeatureExtractor(long_query_chars=18))
    with pytest.raises(ConfigurationError, match="differs from persisted"):
        SimilarityRouter(
            (make_model("alpha"), make_model("beta")),
            loaded,
            extractor=QueryFeatureExtractor(long_query_chars=18),
        )

    class DerivedExtractor(QueryFeatureExtractor):
        pass

    with pytest.raises(ConfigurationError, match="built-in"):
        SimilarityModel.fit(_training(), extractor=DerivedExtractor())


def test_similarity_state_requires_bias_and_known_feature_schema() -> None:
    model = SimilarityModel.fit(_training())
    payload = model.to_dict()
    bias_index = payload["feature_names"].index("bias")
    payload["feature_names"].pop(bias_index)
    payload["inverse_document_frequency"].pop(bias_index)
    for route in payload["routes"].values():
        route["prototype"].pop(bias_index)
        norm = math.sqrt(sum(value * value for value in route["prototype"]))
        route["prototype"] = [value / norm for value in route["prototype"]]
    _resign(payload)
    with pytest.raises(PersistenceError, match="requires bias"):
        SimilarityModel.from_dict(payload)

    with pytest.raises(ConfigurationError, match="feature schema"):
        SimilarityModel(
            config=SimilarityFeatureConfig(),
            training_sha256="0" * 64,
            feature_names=("bias", "unknown:feature"),
            inverse_document_frequency=(1.0, 1.0),
            prototypes={"alpha": (1.0, 0.0)},
            route_counts={"alpha": 1},
        )


def test_similarity_save_checks_size_before_replacing_existing_state(tmp_path) -> None:
    save_defaults = SimilarityModel.save.__kwdefaults__
    load_defaults = SimilarityModel.load.__kwdefaults__
    snapshot_defaults = SimilarityModel.load_with_sha256.__kwdefaults__
    assert save_defaults is not None
    assert load_defaults is not None
    assert snapshot_defaults is not None
    assert save_defaults["max_state_bytes"] == 512 * 1024 * 1024
    assert load_defaults["max_state_bytes"] == save_defaults["max_state_bytes"]
    assert snapshot_defaults["max_state_bytes"] == save_defaults["max_state_bytes"]

    path = tmp_path / "model.json"
    path.write_bytes(b"preserve-this-state")
    with pytest.raises(PersistenceError, match="exceeds 64 bytes"):
        SimilarityModel.fit(_training()).save(path, max_state_bytes=64)
    assert path.read_bytes() == b"preserve-this-state"


def test_default_state_envelope_round_trips_real_fit_larger_than_sixteen_mib(tmp_path) -> None:
    records = 5_200
    outcome = TraceOutcome(0.5, 0.0, 1.0, True)
    rows: list[RouteTrace] = []
    for index in range(records):
        prefix = f"model-{index}-"
        model_id = prefix + "\0" * (512 - len(prefix))
        rows.append(
            RouteTrace(
                RouteRequest("same", request_id=f"large-{index}"),
                {model_id: outcome},
                preferred_model=model_id,
            )
        )
    model = SimilarityModel.fit(
        rows,
        config=SimilarityFeatureConfig(
            max_records=records,
            max_feature_name_characters=512,
        ),
    )
    path = tmp_path / "larger-than-old-default.json"

    model.save(path)
    assert path.stat().st_size > 16 * 1024 * 1024
    loaded = SimilarityModel.load(path)

    assert loaded.config == model.config
    assert loaded.training_sha256 == model.training_sha256
    assert loaded.feature_names == model.feature_names
    assert loaded.inverse_document_frequency == model.inverse_document_frequency
    assert loaded.route_counts == model.route_counts
    assert loaded.prototypes == model.prototypes
    request = RouteRequest("same", request_id="large-probe")
    assert loaded.rank(request, top_features=0) == model.rank(request, top_features=0)


def test_partitions_reject_duplicate_request_id_even_with_other_group_key() -> None:
    training = (_trace("same", "train", "alpha", user_id="training-user"),)
    calibration = (_trace("same", "calibrate", "alpha", user_id="calibration-user"),)
    with pytest.raises(ConfigurationError, match="overlap by request_id"):
        fit_calibrate_evaluate(training, calibration, group_by="user_id")


def test_allowed_model_iterable_is_bounded_by_consumed_items() -> None:
    model = SimilarityModel.fit(
        (_trace("a", "alpha", "alpha"),),
        config=SimilarityFeatureConfig(max_records=2),
    )

    def repeated():
        yield "alpha"
        yield "alpha"
        yield "alpha"

    with pytest.raises(ConfigurationError, match="exceed configured bounds"):
        model.rank(RouteRequest("alpha"), allowed_model_ids=repeated())


def test_top_features_zero_skips_contribution_sorting(monkeypatch) -> None:
    model = SimilarityModel.fit(_training())
    real_sorted = builtins.sorted
    observed_sizes: list[int] = []

    def observing_sorted(values, *args, **kwargs):
        try:
            observed_sizes.append(len(values))
        except TypeError:
            observed_sizes.append(-1)
        return real_sorted(values, *args, **kwargs)

    monkeypatch.setattr(builtins, "sorted", observing_sorted)
    matches = model.rank(RouteRequest("debug python api", task_hint="code"), top_features=0)

    assert matches
    assert observed_sizes
    assert all(size <= len(model.prototypes) for size in observed_sizes)
    assert all(match.contributions == () for match in matches)


def test_similarity_cli_rejects_training_path_collisions(tmp_path, capsys) -> None:
    training = tmp_path / "training.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    write_traces(training, _training())
    write_traces(
        calibration,
        (
            _trace("c1", "python function", "alpha"),
            _trace("c2", "matrix proof", "beta"),
        ),
    )
    before = training.read_bytes()
    assert (
        main(
            [
                "train-similarity",
                "--train-traces",
                str(training),
                "--calibration-traces",
                str(calibration),
                "--output",
                str(training),
            ]
        )
        == 2
    )
    assert "paths collide" in capsys.readouterr().err
    assert training.read_bytes() == before

    same_output = tmp_path / "same.json"
    assert (
        main(
            [
                "train-similarity",
                "--train-traces",
                str(training),
                "--calibration-traces",
                str(calibration),
                "--output",
                str(same_output),
                "--report",
                str(same_output),
            ]
        )
        == 2
    )
    assert "paths collide" in capsys.readouterr().err
    assert not same_output.exists()


def test_similarity_cli_preflights_report_before_replacing_model(tmp_path, capsys) -> None:
    training = tmp_path / "training.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    write_traces(training, _training())
    write_traces(
        calibration,
        (
            _trace("c1", "python function", "alpha"),
            _trace("c2", "matrix proof", "beta"),
        ),
    )
    model = tmp_path / "model.json"
    model.write_bytes(b"old-model")
    report = tmp_path / "report.json"
    report.mkdir()

    assert (
        main(
            [
                "train-similarity",
                "--train-traces",
                str(training),
                "--calibration-traces",
                str(calibration),
                "--output",
                str(model),
                "--report",
                str(report),
            ]
        )
        == 2
    )
    assert "directory" in capsys.readouterr().err
    assert model.read_bytes() == b"old-model"
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob(".*.bak")) == []


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_similarity_cli_rolls_back_model_and_report_after_partial_commit(
    tmp_path, capsys, monkeypatch, error_type: type[BaseException]
) -> None:
    training = tmp_path / "training.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    write_traces(training, _training())
    write_traces(
        calibration,
        (
            _trace("c1", "python function", "alpha"),
            _trace("c2", "matrix proof", "beta"),
        ),
    )
    model = tmp_path / "model.json"
    report = tmp_path / "report.json"
    model.write_bytes(b"old-model")
    report.write_bytes(b"old-report")
    real_replace = persistence_module.os.replace
    calls = 0

    def interrupt_second_install(source, destination) -> None:
        nonlocal calls
        calls += 1
        real_replace(source, destination)
        if calls == 4:
            raise error_type("injected report install failure")

    monkeypatch.setattr(persistence_module.os, "replace", interrupt_second_install)
    arguments = [
        "train-similarity",
        "--train-traces",
        str(training),
        "--calibration-traces",
        str(calibration),
        "--output",
        str(model),
        "--report",
        str(report),
    ]
    if error_type is OSError:
        assert main(arguments) == 2
        assert "injected report install failure" in capsys.readouterr().err
    else:
        with pytest.raises(KeyboardInterrupt, match="injected report install failure"):
            main(arguments)

    assert model.read_bytes() == b"old-model"
    assert report.read_bytes() == b"old-report"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "calibration.jsonl",
        "model.json",
        "report.json",
        "training.jsonl",
    ]


def test_similarity_cli_normalizes_deep_trace_json(tmp_path, capsys) -> None:
    training = tmp_path / "deep.jsonl"
    training.write_text("[" * 1_200 + "0" + "]" * 1_200 + "\n", encoding="utf-8")

    assert (
        main(
            [
                "train-similarity",
                "--train-traces",
                str(training),
                "--calibration-traces",
                str(Path(__file__).parents[1] / "examples" / "traces.jsonl"),
                "--output",
                str(tmp_path / "model.json"),
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "nesting exceeds" in error
    assert "Traceback" not in error


def test_benchmark_rejects_similarity_state_output_collision(
    tmp_path, capsys, make_model: Callable[..., ModelCandidate]
) -> None:
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps({"models": [make_model("alpha").to_dict(), make_model("beta").to_dict()]}),
        encoding="utf-8",
    )
    traces = tmp_path / "traces.jsonl"
    write_traces(traces, _training())
    output = tmp_path / "benchmark"
    output.mkdir()
    state = output / "benchmark.json"
    SimilarityModel.fit(_training()).save(state)
    before = state.read_bytes()

    assert (
        main(
            [
                "benchmark",
                "--models",
                str(models),
                "--traces",
                str(traces),
                "--policy",
                "similarity",
                "--similarity-model",
                str(state),
                "--bootstrap-samples",
                "100",
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert "paths collide" in capsys.readouterr().err
    assert state.read_bytes() == before


def test_benchmark_preflights_every_output_before_writing(
    tmp_path, capsys, make_model: Callable[..., ModelCandidate]
) -> None:
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps({"models": [make_model("alpha").to_dict(), make_model("beta").to_dict()]}),
        encoding="utf-8",
    )
    traces = tmp_path / "traces.jsonl"
    write_traces(traces, _training())
    output = tmp_path / "benchmark"
    output.mkdir()
    (output / "benchmark.html").mkdir()

    assert (
        main(
            [
                "benchmark",
                "--models",
                str(models),
                "--traces",
                str(traces),
                "--policy",
                "rule",
                "--bootstrap-samples",
                "100",
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert "directory" in capsys.readouterr().err
    assert not (output / "benchmark.json").exists()
    assert not (output / "benchmark.csv").exists()


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_benchmark_rolls_back_all_reports_after_partial_commit(
    tmp_path,
    capsys,
    monkeypatch,
    make_model: Callable[..., ModelCandidate],
    error_type: type[BaseException],
) -> None:
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps({"models": [make_model("alpha").to_dict(), make_model("beta").to_dict()]}),
        encoding="utf-8",
    )
    traces = tmp_path / "traces.jsonl"
    write_traces(traces, _training())
    output = tmp_path / "benchmark"
    output.mkdir()
    previous = {
        "benchmark.json": b"old-json",
        "benchmark.csv": b"old-csv",
        "benchmark.html": b"old-html",
    }
    for name, content in previous.items():
        (output / name).write_bytes(content)
    real_replace = persistence_module.os.replace
    calls = 0

    def interrupt_second_install(source, destination) -> None:
        nonlocal calls
        calls += 1
        real_replace(source, destination)
        if calls == 5:
            raise error_type("injected benchmark install failure")

    monkeypatch.setattr(persistence_module.os, "replace", interrupt_second_install)
    arguments = [
        "benchmark",
        "--models",
        str(models),
        "--traces",
        str(traces),
        "--policy",
        "rule",
        "--bootstrap-samples",
        "100",
        "--output-dir",
        str(output),
    ]
    if error_type is OSError:
        assert main(arguments) == 2
        assert "injected benchmark install failure" in capsys.readouterr().err
    else:
        with pytest.raises(KeyboardInterrupt, match="injected benchmark install failure"):
            main(arguments)

    assert {name: (output / name).read_bytes() for name in previous} == previous
    assert sorted(path.name for path in output.iterdir()) == sorted(previous)


def test_benchmark_hashes_same_similarity_snapshot_it_executes(
    tmp_path,
    capsys,
    monkeypatch,
    make_model: Callable[..., ModelCandidate],
) -> None:
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps({"models": [make_model("alpha").to_dict(), make_model("beta").to_dict()]}),
        encoding="utf-8",
    )
    traces = tmp_path / "traces.jsonl"
    write_traces(traces, _training())
    state = tmp_path / "similarity.json"
    SimilarityModel.fit(_training()).save(state)
    expected_digest = hashlib.sha256(state.read_bytes()).hexdigest()
    original_load = SimilarityModel.load_with_sha256

    def load_then_replace(path, **kwargs):
        snapshot = original_load(path, **kwargs)
        Path(path).write_text("{}", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(SimilarityModel, "load_with_sha256", load_then_replace)
    output = tmp_path / "output"
    assert (
        main(
            [
                "benchmark",
                "--models",
                str(models),
                "--traces",
                str(traces),
                "--policy",
                "similarity",
                "--similarity-model",
                str(state),
                "--bootstrap-samples",
                "100",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["manifest"]["input_sha256"]["similarity_model"] == expected_digest
    assert report["policies"]["similarity"]["routed"] == len(_training())


def test_benchmark_hashes_same_catalog_snapshot_it_executes(
    tmp_path,
    capsys,
    monkeypatch,
    make_model: Callable[..., ModelCandidate],
) -> None:
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps({"models": [make_model("alpha").to_dict(), make_model("beta").to_dict()]}),
        encoding="utf-8",
    )
    expected_digest = hashlib.sha256(models.read_bytes()).hexdigest()
    traces = tmp_path / "traces.jsonl"
    write_traces(traces, _training())
    original_load = cli_module._load_models_with_sha256

    def load_then_replace(path):
        snapshot = original_load(path)
        Path(path).write_text("{}", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(cli_module, "_load_models_with_sha256", load_then_replace)
    output = tmp_path / "output"
    assert (
        main(
            [
                "benchmark",
                "--models",
                str(models),
                "--traces",
                str(traces),
                "--policy",
                "rule",
                "--bootstrap-samples",
                "100",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["manifest"]["input_sha256"]["models"] == expected_digest
    assert report["policies"]["rule"]["routed"] == len(_training())


def test_benchmark_separates_reversed_file_and_canonical_trace_digests(
    tmp_path,
    capsys,
    make_model: Callable[..., ModelCandidate],
) -> None:
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps({"models": [make_model("alpha").to_dict(), make_model("beta").to_dict()]}),
        encoding="utf-8",
    )
    reversed_rows = tuple(reversed(_training()[:2]))
    traces = tmp_path / "reversed.jsonl"
    traces.write_text(
        "".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in reversed_rows),
        encoding="utf-8",
    )
    parsed = load_traces(traces)
    file_digest = file_sha256(traces)
    canonical_digest = traces_sha256(parsed)
    sorted_digest = traces_sha256(tuple(sorted(parsed, key=lambda row: row.request.request_id)))
    assert file_digest != canonical_digest
    assert canonical_digest != sorted_digest

    output = tmp_path / "output"
    assert (
        main(
            [
                "benchmark",
                "--models",
                str(models),
                "--traces",
                str(traces),
                "--policy",
                "rule",
                "--bootstrap-samples",
                "100",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads(capsys.readouterr().out)["manifest"]
    assert manifest["dataset_file_sha256"] == file_digest
    assert manifest["dataset_canonical_sha256"] == canonical_digest
    assert manifest["dataset_canonical_sha256"] != sorted_digest
