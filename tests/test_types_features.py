from __future__ import annotations

from collections.abc import Callable

import pytest

from facetroute import (
    CONTEXT_DIMENSION,
    ConfigurationError,
    ModelCandidate,
    QueryFeatureExtractor,
    RouteRequest,
    UserPreferences,
)


def test_model_rejects_empty_identifier(make_model: Callable[..., ModelCandidate]) -> None:
    with pytest.raises(ConfigurationError, match="model_id"):
        make_model(" ")


def test_model_rejects_quality_outside_unit_interval(
    make_model: Callable[..., ModelCandidate],
) -> None:
    with pytest.raises(ConfigurationError, match="between 0 and 1"):
        make_model(quality_by_task={"default": 1.1})


def test_model_rejects_inverted_latency_percentiles(
    make_model: Callable[..., ModelCandidate],
) -> None:
    with pytest.raises(ConfigurationError, match="p95"):
        make_model(latency_ms_p50=300, latency_ms_p95=200)


def test_cost_estimate_uses_separate_input_output_prices(
    make_model: Callable[..., ModelCandidate],
) -> None:
    model = make_model(input_cost_per_million=2, output_cost_per_million=5)
    assert model.estimate_cost(1_000, 200) == pytest.approx(0.003)


def test_quality_uses_default_fallback(make_model: Callable[..., ModelCandidate]) -> None:
    model = make_model(quality_by_task={"default": 0.61, "code": 0.8})
    assert model.quality_for("unknown") == 0.61


def test_quality_uses_mean_without_default(make_model: Callable[..., ModelCandidate]) -> None:
    model = make_model(quality_by_task={"code": 0.8, "math": 0.6})
    assert model.quality_for("general") == pytest.approx(0.7)


def test_model_round_trip_normalizes_names(make_model: Callable[..., ModelCandidate]) -> None:
    model = make_model(capabilities=frozenset({" Text ", "CODE"}), regions=frozenset({" US "}))
    restored = ModelCandidate.from_dict(model.to_dict())
    assert restored == model
    assert restored.capabilities == frozenset({"text", "code"})


def test_model_rejects_string_collection_and_non_boolean_json_fields(
    make_model: Callable[..., ModelCandidate],
) -> None:
    payload = make_model().to_dict()
    payload["capabilities"] = "text"
    with pytest.raises(ConfigurationError, match="collection of strings"):
        ModelCandidate.from_dict(payload)

    payload = make_model().to_dict()
    payload["enabled"] = "false"
    with pytest.raises(ConfigurationError, match="boolean"):
        ModelCandidate.from_dict(payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"display_name": "  "}, "display_name"),
        ({"context_window": True}, "context_window"),
        ({"context_window": 0}, "context_window"),
        ({"quality_by_task": {}}, "quality_by_task"),
        ({"quality_by_task": {" ": 0.5}}, "quality task names"),
        ({"input_cost_per_million": float("inf")}, "finite non-negative"),
        ({"capabilities": ["text", 4]}, "only strings"),
        ({"capabilities": [" ", "text"]}, ""),
        ({"regions": 3}, "collection of strings"),
    ],
)
def test_model_rejects_invalid_direct_domain_inputs(
    make_model: Callable[..., ModelCandidate], changes: dict[str, object], message: str
) -> None:
    if message:
        with pytest.raises(ConfigurationError, match=message):
            make_model(**changes)
    else:
        assert make_model(**changes).capabilities == frozenset({"text"})


def test_model_rejects_negative_cost_estimation_tokens(
    make_model: Callable[..., ModelCandidate],
) -> None:
    with pytest.raises(ConfigurationError, match="token counts"):
        make_model().estimate_cost(-1, 1)


def test_model_direct_api_rejects_non_boolean_flags(
    make_model: Callable[..., ModelCandidate],
) -> None:
    with pytest.raises(ConfigurationError, match="supports_tools"):
        make_model(supports_tools="false")


def test_profile_normalizes_objective_weights() -> None:
    profile = UserPreferences(user_id="u", quality_weight=6, cost_weight=3, latency_weight=1)
    assert profile.objective_weights("general") == pytest.approx((0.6, 0.3, 0.1))


def test_profile_rejects_preferred_blocked_overlap() -> None:
    with pytest.raises(ConfigurationError, match="both preferred and blocked"):
        UserPreferences(
            user_id="u",
            preferred_models=frozenset({"x"}),
            blocked_models=frozenset({"x"}),
        )


def test_profile_applies_task_weight_override() -> None:
    profile = UserPreferences(
        user_id="u",
        task_weight_overrides={"math": {"quality": 8, "cost": 1, "latency": 1}},
    )
    assert profile.objective_weights("math") == pytest.approx((0.8, 0.1, 0.1))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"user_id": "  "}, "user_id"),
        ({"quality_weight": 0, "cost_weight": 0, "latency_weight": 0}, "positive"),
        ({"preferred_models": ["m", 9]}, "only strings"),
        ({"blocked_models": "m"}, "collection of strings"),
        ({"preferred_models": [" ", "m"]}, ""),
        ({"task_weight_overrides": {"math": {"energy": 1}}}, "unsupported"),
    ],
)
def test_profile_domain_input_boundaries(changes: dict[str, object], message: str) -> None:
    if message:
        with pytest.raises(ConfigurationError, match=message):
            UserPreferences(**{"user_id": "u", **changes})
    else:
        assert UserPreferences(**{"user_id": "u", **changes}).preferred_models == frozenset({"m"})


def test_profile_rejects_zero_task_specific_objective() -> None:
    profile = UserPreferences(
        user_id="u",
        task_weight_overrides={"math": {"quality": 0, "cost": 0, "latency": 0}},
    )
    with pytest.raises(ConfigurationError, match="sum to zero"):
        profile.objective_weights("math")


def test_request_rejects_empty_query() -> None:
    with pytest.raises(ConfigurationError, match="query"):
        RouteRequest(query="  ")


def test_request_rejects_unknown_sensitivity() -> None:
    with pytest.raises(ConfigurationError, match="sensitivity"):
        RouteRequest(query="hello", sensitivity="secret")


def test_request_rejects_empty_request_id() -> None:
    with pytest.raises(ConfigurationError, match="request_id"):
        RouteRequest(query="hello", request_id=" ")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"user_id": "  "}, "user_id"),
        ({"expected_output_tokens": True}, "integer"),
        ({"expected_output_tokens": -1}, "negative"),
        ({"context_tokens": True}, "integer"),
        ({"context_tokens": -1}, "negative"),
    ],
)
def test_request_rejects_invalid_token_and_identity_inputs(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        RouteRequest(query="hello", **changes)


def test_request_direct_api_rejects_non_boolean_flags() -> None:
    with pytest.raises(ConfigurationError, match="must be booleans"):
        RouteRequest(query="hello", needs_tools="false")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "task"),
    [
        ("Please debug: def add(a, b): return a+b", "code"),
        ("Calculate x = 3 * 7 and explain the equation", "math"),
        ("Translate this paragraph into Spanish", "translation"),
        ("Summarize this article in one sentence", "summarization"),
        ("Hello there", "general"),
    ],
)
def test_feature_extractor_classifies_common_tasks(text: str, task: str) -> None:
    assert QueryFeatureExtractor().extract(RouteRequest(query=text)).task == task


def test_task_hint_overrides_inference() -> None:
    request = RouteRequest(query="def x(): pass", task_hint="summarization")
    assert QueryFeatureExtractor().extract(request).task == "summarization"


def test_explicit_context_tokens_override_estimate() -> None:
    features = QueryFeatureExtractor().extract(RouteRequest(query="short", context_tokens=9_000))
    assert features.token_estimate == 9_000


def test_zero_context_tokens_are_preserved() -> None:
    features = QueryFeatureExtractor().extract(RouteRequest(query="short", context_tokens=0))
    assert features.token_estimate == 0


def test_tool_and_json_requirements_become_capabilities() -> None:
    request = RouteRequest(query="use a tool", needs_tools=True, needs_json=True)
    features = QueryFeatureExtractor().extract(request)
    assert {"text", "tools", "json"}.issubset(features.required_capabilities)


def test_multistep_language_increases_difficulty() -> None:
    extractor = QueryFeatureExtractor()
    simple = extractor.extract(RouteRequest(query="Describe rain."))
    complex_features = extractor.extract(
        RouteRequest(query="First compare both designs and then derive a step-by-step plan.")
    )
    assert complex_features.has_multistep_language
    assert complex_features.difficulty > simple.difficulty


def test_context_vector_has_stable_bounded_shape(default_profile: UserPreferences) -> None:
    extractor = QueryFeatureExtractor()
    features = extractor.extract(RouteRequest(query="Explain this step by step", user_id="u"))
    vector = extractor.context_vector(features, default_profile)
    assert len(vector) == CONTEXT_DIMENSION
    assert all(0 <= value <= 1 for value in vector)
