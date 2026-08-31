from __future__ import annotations

from collections.abc import Callable

import pytest

from facetroute import (
    ConfigurationError,
    ConstraintEngine,
    ModelCandidate,
    MultiObjectiveScorer,
    QueryFeatureExtractor,
    RouteRequest,
    RoutingRule,
    UserPreferences,
    dominates,
    pareto_front,
)
from facetroute.rules import match_rules


def _filter(
    models: tuple[ModelCandidate, ...],
    request: RouteRequest,
    profile: UserPreferences,
):
    features = QueryFeatureExtractor().extract(request)
    return ConstraintEngine().filter(models, request, features, profile), features


def test_disabled_model_is_rejected(make_model: Callable[..., ModelCandidate]) -> None:
    result, _ = _filter((make_model(enabled=False),), RouteRequest(query="hi"), UserPreferences("u"))
    assert "disabled" in result.rejected["balanced"][0]


def test_blocked_model_is_rejected(make_model: Callable[..., ModelCandidate]) -> None:
    result, _ = _filter(
        (make_model(),), RouteRequest(query="hi"), UserPreferences("u", blocked_models=frozenset({"balanced"}))
    )
    assert any("blocked" in reason for reason in result.rejected["balanced"])


def test_missing_capability_is_reported(make_model: Callable[..., ModelCandidate]) -> None:
    model = make_model(capabilities=frozenset({"text"}))
    result, _ = _filter(
        (model,), RouteRequest(query="calculate x = 2 + 2", task_hint="math"), UserPreferences("u")
    )
    assert "math" in " ".join(result.rejected["balanced"])


def test_context_limit_counts_expected_output(make_model: Callable[..., ModelCandidate]) -> None:
    model = make_model(context_window=100)
    request = RouteRequest(query="x", context_tokens=80, expected_output_tokens=30)
    result, _ = _filter((model,), request, UserPreferences("u"))
    assert any("context limit" in reason for reason in result.rejected["balanced"])


def test_tool_support_is_hard_constraint(make_model: Callable[..., ModelCandidate]) -> None:
    model = make_model(capabilities=frozenset({"text", "tools"}), supports_tools=False)
    result, _ = _filter((model,), RouteRequest(query="tool", needs_tools=True), UserPreferences("u"))
    assert any("tool calling" in reason for reason in result.rejected["balanced"])


def test_tool_support_does_not_require_duplicate_capability(
    make_model: Callable[..., ModelCandidate],
) -> None:
    model = make_model(capabilities=frozenset({"text"}), supports_tools=True)
    result, _ = _filter((model,), RouteRequest(query="tool", needs_tools=True), UserPreferences("u"))
    assert result.eligible == (model,)


def test_json_support_does_not_require_duplicate_capability(
    make_model: Callable[..., ModelCandidate],
) -> None:
    model = make_model(capabilities=frozenset({"text"}), supports_json=True)
    result, _ = _filter((model,), RouteRequest(query="json", needs_json=True), UserPreferences("u"))
    assert result.eligible == (model,)


def test_request_cannot_override_profile_required_region(
    make_model: Callable[..., ModelCandidate],
) -> None:
    model = make_model(regions=frozenset({"eu"}))
    profile = UserPreferences("u", required_region="us")
    result, _ = _filter((model,), RouteRequest(query="hi", region="eu"), profile)
    assert result.eligible == ()
    assert "conflicts" in " ".join(result.rejected["balanced"])


def test_restricted_request_requires_local_metadata(make_model: Callable[..., ModelCandidate]) -> None:
    remote = make_model("remote", metadata={"local": False})
    local = make_model("local", metadata={"local": True})
    result, _ = _filter(
        (remote, local), RouteRequest(query="private", sensitivity="restricted"), UserPreferences("u")
    )
    assert result.eligible == (local,)
    assert "restricted" in " ".join(result.rejected["remote"])


def test_restricted_request_requires_literal_boolean_local_marker(
    make_model: Callable[..., ModelCandidate],
) -> None:
    misleading = make_model(metadata={"local": "false"})
    result, _ = _filter(
        (misleading,),
        RouteRequest(query="private", sensitivity="restricted"),
        UserPreferences("u"),
    )
    assert result.eligible == ()
    assert "restricted" in " ".join(result.rejected["balanced"])


def test_request_and_profile_cost_limits_use_stricter_value(make_model: Callable[..., ModelCandidate]) -> None:
    model = make_model(input_cost_per_million=10, output_cost_per_million=10)
    request = RouteRequest(query="x", context_tokens=1000, expected_output_tokens=0, max_cost_usd=1)
    profile = UserPreferences("u", max_cost_usd=0.005)
    result, _ = _filter((model,), request, profile)
    assert any("exceeds limit" in reason for reason in result.rejected["balanced"])


def test_latency_and_quality_limits_are_enforced(make_model: Callable[..., ModelCandidate]) -> None:
    model = make_model(latency_ms_p95=500, quality_by_task={"default": 0.5})
    profile = UserPreferences("u", max_latency_ms=300, minimum_quality=0.7)
    result, _ = _filter((model,), RouteRequest(query="hi"), profile)
    reasons = " ".join(result.rejected["balanced"])
    assert "latency" in reasons and "quality" in reasons


def test_eligible_models_include_estimated_cost(make_model: Callable[..., ModelCandidate]) -> None:
    model = make_model()
    result, _ = _filter((model,), RouteRequest(query="hello", context_tokens=100), UserPreferences("u"))
    assert result.eligible == (model,)
    assert result.estimated_costs["balanced"] > 0


def test_scorer_rewards_explicit_preference(three_models: tuple[ModelCandidate, ...]) -> None:
    request = RouteRequest(query="hello", user_id="u")
    profile = UserPreferences(
        "u", quality_weight=1, cost_weight=0, latency_weight=0, preferred_models=frozenset({"cheap"})
    )
    filtered, features = _filter(three_models, request, profile)
    scored = MultiObjectiveScorer(preferred_model_bonus=0.5).score(
        filtered.eligible, request, features, profile, filtered.estimated_costs
    )
    assert scored[0].candidate.model_id == "cheap"
    assert scored[0].breakdown.preference_bonus == 0.5


def test_scorer_cost_utility_orders_cheaper_model(three_models: tuple[ModelCandidate, ...]) -> None:
    request = RouteRequest(query="hello")
    profile = UserPreferences("default", quality_weight=0, cost_weight=1, latency_weight=0)
    filtered, features = _filter(three_models, request, profile)
    scored = MultiObjectiveScorer().score(
        filtered.eligible, request, features, profile, filtered.estimated_costs
    )
    assert scored[0].candidate.model_id == "cheap"
    assert scored[0].breakdown.cost_utility == 1


def test_equal_scores_break_ties_by_model_id(make_model: Callable[..., ModelCandidate]) -> None:
    models = (make_model("z"), make_model("a"))
    request = RouteRequest(query="hello")
    profile = UserPreferences("default")
    filtered, features = _filter(models, request, profile)
    scored = MultiObjectiveScorer().score(models, request, features, profile, filtered.estimated_costs)
    assert [item.candidate.model_id for item in scored] == ["a", "z"]


def test_rule_match_respects_task_and_difficulty() -> None:
    features = QueryFeatureExtractor().extract(
        RouteRequest(query="plan step by step", task_hint="reasoning")
    )
    rule = RoutingRule(
        name="deep", prefer_models=("quality",), tasks=frozenset({"reasoning"}), minimum_difficulty=0.3
    )
    bonuses, names = match_rules((rule,), features, {"quality"})
    assert bonuses == {"quality": rule.bonus}
    assert names == ("deep",)


def test_rule_never_applies_to_ineligible_model() -> None:
    features = QueryFeatureExtractor().extract(RouteRequest(query="hello"))
    rule = RoutingRule(name="x", prefer_models=("blocked",))
    bonuses, names = match_rules((rule,), features, {"other"})
    assert bonuses == {}
    assert names == ()


def test_rule_json_rejects_string_instead_of_array() -> None:
    with pytest.raises(ConfigurationError, match="collection of strings"):
        RoutingRule.from_dict({"name": "bad", "prefer_models": "model-a"})


def test_pareto_front_removes_strictly_dominated_model(make_model: Callable[..., ModelCandidate]) -> None:
    winner = make_model("winner", quality_by_task={"default": 0.8}, latency_ms_p95=100)
    loser = make_model(
        "loser",
        quality_by_task={"default": 0.7},
        latency_ms_p95=200,
        input_cost_per_million=2,
        output_cost_per_million=3,
    )
    features = QueryFeatureExtractor().extract(RouteRequest(query="hello"))
    costs = {"winner": 0.001, "loser": 0.002}
    assert dominates(winner, loser, features, costs)
    assert pareto_front((winner, loser), features, costs) == (winner,)


def test_equal_pareto_points_remain_on_front(make_model: Callable[..., ModelCandidate]) -> None:
    left = make_model("left")
    right = make_model("right")
    features = QueryFeatureExtractor().extract(RouteRequest(query="hello"))
    assert pareto_front((right, left), features, {"left": 1, "right": 1}) == (left, right)
