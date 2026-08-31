from __future__ import annotations

from collections.abc import Callable

import pytest

from facetroute import (
    BatchRouter,
    ConfigurationError,
    FeedbackEvent,
    LinUCBPolicy,
    LinUCBRouter,
    ModelCandidate,
    NoEligibleModelError,
    ParetoRouter,
    RouteRequest,
    RoutingRule,
    RuleRouter,
    UserPreferences,
)
from facetroute.errors import PersistenceError


def test_rule_router_returns_auditable_decision(three_models: tuple[ModelCandidate, ...]) -> None:
    rule = RoutingRule(
        name="favor-quality", prefer_models=("quality",), tasks=frozenset({"reasoning"}), bonus=0.2
    )
    router = RuleRouter(three_models, rules=(rule,))
    decision = router.route(RouteRequest(query="plan this step by step", task_hint="reasoning"))
    assert decision.policy == "rule"
    assert decision.selected_model == "quality"
    assert decision.matched_rules == ("favor-quality",)
    assert decision.explanation
    assert len(decision.to_dict()["alternatives"]) == 2


def test_rule_router_default_profile_uses_request_user(three_models: tuple[ModelCandidate, ...]) -> None:
    decision = RuleRouter(three_models).route(RouteRequest(query="hello", user_id="new-user"))
    assert decision.user_id == "new-user"


def test_rule_router_reports_all_rejection_reasons(make_model: Callable[..., ModelCandidate]) -> None:
    router = RuleRouter((make_model(enabled=False),))
    with pytest.raises(NoEligibleModelError) as captured:
        router.route(RouteRequest(query="hello"))
    assert "balanced" in captured.value.reasons


def test_pareto_router_exposes_front_in_features(make_model: Callable[..., ModelCandidate]) -> None:
    better = make_model("better", quality_by_task={"default": 0.9}, latency_ms_p95=100)
    worse = make_model(
        "worse",
        quality_by_task={"default": 0.5},
        latency_ms_p95=400,
        input_cost_per_million=3,
    )
    decision = ParetoRouter((better, worse)).route(RouteRequest(query="hello"))
    assert decision.selected_model == "better"
    assert decision.feature_summary["pareto_front"] == ["better"]
    assert decision.feature_summary["pareto_dominated"] == ["worse"]
    assert "non-dominated" in decision.explanation[0]


def test_route_many_collects_errors_without_reordering(
    make_model: Callable[..., ModelCandidate],
) -> None:
    router = RuleRouter((make_model(context_window=20),))
    requests = (
        RouteRequest(query="ok", context_tokens=1, expected_output_tokens=1, request_id="ok"),
        RouteRequest(query="too long", context_tokens=30, expected_output_tokens=1, request_id="bad"),
        RouteRequest(query="fine", context_tokens=2, expected_output_tokens=1, request_id="fine"),
    )
    result = router.route_many(requests, fail_fast=False)
    assert [item.request_id for item in result.decisions] == ["ok", "fine"]
    assert set(result.errors) == {1}


def test_batch_adapter_fail_fast_propagates(make_model: Callable[..., ModelCandidate]) -> None:
    router = BatchRouter(RuleRouter((make_model(enabled=False),)))
    with pytest.raises(NoEligibleModelError):
        router.route((RouteRequest(query="hello"),), fail_fast=True)


def test_router_rejects_duplicate_candidate_ids(make_model: Callable[..., ModelCandidate]) -> None:
    with pytest.raises(ValueError, match="unique"):
        RuleRouter((make_model("same"), make_model("same")))


def test_linucb_rejects_wrong_context_dimension() -> None:
    policy = LinUCBPolicy(("a",), dimension=2)
    with pytest.raises(ConfigurationError, match="dimension"):
        policy.score("a", (1.0,))


def test_linucb_rejects_duplicate_or_blank_arm_identifiers() -> None:
    with pytest.raises(ConfigurationError, match="unique"):
        LinUCBPolicy(("a", " a "))
    with pytest.raises(ConfigurationError, match="non-empty"):
        LinUCBPolicy((" ",))


def test_linucb_rejects_invalid_exploration_scale() -> None:
    policy = LinUCBPolicy(("a",), dimension=1)
    with pytest.raises(ConfigurationError, match="exploration_scale"):
        policy.score("a", (1.0,), float("nan"))


def test_linucb_update_increases_prediction() -> None:
    policy = LinUCBPolicy(("a",), dimension=2, alpha=0)
    before = policy.score("a", (1.0, 0.0))[1]
    for _ in range(4):
        policy.update("a", (1.0, 0.0), 1.0)
    after = policy.score("a", (1.0, 0.0))[1]
    assert after > before
    assert policy.arms["a"].updates == 4


def test_linucb_rejects_unknown_arm_and_reward() -> None:
    policy = LinUCBPolicy(("a",), dimension=1)
    with pytest.raises(ConfigurationError, match="unknown"):
        policy.score("b", (1.0,))
    with pytest.raises(ConfigurationError, match="reward"):
        policy.update("a", (1.0,), 1.5)


def test_linucb_state_round_trip(tmp_path) -> None:
    path = tmp_path / "state.json"
    policy = LinUCBPolicy(("b", "a"), dimension=2, alpha=0.4, ridge=2)
    policy.update("a", (1.0, 0.5), 0.8)
    policy.save(path)
    restored = LinUCBPolicy.load(path)
    assert restored.to_dict() == policy.to_dict()
    assert restored.score("a", (1.0, 0.5)) == pytest.approx(policy.score("a", (1.0, 0.5)))


def test_linucb_rejects_invalid_persisted_matrix_shape() -> None:
    payload = LinUCBPolicy(("a",), dimension=2).to_dict()
    payload["arms"]["a"]["inverse_covariance"] = [[1.0]]
    with pytest.raises(PersistenceError, match="shape"):
        LinUCBPolicy.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reward_vector", [float("nan"), 0.0], "Non-finite reward"),
        ("updates", -1, "update count"),
    ],
)
def test_linucb_rejects_corrupt_persisted_arm_values(field, value, message) -> None:
    payload = LinUCBPolicy(("a",), dimension=2).to_dict()
    payload["arms"]["a"][field] = value
    with pytest.raises(PersistenceError, match=message):
        LinUCBPolicy.from_dict(payload)


def test_linucb_rejects_non_positive_definite_persisted_covariance() -> None:
    payload = LinUCBPolicy(("a",), dimension=2).to_dict()
    payload["arms"]["a"]["inverse_covariance"] = [[1.0, 2.0], [2.0, 1.0]]
    with pytest.raises(PersistenceError, match="positive definite"):
        LinUCBPolicy.from_dict(payload)


def test_linucb_rejects_untrimmed_persisted_arm_identifier() -> None:
    payload = LinUCBPolicy(("a",), dimension=2).to_dict()
    payload["arms"][" a "] = payload["arms"].pop("a")
    with pytest.raises(PersistenceError, match="trimmed"):
        LinUCBPolicy.from_dict(payload)


def test_linucb_router_returns_context_vector(three_models: tuple[ModelCandidate, ...]) -> None:
    decision = LinUCBRouter(three_models).route(RouteRequest(query="hello", user_id="u"))
    assert decision.policy == "linucb"
    assert len(decision.context_vector) == 16
    assert "predicted reward" in decision.explanation[0]


def test_linucb_router_rejects_incompatible_state_dimension(
    three_models: tuple[ModelCandidate, ...],
) -> None:
    policy = LinUCBPolicy((model.model_id for model in three_models), dimension=2)
    with pytest.raises(ConfigurationError, match="dimension must be 16"):
        LinUCBRouter(three_models, policy=policy)


def test_linucb_router_feedback_updates_selected_arm(three_models: tuple[ModelCandidate, ...]) -> None:
    router = LinUCBRouter(three_models)
    decision = router.route(RouteRequest(query="hello", user_id="u", request_id="r"))
    event = FeedbackEvent(
        request_id="r",
        user_id="u",
        model_id=decision.selected_model,
        reward=0.9,
        policy="linucb",
        context_vector=decision.context_vector,
    )
    router.update_feedback(event)
    assert router.policy.arms[decision.selected_model].updates == 1


def test_linucb_router_rejects_other_policy_feedback(three_models: tuple[ModelCandidate, ...]) -> None:
    router = LinUCBRouter(three_models)
    event = FeedbackEvent(
        request_id="r", user_id="u", model_id="cheap", reward=0.5, policy="rule"
    )
    with pytest.raises(ConfigurationError, match="does not match"):
        router.update_feedback(event)


def test_linucb_router_rejects_non_finite_prior(
    three_models: tuple[ModelCandidate, ...],
) -> None:
    with pytest.raises(ConfigurationError, match="prior_weight"):
        LinUCBRouter(three_models, prior_weight=float("nan"))


def test_user_exploration_weight_scales_bandit_confidence(
    three_models: tuple[ModelCandidate, ...],
) -> None:
    profiles = {
        "off": UserPreferences("off", exploration_weight=0),
        "on": UserPreferences("on", exploration_weight=2),
    }
    router = LinUCBRouter(three_models, profiles)
    off = router.route(RouteRequest(query="hello", user_id="off"))
    on = router.route(RouteRequest(query="hello", user_id="on"))
    assert on.score > off.score
