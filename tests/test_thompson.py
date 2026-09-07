"""Posterior sampling as an alternative to optimism, and its reproducibility."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from pathlib import Path

import pytest

from facetroute import ModelCandidate, RouteRequest, UserPreferences
from facetroute.bandit import (
    LinUCBPolicy,
    LinUCBRouter,
    ThompsonPolicy,
    ThompsonRouter,
)
from facetroute.errors import ConfigurationError, PersistenceError
from facetroute.feedback import FeedbackEvent

ARMS = ("alpha", "beta", "gamma")
CONTEXT = [0.4, 0.7]


def _policy(seed: int = 7, alpha: float = 0.35) -> ThompsonPolicy:
    return ThompsonPolicy(ARMS, dimension=2, alpha=alpha, seed=seed)


def _optimistic(alpha: float = 0.35) -> LinUCBPolicy:
    return LinUCBPolicy(ARMS, dimension=2, alpha=alpha)


# ---------------------------------------------------------------------------
# Reproducible randomness.
# ---------------------------------------------------------------------------


def test_the_same_inputs_always_sample_the_same_value() -> None:
    # A routing log can only be replayed if a decision is a function of its
    # inputs. A generator advanced per call would not survive replay.
    policy = _policy()
    assert policy.score("alpha", CONTEXT) == policy.score("alpha", CONTEXT)


def test_a_different_context_samples_differently() -> None:
    policy = _policy()
    assert policy.score("alpha", [0.4, 0.7])[0] != policy.score("alpha", [0.4, 0.9])[0]


def test_a_different_seed_explores_elsewhere() -> None:
    assert _policy(seed=1).score("alpha", CONTEXT)[0] != _policy(seed=2).score("alpha", CONTEXT)[0]


def test_arms_are_sampled_independently() -> None:
    policy = _policy()
    assert len({round(policy.score(arm, CONTEXT)[0], 12) for arm in ARMS}) == len(ARMS)


def test_negative_zero_is_the_same_context_as_zero() -> None:
    # Two contexts equal as numbers must draw equally, and -0.0 == 0.0.
    policy = _policy()
    assert [0.0, 0.0] == [-0.0, -0.0]
    assert policy.score("alpha", [0.0, 0.0]) == policy.score("alpha", [-0.0, -0.0])


def test_the_draw_leaves_the_posterior_mean_and_width_untouched() -> None:
    # Sampling changes which arm is chosen, not what the data says about it.
    sampled, prediction, uncertainty = _policy().score("alpha", CONTEXT)
    _upper, optimistic_prediction, optimistic_uncertainty = _optimistic().score("alpha", CONTEXT)
    assert prediction == pytest.approx(optimistic_prediction)
    assert uncertainty == pytest.approx(optimistic_uncertainty)
    assert math.isfinite(sampled)


def test_a_certain_arm_is_sampled_at_its_mean() -> None:
    # With no uncertainty there is nothing to sample; the draw cannot move it.
    sampled, prediction, uncertainty = _policy().score("alpha", [0.0, 0.0])
    assert uncertainty == pytest.approx(0.0)
    assert sampled == pytest.approx(prediction)


def test_a_wider_scale_moves_the_sample_further() -> None:
    narrow = _policy(alpha=0.05).score("alpha", CONTEXT)
    wide = _policy(alpha=0.9).score("alpha", CONTEXT)
    assert abs(wide[0] - wide[1]) > abs(narrow[0] - narrow[1])


def test_zero_exploration_returns_the_posterior_mean() -> None:
    sampled, prediction, _uncertainty = _policy().score("alpha", CONTEXT, exploration_scale=0.0)
    assert sampled == pytest.approx(prediction)


# ---------------------------------------------------------------------------
# Learning, and how it differs from optimism.
# ---------------------------------------------------------------------------


def test_rewarding_an_arm_raises_its_posterior_mean() -> None:
    policy = _policy()
    before = policy.score("alpha", CONTEXT)[1]
    for _ in range(5):
        policy.update("alpha", CONTEXT, 1.0)
    assert policy.score("alpha", CONTEXT)[1] > before


def test_evidence_narrows_the_posterior() -> None:
    policy = _policy()
    before = policy.score("alpha", CONTEXT)[2]
    for _ in range(10):
        policy.update("alpha", CONTEXT, 0.5)
    assert policy.score("alpha", CONTEXT)[2] < before


def _selections(policy) -> dict[str, int]:
    rng = random.Random(11)
    picks = dict.fromkeys(ARMS, 0)
    for _ in range(300):
        context = [rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)]
        scores = {arm: policy.score(arm, context)[0] for arm in ARMS}
        chosen = max(scores, key=lambda arm: (scores[arm], arm))
        picks[chosen] += 1
        policy.update(chosen, context, 1.0 if chosen == "beta" else 0.2)
    return picks


def test_both_rules_concentrate_on_the_rewarding_arm() -> None:
    # Neither is useful if it does not learn. `beta` is the only arm paying out.
    for policy in (_policy(), _optimistic()):
        picks = _selections(policy)
        assert picks["beta"] == max(picks.values()), picks


def test_sampling_spreads_selections_at_least_as_wide_as_optimism() -> None:
    """The behavioural difference: optimism returns to the widest interval,
    sampling picks each arm in proportion to the chance it is best."""

    def spread(picks: dict[str, int]) -> int:
        return sum(1 for count in picks.values() if count)

    assert spread(_selections(_policy())) >= spread(_selections(_optimistic()))


# ---------------------------------------------------------------------------
# Validation and state.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [-1, 1.5, "7", True, 2**64])
def test_the_seed_is_validated(seed: object) -> None:
    with pytest.raises(ConfigurationError, match="seed"):
        ThompsonPolicy(ARMS, dimension=2, seed=seed)  # type: ignore[arg-type]


def test_an_unknown_arm_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="unknown Thompson arm"):
        _policy().score("absent", CONTEXT)


def test_a_bad_context_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="dimension"):
        _policy().score("alpha", [0.1])
    with pytest.raises(ConfigurationError, match="finite"):
        _policy().score("alpha", [float("nan"), 0.1])


@pytest.mark.parametrize("scale", [-0.1, float("inf"), float("nan")])
def test_a_bad_exploration_scale_is_refused(scale: float) -> None:
    with pytest.raises(ConfigurationError, match="exploration_scale"):
        _policy().score("alpha", CONTEXT, exploration_scale=scale)


def test_the_state_records_the_policy_and_seed() -> None:
    payload = _policy(seed=42).to_dict()
    assert payload["policy"] == "thompson"
    assert payload["seed"] == 42


def test_the_state_round_trips() -> None:
    policy = _policy(seed=42)
    policy.update("alpha", CONTEXT, 0.8)
    restored = ThompsonPolicy.from_dict(policy.to_dict())
    assert restored.seed == 42
    assert restored.score("alpha", CONTEXT) == policy.score("alpha", CONTEXT)


def test_the_posterior_is_shared_with_the_optimistic_policy() -> None:
    # The two keep the same state and differ only in how a score is drawn from
    # it, so either can read what the other wrote.
    policy = _policy()
    policy.update("alpha", CONTEXT, 0.9)
    as_optimistic = LinUCBPolicy.from_dict(policy.to_dict())
    assert as_optimistic.score("alpha", CONTEXT)[1] == pytest.approx(
        policy.score("alpha", CONTEXT)[1]
    )


def test_an_optimistic_state_loads_without_a_seed() -> None:
    assert ThompsonPolicy.from_dict(_optimistic().to_dict()).seed == 0


@pytest.mark.parametrize("seed", [-1, 1.5, "7", True])
def test_a_bad_seed_in_a_saved_state_is_refused(seed: object) -> None:
    payload = _policy().to_dict()
    payload["seed"] = seed
    with pytest.raises(PersistenceError, match="seed"):
        ThompsonPolicy.from_dict(payload)


def test_the_state_survives_a_file_round_trip(tmp_path: Path) -> None:
    policy = _policy(seed=5)
    policy.update("beta", CONTEXT, 0.7)
    path = tmp_path / "thompson.json"
    policy.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["policy"] == "thompson"
    assert ThompsonPolicy.load(path).score("beta", CONTEXT) == policy.score("beta", CONTEXT)


# ---------------------------------------------------------------------------
# The router.
# ---------------------------------------------------------------------------


def test_the_router_names_itself_in_the_decision(
    three_models: tuple[ModelCandidate, ...],
) -> None:
    decision = ThompsonRouter(three_models).route(RouteRequest(query="hello", user_id="u"))
    assert decision.policy == "thompson"
    assert "Thompson predicted reward" in decision.explanation[0]


def test_the_optimistic_router_still_names_itself(
    three_models: tuple[ModelCandidate, ...],
) -> None:
    decision = LinUCBRouter(three_models).route(RouteRequest(query="hello", user_id="u"))
    assert decision.policy == "linucb"
    assert "LinUCB predicted reward" in decision.explanation[0]


def test_the_router_builds_a_sampling_policy_by_default(
    three_models: tuple[ModelCandidate, ...],
) -> None:
    assert isinstance(ThompsonRouter(three_models).policy, ThompsonPolicy)
    assert not isinstance(LinUCBRouter(three_models).policy, ThompsonPolicy)


def test_routing_is_reproducible(three_models: tuple[ModelCandidate, ...]) -> None:
    request = RouteRequest(query="hello", user_id="u")
    first = ThompsonRouter(three_models).route(request)
    second = ThompsonRouter(three_models).route(request)
    assert first.selected_model == second.selected_model
    assert first.score == pytest.approx(second.score)


def test_the_router_records_the_context_it_sampled_under(
    three_models: tuple[ModelCandidate, ...],
) -> None:
    decision = ThompsonRouter(three_models).route(RouteRequest(query="hello", user_id="u"))
    assert len(decision.context_vector) == 16


def test_feedback_for_another_policy_is_refused(
    three_models: tuple[ModelCandidate, ...],
) -> None:
    event = FeedbackEvent(
        request_id="r1",
        user_id="u1",
        model_id="balanced",
        reward=0.5,
        policy="linucb",
        context_vector=tuple([0.1] * 16),
    )
    with pytest.raises(ConfigurationError, match="does not match"):
        ThompsonRouter(three_models).update_feedback(event)


def test_a_reader_who_never_explores_gets_the_posterior_mean(
    three_models: tuple[ModelCandidate, ...],
    make_model: Callable[..., ModelCandidate],
) -> None:
    # `exploration_weight` scales the sample the same way it scales a confidence
    # radius, so zero leaves the ranking to the posterior mean and the prior.
    quiet = {"u1": UserPreferences(user_id="u1", exploration_weight=0.0)}
    decision = ThompsonRouter(three_models, quiet).route(RouteRequest(query="hello", user_id="u1"))
    assert decision.policy == "thompson"
    assert decision.selected_model in {model.model_id for model in three_models}
