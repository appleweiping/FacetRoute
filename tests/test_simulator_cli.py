from __future__ import annotations

import json
from pathlib import Path

import pytest

from facetroute import (
    EvaluationReport,
    FeedbackLog,
    LinUCBPolicy,
    LinUCBRouter,
    ModelCandidate,
    OfflineSimulator,
    RouteRequest,
    RuleRouter,
)
from facetroute.cli import build_parser, main
from facetroute.features import CONTEXT_DIMENSION

PROJECT_ROOT = Path(__file__).parents[1]
EXAMPLES = PROJECT_ROOT / "examples"


def _requests() -> tuple[RouteRequest, ...]:
    return (
        RouteRequest("Write a Python parser", user_id="u", request_id="r1"),
        RouteRequest("Explain this arithmetic proof", user_id="u", request_id="r2"),
    )


def test_simulation_is_reproducible(three_models, default_profile):
    profiles = {"u": default_profile}
    first = OfflineSimulator(RuleRouter(three_models, profiles), three_models, seed=41)
    second = OfflineSimulator(RuleRouter(three_models, profiles), three_models, seed=41)

    first_observations, first_report = first.run(_requests())
    second_observations, second_report = second.run(_requests())

    assert first_report == second_report
    assert [item.feedback.reward for item in first_observations] == [
        item.feedback.reward for item in second_observations
    ]
    assert [item.feedback.latency_ms for item in first_observations] == [
        item.feedback.latency_ms for item in second_observations
    ]


def test_zero_reward_never_counts_as_success(make_model):
    model = make_model("zero", quality_by_task={"default": 0.0})
    simulator = OfflineSimulator(RuleRouter((model,)), (model,))

    class BoundaryRandom:
        @staticmethod
        def uniform(low, _high):
            return 0.0 if low < 0 else low

        @staticmethod
        def random():
            return 0.0

    simulator.random = BoundaryRandom()  # type: ignore[assignment]
    observations, report = simulator.run((RouteRequest("hello"),))

    assert observations[0].feedback.reward == 0.0
    assert not observations[0].feedback.success
    assert report.success_rate == 0.0


def test_simulation_persists_feedback(three_models, default_profile, tmp_path):
    log = FeedbackLog(tmp_path / "feedback.jsonl")
    simulator = OfflineSimulator(
        RuleRouter(three_models, {"u": default_profile}),
        three_models,
        seed=2,
        feedback_log=log,
    )

    observations, report = simulator.run(_requests())

    assert len(tuple(log.iter_events())) == len(observations)
    assert report.routed_requests == len(observations)
    assert sum(report.selection_counts.values()) == len(observations)


def test_simulation_side_effect_failure_preserves_report_accounting(
    three_models, tmp_path, monkeypatch
):
    log = FeedbackLog(tmp_path / "feedback.jsonl")

    def fail_append(_event):
        raise OSError("disk unavailable")

    monkeypatch.setattr(log, "append", fail_append)
    observations, report = OfflineSimulator(
        RuleRouter(three_models), three_models, feedback_log=log
    ).run([RouteRequest("hello")])

    assert observations == ()
    assert report.total_requests == report.routed_requests + report.failed_requests == 1
    assert "disk unavailable" in report.failures[0]


def test_simulation_can_train_linucb(three_models, default_profile):
    router = LinUCBRouter(three_models, {"u": default_profile})

    observations, report = OfflineSimulator(router, three_models, seed=9).run(
        _requests(), learn=True
    )

    assert report.failed_requests == 0
    assert sum(arm.updates for arm in router.policy.arms.values()) == len(observations)


def test_simulation_rejects_learning_with_static_router(three_models):
    simulator = OfflineSimulator(RuleRouter(three_models), three_models)
    with pytest.raises(ValueError, match="online updates"):
        simulator.run(_requests(), learn=True)


def test_simulation_reports_constraint_failure(three_models):
    request = RouteRequest(
        "Use an unavailable image model",
        required_capabilities=frozenset({"vision"}),
    )
    observations, report = OfflineSimulator(
        RuleRouter(three_models), three_models
    ).run([request])

    assert observations == ()
    assert report.failed_requests == 1
    assert report.routed_requests == 0
    assert 0 in report.failures


def test_simulation_rejects_empty_catalog():
    with pytest.raises(ValueError, match="at least one"):
        OfflineSimulator(RuleRouter([]), [])


def test_evaluation_report_save_round_trip(tmp_path):
    report = EvaluationReport(
        total_requests=2,
        routed_requests=1,
        failed_requests=1,
        average_reward=0.8,
        average_cost_usd=0.002,
        p95_latency_ms=120.0,
        success_rate=1.0,
        average_quality_regret=0.1,
        selection_counts={"small": 1},
        failures={1: "no match"},
    )
    output = tmp_path / "nested" / "report.json"

    report.save(output)

    assert json.loads(output.read_text(encoding="utf-8")) == report.to_dict()


def test_cli_parser_exposes_all_workflows():
    parser = build_parser()
    help_text = parser.format_help()

    for command in ("route", "simulate", "feedback", "report"):
        assert command in help_text


def test_cli_route_end_to_end(capsys):
    exit_code = main(
        [
            "route",
            "--models",
            str(EXAMPLES / "models.json"),
            "--preferences",
            str(EXAMPLES / "preferences.json"),
            "--rules",
            str(EXAMPLES / "rules.json"),
            "--query",
            "Write a Python function and explain it",
            "--user",
            "builder",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["selected_model"]
    assert payload["explanation"]


def test_cli_route_reports_bad_json(capsys):
    exit_code = main(
        [
            "route",
            "--models",
            str(EXAMPLES / "models.json"),
            "--request-json",
            "not-json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "facetroute: error:" in captured.err


def test_cli_simulate_writes_all_requested_artifacts(tmp_path, capsys):
    state = tmp_path / "state.json"
    feedback = tmp_path / "feedback.jsonl"
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "simulate",
            "--models",
            str(EXAMPLES / "models.json"),
            "--preferences",
            str(EXAMPLES / "preferences.json"),
            "--queries",
            str(EXAMPLES / "queries.jsonl"),
            "--policy",
            "linucb",
            "--state",
            str(state),
            "--feedback-log",
            str(feedback),
            "--output",
            str(output),
            "--learn",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert printed == json.loads(output.read_text(encoding="utf-8"))
    assert state.exists()
    assert len(tuple(FeedbackLog(feedback).iter_events())) == printed["routed_requests"]


def test_cli_feedback_updates_policy_and_report(tmp_path, capsys):
    state = tmp_path / "state.json"
    log = tmp_path / "feedback.jsonl"
    policy = LinUCBPolicy(["small"])
    policy.save(state)
    context = [1.0] + [0.0] * (CONTEXT_DIMENSION - 1)

    feedback_exit = main(
        [
            "feedback",
            "--log",
            str(log),
            "--request-id",
            "request-1",
            "--user",
            "reader",
            "--model",
            "small",
            "--reward",
            "0.75",
            "--policy",
            "linucb",
            "--context",
            json.dumps(context),
            "--state",
            str(state),
        ]
    )
    feedback_payload = json.loads(capsys.readouterr().out)

    report_exit = main(["report", "--log", str(log)])
    report_payload = json.loads(capsys.readouterr().out)

    assert feedback_exit == 0
    assert feedback_payload["reward"] == 0.75
    assert LinUCBPolicy.load(state).arms["small"].updates == 1
    assert report_exit == 0
    assert report_payload["total_events"] == 1
    assert report_payload["models"]["small"]["average_reward"] == 0.75


def test_cli_rejects_state_update_for_non_bandit_policy(tmp_path, capsys):
    exit_code = main(
        [
            "feedback",
            "--log",
            str(tmp_path / "feedback.jsonl"),
            "--request-id",
            "request-1",
            "--user",
            "reader",
            "--model",
            "small",
            "--reward",
            "0.5",
            "--policy",
            "rule",
            "--state",
            str(tmp_path / "state.json"),
        ]
    )

    assert exit_code == 2
    assert "only valid" in capsys.readouterr().err


def test_simulator_uses_declared_candidate_cost():
    model = ModelCandidate(
        model_id="solo",
        display_name="Solo",
        capabilities=frozenset({"text"}),
        input_cost_per_million=2.0,
        output_cost_per_million=4.0,
        latency_ms_p50=10,
        latency_ms_p95=20,
        context_window=1000,
        quality_by_task={"default": 0.8},
    )
    request = RouteRequest("hello", expected_output_tokens=10, context_tokens=10)

    observations, report = OfflineSimulator(RuleRouter([model]), [model], seed=3).run(
        [request]
    )

    assert len(observations) == 1
    assert report.average_cost_usd == pytest.approx(0.00006)


def test_simulator_oracle_excludes_models_rejected_by_hard_constraints(make_model):
    local = make_model(
        "local", quality_by_task={"default": 0.4}, metadata={"local": True}
    )
    remote = make_model(
        "remote", quality_by_task={"default": 0.9}, metadata={"local": False}
    )
    observations, report = OfflineSimulator(
        RuleRouter((local, remote)), (local, remote), seed=3
    ).run([RouteRequest("private", sensitivity="restricted")])

    assert observations[0].oracle_quality == 0.4
    assert observations[0].regret == 0.0
    assert report.average_quality_regret == 0.0


def test_cli_linucb_feedback_requires_context(tmp_path, capsys):
    exit_code = main(
        [
            "feedback",
            "--log",
            str(tmp_path / "feedback.jsonl"),
            "--request-id",
            "request-1",
            "--user",
            "reader",
            "--model",
            "small",
            "--reward",
            "0.5",
            "--policy",
            "linucb",
        ]
    )

    assert exit_code == 2
    assert "--context is required" in capsys.readouterr().err
    assert not (tmp_path / "feedback.jsonl").exists()
