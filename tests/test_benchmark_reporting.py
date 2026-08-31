from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from facetroute.bandit import LinUCBRouter
from facetroute.benchmark import BenchmarkRunner, PolicySpec
from facetroute.errors import ConfigurationError
from facetroute.reporting import (
    benchmark_rows,
    write_benchmark_csv,
    write_benchmark_html,
    write_json,
)
from facetroute.routers import ParetoRouter, RuleRouter
from facetroute.traces import RouteTrace, TraceOutcome
from facetroute.types import ModelCandidate, RouteRequest


def _traces() -> tuple[RouteTrace, ...]:
    return tuple(
        RouteTrace(
            RouteRequest(
                f"request {index}",
                request_id=f"r{index}",
                max_cost_usd=0.0001 if index == 0 else None,
            ),
            {
                "cheap": TraceOutcome(0.55 + index * 0.02, 0.001, 70, True),
                "balanced": TraceOutcome(0.75, 0.004, 150, True),
                "quality": TraceOutcome(0.95, 0.03, 700, index != 1),
            },
        )
        for index in range(4)
    )


def test_benchmark_compares_policies_with_reproducible_intervals(
    three_models: tuple[ModelCandidate, ...],
):
    policies = (
        PolicySpec("rule", router=RuleRouter(three_models)),
        PolicySpec("pareto", router=ParetoRouter(three_models)),
        PolicySpec("fixed-quality", fixed_model="quality"),
    )
    first = BenchmarkRunner(three_models, seed=91, bootstrap_samples=200).run(_traces(), policies)
    second = BenchmarkRunner(three_models, seed=91, bootstrap_samples=200).run(
        _traces(),
        (
            PolicySpec("rule", router=RuleRouter(three_models)),
            PolicySpec("pareto", router=ParetoRouter(three_models)),
            PolicySpec("fixed-quality", fixed_model="quality"),
        ),
    )

    assert first.to_dict() == second.to_dict()
    assert first.manifest.records == 4
    assert len(first.manifest.dataset_sha256) == 64
    fixed = first.policies["fixed-quality"]
    assert fixed.constraint_violation_rate.estimate == 0.25
    assert fixed.average_quality.estimate == pytest.approx(0.95)
    assert fixed.average_quality.lower <= fixed.average_quality.estimate
    assert fixed.average_quality.upper >= fixed.average_quality.estimate


def test_online_linucb_benchmark_learns_from_observed_quality(three_models):
    router = LinUCBRouter(three_models)
    report = BenchmarkRunner(three_models, bootstrap_samples=100).run(
        _traces(),
        (PolicySpec("linucb-online", router=router, learn_online=True),),
    )

    assert report.policies["linucb-online"].routed == 4
    assert sum(arm.updates for arm in router.policy.arms.values()) == 4


def test_benchmark_rejects_shared_router_state_between_policy_arms(three_models):
    router = LinUCBRouter(three_models)

    with pytest.raises(ConfigurationError, match="distinct router instances"):
        BenchmarkRunner(three_models, bootstrap_samples=100).run(
            _traces(),
            (
                PolicySpec("first", router=router, learn_online=True),
                PolicySpec("second", router=router),
            ),
        )


def test_benchmark_records_router_failures_and_missing_outcomes(three_models):
    class BrokenRouter:
        def route(self, _request: RouteRequest):
            raise RuntimeError("deliberate failure")

    report = BenchmarkRunner(three_models, bootstrap_samples=100).run(
        _traces(), (PolicySpec("broken", router=BrokenRouter()),)
    )
    metrics = report.policies["broken"]

    assert metrics.routed == 0
    assert metrics.failure_rate.estimate == 1.0
    assert metrics.average_quality.estimate is None
    assert len(metrics.errors) == 4
    assert set(metrics.errors.values()) == {"RuntimeError: policy evaluation failed"}


@pytest.mark.parametrize(
    "factory, message",
    [
        (
            lambda models: PolicySpec("x", router=RuleRouter(models), fixed_model="cheap"),
            "exactly one",
        ),
        (
            lambda _models: PolicySpec("x", fixed_model="cheap", learn_online=True),
            "online learning",
        ),
        (lambda _models: PolicySpec("", fixed_model="cheap"), "cannot be empty"),
    ],
)
def test_policy_spec_validation(three_models, factory: Callable, message: str):
    with pytest.raises(ConfigurationError, match=message):
        factory(three_models)


def test_benchmark_validates_configuration_and_trace_catalog(three_models, make_model):
    with pytest.raises(ConfigurationError, match="at least one model"):
        BenchmarkRunner(())
    with pytest.raises(ConfigurationError, match="unique"):
        BenchmarkRunner((three_models[0], three_models[0]))
    with pytest.raises(ConfigurationError, match="bootstrap_samples"):
        BenchmarkRunner(three_models, bootstrap_samples=99)
    with pytest.raises(ConfigurationError, match="bootstrap_samples"):
        BenchmarkRunner(three_models, bootstrap_samples="100")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="confidence_level"):
        BenchmarkRunner(three_models, confidence_level=True)
    runner = BenchmarkRunner(three_models, bootstrap_samples=100)
    with pytest.raises(ConfigurationError, match="at least one policy"):
        runner.run(_traces(), ())
    with pytest.raises(ConfigurationError, match="policy names"):
        runner.run(
            _traces(),
            (PolicySpec("same", fixed_model="cheap"), PolicySpec("same", fixed_model="balanced")),
        )
    with pytest.raises(ConfigurationError, match="unknown model"):
        runner.run(_traces(), (PolicySpec("missing", fixed_model="missing"),))
    unknown_trace = RouteTrace(RouteRequest("x"), {"unknown": TraceOutcome(0.5, 0, 1, True)})
    with pytest.raises(ConfigurationError, match="contains unknown"):
        runner.run((unknown_trace,), (PolicySpec("cheap", fixed_model="cheap"),))
    with pytest.raises(ConfigurationError, match="input digest"):
        runner.run(
            _traces(),
            (PolicySpec("cheap", fixed_model="cheap"),),
            input_sha256={"models": "not-a-digest"},
        )


def test_report_writers_produce_machine_and_human_readable_artifacts(three_models, tmp_path):
    report = BenchmarkRunner(three_models, bootstrap_samples=100).run(
        _traces(),
        (
            PolicySpec("rule<&", router=RuleRouter(three_models)),
            PolicySpec("fixed", fixed_model="cheap"),
        ),
    )
    json_path = tmp_path / "nested" / "benchmark.json"
    csv_path = tmp_path / "benchmark.csv"
    html_path = tmp_path / "benchmark.html"

    write_json(json_path, report.to_dict())
    write_benchmark_csv(csv_path, report)
    write_benchmark_html(html_path, report)

    assert json.loads(json_path.read_text(encoding="utf-8"))["manifest"]["records"] == 4
    assert "average_quality_ci_lower" in csv_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "rule&lt;&amp;" in html
    assert len(benchmark_rows(report)) == 2
