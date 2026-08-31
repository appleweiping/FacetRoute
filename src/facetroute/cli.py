"""Command-line interface for fully offline routing workflows."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from ._json import loads_strict
from .bandit import LinUCBPolicy, LinUCBRouter
from .benchmark import BenchmarkRunner, PolicySpec
from .calibration import ThresholdCalibrator
from .config import load_models, load_preferences, load_requests, load_rules, request_from_dict
from .errors import FacetRouteError
from .feedback import FeedbackEvent, FeedbackLog
from .reporting import (
    write_benchmark_csv,
    write_benchmark_html,
    write_calibration_csv,
    write_json,
)
from .routers import ParetoRouter, Router, RuleRouter
from .server import create_server
from .simulator import OfflineSimulator
from .traces import file_sha256, load_traces
from .types import ModelCandidate, RouteRequest, UserPreferences


def _add_catalog_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", required=True, help="JSON model catalog")
    parser.add_argument("--preferences", help="JSON user-profile file")
    parser.add_argument("--rules", help="JSON routing-rule file")
    parser.add_argument("--policy", choices=("rule", "pareto", "linucb"), default="rule")
    parser.add_argument("--state", help="LinUCB JSON state path")
    parser.add_argument("--alpha", type=float, default=0.35, help="LinUCB exploration factor")
    parser.add_argument(
        "--prior-weight", type=float, default=0.2, help="deterministic prior in LinUCB"
    )


def _build_router(
    policy_name: str,
    models: tuple[ModelCandidate, ...],
    preferences: dict[str, UserPreferences],
    rules_path: str | None,
    state_path: str | None,
    alpha: float,
    prior_weight: float,
) -> Router:
    rules = load_rules(rules_path)
    if policy_name == "rule":
        return RuleRouter(models, preferences, rules)
    if policy_name == "pareto":
        return ParetoRouter(models, preferences, rules)
    if state_path and Path(state_path).exists():
        bandit = LinUCBPolicy.load(state_path)
    else:
        bandit = LinUCBPolicy((model.model_id for model in models), alpha=alpha)
    return LinUCBRouter(
        models,
        preferences,
        rules,
        policy=bandit,
        prior_weight=prior_weight,
    )


def _route_request_from_args(args: argparse.Namespace) -> RouteRequest:
    if args.request_json:
        payload = loads_strict(args.request_json)
        if not isinstance(payload, dict):
            raise ValueError("--request-json must be a JSON object")
        return request_from_dict(payload)
    return RouteRequest(
        query=args.query,
        user_id=args.user,
        expected_output_tokens=args.output_tokens,
        required_capabilities=frozenset(args.require_capability or []),
        max_cost_usd=args.max_cost,
        max_latency_ms=args.max_latency,
        region=args.region,
        needs_tools=args.needs_tools,
        needs_json=args.needs_json,
        sensitivity=args.sensitivity,
        task_hint=args.task,
        context_tokens=args.context_tokens,
    )


def _run_route(args: argparse.Namespace) -> int:
    models = load_models(args.models)
    preferences = load_preferences(args.preferences)
    router = _build_router(
        args.policy,
        models,
        preferences,
        args.rules,
        args.state,
        args.alpha,
        args.prior_weight,
    )
    decision = router.route(_route_request_from_args(args))
    if args.state and isinstance(router, LinUCBRouter):
        router.save_state(args.state)
    print(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _run_simulate(args: argparse.Namespace) -> int:
    models = load_models(args.models)
    preferences = load_preferences(args.preferences)
    router = _build_router(
        args.policy,
        models,
        preferences,
        args.rules,
        args.state,
        args.alpha,
        args.prior_weight,
    )
    requests = load_requests(args.queries)
    log = FeedbackLog(args.feedback_log) if args.feedback_log else None
    simulator = OfflineSimulator(router, models, seed=args.seed, feedback_log=log)
    _, report = simulator.run(requests, learn=args.learn)
    if args.state and isinstance(router, LinUCBRouter):
        router.save_state(args.state)
    if args.output:
        report.save(args.output)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report.routed_requests else 1


def _parse_context(value: str) -> tuple[float, ...]:
    payload = loads_strict(value)
    if not isinstance(payload, list):
        raise ValueError("--context must be a JSON list of numbers")
    return tuple(float(item) for item in payload)


def _run_feedback(args: argparse.Namespace) -> int:
    if args.policy == "linucb" and not args.context:
        raise ValueError("--context is required for --policy linucb feedback")
    event = FeedbackEvent(
        request_id=args.request_id,
        user_id=args.user,
        model_id=args.model,
        reward=args.reward,
        policy=args.policy,
        context_vector=_parse_context(args.context) if args.context else (),
        success=args.success,
        latency_ms=args.latency_ms,
        cost_usd=args.cost_usd,
    )
    policy: LinUCBPolicy | None = None
    if args.state:
        if args.policy != "linucb":
            raise ValueError("--state updates are only valid for --policy linucb")
        policy = LinUCBPolicy.load(args.state)
        policy.update(event.model_id, event.context_vector, event.reward)
    FeedbackLog(args.log).append(event)
    if policy is not None:
        policy.save(args.state)
    print(json.dumps(event.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _run_report(args: argparse.Namespace) -> int:
    summary = FeedbackLog(args.log).summarize()
    payload: dict[str, Any] = {
        "models": {model_id: item.to_dict() for model_id, item in summary.items()},
        "total_events": sum(item.count for item in summary.values()),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _run_calibrate(args: argparse.Namespace) -> int:
    report = ThresholdCalibrator(load_traces(args.traces)).calibrate(
        max_average_cost_usd=args.max_average_cost,
        minimum_average_quality=args.minimum_average_quality,
        dataset_sha256=file_sha256(args.traces),
    )
    payload = report.to_dict()
    if args.output:
        write_json(args.output, payload)
    if args.csv:
        write_calibration_csv(args.csv, report)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    models = load_models(args.models)
    preferences = load_preferences(args.preferences)
    rules = load_rules(args.rules)
    traces = load_traces(args.traces)
    policies: list[PolicySpec] = []
    selected = set(args.policy or ("rule", "pareto", "linucb", "fixed"))
    if "rule" in selected:
        policies.append(PolicySpec("rule", router=RuleRouter(models, preferences, rules)))
    if "pareto" in selected:
        policies.append(PolicySpec("pareto", router=ParetoRouter(models, preferences, rules)))
    if "linucb" in selected:
        policies.append(
            PolicySpec(
                "linucb-online",
                router=LinUCBRouter(
                    models,
                    preferences,
                    rules,
                    policy=LinUCBPolicy((model.model_id for model in models), alpha=args.alpha),
                    prior_weight=args.prior_weight,
                ),
                learn_online=True,
            )
        )
    fixed_models = args.fixed_model
    if "fixed" in selected and not fixed_models:
        fixed_models = [model.model_id for model in models]
    known = {model.model_id for model in models}
    for model_id in fixed_models:
        if model_id not in known:
            raise ValueError(f"unknown --fixed-model: {model_id}")
        policies.append(PolicySpec(f"fixed:{model_id}", fixed_model=model_id))
    runner = BenchmarkRunner(
        models,
        preferences,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence,
    )
    input_digests = {"models": file_sha256(args.models)}
    if args.preferences:
        input_digests["preferences"] = file_sha256(args.preferences)
    if args.rules:
        input_digests["rules"] = file_sha256(args.rules)
    report = runner.run(
        traces,
        policies,
        dataset_sha256=file_sha256(args.traces),
        input_sha256=input_digests,
    )
    output = Path(args.output_dir)
    write_json(output / "benchmark.json", report.to_dict())
    write_benchmark_csv(output / "benchmark.csv", report)
    write_benchmark_html(output / "benchmark.html", report)
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _run_serve(args: argparse.Namespace) -> int:
    models = load_models(args.models)
    preferences = load_preferences(args.preferences)
    router = _build_router(
        args.policy,
        models,
        preferences,
        args.rules,
        args.state,
        args.alpha,
        args.prior_weight,
    )
    token = os.environ.get(args.token_env) if args.token_env else None
    if not _is_loopback(args.host) and token is None and not args.allow_unauthenticated_nonloopback:
        raise ValueError(
            "non-loopback binding requires a bearer token environment variable or "
            "--allow-unauthenticated-nonloopback"
        )
    server = create_server(
        router,
        models,
        host=args.host,
        port=args.port,
        max_body_bytes=args.max_body_bytes,
        max_concurrency=args.max_concurrency,
        request_timeout_seconds=args.request_timeout,
        bearer_token=token,
    )
    raw_host, port = server.server_address[:2]
    host = raw_host.decode("ascii") if isinstance(raw_host, bytes) else str(raw_host)
    print(f"FacetRoute listening on http://{host}:{port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facetroute",
        description="Offline-first personalized routing for declared LLM candidates.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    route = commands.add_parser("route", help="route one request")
    _add_catalog_arguments(route)
    input_group = route.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--query")
    input_group.add_argument("--request-json")
    route.add_argument("--user", default="default")
    route.add_argument("--output-tokens", type=int, default=256)
    route.add_argument("--context-tokens", type=int)
    route.add_argument("--require-capability", action="append")
    route.add_argument("--max-cost", type=float)
    route.add_argument("--max-latency", type=float)
    route.add_argument("--region")
    route.add_argument("--needs-tools", action="store_true")
    route.add_argument("--needs-json", action="store_true")
    route.add_argument(
        "--sensitivity", choices=("normal", "sensitive", "restricted"), default="normal"
    )
    route.add_argument("--task")
    route.set_defaults(handler=_run_route)

    simulate = commands.add_parser("simulate", help="run deterministic offline evaluation")
    _add_catalog_arguments(simulate)
    simulate.add_argument("--queries", required=True, help="JSONL request set")
    simulate.add_argument("--seed", type=int, default=7)
    simulate.add_argument("--learn", action="store_true", help="update LinUCB after each event")
    simulate.add_argument("--feedback-log")
    simulate.add_argument("--output", help="write report JSON")
    simulate.set_defaults(handler=_run_simulate)

    feedback = commands.add_parser("feedback", help="append feedback and optionally update LinUCB")
    feedback.add_argument("--log", required=True)
    feedback.add_argument("--request-id", required=True)
    feedback.add_argument("--user", required=True)
    feedback.add_argument("--model", required=True)
    feedback.add_argument("--reward", required=True, type=float)
    feedback.add_argument("--policy", choices=("rule", "pareto", "linucb"), required=True)
    feedback.add_argument("--context", help="JSON numeric vector; required for state update")
    feedback.add_argument("--state", help="existing LinUCB state to update")
    feedback.add_argument("--success", action=argparse.BooleanOptionalAction, default=True)
    feedback.add_argument("--latency-ms", type=float)
    feedback.add_argument("--cost-usd", type=float)
    feedback.set_defaults(handler=_run_feedback)

    report = commands.add_parser("report", help="summarize a feedback JSONL log")
    report.add_argument("--log", required=True)
    report.set_defaults(handler=_run_report)

    calibrate = commands.add_parser(
        "calibrate", help="calibrate a strong/weak score threshold from strict traces"
    )
    calibrate.add_argument("--traces", required=True, help="strict JSONL route trace")
    calibrate.add_argument("--max-average-cost", type=float)
    calibrate.add_argument("--minimum-average-quality", type=float)
    calibrate.add_argument("--output", help="write calibration JSON")
    calibrate.add_argument("--csv", help="write cost-quality curve CSV")
    calibrate.set_defaults(handler=_run_calibrate)

    benchmark = commands.add_parser(
        "benchmark", help="compare routers against observed counterfactual outcomes"
    )
    benchmark.add_argument("--models", required=True)
    benchmark.add_argument("--preferences")
    benchmark.add_argument("--rules")
    benchmark.add_argument("--traces", required=True)
    benchmark.add_argument(
        "--policy",
        action="append",
        choices=("rule", "pareto", "linucb", "fixed"),
        default=[],
        help="policy to include; repeatable (default: all)",
    )
    benchmark.add_argument("--fixed-model", action="append", default=[])
    benchmark.add_argument("--alpha", type=float, default=0.35)
    benchmark.add_argument("--prior-weight", type=float, default=0.2)
    benchmark.add_argument("--seed", type=int, default=17)
    benchmark.add_argument("--bootstrap-samples", type=int, default=1000)
    benchmark.add_argument("--confidence", type=float, default=0.95)
    benchmark.add_argument("--output-dir", required=True)
    benchmark.set_defaults(handler=_run_benchmark)

    serve = commands.add_parser(
        "serve", help="serve routing decisions without proxying provider requests"
    )
    _add_catalog_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--max-body-bytes", type=int, default=262_144)
    serve.add_argument("--max-concurrency", type=int, default=32)
    serve.add_argument("--request-timeout", type=float, default=10.0)
    serve.add_argument(
        "--token-env",
        default="FACETROUTE_BEARER_TOKEN",
        help="environment variable containing an optional bearer token",
    )
    serve.add_argument("--allow-unauthenticated-nonloopback", action="store_true")
    serve.set_defaults(handler=_run_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FacetRouteError, ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"facetroute: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
