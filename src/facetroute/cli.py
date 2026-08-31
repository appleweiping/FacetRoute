"""Command-line interface for fully offline routing workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .bandit import LinUCBPolicy, LinUCBRouter
from .config import load_models, load_preferences, load_requests, load_rules, request_from_dict
from .errors import FacetRouteError
from .feedback import FeedbackEvent, FeedbackLog
from .routers import ParetoRouter, Router, RuleRouter
from .simulator import OfflineSimulator
from .types import ModelCandidate, RouteRequest, UserPreferences


def _add_catalog_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", required=True, help="JSON model catalog")
    parser.add_argument("--preferences", help="JSON user-profile file")
    parser.add_argument("--rules", help="JSON routing-rule file")
    parser.add_argument(
        "--policy", choices=("rule", "pareto", "linucb"), default="rule"
    )
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
        payload = json.loads(args.request_json)
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
    payload = json.loads(value)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facetroute",
        description="Offline-first personalized routing for declared LLM candidates.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FacetRouteError, ValueError, json.JSONDecodeError) as exc:
        print(f"facetroute: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
