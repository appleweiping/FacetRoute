"""Command-line interface for offline evaluation and optional provider serving."""

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
from .bandit import LinUCBPolicy, LinUCBRouter, ThompsonPolicy, ThompsonRouter
from .benchmark import BenchmarkRunner, PolicySpec
from .benchmark_formats import BenchmarkFormat, load_benchmark_examples, write_benchmark_examples
from .calibration import ThresholdCalibrator
from .config import (
    _load_models_with_sha256,
    _load_preferences_with_sha256,
    _load_rules_with_sha256,
    load_models,
    load_preferences,
    load_requests,
    load_rules,
    request_from_dict,
)
from .errors import FacetRouteError
from .factorization import (
    FactorizationConfig,
    FactorizationRouter,
    PairwiseFactorModel,
    evaluate_held_out,
)
from .feedback import FeedbackEvent, FeedbackLog
from .persistence import _atomic_write_bytes_bundle, _json_bytes
from .providers import load_provider_registry
from .reporting import (
    write_benchmark_bundle,
    write_calibration_csv,
    write_json,
)
from .routers import ParetoRouter, Router, RuleRouter
from .server import create_server
from .similarity import (
    SimilarityFeatureConfig,
    SimilarityModel,
    SimilarityRouter,
    fit_calibrate_evaluate,
)
from .simulator import OfflineSimulator
from .splitting import split_traces, write_trace_partitions
from .traces import RouteTrace, _load_traces_with_sha256, load_traces
from .types import ModelCandidate, RouteRequest, UserPreferences


def _add_catalog_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", required=True, help="JSON model catalog")
    parser.add_argument("--preferences", help="JSON user-profile file")
    parser.add_argument("--rules", help="JSON routing-rule file")
    parser.add_argument(
        "--policy",
        choices=("rule", "pareto", "linucb", "thompson", "similarity", "factorization"),
        default="rule",
    )
    parser.add_argument("--state", help="LinUCB JSON state path")
    parser.add_argument(
        "--similarity-model", help="trained similarity JSON state (required for similarity)"
    )
    parser.add_argument("--factor-model", help="trained pairwise-factorization JSON state")
    parser.add_argument("--alpha", type=float, default=0.35, help="LinUCB exploration factor")
    parser.add_argument(
        "--prior-weight", type=float, default=0.2, help="deterministic prior in LinUCB"
    )


def _paths_collide(left: Path, right: Path) -> bool:
    try:
        if left.resolve() == right.resolve():
            return True
        return left.exists() and right.exists() and left.samefile(right)
    except (OSError, RuntimeError):
        return False


def _require_disjoint_paths(paths: dict[str, str | Path | None]) -> None:
    present = [(name, Path(path)) for name, path in paths.items() if path is not None]
    for index, (left_name, left_path) in enumerate(present):
        for right_name, right_path in present[index + 1 :]:
            if _paths_collide(left_path, right_path):
                raise ValueError(
                    f"paths collide: {left_name}={left_path} and {right_name}={right_path}"
                )


def _build_router(
    policy_name: str,
    models: tuple[ModelCandidate, ...],
    preferences: dict[str, UserPreferences],
    rules_path: str | None,
    state_path: str | None,
    alpha: float,
    prior_weight: float,
    similarity_model_path: str | None,
    factor_model_path: str | None,
) -> Router:
    rules = load_rules(rules_path)
    if policy_name == "rule":
        return RuleRouter(models, preferences, rules)
    if policy_name == "pareto":
        return ParetoRouter(models, preferences, rules)
    if policy_name == "similarity":
        if not similarity_model_path:
            raise ValueError("--similarity-model is required for --policy similarity")
        if state_path:
            raise ValueError("--state is only valid for linucb or thompson")
        return SimilarityRouter(
            models,
            SimilarityModel.load(similarity_model_path),
            preferences,
            rules,
        )
    if policy_name == "factorization":
        if not factor_model_path:
            raise ValueError("--factor-model is required for --policy factorization")
        if state_path:
            raise ValueError("--state is only valid for linucb or thompson")
        return FactorizationRouter(
            models,
            PairwiseFactorModel.load(factor_model_path),
            preferences,
            rules,
        )
    if policy_name not in {"linucb", "thompson"}:
        raise ValueError(f"unknown routing policy: {policy_name}")
    # Both bandits keep the same per-arm posterior, so a saved state loads under
    # either. Only the way a score is drawn from it differs.
    thompson = policy_name == "thompson"
    policy_type = ThompsonPolicy if thompson else LinUCBPolicy
    if state_path and Path(state_path).exists():
        bandit = policy_type.load(state_path)
    else:
        bandit = policy_type((model.model_id for model in models), alpha=alpha)
    router_type = ThompsonRouter if thompson else LinUCBRouter
    return router_type(
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
        args.similarity_model,
        args.factor_model,
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
        args.similarity_model,
        args.factor_model,
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
    if args.policy in {"linucb", "thompson"} and not args.context:
        raise ValueError(f"--context is required for --policy {args.policy} feedback")
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
        if args.policy not in {"linucb", "thompson"}:
            raise ValueError(
                "--state updates are only valid for --policy linucb or --policy thompson"
            )
        policy = (ThompsonPolicy if args.policy == "thompson" else LinUCBPolicy).load(args.state)
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


def _run_normalize_benchmark(args: argparse.Namespace) -> int:
    examples = load_benchmark_examples(args.input, format=args.format)
    write_benchmark_examples(args.output, examples)
    formats = {example.format.value for example in examples}
    print(f"normalized {len(examples)} {next(iter(formats))} examples to {args.output}")
    return 0


def _run_calibrate(args: argparse.Namespace) -> int:
    calibration, calibration_sha256 = _load_traces_with_sha256(args.traces)
    held_out: tuple[RouteTrace, ...] | None = None
    held_out_sha256: str | None = None
    if args.held_out_traces:
        held_out, held_out_sha256 = _load_traces_with_sha256(args.held_out_traces)
    report = ThresholdCalibrator(calibration).calibrate(
        max_average_cost_usd=args.max_average_cost,
        minimum_average_quality=args.minimum_average_quality,
        dataset_sha256=calibration_sha256,
        held_out_traces=held_out,
        held_out_dataset_sha256=held_out_sha256,
        held_out_group_by=args.held_out_group_by,
    )
    payload = report.to_dict()
    if args.output:
        write_json(args.output, payload)
    if args.csv:
        write_calibration_csv(args.csv, report)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _run_split_traces(args: argparse.Namespace) -> int:
    traces = load_traces(args.traces)
    partitions = split_traces(
        traces,
        seed=args.seed,
        train_fraction=args.train_fraction,
        calibration_fraction=args.calibration_fraction,
        group_by=args.group_by,
    )
    manifest = write_trace_partitions(
        args.output_dir,
        partitions,
        source_path=args.traces,
        dataset_name=args.dataset_name,
        source_uri=args.source_uri,
        license_name=args.license,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _run_train_similarity(args: argparse.Namespace) -> int:
    _require_disjoint_paths(
        {
            "train_traces": args.train_traces,
            "calibration_traces": args.calibration_traces,
            "held_out_traces": args.held_out_traces,
            "output": args.output,
            "report": args.report,
        }
    )
    config = SimilarityFeatureConfig(
        max_features=args.max_features,
        min_document_frequency=args.min_document_frequency,
        max_records=args.max_records,
        max_query_characters=args.max_query_characters,
        max_tokens_per_query=args.max_tokens_per_query,
        max_feature_occurrences=args.max_feature_occurrences,
        max_prototype_values=args.max_prototype_values,
        long_query_characters=args.long_query_characters,
    )
    training = load_traces(args.train_traces, max_records=config.max_records)
    calibration = load_traces(args.calibration_traces, max_records=config.max_records)
    held_out = (
        load_traces(args.held_out_traces, max_records=config.max_records)
        if args.held_out_traces
        else None
    )
    model, report = fit_calibrate_evaluate(
        training,
        calibration,
        held_out,
        config=config,
        group_by=args.group_by,
        minimum_coverage=args.minimum_coverage,
    )
    payload = report.to_dict()
    expected_state = model.to_dict()
    writes = {Path(args.output): model._state_bytes()}
    if args.report:
        writes[Path(args.report)] = _json_bytes(payload)

    def validate_model_state(staged_path: Path) -> None:
        if SimilarityModel.load(staged_path).to_dict() != expected_state:
            raise ValueError("staged similarity model did not round-trip")

    _atomic_write_bytes_bundle(
        writes,
        validators={Path(args.output): validate_model_state},
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _run_train_factorization(args: argparse.Namespace) -> int:
    _require_disjoint_paths(
        {
            "train_traces": args.train_traces,
            "held_out_traces": args.held_out_traces,
            "output": args.output,
            "report": args.report,
        }
    )
    config = FactorizationConfig(
        dimension=args.dimension,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        seed=args.seed,
        max_records=args.max_records,
        max_pairs=args.max_pairs,
        max_features=args.max_features,
    )
    training = load_traces(args.train_traces, max_records=config.max_records)
    model = PairwiseFactorModel.fit(training, config=config)
    report: dict[str, object] = {
        "training_sha256": model.encoder.training_sha256,
        "training_records": len(training),
        "training_pairs": model.training_pairs,
        "training_loss": model.final_loss,
        "routes": list(model.route_ids),
    }
    if args.held_out_traces:
        held_out = load_traces(args.held_out_traces, max_records=config.max_records)
        report["held_out"] = evaluate_held_out(model, training, held_out, group_by=args.group_by)
    expected = model.to_dict()
    writes = {Path(args.output): model._state_bytes()}
    if args.report:
        writes[Path(args.report)] = _json_bytes(report)

    def validate_state(staged_path: Path) -> None:
        if PairwiseFactorModel.load(staged_path).to_dict() != expected:
            raise ValueError("staged factorization model did not round-trip")

    _atomic_write_bytes_bundle(
        writes,
        validators={Path(args.output): validate_state},
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    _require_disjoint_paths(
        {
            "models": args.models,
            "preferences": args.preferences,
            "rules": args.rules,
            "traces": args.traces,
            "similarity_model": args.similarity_model,
            "factor_model": args.factor_model,
            "output_dir": output,
            "benchmark_json": output / "benchmark.json",
            "benchmark_csv": output / "benchmark.csv",
            "benchmark_html": output / "benchmark.html",
        }
    )
    models, models_sha256 = _load_models_with_sha256(args.models)
    if args.preferences:
        preferences, preferences_sha256 = _load_preferences_with_sha256(args.preferences)
    else:
        preferences, preferences_sha256 = {}, None
    if args.rules:
        rules, rules_sha256 = _load_rules_with_sha256(args.rules)
    else:
        rules, rules_sha256 = (), None
    traces, traces_file_sha256 = _load_traces_with_sha256(args.traces)
    policies: list[PolicySpec] = []
    defaults = ["rule", "pareto", "linucb", "thompson", "fixed"]
    if args.similarity_model:
        defaults.append("similarity")
    if args.factor_model:
        defaults.append("factorization")
    selected = set(args.policy or defaults)
    similarity_model: SimilarityModel | None = None
    similarity_model_sha256: str | None = None
    if "similarity" in selected:
        if not args.similarity_model:
            raise ValueError("--similarity-model is required for --policy similarity")
        similarity_model, similarity_model_sha256 = SimilarityModel.load_with_sha256(
            args.similarity_model
        )
    factor_model: PairwiseFactorModel | None = None
    factor_model_sha256: str | None = None
    if "factorization" in selected:
        if not args.factor_model:
            raise ValueError("--factor-model is required for --policy factorization")
        factor_model, factor_model_sha256 = PairwiseFactorModel.load_with_sha256(args.factor_model)
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
    if "thompson" in selected:
        policies.append(
            PolicySpec(
                "thompson-online",
                router=ThompsonRouter(
                    models,
                    preferences,
                    rules,
                    policy=ThompsonPolicy((model.model_id for model in models), alpha=args.alpha),
                    prior_weight=args.prior_weight,
                ),
                learn_online=True,
            )
        )
    if "similarity" in selected:
        if similarity_model is None:
            raise ValueError("--similarity-model is required for --policy similarity")
        policies.append(
            PolicySpec(
                "similarity",
                router=SimilarityRouter(
                    models,
                    similarity_model,
                    preferences,
                    rules,
                ),
            )
        )
    if "factorization" in selected:
        if factor_model is None:
            raise ValueError("--factor-model is required for --policy factorization")
        policies.append(
            PolicySpec(
                "factorization",
                router=FactorizationRouter(models, factor_model, preferences, rules),
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
    input_digests = {"models": models_sha256}
    if preferences_sha256 is not None:
        input_digests["preferences"] = preferences_sha256
    if rules_sha256 is not None:
        input_digests["rules"] = rules_sha256
    if similarity_model_sha256 is not None:
        input_digests["similarity_model"] = similarity_model_sha256
    if factor_model_sha256 is not None:
        input_digests["factor_model"] = factor_model_sha256
    report = runner.run(
        traces,
        policies,
        dataset_file_sha256=traces_file_sha256,
        input_sha256=input_digests,
    )
    write_benchmark_bundle(output, report)
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
        args.similarity_model,
        args.factor_model,
    )
    token = os.environ.get(args.token_env) if args.token_env else None
    if not _is_loopback(args.host) and token is None and not args.allow_unauthenticated_nonloopback:
        raise ValueError(
            "non-loopback binding requires a bearer token environment variable or "
            "--allow-unauthenticated-nonloopback"
        )
    providers = None
    if args.providers:
        providers = load_provider_registry(
            args.providers,
            allow_insecure_http=args.allow_insecure_provider_http,
            max_response_bytes=args.max_provider_response_bytes,
            max_stream_bytes=args.max_provider_stream_bytes,
            max_event_bytes=args.max_provider_event_bytes,
        )
        unknown_models = providers.model_ids - {model.model_id for model in models}
        if unknown_models:
            raise ValueError(
                f"provider configuration references unknown catalog models: {sorted(unknown_models)}"
            )
        missing_models = {model.model_id for model in models if model.enabled} - providers.model_ids
        if missing_models:
            raise ValueError(
                f"provider configuration is missing enabled catalog models: {sorted(missing_models)}"
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
        provider_registry=providers,
        provider_timeout_seconds=args.provider_timeout,
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
    feedback.add_argument(
        "--policy",
        choices=("rule", "pareto", "linucb", "thompson", "similarity", "factorization"),
        required=True,
    )
    feedback.add_argument("--context", help="JSON numeric vector; required for state update")
    feedback.add_argument("--state", help="existing LinUCB state to update")
    feedback.add_argument("--success", action=argparse.BooleanOptionalAction, default=True)
    feedback.add_argument("--latency-ms", type=float)
    feedback.add_argument("--cost-usd", type=float)
    feedback.set_defaults(handler=_run_feedback)

    report = commands.add_parser("report", help="summarize a feedback JSONL log")
    report.add_argument("--log", required=True)
    report.set_defaults(handler=_run_report)

    normalize = commands.add_parser(
        "normalize-benchmark",
        help="validate and canonicalize MMLU, GSM8K, or MT-Bench JSON/JSONL",
    )
    normalize.add_argument("--input", required=True, help="benchmark JSON or JSONL")
    normalize.add_argument("--output", required=True, help="canonical JSONL output")
    normalize.add_argument(
        "--format",
        choices=("auto", *(item.value for item in BenchmarkFormat)),
        default="auto",
    )
    normalize.set_defaults(handler=_run_normalize_benchmark)

    calibrate = commands.add_parser(
        "calibrate", help="calibrate a strong/weak score threshold from strict traces"
    )
    calibrate.add_argument("--traces", required=True, help="strict JSONL route trace")
    calibrate.add_argument(
        "--held-out-traces",
        help="disjoint strict JSONL trace evaluated once at the selected threshold",
    )
    calibrate.add_argument(
        "--held-out-group-by",
        help="leakage unit: request_id (default), user_id, or metadata:<field>",
    )
    calibrate.add_argument("--max-average-cost", type=float)
    calibrate.add_argument("--minimum-average-quality", type=float)
    calibrate.add_argument("--output", help="write calibration JSON")
    calibrate.add_argument("--csv", help="write cost-quality curve CSV")
    calibrate.set_defaults(handler=_run_calibrate)

    split = commands.add_parser(
        "split-traces", help="create deterministic group-disjoint experiment partitions"
    )
    split.add_argument("--traces", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--dataset-name", required=True)
    split.add_argument("--source-uri", required=True)
    split.add_argument("--license", required=True)
    split.add_argument("--seed", type=int, default=17)
    split.add_argument("--train-fraction", type=float, default=0.6)
    split.add_argument("--calibration-fraction", type=float, default=0.2)
    split.add_argument(
        "--group-by",
        default="request_id",
        help="request_id, user_id, or metadata:<field>",
    )
    split.set_defaults(handler=_run_split_traces)

    train_similarity = commands.add_parser(
        "train-similarity",
        help="fit and calibrate a versioned similarity model from disjoint traces",
    )
    train_similarity.add_argument("--train-traces", required=True)
    train_similarity.add_argument("--calibration-traces", required=True)
    train_similarity.add_argument("--held-out-traces")
    train_similarity.add_argument(
        "--group-by",
        default="request_id",
        help="leakage unit: request_id, user_id, or metadata:<field>",
    )
    train_similarity.add_argument("--minimum-coverage", type=float, default=0.5)
    train_similarity.add_argument("--max-features", type=int, default=2_048)
    train_similarity.add_argument("--min-document-frequency", type=int, default=1)
    train_similarity.add_argument("--max-records", type=int, default=100_000)
    train_similarity.add_argument("--max-query-characters", type=int, default=100_000)
    train_similarity.add_argument("--max-tokens-per-query", type=int, default=2_048)
    train_similarity.add_argument("--max-feature-occurrences", type=int, default=1_000_000)
    train_similarity.add_argument("--max-prototype-values", type=int, default=250_000)
    train_similarity.add_argument("--long-query-characters", type=int, default=1_200)
    train_similarity.add_argument("--output", required=True, help="similarity model JSON")
    train_similarity.add_argument("--report", help="optional calibration report JSON")
    train_similarity.set_defaults(handler=_run_train_similarity)

    train_factorization = commands.add_parser(
        "train-factorization",
        help="fit a deterministic pairwise low-rank router on labelled traces",
    )
    train_factorization.add_argument("--train-traces", required=True)
    train_factorization.add_argument("--held-out-traces")
    train_factorization.add_argument("--group-by", default="request_id")
    train_factorization.add_argument("--dimension", type=int, default=8)
    train_factorization.add_argument("--epochs", type=int, default=30)
    train_factorization.add_argument("--learning-rate", type=float, default=0.05)
    train_factorization.add_argument("--regularization", type=float, default=0.0001)
    train_factorization.add_argument("--seed", type=int, default=17)
    train_factorization.add_argument("--max-records", type=int, default=10_000)
    train_factorization.add_argument("--max-pairs", type=int, default=100_000)
    train_factorization.add_argument("--max-features", type=int, default=512)
    train_factorization.add_argument("--output", required=True)
    train_factorization.add_argument("--report")
    train_factorization.set_defaults(handler=_run_train_factorization)

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
        choices=("rule", "pareto", "linucb", "thompson", "similarity", "factorization", "fixed"),
        default=[],
        help="policy to include; repeatable (default: all)",
    )
    benchmark.add_argument("--fixed-model", action="append", default=[])
    benchmark.add_argument("--alpha", type=float, default=0.35)
    benchmark.add_argument("--prior-weight", type=float, default=0.2)
    benchmark.add_argument("--similarity-model", help="trained similarity JSON state")
    benchmark.add_argument("--factor-model", help="trained factorization JSON state")
    benchmark.add_argument("--seed", type=int, default=17)
    benchmark.add_argument("--bootstrap-samples", type=int, default=1000)
    benchmark.add_argument("--confidence", type=float, default=0.95)
    benchmark.add_argument("--output-dir", required=True)
    benchmark.set_defaults(handler=_run_benchmark)

    serve = commands.add_parser(
        "serve", help="serve routing decisions and an optional chat-completions proxy"
    )
    _add_catalog_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--max-body-bytes", type=int, default=262_144)
    serve.add_argument("--max-concurrency", type=int, default=32)
    serve.add_argument("--request-timeout", type=float, default=10.0)
    serve.add_argument(
        "--providers",
        help="JSON catalog-model to OpenAI-compatible provider bindings",
    )
    serve.add_argument("--provider-timeout", type=float, default=60.0)
    serve.add_argument("--max-provider-response-bytes", type=int, default=16 * 1024 * 1024)
    serve.add_argument("--max-provider-stream-bytes", type=int, default=64 * 1024 * 1024)
    serve.add_argument("--max-provider-event-bytes", type=int, default=1024 * 1024)
    serve.add_argument(
        "--allow-insecure-provider-http",
        action="store_true",
        help="allow plain HTTP to non-loopback providers (unsafe)",
    )
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
