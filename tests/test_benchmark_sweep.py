"""Hand-computed threshold and adversarial cache oracles."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

import facetroute.benchmark_sweep as sweep_module
from facetroute.benchmark_sweep import (
    SweepConfig,
    cached_threshold_sweep,
    load_sweep_traces,
    run_threshold_sweep,
)
from facetroute.cli import main
from facetroute.errors import ConfigurationError, PersistenceError
from facetroute.traces import RouteTrace, TraceOutcome, write_traces
from facetroute.types import ModelCandidate, RouteRequest


def _traces() -> tuple[RouteTrace, ...]:
    values = (
        ("a", 0.9, 0.4, 0.9, 0.4),
        ("b", 0.6, 0.6, 0.5, 0.5),
        ("c", 0.2, 0.8, 0.9, 0.6),
    )
    return tuple(
        RouteTrace(
            RouteRequest(f"synthetic query {name}", request_id=name),
            {
                "weak": TraceOutcome(weak_quality, 0.1, 10.0, True),
                "strong": TraceOutcome(strong_quality, strong_cost, 30.0, name != "b"),
            },
            route_score=score,
            strong_model="strong",
            weak_model="weak",
        )
        for name, score, weak_quality, strong_quality, strong_cost in values
    )


def _models(make_model: object) -> tuple[ModelCandidate, ...]:
    factory = make_model
    return (factory("weak"), factory("strong"))  # type: ignore[operator]


def _extreme_traces() -> tuple[RouteTrace, ...]:
    maximum = sys.float_info.max
    return tuple(
        RouteTrace(
            RouteRequest(f"extreme {index}", request_id=f"extreme-{index}"),
            {
                "weak": TraceOutcome(0.5, maximum, maximum, True),
                "strong": TraceOutcome(0.8, strong_value, strong_value, True),
            },
            route_score=score,
            strong_model="strong",
            weak_model="weak",
        )
        for index, (score, strong_value) in enumerate(((0.8, maximum), (0.2, 0.0)))
    )


def test_independent_three_row_quality_cost_and_oracle(make_model: object) -> None:
    report = run_threshold_sweep(
        _traces(), _models(make_model), config=SweepConfig(thresholds=(0.5, 0.75))
    )
    by_threshold = {point.threshold: point for point in report.points}
    assert set(by_threshold) == {0.0, 0.5, 0.75, 1.0}
    assert report.weak_baseline == {
        "average_quality": pytest.approx(0.6),
        "average_cost_usd": pytest.approx(0.1),
        "average_latency_ms": 10.0,
        "success_rate": 1.0,
    }
    assert report.strong_baseline["average_quality"] == pytest.approx(2.3 / 3)
    assert report.strong_baseline["average_cost_usd"] == pytest.approx(0.5)
    assert report.strong_baseline["success_rate"] == pytest.approx(2 / 3)
    assert by_threshold[0.5].strong_calls == 2
    assert by_threshold[0.5].average_quality == pytest.approx(2.2 / 3)
    assert by_threshold[0.5].average_cost_usd == pytest.approx(1.0 / 3)
    assert by_threshold[0.5].optimal_quality_at_same_strong_calls == pytest.approx(0.8)
    assert by_threshold[0.5].quality_regret_to_call_budget_oracle == pytest.approx(1 / 15)
    assert by_threshold[0.75].strong_calls == 1
    assert by_threshold[0.75].average_quality == pytest.approx(2.3 / 3)
    assert by_threshold[0.75].average_cost_usd == pytest.approx(0.2)
    assert by_threshold[0.75].quality_regret_to_call_budget_oracle == pytest.approx(0.0)
    assert not by_threshold[0.0].pareto_optimal
    assert not by_threshold[0.5].pareto_optimal
    assert by_threshold[0.75].pareto_optimal
    assert by_threshold[1.0].pareto_optimal


def test_score_ties_and_quantile_grid_are_deterministic(make_model: object) -> None:
    traces = _traces()
    tie = RouteTrace(
        RouteRequest("tie", request_id="tie"),
        traces[0].outcomes,
        route_score=0.9,
        strong_model="strong",
        weak_model="weak",
    )
    config = SweepConfig(bins=10)
    first = run_threshold_sweep((*traces, tie), _models(make_model), config=config)
    second = run_threshold_sweep((*traces, tie), _models(make_model), config=config)
    assert first.to_dict() == second.to_dict()
    assert [point.threshold for point in first.points] == sorted(
        {point.threshold for point in first.points}
    )
    assert first.points[-1].strong_calls == 0
    assert first.points[0].strong_calls == 4


def test_max_finite_outcomes_remain_finite_direct_cached_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_model: object
) -> None:
    traces, models = _extreme_traces(), _models(make_model)
    config = SweepConfig(thresholds=(0.5,))
    report = run_threshold_sweep(traces, models, config=config)
    assert report.weak_baseline["average_cost_usd"] == sys.float_info.max
    assert report.weak_baseline["average_latency_ms"] == sys.float_info.max
    assert report.strong_baseline["average_cost_usd"] == sys.float_info.max / 2
    assert report.strong_baseline["average_latency_ms"] == sys.float_info.max / 2
    for baseline in (report.weak_baseline, report.strong_baseline):
        assert all(math.isfinite(value) for value in baseline.values())
    for point in report.points:
        assert all(math.isfinite(value) for value in point.to_dict().values())

    cache_dir = tmp_path / "cache"
    miss = cached_threshold_sweep(traces, models, cache_dir, config=config)
    hit = cached_threshold_sweep(traces, models, cache_dir, config=config)
    assert not miss.cache_hit and hit.cache_hit
    assert miss.report.to_dict() == hit.report.to_dict() == report.to_dict()

    model_file = tmp_path / "models.json"
    model_file.write_text(
        json.dumps({"models": [model.to_dict() for model in models]}), encoding="utf-8"
    )
    trace_file = tmp_path / "traces.jsonl"
    write_traces(trace_file, traces)
    output = tmp_path / "sweep.json"
    args = [
        "benchmark-sweep",
        "--models",
        str(model_file),
        "--traces",
        str(trace_file),
        "--cache-dir",
        str(tmp_path / "cli-cache"),
        "--output",
        str(output),
        "--threshold",
        "0.5",
    ]
    assert main(args) == 0
    first = output.read_bytes()
    assert "cache miss" in capsys.readouterr().out
    assert main(args) == 0
    assert output.read_bytes() == first
    assert "cache hit" in capsys.readouterr().out
    assert json.loads(first)["weak_baseline"]["average_cost_usd"] == sys.float_info.max


def test_cache_hit_skips_sweep_and_hash_changes_invalidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_model: object
) -> None:
    traces = _traces()
    models = _models(make_model)
    cache = tmp_path / "cache"
    first = cached_threshold_sweep(traces, models, cache, config=SweepConfig(thresholds=(0.5,)))
    assert not first.cache_hit
    assert first.cache_path.exists()
    original = sweep_module.run_threshold_sweep

    def no_recompute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cache hit recomputed sweep")

    monkeypatch.setattr(sweep_module, "run_threshold_sweep", no_recompute)
    hit = cached_threshold_sweep(traces, models, cache, config=SweepConfig(thresholds=(0.5,)))
    assert hit.cache_hit
    assert hit.report.to_dict() == first.report.to_dict()
    monkeypatch.setattr(sweep_module, "run_threshold_sweep", original)
    altered = list(traces)
    payload = altered[0].to_dict()
    payload["outcomes"]["strong"]["quality"] = 0.8
    altered[0] = RouteTrace.from_dict(payload)
    assert (
        cached_threshold_sweep(
            tuple(altered), models, cache, config=SweepConfig(thresholds=(0.5,))
        ).cache_path
        != first.cache_path
    )
    assert (
        cached_threshold_sweep(traces, models, cache, config=SweepConfig(bins=2)).cache_path
        != first.cache_path
    )
    changed_models = (models[0], make_model("strong", latency_ms_p50=150))  # type: ignore[operator]
    assert (
        cached_threshold_sweep(
            traces, changed_models, cache, config=SweepConfig(thresholds=(0.5,))
        ).cache_path
        != first.cache_path
    )
    for digest_name in ("trace_file_sha256", "model_file_sha256"):
        extra = {digest_name: "a" * 64}
        assert (
            cached_threshold_sweep(
                traces, models, cache, config=SweepConfig(thresholds=(0.5,)), **extra
            ).cache_path
            != first.cache_path
        )


def test_corrupt_cache_fails_closed_and_refresh_repairs(tmp_path: Path, make_model: object) -> None:
    traces, models = _traces(), _models(make_model)
    first = cached_threshold_sweep(traces, models, tmp_path)
    envelope = json.loads(first.cache_path.read_text(encoding="utf-8"))
    envelope["result"]["points"][0]["average_quality"] = 0.123
    first.cache_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(PersistenceError, match="checksum mismatch"):
        cached_threshold_sweep(traces, models, tmp_path)
    repaired = cached_threshold_sweep(traces, models, tmp_path, refresh=True)
    assert not repaired.cache_hit
    assert repaired.report.to_dict() == first.report.to_dict()
    first.cache_path.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    with pytest.raises(PersistenceError, match="byte limit"):
        cached_threshold_sweep(traces, models, tmp_path)


def test_cache_alias_and_atomic_failure_leave_existing_entry_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_model: object
) -> None:
    traces, models = _traces(), _models(make_model)
    first = cached_threshold_sweep(traces, models, tmp_path)
    original = first.cache_path.read_bytes()
    with pytest.raises(ConfigurationError, match="must not alias"):
        cached_threshold_sweep(traces, models, tmp_path, protected_paths=(first.cache_path,))

    def fail_write(_writes: object) -> None:
        raise OSError("injected cache install failure")

    monkeypatch.setattr(sweep_module, "_atomic_write_bytes_bundle", fail_write)
    with pytest.raises(PersistenceError, match="injected cache install failure"):
        cached_threshold_sweep(traces, models, tmp_path, refresh=True)
    assert first.cache_path.read_bytes() == original


def test_bounded_trace_loader_and_config_validation(tmp_path: Path) -> None:
    source = tmp_path / "traces.jsonl"
    write_traces(source, _traces())
    config = SweepConfig(max_records=3)
    loaded, digest = load_sweep_traces(source, config=config)
    assert loaded == _traces()
    assert len(digest) == 64
    with pytest.raises(ConfigurationError, match="max_input_bytes"):
        load_sweep_traces(source, config=SweepConfig(max_input_bytes=20))
    with pytest.raises(ConfigurationError, match="exceeds 2 records"):
        load_sweep_traces(source, config=SweepConfig(max_records=2))
    with pytest.raises(ConfigurationError, match="trace line exceeds"):
        load_sweep_traces(source, config=SweepConfig(max_line_bytes=16))
    with pytest.raises(ConfigurationError, match="thresholds must be unique"):
        SweepConfig(thresholds=(0.5, 0.5))
    with pytest.raises(ConfigurationError, match="thresholds must be finite"):
        SweepConfig(thresholds=(float("nan"),))
    with pytest.raises(ConfigurationError, match="bins"):
        SweepConfig(bins=101)
    for bad in (False, "ten", 0):
        with pytest.raises(ConfigurationError, match="max_records"):
            SweepConfig(max_records=bad)  # type: ignore[arg-type]
    for bad in (True, "0.5", -0.1, 1.1, float("inf")):
        with pytest.raises(ConfigurationError, match="thresholds must be finite"):
            SweepConfig(thresholds=(bad,))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="bounded tuple"):
        SweepConfig(thresholds=tuple(i / 1000 for i in range(101)))


def test_sweep_rejects_bad_pair_catalog_and_hashes(make_model: object) -> None:
    traces, models = _traces(), _models(make_model)
    with pytest.raises(ConfigurationError, match="1 to max_records"):
        run_threshold_sweep((), models)
    with pytest.raises(ConfigurationError, match="non-empty ModelCandidate"):
        run_threshold_sweep(traces, ())
    with pytest.raises(ConfigurationError, match="unique"):
        run_threshold_sweep(traces, (models[0], models[0]))
    with pytest.raises(ConfigurationError, match="trace_file_sha256"):
        run_threshold_sweep(traces, models, trace_file_sha256="bad")
    with pytest.raises(ConfigurationError, match="model_file_sha256"):
        run_threshold_sweep(traces, models, model_file_sha256="bad")
    with pytest.raises(ConfigurationError, match="must exist in the catalog"):
        run_threshold_sweep(traces, (models[0],))
    bad = RouteTrace(
        RouteRequest("unscored", request_id="unscored"),
        traces[0].outcomes,
    )
    with pytest.raises(ConfigurationError, match="strong/weak pair"):
        run_threshold_sweep((bad,), models)
    other = RouteTrace(
        RouteRequest("mixed", request_id="mixed"),
        traces[0].outcomes,
        route_score=0.5,
        strong_model="weak",
        weak_model="strong",
    )
    with pytest.raises(ConfigurationError, match="same scored strong/weak"):
        run_threshold_sweep((traces[0], other), models)


def test_cache_rejects_schema_identity_and_nonfinite_values(
    tmp_path: Path, make_model: object
) -> None:
    traces, models = _traces(), _models(make_model)
    first = cached_threshold_sweep(traces, models, tmp_path)
    original = json.loads(first.cache_path.read_text(encoding="utf-8"))
    for field, value, message in (
        ("schema_version", 2, "schema or key"),
        ("key", "0" * 64, "schema or key"),
        ("result_sha256", "0" * 64, "checksum mismatch"),
    ):
        changed = json.loads(json.dumps(original))
        changed[field] = value
        first.cache_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(PersistenceError, match=message):
            cached_threshold_sweep(traces, models, tmp_path)
    changed = json.loads(json.dumps(original))
    changed["result"]["manifest"]["records"] = 100
    first.cache_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(PersistenceError, match="identity mismatch"):
        cached_threshold_sweep(traces, models, tmp_path)
    changed = json.loads(json.dumps(original))
    changed["result"]["points"][0]["average_quality"] = float("nan")
    first.cache_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(PersistenceError, match="invalid"):
        cached_threshold_sweep(traces, models, tmp_path)


def test_cli_smoke_and_output_alias_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_model: object
) -> None:
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps({"models": [model.to_dict() for model in _models(make_model)]}),
        encoding="utf-8",
    )
    traces = tmp_path / "traces.jsonl"
    write_traces(traces, _traces())
    output = tmp_path / "sweep.json"
    args = [
        "benchmark-sweep",
        "--models",
        str(models),
        "--traces",
        str(traces),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--output",
        str(output),
        "--threshold",
        "0.5",
        "--threshold",
        "0.75",
    ]
    assert main(args) == 0
    first = output.read_bytes()
    assert "cache miss" in capsys.readouterr().out
    assert main(args) == 0
    assert output.read_bytes() == first
    assert "cache hit" in capsys.readouterr().out
    assert json.loads(first)["manifest"]["records"] == 3
    assert main([*args[: args.index("--output")], "--output", str(models)]) == 2
    assert models.read_text(encoding="utf-8").startswith("{")
