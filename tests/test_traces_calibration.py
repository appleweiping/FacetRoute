from __future__ import annotations

import json

import pytest

from facetroute.calibration import ThresholdCalibrator
from facetroute.errors import ConfigurationError
from facetroute.traces import (
    RouteTrace,
    TraceOutcome,
    file_sha256,
    load_traces,
    traces_sha256,
    write_traces,
)
from facetroute.types import RouteRequest


def _trace(
    request_id: str,
    score: float,
    *,
    preferred: str = "strong",
    strong_quality: float = 0.9,
    weak_quality: float = 0.6,
    user_id: str = "default",
    metadata: dict[str, object] | None = None,
) -> RouteTrace:
    return RouteTrace(
        request=RouteRequest(
            "local evaluation text",
            request_id=request_id,
            user_id=user_id,
            metadata={} if metadata is None else metadata,
        ),
        outcomes={
            "strong": TraceOutcome(strong_quality, 0.02, 400, True),
            "weak": TraceOutcome(weak_quality, 0.002, 80, weak_quality > 0.5),
        },
        preferred_model=preferred,
        route_score=score,
        strong_model="strong",
        weak_model="weak",
    )


def test_trace_round_trip_through_strict_jsonl(tmp_path):
    source = tmp_path / "traces.jsonl"
    records = (_trace("a", 0.8), _trace("b", 0.2, preferred="weak"))
    source.write_text(
        "\n".join(json.dumps(record.to_dict()) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_traces(source)

    assert [item.request.request_id for item in loaded] == ["a", "b"]
    assert loaded[0].outcomes["strong"].quality == 0.9
    assert len(file_sha256(source)) == 64


def test_trace_round_trip_preserves_request_metadata(tmp_path):
    source = tmp_path / "metadata.jsonl"
    trace = RouteTrace(
        RouteRequest(
            "private request",
            request_id="metadata",
            sensitivity="restricted",
            metadata={"tenant": "local-eval", "nested": {"fold": 2}},
        ),
        {"local": TraceOutcome(0.8, 0.0, 10, True)},
    )
    source.write_text(json.dumps(trace.to_dict()) + "\n", encoding="utf-8")

    loaded = load_traces(source)

    assert loaded[0].request.metadata == trace.request.metadata


@pytest.mark.parametrize(
    "line, message",
    [
        ('{"request":{},"request":{},"outcomes":{}}', "duplicate"),
        ('{"request":{"query":"x"},"outcomes":{"m":{"quality":NaN}}}', "non-finite"),
        ('{"request":{"query":"x"},"outcomes":{},"surprise":1}', "unknown trace"),
        ("[]", "JSON object"),
    ],
)
def test_trace_loader_rejects_ambiguous_or_invalid_json(tmp_path, line, message):
    source = tmp_path / "bad.jsonl"
    source.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        load_traces(source)


def test_trace_loader_rejects_duplicate_ids_and_limits(tmp_path):
    source = tmp_path / "duplicate.jsonl"
    line = json.dumps(_trace("same", 0.5).to_dict())
    source.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="duplicate request_id"):
        load_traces(source)
    with pytest.raises(ConfigurationError, match="exceeds 1 records"):
        load_traces(source, max_records=1)
    with pytest.raises(ConfigurationError, match="exceeds 5 bytes"):
        load_traces(source, max_line_bytes=5)
    with pytest.raises(ValueError, match="positive"):
        load_traces(source, max_records=0)


def test_trace_schema_validates_pair_ids_outcomes_and_numbers():
    request = RouteRequest("x", request_id="r")
    outcome = TraceOutcome(0.5, 0.1, 10, True)
    with pytest.raises(ConfigurationError, match="provided together"):
        RouteTrace(request, {"a": outcome}, strong_model="a")
    with pytest.raises(ConfigurationError, match="must differ"):
        RouteTrace(
            request,
            {"a": outcome},
            route_score=0.2,
            strong_model="a",
            weak_model="a",
        )
    with pytest.raises(ConfigurationError, match=r"\[0, 1\)"):
        RouteTrace(request, {"a": outcome}, route_score=1.0)
    with pytest.raises(ConfigurationError, match="preferred_model"):
        RouteTrace(request, {"a": outcome}, preferred_model="missing")
    with pytest.raises(ConfigurationError, match="between 0 and 1"):
        TraceOutcome(2.0, 0.0, 0.0, True)
    with pytest.raises(ConfigurationError, match="success"):
        TraceOutcome.from_dict({"quality": 0.5, "cost_usd": 0, "latency_ms": 1, "success": 1})
    with pytest.raises(ConfigurationError, match="TraceOutcome"):
        RouteTrace(request, {"a": object()})  # type: ignore[dict-item]
    with pytest.raises(ConfigurationError, match="trimmed"):
        RouteTrace(request, {" a ": outcome})


def test_trace_loader_requires_stable_request_id(tmp_path):
    source = tmp_path / "unstable.jsonl"
    source.write_text(
        json.dumps(
            {
                "request": {"query": "x"},
                "outcomes": {
                    "a": {
                        "quality": 0.5,
                        "cost_usd": 0,
                        "latency_ms": 1,
                        "success": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="required for reproducibility"):
        load_traces(source)

    source.write_text(
        json.dumps(
            {
                "request_id": "r",
                "request": {"query": "x", "request_id": "r", "unknown": True},
                "outcomes": {
                    "a": {
                        "quality": 0.5,
                        "cost_usd": 0,
                        "latency_ms": 1,
                        "success": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="unknown trace request"):
        load_traces(source)


def test_calibrator_builds_cost_quality_pareto_curve_and_recommendation():
    traces = (
        _trace("a", 0.8, preferred="strong", strong_quality=0.95),
        _trace("b", 0.3, preferred="weak", strong_quality=0.7),
        _trace("c", 0.6, preferred="strong", strong_quality=0.9),
    )

    report = ThresholdCalibrator(traces).calibrate(max_average_cost_usd=0.014)

    assert report.records == 3
    assert report.strong_model == "strong"
    assert report.weak_model == "weak"
    assert any(point.pareto_optimal for point in report.points)
    chosen = next(
        point for point in report.points if point.threshold == report.recommended_threshold
    )
    assert chosen.average_cost_usd <= 0.014
    payload = report.to_dict()
    assert payload["schema_version"] == 1
    assert "held_out" not in payload


def test_calibrator_evaluates_selected_threshold_once_on_disjoint_holdout():
    calibration = (
        _trace("cal-a", 0.9, preferred="strong", user_id="cal-user-a"),
        _trace("cal-b", 0.2, preferred="weak", user_id="cal-user-b"),
    )
    held_out = (
        _trace(
            "test-a",
            0.8,
            preferred="strong",
            strong_quality=0.95,
            user_id="test-user-a",
        ),
        _trace(
            "test-b",
            0.1,
            preferred="weak",
            strong_quality=0.7,
            user_id="test-user-b",
        ),
    )

    report = ThresholdCalibrator(calibration).calibrate(
        held_out_traces=held_out, held_out_group_by="user_id"
    )

    assert report.held_out is not None
    assert report.held_out.threshold == report.recommended_threshold
    assert report.held_out_records == 2
    assert report.held_out_dataset_sha256 == traces_sha256(held_out)
    assert report.to_dict()["schema_version"] == 2
    assert report.to_dict()["held_out"] == {
        "records": 2,
        "dataset_sha256": traces_sha256(held_out),
        "leakage_check": {
            "group_by": "user_id",
            "calibration_groups": 2,
            "held_out_groups": 2,
        },
        "metrics": report.held_out.to_dict(),
    }


def test_calibration_without_holdout_retains_v1_wire_schema() -> None:
    report = ThresholdCalibrator((_trace("legacy", 0.8),)).calibrate()

    payload = report.to_dict()

    assert payload["schema_version"] == 1
    assert set(payload) == {
        "schema_version",
        "records",
        "strong_model",
        "weak_model",
        "recommended_threshold",
        "selection_reason",
        "dataset_sha256",
        "points",
    }


def test_calibrator_refuses_holdout_leakage_pair_drift_and_bad_digest():
    calibration = (_trace("same", 0.9),)
    with pytest.raises(ConfigurationError, match="overlap"):
        ThresholdCalibrator(calibration).calibrate(held_out_traces=calibration)

    different_pair = RouteTrace(
        RouteRequest("x", request_id="held-out"),
        {
            "large": TraceOutcome(0.9, 1, 1, True),
            "small": TraceOutcome(0.5, 0, 1, True),
        },
        preferred_model="large",
        route_score=0.5,
        strong_model="large",
        weak_model="small",
    )
    with pytest.raises(ConfigurationError, match="model pair"):
        ThresholdCalibrator(calibration).calibrate(held_out_traces=(different_pair,))
    with pytest.raises(ConfigurationError, match="held_out_dataset_sha256"):
        ThresholdCalibrator(calibration).calibrate(
            held_out_traces=(_trace("held-out", 0.1),), held_out_dataset_sha256="bad"
        )
    with pytest.raises(ConfigurationError, match="require held_out_traces"):
        ThresholdCalibrator(calibration).calibrate(held_out_group_by="user_id")


def test_calibrator_can_reject_user_or_metadata_group_leakage() -> None:
    calibration = (_trace("cal", 0.9, user_id="shared", metadata={"domain": "shared-domain"}),)
    held_out_same_user = (
        _trace("test", 0.1, user_id="shared", metadata={"domain": "other-domain"}),
    )
    with pytest.raises(ConfigurationError, match="overlap by user_id"):
        ThresholdCalibrator(calibration).calibrate(
            held_out_traces=held_out_same_user,
            held_out_group_by="user_id",
        )

    held_out_same_domain = (
        _trace("test", 0.1, user_id="other", metadata={"domain": "shared-domain"}),
    )
    with pytest.raises(ConfigurationError, match="overlap by metadata:domain"):
        ThresholdCalibrator(calibration).calibrate(
            held_out_traces=held_out_same_domain,
            held_out_group_by="metadata:domain",
        )


def test_trace_canonical_writer_round_trips_and_fingerprints(tmp_path):
    path = tmp_path / "canonical.jsonl"
    traces = (_trace("a", 0.8), _trace("b", 0.2))
    write_traces(path, traces)
    assert load_traces(path) == traces
    assert traces_sha256(traces) == file_sha256(path)
    with pytest.raises(ConfigurationError, match="empty"):
        write_traces(path, ())
    with pytest.raises(ConfigurationError, match="empty"):
        traces_sha256(())


def test_calibrator_can_optimize_label_accuracy_or_quality_floor():
    traces = (
        _trace("a", 0.9, preferred="strong"),
        _trace("b", 0.1, preferred="weak"),
    )
    label_report = ThresholdCalibrator(traces).calibrate()
    floor_report = ThresholdCalibrator(traces).calibrate(minimum_average_quality=0.7)

    label_point = next(
        point
        for point in label_report.points
        if point.threshold == label_report.recommended_threshold
    )
    assert label_point.preference_accuracy == 1.0
    floor_point = next(
        point
        for point in floor_report.points
        if point.threshold == floor_report.recommended_threshold
    )
    assert floor_point.average_quality >= 0.7


def test_calibrator_rejects_inconsistent_or_infeasible_data():
    bare = RouteTrace(
        RouteRequest("x"),
        {"a": TraceOutcome(0.5, 0, 1, True)},
    )
    with pytest.raises(ConfigurationError, match="require strong_model"):
        ThresholdCalibrator((bare,))
    with pytest.raises(ConfigurationError, match="same model pair"):
        ThresholdCalibrator(
            (
                _trace("a", 0.5),
                RouteTrace(
                    RouteRequest("x", request_id="b"),
                    {
                        "s": TraceOutcome(0.9, 1, 1, True),
                        "w": TraceOutcome(0.5, 0, 1, True),
                    },
                    route_score=0.5,
                    strong_model="s",
                    weak_model="w",
                ),
            )
        )
    with pytest.raises(ConfigurationError, match="no threshold"):
        ThresholdCalibrator((_trace("a", 0.5),)).calibrate(
            max_average_cost_usd=0.0,
            minimum_average_quality=1.0,
        )
    unlabelled = RouteTrace(
        RouteRequest("x"),
        {
            "strong": TraceOutcome(0.9, 1, 1, True),
            "weak": TraceOutcome(0.5, 0, 1, True),
        },
        route_score=0.5,
        strong_model="strong",
        weak_model="weak",
    )
    with pytest.raises(ConfigurationError, match="provide a cost/quality bound"):
        ThresholdCalibrator((unlabelled,)).calibrate()
    third_label = RouteTrace(
        RouteRequest("x"),
        {
            "strong": TraceOutcome(0.9, 1, 1, True),
            "weak": TraceOutcome(0.5, 0, 1, True),
            "third": TraceOutcome(0.7, 0.5, 1, True),
        },
        preferred_model="third",
        route_score=0.5,
        strong_model="strong",
        weak_model="weak",
    )
    with pytest.raises(ConfigurationError, match="strong or weak"):
        ThresholdCalibrator((third_label,))
    with pytest.raises(ConfigurationError, match="max_average_cost_usd"):
        ThresholdCalibrator((_trace("bound", 0.5),)).calibrate(max_average_cost_usd=True)
