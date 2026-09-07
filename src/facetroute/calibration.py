"""Threshold calibration and cost-quality Pareto analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import ConfigurationError
from .splitting import _group_key
from .traces import RouteTrace, traces_sha256


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    threshold: float
    strong_fraction: float
    average_quality: float
    average_cost_usd: float
    average_latency_ms: float
    success_rate: float
    preference_accuracy: float | None
    pareto_optimal: bool = False

    def to_dict(self) -> dict[str, float | bool | None]:
        return {
            "threshold": self.threshold,
            "strong_fraction": self.strong_fraction,
            "average_quality": self.average_quality,
            "average_cost_usd": self.average_cost_usd,
            "average_latency_ms": self.average_latency_ms,
            "success_rate": self.success_rate,
            "preference_accuracy": self.preference_accuracy,
            "pareto_optimal": self.pareto_optimal,
        }


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    records: int
    strong_model: str
    weak_model: str
    points: tuple[CalibrationPoint, ...]
    recommended_threshold: float
    selection_reason: str
    dataset_sha256: str
    held_out: CalibrationPoint | None = None
    held_out_records: int | None = None
    held_out_dataset_sha256: str | None = None
    held_out_group_by: str | None = None
    calibration_groups: int | None = None
    held_out_groups: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": 2 if self.held_out is not None else 1,
            "records": self.records,
            "strong_model": self.strong_model,
            "weak_model": self.weak_model,
            "recommended_threshold": self.recommended_threshold,
            "selection_reason": self.selection_reason,
            "dataset_sha256": self.dataset_sha256,
            "points": [point.to_dict() for point in self.points],
        }
        if self.held_out is not None:
            result["held_out"] = {
                "records": self.held_out_records,
                "dataset_sha256": self.held_out_dataset_sha256,
                "leakage_check": {
                    "group_by": self.held_out_group_by,
                    "calibration_groups": self.calibration_groups,
                    "held_out_groups": self.held_out_groups,
                },
                "metrics": self.held_out.to_dict(),
            }
        return result


def _dominates(left: CalibrationPoint, right: CalibrationPoint) -> bool:
    no_worse = (
        left.average_quality >= right.average_quality
        and left.average_cost_usd <= right.average_cost_usd
    )
    strictly_better = (
        left.average_quality > right.average_quality
        or left.average_cost_usd < right.average_cost_usd
    )
    return no_worse and strictly_better


class ThresholdCalibrator:
    """Calibrate a strong/weak threshold against counterfactual observations."""

    def __init__(self, traces: tuple[RouteTrace, ...]) -> None:
        if not traces:
            raise ConfigurationError("calibration requires at least one trace")
        first = traces[0]
        if first.strong_model is None or first.weak_model is None:
            raise ConfigurationError("calibration traces require strong_model and weak_model")
        self.strong_model = first.strong_model
        self.weak_model = first.weak_model
        for trace in traces:
            if (trace.strong_model, trace.weak_model) != (
                self.strong_model,
                self.weak_model,
            ):
                raise ConfigurationError("all calibration traces must use the same model pair")
            if trace.route_score is None:
                raise ConfigurationError("all calibration traces require route_score")
            if trace.preferred_model is not None and trace.preferred_model not in {
                self.strong_model,
                self.weak_model,
            }:
                raise ConfigurationError(
                    "calibration preferred_model must be the strong or weak model"
                )
        self.traces = traces

    def calibrate(
        self,
        *,
        max_average_cost_usd: float | None = None,
        minimum_average_quality: float | None = None,
        dataset_sha256: str | None = None,
        held_out_traces: tuple[RouteTrace, ...] | None = None,
        held_out_dataset_sha256: str | None = None,
        held_out_group_by: str | None = None,
    ) -> CalibrationReport:
        if max_average_cost_usd is not None and (
            isinstance(max_average_cost_usd, bool)
            or not isinstance(max_average_cost_usd, (int, float))
            or not math.isfinite(max_average_cost_usd)
            or max_average_cost_usd < 0
        ):
            raise ConfigurationError("max_average_cost_usd must be finite and non-negative")
        if minimum_average_quality is not None and (
            isinstance(minimum_average_quality, bool)
            or not isinstance(minimum_average_quality, (int, float))
            or not math.isfinite(minimum_average_quality)
            or not 0 <= minimum_average_quality <= 1
        ):
            raise ConfigurationError("minimum_average_quality must be between 0 and 1")
        digest = traces_sha256(self.traces) if dataset_sha256 is None else dataset_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ConfigurationError("dataset_sha256 must be a lowercase SHA-256 digest")
        thresholds = sorted(
            {
                0.0,
                1.0,
                *(trace.route_score for trace in self.traces if trace.route_score is not None),
            }
        )
        raw_points = tuple(self._point(threshold) for threshold in thresholds)
        points = tuple(
            CalibrationPoint(
                threshold=point.threshold,
                strong_fraction=point.strong_fraction,
                average_quality=point.average_quality,
                average_cost_usd=point.average_cost_usd,
                average_latency_ms=point.average_latency_ms,
                success_rate=point.success_rate,
                preference_accuracy=point.preference_accuracy,
                pareto_optimal=not any(
                    _dominates(other, point) for other in raw_points if other is not point
                ),
            )
            for point in raw_points
        )
        feasible = [
            point
            for point in points
            if (max_average_cost_usd is None or point.average_cost_usd <= max_average_cost_usd)
            and (
                minimum_average_quality is None or point.average_quality >= minimum_average_quality
            )
        ]
        if not feasible:
            raise ConfigurationError("no threshold satisfies the requested calibration bounds")
        if max_average_cost_usd is not None:
            selected = min(
                feasible,
                key=lambda item: (-item.average_quality, item.average_cost_usd, item.threshold),
            )
            reason = "highest observed quality within the average-cost bound"
        elif minimum_average_quality is not None:
            selected = min(
                feasible,
                key=lambda item: (item.average_cost_usd, -item.average_quality, item.threshold),
            )
            reason = "lowest observed cost meeting the average-quality bound"
        else:
            # Without a deployment constraint, avoid inventing an arbitrary
            # cost/quality exchange rate. Preference accuracy is a direct label.
            labelled = [item for item in feasible if item.preference_accuracy is not None]
            if not labelled:
                raise ConfigurationError("provide a cost/quality bound or preferred_model labels")
            selected = min(
                labelled,
                key=lambda item: (
                    -(item.preference_accuracy or 0.0),
                    item.average_cost_usd,
                    item.threshold,
                ),
            )
            reason = "highest preference-label accuracy, then lowest observed cost"
        held_out_point: CalibrationPoint | None = None
        held_out_records: int | None = None
        held_out_digest: str | None = None
        audited_group_by: str | None = None
        calibration_group_count: int | None = None
        held_out_group_count: int | None = None
        if held_out_traces is None and (
            held_out_dataset_sha256 is not None or held_out_group_by is not None
        ):
            raise ConfigurationError(
                "held_out_dataset_sha256 and held_out_group_by require held_out_traces"
            )
        if held_out_traces is not None:
            evaluator = ThresholdCalibrator(held_out_traces)
            if (evaluator.strong_model, evaluator.weak_model) != (
                self.strong_model,
                self.weak_model,
            ):
                raise ConfigurationError("held-out traces must use the calibration model pair")
            audited_group_by = "request_id" if held_out_group_by is None else held_out_group_by
            if not isinstance(audited_group_by, str):
                raise ConfigurationError(
                    "held_out_group_by must be request_id, user_id, or metadata:<field>"
                )
            calibration_groups = {_group_key(trace, audited_group_by) for trace in self.traces}
            held_out_groups = {_group_key(trace, audited_group_by) for trace in held_out_traces}
            overlap = sorted(calibration_groups & held_out_groups)
            if overlap:
                preview = ", ".join(overlap[:3])
                raise ConfigurationError(
                    f"calibration and held-out traces overlap by {audited_group_by}: {preview}"
                )
            calibration_group_count = len(calibration_groups)
            held_out_group_count = len(held_out_groups)
            held_out_digest = (
                traces_sha256(held_out_traces)
                if held_out_dataset_sha256 is None
                else held_out_dataset_sha256
            )
            if (
                not isinstance(held_out_digest, str)
                or len(held_out_digest) != 64
                or any(char not in "0123456789abcdef" for char in held_out_digest)
            ):
                raise ConfigurationError(
                    "held_out_dataset_sha256 must be a lowercase SHA-256 digest"
                )
            held_out_point = evaluator._point(selected.threshold)
            held_out_records = len(held_out_traces)

        return CalibrationReport(
            records=len(self.traces),
            strong_model=self.strong_model,
            weak_model=self.weak_model,
            points=points,
            recommended_threshold=selected.threshold,
            selection_reason=reason,
            dataset_sha256=digest,
            held_out=held_out_point,
            held_out_records=held_out_records,
            held_out_dataset_sha256=held_out_digest,
            held_out_group_by=audited_group_by,
            calibration_groups=calibration_group_count,
            held_out_groups=held_out_group_count,
        )

    def _point(self, threshold: float) -> CalibrationPoint:
        selected: list[tuple[str, RouteTrace]] = []
        for trace in self.traces:
            model_id = (
                self.strong_model
                if trace.route_score is not None and trace.route_score >= threshold
                else self.weak_model
            )
            selected.append((model_id, trace))
        outcomes = [trace.outcomes[model_id] for model_id, trace in selected]
        labelled = [
            model_id == trace.preferred_model
            for model_id, trace in selected
            if trace.preferred_model is not None
        ]
        count = len(outcomes)
        return CalibrationPoint(
            threshold=threshold,
            strong_fraction=sum(model_id == self.strong_model for model_id, _ in selected) / count,
            average_quality=sum(item.quality for item in outcomes) / count,
            average_cost_usd=sum(item.cost_usd for item in outcomes) / count,
            average_latency_ms=sum(item.latency_ms for item in outcomes) / count,
            success_rate=sum(item.success for item in outcomes) / count,
            preference_accuracy=(sum(labelled) / len(labelled) if labelled else None),
        )
