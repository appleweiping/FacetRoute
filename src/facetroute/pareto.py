"""Pareto-front utilities for quality/cost/latency trade-offs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .types import ModelCandidate, QueryFeatures


def dominates(
    left: ModelCandidate,
    right: ModelCandidate,
    features: QueryFeatures,
    estimated_costs: Mapping[str, float],
) -> bool:
    """Return whether ``left`` is no worse everywhere and better somewhere."""

    left_values = (
        left.quality_for(features.task),
        -estimated_costs[left.model_id],
        -left.latency_ms_p95,
    )
    right_values = (
        right.quality_for(features.task),
        -estimated_costs[right.model_id],
        -right.latency_ms_p95,
    )
    return all(a >= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a > b for a, b in zip(left_values, right_values, strict=True)
    )


def pareto_front(
    candidates: Iterable[ModelCandidate],
    features: QueryFeatures,
    estimated_costs: Mapping[str, float],
) -> tuple[ModelCandidate, ...]:
    """Return non-dominated candidates in stable model-id order."""

    options = tuple(candidates)
    front = [
        candidate
        for candidate in options
        if not any(
            other.model_id != candidate.model_id
            and dominates(other, candidate, features, estimated_costs)
            for other in options
        )
    ]
    return tuple(sorted(front, key=lambda item: item.model_id))
