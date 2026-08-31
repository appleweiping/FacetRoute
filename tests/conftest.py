from __future__ import annotations

from collections.abc import Callable

import pytest

from facetroute import ModelCandidate, UserPreferences


@pytest.fixture
def make_model() -> Callable[..., ModelCandidate]:
    def factory(model_id: str = "balanced", **overrides: object) -> ModelCandidate:
        values: dict[str, object] = {
            "model_id": model_id,
            "display_name": model_id.title(),
            "capabilities": frozenset({"text", "code", "math", "reasoning", "json"}),
            "input_cost_per_million": 1.0,
            "output_cost_per_million": 2.0,
            "latency_ms_p50": 100.0,
            "latency_ms_p95": 200.0,
            "context_window": 16_384,
            "quality_by_task": {"default": 0.7, "code": 0.75, "math": 0.72},
            "regions": frozenset({"us", "eu"}),
            "supports_tools": False,
            "supports_json": True,
            "enabled": True,
            "metadata": {},
        }
        values.update(overrides)
        return ModelCandidate(**values)  # type: ignore[arg-type]

    return factory


@pytest.fixture
def three_models(make_model: Callable[..., ModelCandidate]) -> tuple[ModelCandidate, ...]:
    return (
        make_model(
            "cheap",
            input_cost_per_million=0.1,
            output_cost_per_million=0.2,
            latency_ms_p50=60,
            latency_ms_p95=100,
            quality_by_task={"default": 0.55, "code": 0.6, "math": 0.5},
            metadata={"local": True},
            regions=frozenset({"local", "us"}),
        ),
        make_model("balanced"),
        make_model(
            "quality",
            input_cost_per_million=3.0,
            output_cost_per_million=8.0,
            latency_ms_p50=400,
            latency_ms_p95=900,
            context_window=65_536,
            quality_by_task={"default": 0.92, "code": 0.95, "math": 0.97},
            supports_tools=True,
        ),
    )


@pytest.fixture
def default_profile() -> UserPreferences:
    return UserPreferences(user_id="u", quality_weight=0.6, cost_weight=0.25, latency_weight=0.15)
