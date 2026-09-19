"""Adversarial factorization boundaries independent of the happy-path oracle."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import replace

import pytest

import facetroute.factorization as factorization
from facetroute import RouteRequest
from facetroute.cli import main
from facetroute.errors import ConfigurationError, PersistenceError
from facetroute.factorization import (
    FactorizationConfig,
    FactorizationRouter,
    PairwiseFactorModel,
    evaluate_held_out,
)
from facetroute.traces import RouteTrace, TraceOutcome, write_traces


def _trace(request_id: str, winner: str, *, user_id: str | None = None) -> RouteTrace:
    return RouteTrace(
        RouteRequest(
            "debug python code" if winner == "alpha" else "prove algebra theorem",
            request_id=request_id,
            user_id=user_id or request_id,
        ),
        {
            "alpha": TraceOutcome(0.8, 0.001, 100, True),
            "beta": TraceOutcome(0.8, 0.001, 100, True),
        },
        preferred_model=winner,
    )


def _training() -> tuple[RouteTrace, ...]:
    return (_trace("train-a", "alpha"), _trace("train-b", "beta"))


def _model() -> PairwiseFactorModel:
    return PairwiseFactorModel.fit(
        _training(), config=FactorizationConfig(dimension=2, epochs=3, learning_rate=0.1)
    )


def _resign(state: dict[str, object]) -> None:
    unsigned = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def test_fit_and_held_out_report_cover_both_observed_top1_outcomes() -> None:
    training = _training()
    model = _model()
    query = RouteRequest("debug python code", request_id="held", user_id="new-user")
    ranking = model.scores(query)
    assert {route for route, _score in ranking} == {"alpha", "beta"}
    observed = {route: TraceOutcome(0.8, 0.001, 100, True) for route, _score in ranking}
    first = RouteTrace(query, observed, preferred_model=ranking[0][0])
    second = RouteTrace(query, observed, preferred_model=ranking[1][0])
    before = model.to_dict()
    assert evaluate_held_out(model, training, (first,))["top1_accuracy"] == 1.0
    assert evaluate_held_out(model, training, (second,))["top1_accuracy"] == 0.0
    assert model.to_dict() == before


def test_audit_rejects_cross_partition_request_id_and_unseen_preferred_route() -> None:
    model = _model()
    training = _training()
    reused_id = _trace("train-a", "alpha", user_id="held-user")
    with pytest.raises(ConfigurationError, match="request ids overlap"):
        evaluate_held_out(model, training, (reused_id,), group_by="user_id")

    unseen = RouteTrace(
        RouteRequest("novel topic", request_id="held-unknown", user_id="held-user"),
        {
            "alpha": TraceOutcome(0.8, 0.001, 100, True),
            "gamma": TraceOutcome(0.8, 0.001, 100, True),
        },
        preferred_model="gamma",
    )
    with pytest.raises(ConfigurationError, match="label was unseen"):
        evaluate_held_out(model, training, (unseen,))
    with pytest.raises(ConfigurationError, match="PairwiseFactorModel"):
        evaluate_held_out(object(), training, (unseen,))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="PairwiseFactorModel"):
        FactorizationRouter((), object())  # type: ignore[arg-type]


def test_fit_rejects_malformed_routes_and_preflight_pair_budget() -> None:
    one = RouteTrace(
        RouteRequest("debug", request_id="only"),
        {"alpha": TraceOutcome(0.8, 0.001, 100, True)},
        preferred_model="alpha",
    )
    with pytest.raises(ConfigurationError, match="at least two observed routes"):
        PairwiseFactorModel.fit((one,))

    bad_id = RouteTrace(
        RouteRequest("debug", request_id="bad"),
        {
            "alpha": TraceOutcome(0.8, 0.001, 100, True),
            "b" * 513: TraceOutcome(0.8, 0.001, 100, True),
        },
        preferred_model="alpha",
    )
    with pytest.raises(ConfigurationError, match="bounded UTF-8"):
        PairwiseFactorModel.fit((bad_id,))
    with pytest.raises(ConfigurationError, match="pair limit"):
        PairwiseFactorModel.fit(_training(), config=FactorizationConfig(max_pairs=1))


def test_optimizer_fails_closed_on_nonfinite_initial_or_updated_factors(monkeypatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(factorization, "_weight", lambda _seed, _label: math.inf)
        with pytest.raises(ConfigurationError, match="non-finite score"):
            PairwiseFactorModel.fit(_training(), config=FactorizationConfig(epochs=1))
    with monkeypatch.context() as patch:
        patch.setattr(factorization, "_logistic_derivative", lambda _delta: math.inf)
        with pytest.raises(ConfigurationError, match="non-finite factors"):
            PairwiseFactorModel.fit((_trace("one", "alpha"),), config=FactorizationConfig(epochs=1))


def test_constructor_and_resigned_state_reject_shape_and_schema_confusion() -> None:
    model = _model()
    with pytest.raises(ConfigurationError, match="matrices must be tuples"):
        replace(model, projection=list(model.projection))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="matrices must be tuples"):
        replace(model, route_factors=list(model.route_factors))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="config fields"):
        FactorizationConfig.from_dict({"epochs": 3})

    malformed = model.to_dict()
    malformed["unexpected"] = 1
    with pytest.raises(PersistenceError, match="invalid fields"):
        PairwiseFactorModel.from_dict(malformed)
    tampered = model.to_dict()
    tampered["training_pairs"] = model.training_pairs + 1
    with pytest.raises(PersistenceError, match="integrity"):
        PairwiseFactorModel.from_dict(tampered)
    wrong_schema = model.to_dict()
    wrong_schema["schema_version"] = 2
    _resign(wrong_schema)
    with pytest.raises(PersistenceError, match="unsupported"):
        PairwiseFactorModel.from_dict(wrong_schema)


def test_load_and_cli_reject_corrupt_state_and_aliases_without_writing(tmp_path, capsys) -> None:
    state = tmp_path / "state.json"
    state.write_bytes(b'{"format":"a","format":"b"}')
    with pytest.raises(PersistenceError, match="duplicate"):
        PairwiseFactorModel.load(state)
    state.write_bytes(b"\xff")
    with pytest.raises(PersistenceError, match="decode"):
        PairwiseFactorModel.load(state)

    traces = tmp_path / "training.jsonl"
    write_traces(traces, _training())
    output = tmp_path / "output.json"
    os.link(traces, output)
    before = traces.read_bytes()
    assert (
        main(["train-factorization", "--train-traces", str(traces), "--output", str(output)]) == 2
    )
    assert "paths collide" in capsys.readouterr().err
    assert traces.read_bytes() == before
    other_output = tmp_path / "other.json"
    assert (
        main(
            [
                "train-factorization",
                "--train-traces",
                str(traces),
                "--dimension",
                "0",
                "--output",
                str(other_output),
            ]
        )
        == 2
    )
    assert "dimension" in capsys.readouterr().err
    assert not other_output.exists()
