"""Independent pairwise-factorization arithmetic, integrity, and routing gates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace

import pytest

import facetroute.factorization as factorization_module
from facetroute import RouteRequest
from facetroute.cli import main
from facetroute.errors import ConfigurationError, NoEligibleModelError, PersistenceError
from facetroute.factorization import (
    FactorizationConfig,
    FactorizationRouter,
    PairwiseFactorModel,
    evaluate_held_out,
)
from facetroute.similarity import SimilarityFeatureConfig, SimilarityModel
from facetroute.traces import RouteTrace, TraceOutcome, write_traces


def _trace(request_id: str, query: str, winner: str, *, user_id: str | None = None) -> RouteTrace:
    outcomes = {
        "alpha": TraceOutcome(0.8, 0.001, 100, True),
        "beta": TraceOutcome(0.8, 0.001, 100, True),
    }
    if winner not in outcomes:
        outcomes[winner] = TraceOutcome(0.8, 0.001, 100, True)
    return RouteTrace(
        RouteRequest(query, request_id=request_id, user_id=user_id or request_id),
        outcomes,
        preferred_model=winner,
    )


def _training() -> tuple[RouteTrace, ...]:
    return (
        _trace("a1", "python debugging test", "alpha"),
        _trace("a2", "python code function", "alpha"),
        _trace("b1", "algebra theorem proof", "beta"),
        _trace("b2", "math algebra equation", "beta"),
    )


def _model() -> PairwiseFactorModel:
    return PairwiseFactorModel.fit(
        _training(),
        config=FactorizationConfig(dimension=4, epochs=90, learning_rate=0.15),
    )


def _resign(payload: dict[str, object]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "state_sha256"}
    payload["state_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def test_pairwise_logistic_extremes_remain_finite() -> None:
    assert factorization_module._logistic_loss(-1000.0) == pytest.approx(1000.0)
    assert factorization_module._logistic_loss(1000.0) == pytest.approx(0.0)
    assert factorization_module._logistic_derivative(-1000.0) == pytest.approx(1.0)
    assert factorization_module._logistic_derivative(1000.0) == pytest.approx(0.0)


def test_factorization_matches_independent_bilinear_oracle() -> None:
    config = FactorizationConfig(dimension=2)
    encoder = SimilarityModel.fit(
        _training(),
        config=SimilarityFeatureConfig(
            max_features=config.max_features,
            max_records=config.max_records,
            max_prototype_values=250_000,
        ),
    )
    names = encoder.feature_names
    projection = (
        tuple(1.0 if name == "term:python" else 0.0 for name in names),
        tuple(1.0 if name == "term:algebra" else 0.0 for name in names),
    )
    model = PairwiseFactorModel(
        config,
        encoder,
        ("alpha", "beta"),
        ((2.0, -0.5), (-1.0, 3.0)),
        projection,
        4,
        0.5,
    )
    request = RouteRequest("python debugging", request_id="q")
    vector = encoder.vectorize(request)
    expected_alpha = (
        2.0 * vector[names.index("term:python")] - 0.5 * vector[names.index("term:algebra")]
    )
    expected_beta = -vector[names.index("term:python")] + 3.0 * vector[names.index("term:algebra")]
    scores = dict(model.scores(request))
    assert scores["alpha"] == pytest.approx(expected_alpha, abs=1e-15)
    assert scores["beta"] == pytest.approx(expected_beta, abs=1e-15)


def test_factorization_fit_is_order_invariant_and_generalizes_on_held_out() -> None:
    training = _training()
    config = FactorizationConfig(dimension=4, epochs=90, learning_rate=0.15)
    model = PairwiseFactorModel.fit(training, config=config)
    reordered = PairwiseFactorModel.fit(tuple(reversed(training)), config=config)
    assert model.to_dict() == reordered.to_dict()
    held_out = (
        _trace("a3", "debug python function", "alpha"),
        _trace("b3", "prove math theorem", "beta"),
    )
    report = evaluate_held_out(model, training, held_out)
    assert report["top1_accuracy"] == 1.0
    assert report["pairwise_accuracy"] == 1.0
    assert isinstance(report["pairwise_log_loss"], float)
    assert report["training_sha256"] == model.encoder.training_sha256
    assert model.to_dict() == reordered.to_dict()


def test_factorization_one_epoch_matches_independent_sgd_oracle() -> None:
    traces = (
        _trace("a", "python debugging", "alpha"),
        _trace("b", "algebra proof", "beta"),
    )
    config = FactorizationConfig(
        dimension=2, epochs=1, learning_rate=0.1, regularization=0.01, seed=5
    )
    fitted = PairwiseFactorModel.fit(traces, config=config)
    features = fitted.encoder.feature_names

    def initial(label: str) -> float:
        digest = hashlib.sha256(f"5\0{label}".encode()).digest()
        return (2 * int.from_bytes(digest[:8], "big") / 2**64 - 1) * 0.05

    projection = [
        [initial(f"projection:{axis}:{feature}") for feature in features] for axis in range(2)
    ]
    rows = [[initial(f"route:{route}:{axis}") for axis in range(2)] for route in fitted.route_ids]
    for trace in traces:
        winner = fitted.route_ids.index(trace.preferred_model)
        loser = 1 - winner
        query = fitted.encoder.vectorize(trace.request)
        embedding = [
            math.fsum(p * x for p, x in zip(row, query, strict=True)) for row in projection
        ]
        difference = [rows[winner][axis] - rows[loser][axis] for axis in range(2)]
        margin = math.fsum(d * e for d, e in zip(difference, embedding, strict=True))
        gradient = 1 / (1 + math.exp(margin))
        for axis in range(2):
            old_winner = rows[winner][axis]
            old_loser = rows[loser][axis]
            rows[winner][axis] = old_winner + 0.1 * (gradient * embedding[axis] - 0.01 * old_winner)
            rows[loser][axis] = old_loser + 0.1 * (-gradient * embedding[axis] - 0.01 * old_loser)
            for index, value in enumerate(query):
                if value == 0.0:
                    continue
                old = projection[axis][index]
                projection[axis][index] = old + 0.1 * (
                    gradient * difference[axis] * value - 0.01 * old
                )
    for actual, expected in zip(fitted.route_factors, rows, strict=True):
        assert actual == pytest.approx(expected, abs=1e-15)
    for actual, expected in zip(fitted.projection, projection, strict=True):
        assert actual == pytest.approx(expected, abs=1e-15)


def test_factorization_held_out_rejects_group_leakage_and_wrong_training() -> None:
    training = _training()
    model = _model()
    with pytest.raises(ConfigurationError, match="overlap"):
        evaluate_held_out(
            model,
            training,
            (_trace("other", "new math", "beta", user_id="a1"),),
            group_by="user_id",
        )
    with pytest.raises(ConfigurationError, match="do not match"):
        evaluate_held_out(model, training[:-1], (_trace("b3", "math", "beta"),))


def test_factorization_held_out_rejects_unseen_route_and_missing_label() -> None:
    training = _training()
    model = _model()
    unseen_outcome = RouteTrace(
        RouteRequest("math", request_id="new"),
        {
            "beta": TraceOutcome(0.8, 0.001, 100, True),
            "gamma": TraceOutcome(0.8, 0.001, 100, True),
        },
        preferred_model="beta",
    )
    with pytest.raises(ConfigurationError, match="unseen factorization routes"):
        evaluate_held_out(model, training, (unseen_outcome,))
    missing_label = RouteTrace(
        RouteRequest("math", request_id="unlabelled"),
        {"beta": TraceOutcome(0.8, 0.001, 100, True)},
    )
    with pytest.raises(ConfigurationError, match="preferred_model"):
        evaluate_held_out(model, training, (missing_label,))


def test_factorization_training_rejects_pair_and_route_exhaustion_early() -> None:
    with pytest.raises(ConfigurationError, match="pair limit"):
        PairwiseFactorModel.fit(_training(), config=FactorizationConfig(max_pairs=1))
    many_outcomes = {
        name: TraceOutcome(0.8, 0.001, 100, True) for name in ("a", "b", "c", "d", "e")
    }
    one_trace = RouteTrace(
        RouteRequest("python math", request_id="one"),
        many_outcomes,
        preferred_model="a",
    )
    with pytest.raises(ConfigurationError, match="route limit"):
        PairwiseFactorModel.fit((one_trace,), config=FactorizationConfig(max_records=4))


def test_factorization_training_rejects_degenerate_data_and_work_budget(monkeypatch) -> None:
    with pytest.raises(ConfigurationError, match="at least two observed routes"):
        PairwiseFactorModel.fit(
            (
                RouteTrace(
                    RouteRequest("python", request_id="one"),
                    {"alpha": TraceOutcome(0.8, 0.001, 100, True)},
                    preferred_model="alpha",
                ),
            )
        )
    without_comparison = (
        RouteTrace(
            RouteRequest("python", request_id="a"),
            {"alpha": TraceOutcome(0.8, 0.001, 100, True)},
            preferred_model="alpha",
        ),
        RouteTrace(
            RouteRequest("algebra", request_id="b"),
            {"beta": TraceOutcome(0.8, 0.001, 100, True)},
            preferred_model="beta",
        ),
    )
    with pytest.raises(ConfigurationError, match="labelled comparison"):
        PairwiseFactorModel.fit(without_comparison)
    with pytest.raises(ConfigurationError, match="FactorizationConfig"):
        PairwiseFactorModel.fit(_training(), config=object())
    monkeypatch.setattr(factorization_module, "_MAX_TRAINING_UPDATES", 1)
    with pytest.raises(ConfigurationError, match="work limit"):
        PairwiseFactorModel.fit(_training())


def test_factorization_state_round_trip_and_rejects_tampering(tmp_path) -> None:
    model = _model()
    path = tmp_path / "factorization.json"
    model.save(path)
    assert PairwiseFactorModel.load(path).to_dict() == model.to_dict()
    payload = model.to_dict()
    payload["route_factors"][0][0] += 1.0
    with pytest.raises(PersistenceError, match="integrity"):
        PairwiseFactorModel.from_dict(payload)


def test_factorization_rejects_invalid_resource_limits_and_nan() -> None:
    with pytest.raises(ConfigurationError, match="dimension"):
        FactorizationConfig(dimension=True)
    with pytest.raises(ConfigurationError, match="factor limit"):
        FactorizationConfig(dimension=64, max_features=8192)
    with pytest.raises(ConfigurationError, match="learning_rate"):
        FactorizationConfig(learning_rate=math.inf)
    with pytest.raises(ConfigurationError, match="route factor"):
        replace(_model(), route_factors=((math.nan,) * 4, (0.0,) * 4))


def test_factorization_rejects_encoder_labels_outside_factor_routes() -> None:
    model = _model()
    other_encoder = SimilarityModel.fit(
        (
            _trace("g", "python debugging", "gamma"),
            _trace("d", "algebra theorem", "delta"),
        ),
        config=model.encoder.config,
    )
    with pytest.raises(ConfigurationError, match="encoder references routes"):
        replace(model, encoder=other_encoder)
    payload = model.to_dict()
    payload["encoder"] = other_encoder.to_dict()
    _resign(payload)
    with pytest.raises(PersistenceError, match="encoder references routes"):
        PairwiseFactorModel.from_dict(payload)


def test_factorization_rejects_encoder_configuration_drift() -> None:
    model = _model()
    altered_config = replace(model.config, max_features=16, max_records=100)
    with pytest.raises(ConfigurationError, match="encoder configuration"):
        replace(model, config=altered_config)
    payload = model.to_dict()
    payload["config"] = altered_config.to_dict()
    _resign(payload)
    with pytest.raises(PersistenceError, match="encoder configuration"):
        PairwiseFactorModel.from_dict(payload)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"config": None}, "config"),
        ({"encoder": None}, "encoder"),
        ({"route_ids": []}, "tuple"),
        ({"route_ids": ("alpha",), "route_factors": ((0.0,) * 4,)}, "sorted"),
        ({"route_ids": ("beta", "alpha")}, "sorted"),
        ({"route_ids": ("alpha", "alpha")}, "sorted"),
        ({"route_ids": (" alpha", "beta")}, "sorted"),
        ({"route_factors": ()}, "dimensions"),
        ({"route_factors": ((0.0,), (0.0,))}, "dimensions"),
        ({"projection": ()}, "dimensions"),
        ({"projection": ((0.0,),) * 4}, "dimensions"),
        ({"training_pairs": 0}, "training_pairs"),
        ({"final_loss": math.inf}, "final_loss"),
    ],
)
def test_factorization_constructor_fails_closed_on_shape_and_settings(change, message) -> None:
    with pytest.raises(ConfigurationError, match=message):
        replace(_model(), **change)


def test_factorization_config_rejects_unknown_fields() -> None:
    with pytest.raises(ConfigurationError, match="fields"):
        FactorizationConfig.from_dict({"dimension": 2})


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("format", "other", "schema"),
        ("schema_version", True, "schema"),
        ("route_ids", ["beta", "alpha"], "sorted"),
        ("route_ids", ["alpha"], "sorted"),
        ("route_factors", [], "dimensions"),
        ("projection", [], "dimensions"),
        ("training_pairs", 0, "training_pairs"),
        ("config", {"dimension": 2}, "config fields"),
    ],
)
def test_factorization_resigned_invalid_state_is_rejected(key, value, message) -> None:
    payload = _model().to_dict()
    payload[key] = value
    _resign(payload)
    with pytest.raises(PersistenceError, match=message):
        PairwiseFactorModel.from_dict(payload)


def test_factorization_rejects_non_object_state_and_oversize_load(tmp_path) -> None:
    with pytest.raises(PersistenceError, match="invalid fields"):
        PairwiseFactorModel.from_dict(None)
    huge = tmp_path / "huge.json"
    with huge.open("wb") as handle:
        handle.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(PersistenceError, match="byte limit"):
        PairwiseFactorModel.load(huge)
    with pytest.raises(PersistenceError, match="Cannot read"):
        PairwiseFactorModel.load(tmp_path / "missing.json")


def test_factorization_load_rejects_duplicate_keys_bad_utf8_and_nested_json(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(b'{"format":"x","format":"y"}')
    with pytest.raises(PersistenceError, match="duplicate"):
        PairwiseFactorModel.load(path)
    path.write_bytes(b"\xff")
    with pytest.raises(PersistenceError, match="decode"):
        PairwiseFactorModel.load(path)
    path.write_bytes(b"[" * 300 + b"0" + b"]" * 300)
    with pytest.raises(PersistenceError, match="decode"):
        PairwiseFactorModel.load(path)


def test_factorization_constructor_state_bound_and_unpaired_surrogate(monkeypatch) -> None:
    model = _model()
    monkeypatch.setattr(factorization_module, "_MAX_STATE_BYTES", 64)
    with pytest.raises(ConfigurationError, match="state exceeds"):
        replace(model)
    monkeypatch.undo()
    with pytest.raises(ConfigurationError, match="UTF-8"):
        replace(model, route_ids=("alpha", "\ud800"))


def test_factorization_fit_and_audit_reject_unpaired_surrogates_cleanly() -> None:
    bad_query = _trace("bad-query", "python \ud800 debugging", "alpha")
    with pytest.raises(ConfigurationError, match="strict JSON"):
        PairwiseFactorModel.fit((*_training(), bad_query))

    bad_route = RouteTrace(
        RouteRequest("python debugging", request_id="bad-route"),
        {
            "alpha": TraceOutcome(0.8, 0.001, 100, True),
            "\ud800": TraceOutcome(0.8, 0.001, 100, True),
        },
        preferred_model="alpha",
    )
    with pytest.raises(ConfigurationError, match="bounded UTF-8"):
        PairwiseFactorModel.fit((*_training(), bad_route))

    with pytest.raises(ConfigurationError, match="strict JSON"):
        evaluate_held_out(_model(), _training(), (bad_query,))


def test_factorization_held_out_reports_no_pairwise_comparisons() -> None:
    training = _training()
    only_one = RouteTrace(
        RouteRequest("algebra theorem", request_id="single"),
        {"alpha": TraceOutcome(0.8, 0.001, 100, True)},
        preferred_model="alpha",
    )
    model = _model()
    assert model.scores(only_one.request)[0][0] == "beta"
    report = evaluate_held_out(model, training, (only_one,))
    assert report["top1_accuracy"] is None
    assert report["top1_evaluable_records"] == 0
    assert report["pairwise_accuracy"] is None
    assert report["pairwise_log_loss"] is None


def test_factorization_top1_uses_only_observed_held_out_routes() -> None:
    training = (*_training(), _trace("c1", "history novel literature", "gamma"))
    model = PairwiseFactorModel.fit(training)
    query = RouteRequest("python debugging", request_id="observed-only")
    global_winner = model.scores(query)[0][0]
    observed = tuple(route for route in model.route_ids if route != global_winner)
    score_by_route = dict(model.scores(query))
    preferred = sorted(observed, key=lambda route: (-score_by_route[route], route))[0]
    held_out = RouteTrace(
        query,
        {route: TraceOutcome(0.8, 0.001, 100, True) for route in observed},
        preferred_model=preferred,
    )
    report = evaluate_held_out(model, training, (held_out,))
    assert report["top1_evaluable_records"] == 1
    assert report["top1_accuracy"] == 1.0


def test_factorization_router_filters_hard_constraints_before_logits(make_model) -> None:
    model = _model()
    alpha = make_model("alpha", enabled=False)
    beta = make_model("beta")
    router = FactorizationRouter((alpha, beta), model)
    decision = router.route(RouteRequest("python debugging", request_id="q"))
    assert decision.selected_model == "beta"
    assert "alpha" in decision.excluded
    assert decision.feature_summary["factorization"]["used"] is True
    with pytest.raises(NoEligibleModelError):
        FactorizationRouter((alpha, make_model("beta", enabled=False)), model).route(
            RouteRequest("python debugging", request_id="q2")
        )


def test_factorization_router_rejects_unknown_catalog_route(make_model) -> None:
    with pytest.raises(ConfigurationError, match="outside catalog"):
        FactorizationRouter((make_model("alpha"),), _model())


def test_factorization_router_falls_back_when_trained_routes_are_ineligible(make_model) -> None:
    router = FactorizationRouter(
        (
            make_model("alpha", enabled=False),
            make_model("beta", enabled=False),
            make_model("gamma"),
        ),
        _model(),
    )
    decision = router.route(RouteRequest("python debugging", request_id="fallback"))
    assert decision.selected_model == "gamma"
    assert decision.feature_summary["factorization"]["used"] is False
    assert decision.feature_summary["factorization"]["ranking"] == []


def test_factorization_exact_tie_uses_route_identifier_order(make_model) -> None:
    fitted = _model()
    tied = replace(
        fitted,
        route_factors=tuple((0.0,) * fitted.config.dimension for _ in fitted.route_ids),
    )
    decision = FactorizationRouter((make_model("beta"), make_model("alpha")), tied).route(
        RouteRequest("python debugging", request_id="tie")
    )
    assert decision.selected_model == "alpha"
    assert decision.score == 0.0


def test_factorization_cli_train_route_and_benchmark(tmp_path, capsys, make_model) -> None:
    models = tmp_path / "models.json"
    models.write_text(
        json.dumps(
            {
                "models": [
                    make_model("alpha").to_dict(),
                    make_model("beta").to_dict(),
                ]
            }
        ),
        encoding="utf-8",
    )
    training_path = tmp_path / "training.jsonl"
    test_path = tmp_path / "test.jsonl"
    training = _training()
    write_traces(training_path, training)
    write_traces(
        test_path,
        (
            _trace("a3", "debug python function", "alpha"),
            _trace("b3", "prove math theorem", "beta"),
        ),
    )
    state_path = tmp_path / "factor.json"
    report_path = tmp_path / "factor-report.json"
    assert (
        main(
            [
                "train-factorization",
                "--train-traces",
                str(training_path),
                "--held-out-traces",
                str(test_path),
                "--dimension",
                "4",
                "--epochs",
                "90",
                "--learning-rate",
                "0.15",
                "--output",
                str(state_path),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(report_path.read_text(encoding="utf-8"))
    assert report["held_out"]["top1_accuracy"] == 1.0
    assert (
        main(
            [
                "route",
                "--models",
                str(models),
                "--policy",
                "factorization",
                "--factor-model",
                str(state_path),
                "--query",
                "debug python function",
            ]
        )
        == 0
    )
    decision = json.loads(capsys.readouterr().out)
    assert decision["policy"] == "factorization"
    assert decision["selected_model"] == "alpha"
    benchmark_dir = tmp_path / "benchmark"
    assert (
        main(
            [
                "benchmark",
                "--models",
                str(models),
                "--traces",
                str(test_path),
                "--policy",
                "factorization",
                "--factor-model",
                str(state_path),
                "--bootstrap-samples",
                "100",
                "--output-dir",
                str(benchmark_dir),
            ]
        )
        == 0
    )
    benchmark = json.loads(capsys.readouterr().out)
    assert benchmark["policies"]["factorization"]["routed"] == 2
    assert "factor_model" in benchmark["manifest"]["input_sha256"]


def test_factorization_cli_rejects_training_output_collision(tmp_path, capsys) -> None:
    path = tmp_path / "training.jsonl"
    write_traces(path, _training())
    before = path.read_bytes()
    assert main(["train-factorization", "--train-traces", str(path), "--output", str(path)]) == 2
    assert "paths collide" in capsys.readouterr().err
    assert path.read_bytes() == before
