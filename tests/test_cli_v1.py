from __future__ import annotations

import json
from pathlib import Path

from facetroute.cli import _is_loopback, main
from facetroute.traces import RouteTrace, TraceOutcome
from facetroute.types import RouteRequest

PROJECT_ROOT = Path(__file__).parents[1]
EXAMPLES = PROJECT_ROOT / "examples"


def _write_traces(path: Path) -> None:
    records = []
    for index, score in enumerate((0.2, 0.7, 0.9)):
        records.append(
            RouteTrace(
                RouteRequest(
                    f"Explain benchmark request {index}",
                    user_id="default",
                    request_id=f"trace-{index}",
                ),
                {
                    "local-sparrow": TraceOutcome(0.55, 0.0, 100, True),
                    "cobalt-chat": TraceOutcome(0.76, 0.001, 300, True),
                    "marble-reasoner": TraceOutcome(0.92, 0.006, 900, True),
                    "lotus-polyglot": TraceOutcome(0.68, 0.0008, 250, True),
                },
                preferred_model="marble-reasoner" if score > 0.5 else "local-sparrow",
                route_score=score,
                strong_model="marble-reasoner",
                weak_model="local-sparrow",
            ).to_dict()
        )
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def test_cli_calibrate_writes_json_and_csv(tmp_path, capsys):
    traces = tmp_path / "traces.jsonl"
    output = tmp_path / "calibration.json"
    csv = tmp_path / "calibration.csv"
    _write_traces(traces)

    code = main(
        [
            "calibrate",
            "--traces",
            str(traces),
            "--max-average-cost",
            "0.0045",
            "--output",
            str(output),
            "--csv",
            str(csv),
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert printed == json.loads(output.read_text(encoding="utf-8"))
    assert csv.read_text(encoding="utf-8").startswith("threshold,")


def test_cli_benchmark_runs_all_policies_and_artifacts(tmp_path, capsys):
    traces = tmp_path / "traces.jsonl"
    output = tmp_path / "report"
    _write_traces(traces)

    code = main(
        [
            "benchmark",
            "--models",
            str(EXAMPLES / "models.json"),
            "--preferences",
            str(EXAMPLES / "preferences.json"),
            "--rules",
            str(EXAMPLES / "rules.json"),
            "--traces",
            str(traces),
            "--bootstrap-samples",
            "100",
            "--output-dir",
            str(output),
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert set(printed["policies"]) >= {
        "rule",
        "pareto",
        "linucb-online",
        "fixed:local-sparrow",
    }
    assert (output / "benchmark.json").exists()
    assert (output / "benchmark.csv").exists()
    assert (output / "benchmark.html").exists()


def test_cli_benchmark_allows_explicit_policy_and_rejects_unknown_fixed(tmp_path, capsys):
    traces = tmp_path / "traces.jsonl"
    _write_traces(traces)
    code = main(
        [
            "benchmark",
            "--models",
            str(EXAMPLES / "models.json"),
            "--traces",
            str(traces),
            "--policy",
            "fixed",
            "--fixed-model",
            "missing",
            "--bootstrap-samples",
            "100",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 2
    assert "unknown --fixed-model" in capsys.readouterr().err


def test_serve_cli_uses_bounded_server_without_printing_token(tmp_path, monkeypatch, capsys):
    class FakeServer:
        server_address = ("127.0.0.1", 8123)
        served = False
        closed = False

        def serve_forever(self):
            self.served = True

        def server_close(self):
            self.closed = True

    fake = FakeServer()
    captured: dict[str, object] = {}

    def fake_create(_router, _models, **kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr("facetroute.cli.create_server", fake_create)
    monkeypatch.setenv("PRIVATE_ROUTE_TOKEN", "do-not-print-me")
    code = main(
        [
            "serve",
            "--models",
            str(EXAMPLES / "models.json"),
            "--port",
            "8123",
            "--token-env",
            "PRIVATE_ROUTE_TOKEN",
        ]
    )
    stderr = capsys.readouterr().err

    assert code == 0
    assert fake.served and fake.closed
    assert captured["bearer_token"] == "do-not-print-me"
    assert "do-not-print-me" not in stderr


def test_serve_cli_guards_nonloopback_binding(capsys):
    code = main(
        [
            "serve",
            "--models",
            str(EXAMPLES / "models.json"),
            "--host",
            "0.0.0.0",
            "--token-env",
            "MISSING_FACETROUTE_TOKEN",
        ]
    )

    assert code == 2
    assert "non-loopback" in capsys.readouterr().err
    assert _is_loopback("localhost")
    assert _is_loopback("::1")
    assert not _is_loopback("router.example")
