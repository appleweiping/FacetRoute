from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

from facetroute.async_client import AsyncProviderTarget
from facetroute.errors import ConfigurationError
from facetroute.public_score_audit import audit_json, audit_public_scores
from facetroute.public_score_audit_cli import main
from facetroute.public_scores import (
    PublicScoreLimits,
    PublicScoreModel,
    PublicScoreProvenance,
    run_public_scores,
)


class Responses:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    async def complete(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> Mapping[str, Any]:
        self.calls += 1
        return {
            "id": "fixture-completion",
            "object": "chat.completion",
            "model": model,
            "choices": [{"message": {"content": self.responses.pop(0)}}],
        }

    def stream(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> AsyncIterator[Mapping[str, Any]]:
        raise AssertionError("audit fixture never streams")


def _fixture(
    tmp_path: Path, *, complete: bool = True
) -> tuple[Path, Path, Path, Responses, Responses]:
    source = tmp_path / "questions.jsonl"
    source.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "id": "a",
                    "question": "Q alpha",
                    "choices": ["no", "yes"],
                    "answer": 1,
                    "subject": "logic",
                },
                {
                    "id": "b",
                    "question": "Q beta",
                    "choices": ["no", "yes"],
                    "answer": 0,
                    "subject": "logic",
                },
                {
                    "id": "c",
                    "question": "Q gamma",
                    "choices": ["no", "yes"],
                    "answer": 0,
                    "subject": "math",
                },
            )
        ),
        encoding="utf-8",
    )
    raw = source.read_bytes()
    checkpoint = tmp_path / "checkpoint.json"
    weak_provider = Responses(["A", "A", "A"])  # false, true, true
    strong_provider = Responses(["B", "B", "B"])  # true, false, false
    weak = PublicScoreModel(AsyncProviderTarget("weak", "w", weak_provider), "revision-a")
    strong = PublicScoreModel(AsyncProviderTarget("strong", "s", strong_provider), "revision-b")
    asyncio.run(
        run_public_scores(
            source,
            checkpoint,
            provenance=PublicScoreProvenance(
                "fixture:public-score-audit", "MIT", hashlib.sha256(raw).hexdigest()
            ),
            weak=weak,
            strong=strong,
            max_calls=6 if complete else 1,
        )
    )
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        '{"id":"a","strong_win_rate":0.9}\n'
        '{"id":"b","strong_win_rate":0.4}\n'
        '{"id":"c","strong_win_rate":0.1}\n',
        encoding="utf-8",
    )
    return source, checkpoint, scores, weak_provider, strong_provider


def test_hand_oracle_curve_and_aggregate_privacy(tmp_path: Path) -> None:
    source, checkpoint, scores, weak, strong = _fixture(tmp_path)
    calls = (weak.calls, strong.calls)
    report = audit_public_scores(source, checkpoint, scores)
    assert (weak.calls, strong.calls) == calls  # no provider execution in read-only analysis
    assert report.baselines == {"weak_accuracy": 2 / 3, "strong_accuracy": 1 / 3}
    assert report.manifest["source_records"] == report.manifest["evaluated_records"] == 3
    assert report.manifest["official_benchmark_parity"] is False
    assert (
        report.manifest["route_scores_file_sha256"]
        == hashlib.sha256(scores.read_bytes()).hexdigest()
    )
    assert report.domains == {
        "logic": {"records": 2, "weak_accuracy": 0.5, "strong_accuracy": 0.5},
        "math": {"records": 1, "weak_accuracy": 1.0, "strong_accuracy": 0.0},
    }
    points = report.points
    assert [point["threshold"] for point in points] == [None, 0.1, 0.4, 0.9]
    assert [point["strong_calls"] for point in points] == [0, 3, 2, 1]
    assert [point["accuracy"] for point in points] == [2 / 3, 1 / 3, 2 / 3, 1.0]
    assert [point["oracle_accuracy_at_same_calls"] for point in points] == [
        2 / 3,
        1 / 3,
        2 / 3,
        1.0,
    ]
    assert [point["regret_to_oracle"] for point in points] == [0.0, 0.0, 0.0, 0.0]
    weak_outcomes = (0, 1, 1)
    strong_outcomes = (1, 0, 0)
    for point in points:
        count = point["strong_calls"]
        assert point["oracle_accuracy_at_same_calls"] == max(
            sum(
                strong_outcomes[index] if index in chosen else weak_outcomes[index]
                for index in range(3)
            )
            / 3
            for chosen in combinations(range(3), count)
        )
    payload = audit_json(report)
    assert payload == audit_json(audit_public_scores(source, checkpoint, scores))
    assert b"Q alpha" not in payload and b'"response"' not in payload
    assert b'"answer"' not in payload and b"revision-a" in payload

    scores.write_text(
        '{"id":"a","strong_win_rate":0.4}\n'
        '{"id":"b","strong_win_rate":0.9}\n'
        '{"id":"c","strong_win_rate":0.1}\n',
        encoding="utf-8",
    )
    misranked = audit_public_scores(source, checkpoint, scores)
    assert misranked.points[-1]["accuracy"] == 1 / 3
    assert misranked.points[-1]["oracle_accuracy_at_same_calls"] == 1.0
    assert misranked.points[-1]["regret_to_oracle"] == 2 / 3


def test_exclusion_is_pinned_by_normalized_prompt_digest(tmp_path: Path) -> None:
    source, checkpoint, scores, _, _ = _fixture(tmp_path)
    exclusions = tmp_path / "exclude.jsonl"
    exclusions.write_text(
        json.dumps({"prompt_sha256": hashlib.sha256(b"Q gamma").hexdigest()}) + "\n",
        encoding="utf-8",
    )
    report = audit_public_scores(source, checkpoint, scores, exclusions_path=exclusions)
    assert report.manifest["excluded_records"] == 1
    assert report.manifest["evaluated_records"] == 2
    assert (
        report.manifest["exclusions_file_sha256"]
        == hashlib.sha256(exclusions.read_bytes()).hexdigest()
    )
    assert report.baselines == {"weak_accuracy": 0.5, "strong_accuracy": 0.5}
    assert [point["accuracy"] for point in report.points] == [0.5, 0.5, 1.0]
    exclusions.write_text('{"prompt_sha256":"' + "0" * 64 + '"}\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="does not match"):
        audit_public_scores(source, checkpoint, scores, exclusions_path=exclusions)


def test_incomplete_and_corrupt_checkpoints_fail_closed(tmp_path: Path) -> None:
    source, checkpoint, scores, _, _ = _fixture(tmp_path, complete=False)
    with pytest.raises(ConfigurationError, match="completed"):
        audit_public_scores(source, checkpoint, scores)
    source, checkpoint, scores, _, _ = _fixture(tmp_path)
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ConfigurationError, match="SHA-256"):
        audit_public_scores(source, checkpoint, scores)
    source, checkpoint, scores, _, _ = _fixture(tmp_path)
    checkpoint.write_bytes(checkpoint.read_bytes().replace(b'"response":"A"', b'"response":"B"', 1))
    with pytest.raises(ConfigurationError, match="checksum"):
        audit_public_scores(source, checkpoint, scores)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('{"id":"a","strong_win_rate":0.5}\n', "must match"),
        ('{"id":"a","strong_win_rate":0.5}\n{"id":"a","strong_win_rate":0.6}\n', "unique"),
        ('{"id":"a","strong_win_rate":NaN}\n', "invalid strict JSON"),
        ('{"id":"a","strong_win_rate":true}\n', "finite"),
        ('{"id":"a","strong_win_rate":1.1}\n', "finite"),
        ('{"id":"a","strong_win_rate":' + "9" * 400 + "}\n", "finite"),
        ('{"id":"a","strong_win_rate":0.4,"id":"b"}\n', "invalid strict JSON"),
        ('{"id":"a","strong_win_rate":0.5,"extra":0}\n', "require"),
        ('{"id":"a","strong_win_rate":0.5}', "newline-terminated"),
    ],
)
def test_route_score_fail_closed(tmp_path: Path, body: str, message: str) -> None:
    source, checkpoint, scores, _, _ = _fixture(tmp_path)
    scores.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        audit_public_scores(source, checkpoint, scores)


def test_same_path_and_exclusion_validation(tmp_path: Path) -> None:
    source, checkpoint, scores, _, _ = _fixture(tmp_path)
    with pytest.raises(ConfigurationError, match="paths must differ"):
        audit_public_scores(source, checkpoint, source)
    exclusions = tmp_path / "exclude.jsonl"
    row = '{"prompt_sha256":"' + hashlib.sha256(b"Q alpha").hexdigest() + '"}\n'
    exclusions.write_text(row * 2)
    with pytest.raises(ConfigurationError):
        audit_public_scores(source, checkpoint, scores, exclusions_path=exclusions)


def test_resealed_but_inconsistent_result_is_rejected(tmp_path: Path) -> None:
    source, checkpoint, scores, _, _ = _fixture(tmp_path)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    state["results"][0]["correct"] = True
    content = {key: value for key, value in state.items() if key != "sha256"}
    state["sha256"] = hashlib.sha256(
        json.dumps(
            content, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    checkpoint.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="inconsistent"):
        audit_public_scores(source, checkpoint, scores)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: state["manifest"].update(protocol="unsupported"),
        lambda state: state["manifest"].update(models=[]),
        lambda state: state["manifest"]["source"].update(file_sha256="0" * 64),
        lambda state: state["manifest"]["limits"].update(max_records=0),
    ],
)
def test_resealed_malformed_manifests_fail_before_reporting(tmp_path: Path, mutation: Any) -> None:
    source, checkpoint, scores, _, _ = _fixture(tmp_path)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    mutation(state)
    content = {key: value for key, value in state.items() if key != "sha256"}
    state["sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    checkpoint.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        audit_public_scores(source, checkpoint, scores)


def test_tied_scores_move_together_and_excluding_all_fails(tmp_path: Path) -> None:
    source, checkpoint, scores, _, _ = _fixture(tmp_path)
    scores.write_text(
        "".join(json.dumps({"id": id_, "strong_win_rate": 0.5}) + "\n" for id_ in ("a", "b", "c")),
        encoding="utf-8",
    )
    report = audit_public_scores(source, checkpoint, scores)
    assert [point["strong_calls"] for point in report.points] == [0, 3]
    excluded = tmp_path / "all-excluded.jsonl"
    excluded.write_text(
        "".join(
            json.dumps({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}) + "\n"
            for prompt in ("Q alpha", "Q beta", "Q gamma")
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="every source record"):
        audit_public_scores(source, checkpoint, scores, exclusions_path=excluded)


def test_gsm8k_completed_checkpoint_uses_same_verified_audit_path(tmp_path: Path) -> None:
    source = tmp_path / "math.jsonl"
    source.write_text(
        json.dumps({"id": "math-1", "question": "What is 7 plus 8?", "answer": "#### 15"}) + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "math-state.json"
    weak = PublicScoreModel(AsyncProviderTarget("weak", "w", Responses(["#### 14"])), "w1")
    strong = PublicScoreModel(AsyncProviderTarget("strong", "s", Responses(["#### 15"])), "s1")
    asyncio.run(
        run_public_scores(
            source,
            checkpoint,
            provenance=PublicScoreProvenance(
                "fixture:gsm8k-audit", "MIT", hashlib.sha256(source.read_bytes()).hexdigest()
            ),
            weak=weak,
            strong=strong,
        )
    )
    scores = tmp_path / "math-scores.jsonl"
    scores.write_text('{"id":"math-1","strong_win_rate":0.7}\n', encoding="utf-8")
    report = audit_public_scores(source, checkpoint, scores)
    assert report.manifest["format"] == "gsm8k"
    assert report.baselines == {"weak_accuracy": 0.0, "strong_accuracy": 1.0}
    assert report.domains["(unspecified)"]["records"] == 1
    assert [point["accuracy"] for point in report.points] == [0.0, 1.0]


def test_snapshot_and_strict_jsonl_resource_failures(tmp_path: Path, monkeypatch: Any) -> None:
    import facetroute.public_score_audit as audit_module

    source, checkpoint, scores, _, _ = _fixture(tmp_path)
    with pytest.raises(ConfigurationError, match="cannot read"):
        audit_public_scores(source, checkpoint, tmp_path / "missing.jsonl")
    monkeypatch.setattr(audit_module, "_MAX_SCORE_BYTES", 100)
    scores.write_bytes(b"a" * 101)
    with pytest.raises(ConfigurationError, match="byte limit"):
        audit_public_scores(source, checkpoint, scores)
    monkeypatch.setattr(audit_module, "_MAX_SCORE_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(audit_module, "_MAX_SCORE_LINE_BYTES", 100)
    for body in (b"\n", b"x" * 101 + b"\n"):
        scores.write_bytes(body)
        with pytest.raises(ConfigurationError, match="empty or oversized"):
            audit_public_scores(source, checkpoint, scores)
    monkeypatch.setattr(audit_module, "_MAX_SCORE_LINE_BYTES", 32 * 1024 * 1024 + 1024)
    scores.write_text(
        "".join(json.dumps({"id": id_, "strong_win_rate": 0.5}) + "\n" for id_ in ("a", "b", "c")),
        encoding="utf-8",
    )
    raw_checkpoint = checkpoint.read_bytes()
    for body, message in ((b"not-json", "strict JSON"), (b"{}", "schema mismatch")):
        checkpoint.write_bytes(body)
        with pytest.raises(ConfigurationError, match=message):
            audit_public_scores(source, checkpoint, scores)
    checkpoint.write_bytes(raw_checkpoint)
    assert audit_public_scores(source, checkpoint, scores).manifest["evaluated_records"] == 3


def test_audit_accepts_large_valid_id_and_generator_checkpoint_ceiling(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import facetroute.public_score_audit as audit_module

    assert (
        PublicScoreLimits(max_checkpoint_bytes=256 * 1024 * 1024).max_checkpoint_bytes
        == audit_module._MAX_CHECKPOINT_BYTES
    )
    long_id = "x" * (2 * 1024 * 1024)
    source = tmp_path / "long-id.jsonl"
    source.write_text(
        json.dumps({"id": long_id, "question": "Choose yes", "choices": ["no", "yes"], "answer": 1})
        + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "long-id-checkpoint.json"
    asyncio.run(
        run_public_scores(
            source,
            checkpoint,
            provenance=PublicScoreProvenance(
                "fixture:large-id", "MIT", hashlib.sha256(source.read_bytes()).hexdigest()
            ),
            weak=PublicScoreModel(AsyncProviderTarget("weak", "w", Responses(["A"])), "w1"),
            strong=PublicScoreModel(AsyncProviderTarget("strong", "s", Responses(["B"])), "s1"),
        )
    )
    scores = tmp_path / "long-id-scores.jsonl"
    scores.write_text(json.dumps({"id": long_id, "strong_win_rate": 0.8}) + "\n", encoding="utf-8")
    assert len(scores.read_bytes()) > 2 * 1024 * 1024
    assert audit_public_scores(source, checkpoint, scores).points[-1]["accuracy"] == 1.0
    monkeypatch.setattr(audit_module, "_MAX_CHECKPOINT_BYTES", len(checkpoint.read_bytes()) - 1)
    with pytest.raises(ConfigurationError, match="checkpoint exceeds byte limit"):
        audit_public_scores(source, checkpoint, scores)


def test_cli_prints_only_aggregate_and_errors_without_traceback(tmp_path: Path, capfd: Any) -> None:
    source, checkpoint, scores, _, _ = _fixture(tmp_path)
    assert (
        main(
            [
                "--source",
                str(source),
                "--checkpoint",
                str(checkpoint),
                "--route-scores",
                str(scores),
            ]
        )
        == 0
    )
    output, error = capfd.readouterr()
    assert error == ""
    assert json.loads(output)["schema"] == "facet-public-score-audit-v1"
    assert "Q alpha" not in output
    scores.write_text("broken", encoding="utf-8")
    assert (
        main(
            [
                "--source",
                str(source),
                "--checkpoint",
                str(checkpoint),
                "--route-scores",
                str(scores),
            ]
        )
        == 2
    )
    _, error = capfd.readouterr()
    assert "public-score audit:" in error and "Traceback" not in error
