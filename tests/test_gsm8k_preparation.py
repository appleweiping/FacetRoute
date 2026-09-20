from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import AsyncIterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import facetroute.gsm8k_preparation as preparation
import facetroute.gsm8k_preparation_cli as preparation_cli
from facetroute.async_client import AsyncProviderTarget
from facetroute.benchmark_formats import BenchmarkFormat, load_benchmark_examples_bytes
from facetroute.errors import ConfigurationError, FacetRouteError
from facetroute.gsm8k_preparation import (
    GSM8KJSONLPreparationPlan,
    prepare_gsm8k_jsonl,
    verify_gsm8k_jsonl_preparation,
)
from facetroute.gsm8k_preparation_cli import _create_only, main
from facetroute.public_score_audit import audit_public_scores
from facetroute.public_scores import (
    PublicScoreModel,
    PublicScoreProvenance,
    _prompt,
    run_public_scores,
)

TRAIN = (
    b'{"question":"What is one plus one?","answer":"Add them. #### 2"}\n'
    b'{"question":"What is three plus four?","answer":"Add them. #### 7"}\n'
)
TEST = b'{"question":"What is two plus three?","answer":"Work it out. #### 5"}\n'


def _plan(
    train: bytes = TRAIN, test: bytes = TEST, *, shots: int = 2, budget: int = 8192
) -> GSM8KJSONLPreparationPlan:
    return GSM8KJSONLPreparationPlan(
        train=PublicScoreProvenance("fixture:train", "MIT", hashlib.sha256(train).hexdigest()),
        test=PublicScoreProvenance("fixture:test", "MIT", hashlib.sha256(test).hexdigest()),
        requested_shots=shots,
        max_prompt_bytes=budget,
    )


def _prepare(train: bytes = TRAIN, test: bytes = TEST, *, shots: int = 2, budget: int = 8192):
    return prepare_gsm8k_jsonl(train, test, _plan(train, test, shots=shots, budget=budget))


def test_exact_hand_prompt_evidence_and_replay() -> None:
    prepared = _prepare()
    assert (prepared.train_rows, prepared.test_rows, prepared.selected_shots) == (2, 1, 2)
    artifact = prepared.benchmark_jsonl()
    example = load_benchmark_examples_bytes(artifact)[0]
    assert example.format is BenchmarkFormat.GSM8K
    assert example.answer == "Work it out. #### 5"
    assert example.prompt == "What is two plus three?"
    expected = (
        "Worked examples (answers shown only for examples):\n\n"
        "Question: What is one plus one?\nAnswer: Add them. #### 2\n\n"
        "Question: What is three plus four?\nAnswer: Add them. #### 7\n\n"
        "Question: What is two plus three?\nGive the final number after ####."
    )
    assert _prompt(example)[0] == expected
    assert "Work it out. #### 5" not in _prompt(example)[0]
    assert example.to_request().query == "What is two plus three?"
    assert "few_shot_context" not in example.to_request().metadata
    evidence = json.loads(prepared.evidence_json())
    assert evidence["artifact_sha256"] == hashlib.sha256(artifact).hexdigest()
    assert evidence["provider_prompt_sha256"] == [
        {"id": "gsm8k-0001", "sha256": hashlib.sha256(expected.encode()).hexdigest()}
    ]
    assert "What is" not in prepared.evidence_json().decode()
    assert verify_gsm8k_jsonl_preparation(artifact, prepared.evidence_json(), TRAIN, TEST, _plan())


def test_budget_backoff_and_legacy_zero_shot_bytes() -> None:
    two = _prepare(shots=2)
    one = _prepare(shots=1)
    zero = _prepare(shots=0)
    sizes = [len(_prompt(item.examples[0])[0].encode()) for item in (two, one, zero)]
    assert sizes[0] > sizes[1] > sizes[2]
    assert _prepare(budget=sizes[1]).selected_shots == 1
    assert _prepare(budget=sizes[2]).selected_shots == 0
    with pytest.raises(ConfigurationError, match="zero shots"):
        _prepare(budget=sizes[2] - 1)
    legacy = load_benchmark_examples_bytes(TEST)[0]
    assert _prompt(legacy) == (
        "Question: What is two plus three?\nGive the final number after ####.",
        256,
    )


def test_budget_backoff_uses_longest_test_question() -> None:
    extra = (
        b'{"question":"What is a substantially longer additional question?","answer":"#### 9"}\n'
    )
    test = TEST + extra
    one = _prepare(test=test, shots=1)
    longest = max(len(_prompt(example)[0].encode()) for example in one.examples)
    chosen = _prepare(test=test, budget=longest)
    assert chosen.selected_shots == 1 and len(chosen.examples) == 2
    assert all(len(_prompt(example)[0].encode()) <= longest for example in chosen.examples)


def test_test_answer_perturbation_changes_labels_not_provider_payload() -> None:
    altered = TEST.replace(b"#### 5", b"#### 6")
    original, changed = _prepare(), _prepare(test=altered)
    assert original.examples[0].answer != changed.examples[0].answer
    assert _prompt(original.examples[0]) == _prompt(changed.examples[0])
    assert (
        json.loads(original.evidence_json())["provider_prompt_sha256"]
        == json.loads(changed.evidence_json())["provider_prompt_sha256"]
    )


class _FakeProvider:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> Mapping[str, Any]:
        self.calls.append(dict(payload))
        await asyncio.sleep(0)
        return {
            "id": "fixture",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": self.answer}}],
        }

    async def stream(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> AsyncIterator[Mapping[str, Any]]:
        if False:  # pragma: no cover - protocol-only fake
            yield {}


def test_injected_provider_checkpoint_and_audit_answer_oracle(tmp_path: Path) -> None:
    async def run_case(test: bytes, name: str):
        prepared = _prepare(test=test)
        source = tmp_path / f"{name}.jsonl"
        source.write_bytes(prepared.benchmark_jsonl())
        weak_provider, strong_provider = _FakeProvider("#### 5"), _FakeProvider("#### 6")
        weak = PublicScoreModel(AsyncProviderTarget("weak", "weak-upstream", weak_provider), "r1")
        strong = PublicScoreModel(
            AsyncProviderTarget("strong", "strong-upstream", strong_provider), "r1"
        )
        checkpoint = tmp_path / f"{name}-checkpoint.json"
        kwargs = dict(
            provenance=PublicScoreProvenance(
                "fixture:prepared-gsm8k", "MIT", hashlib.sha256(source.read_bytes()).hexdigest()
            ),
            weak=weak,
            strong=strong,
        )
        pending = await run_public_scores(source, checkpoint, max_calls=1, **kwargs)
        assert not pending.finished and pending.completed_calls == 1
        progress = await run_public_scores(source, checkpoint, max_calls=1, **kwargs)
        assert progress.finished
        scores = tmp_path / f"{name}-scores.jsonl"
        scores.write_text('{"id":"gsm8k-0001","strong_win_rate":0.7}\n', encoding="utf-8")
        audit = audit_public_scores(source, checkpoint, scores)
        return progress, weak_provider.calls, strong_provider.calls, audit

    original = asyncio.run(run_case(TEST, "original"))
    altered = asyncio.run(run_case(TEST.replace(b"#### 5", b"#### 6"), "altered"))
    assert (original[0].weak_correct, original[0].strong_correct) == (1, 0)
    assert (altered[0].weak_correct, altered[0].strong_correct) == (0, 1)
    assert original[1:3] == altered[1:3]
    assert original[3].baselines == {"weak_accuracy": 1.0, "strong_accuracy": 0.0}
    assert altered[3].baselines == {"weak_accuracy": 0.0, "strong_accuracy": 1.0}


@pytest.mark.parametrize(
    "train,test,reason",
    [
        (TRAIN + TRAIN[: TRAIN.index(b"\n") + 1], TEST, "duplicate"),
        (TRAIN, TEST.replace(b"What is two plus three?", b"WHAT  IS  ONE PLUS ONE?"), "overlap"),
        (TRAIN.replace(b"#### 2", b"no final number"), TEST, "final"),
        (TRAIN, TEST.replace(b"#### 5", b"#### NaN"), "final"),
        (TRAIN + b"\n", TEST, "blank"),
        (b"\xff", TEST, "UTF-8"),
        (b'{"question":"q","answer":"#### 1","extra":1}\n', TEST, "schema"),
    ],
)
def test_malformed_or_leaky_source_rejected(train: bytes, test: bytes, reason: str) -> None:
    with pytest.raises(ConfigurationError, match=reason):
        _prepare(train, test)


def test_hash_budget_type_and_replay_guards() -> None:
    with pytest.raises(ConfigurationError, match="SHA-256"):
        prepare_gsm8k_jsonl(TRAIN + b" ", TEST, _plan())
    with pytest.raises(ConfigurationError, match="byte snapshots"):
        prepare_gsm8k_jsonl("not bytes", TEST, _plan())
    with pytest.raises(ConfigurationError):
        GSM8KJSONLPreparationPlan(_plan().train, _plan().test, requested_shots=True)
    with pytest.raises(ConfigurationError):
        GSM8KJSONLPreparationPlan(_plan().train, _plan().test, max_prompt_bytes=31)
    with pytest.raises(ConfigurationError, match="train/test require"):
        GSM8KJSONLPreparationPlan("not provenance", _plan().test)
    with pytest.raises(ConfigurationError, match="explicit plan"):
        prepare_gsm8k_jsonl(TRAIN, TEST, "not plan")
    with pytest.raises(ConfigurationError, match="available train"):
        _prepare(shots=3)
    with pytest.raises(ConfigurationError, match="byte snapshot"):
        _prepare(train=TRAIN + b" " * preparation.MAX_SOURCE_BYTES)
    prepared = _prepare()
    assert not verify_gsm8k_jsonl_preparation(
        prepared.benchmark_jsonl() + b" ", prepared.evidence_json(), TRAIN, TEST, _plan()
    )
    assert not verify_gsm8k_jsonl_preparation(
        prepared.benchmark_jsonl(), prepared.evidence_json() + b" ", TRAIN, TEST, _plan()
    )
    with pytest.raises(ConfigurationError, match="byte snapshots"):
        verify_gsm8k_jsonl_preparation("not bytes", prepared.evidence_json(), TRAIN, TEST, _plan())


@pytest.mark.parametrize("oversized_train", [True, False])
def test_direct_api_rejects_oversized_source_before_hash(
    monkeypatch: pytest.MonkeyPatch, oversized_train: bool
) -> None:
    plan = _plan()
    oversized = b"x" * (preparation.MAX_SOURCE_BYTES + 1)

    def unexpected_hash(_raw: bytes) -> str:
        pytest.fail("oversized source reached SHA-256")

    monkeypatch.setattr(preparation, "_sha", unexpected_hash)
    train, test = (oversized, TEST) if oversized_train else (TRAIN, oversized)
    with pytest.raises(ConfigurationError, match="byte snapshot"):
        prepare_gsm8k_jsonl(train, test, plan)


def test_bom_empty_rows_line_count_fields_and_output_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    for source, reason in (
        (b"\xef\xbb\xbf" + TRAIN, "BOM"),
        (b"", "no rows"),
        (TRAIN + b"\n", "blank"),
        (b"not JSON\n", "strict JSON"),
        (b'{"question":null,"answer":"#### 2"}\n', "non-empty"),
        (b'{"question":"q\\tq","answer":"#### 2"}\n', "controls"),
        (b'{"question":"q","answer":"step\\twrong #### 2"}\n', "controls"),
        (b'{"question":"q","answer":"#### 2","answer":"#### 3"}\n', "strict JSON"),
    ):
        with pytest.raises(ConfigurationError, match=reason):
            _prepare(train=source)
    long_question = json.dumps({"question": "Q" * 4097, "answer": "#### 2"}).encode() + b"\n"
    with pytest.raises(ConfigurationError, match="field byte limit"):
        _prepare(train=long_question)
    many = b"".join(
        json.dumps({"question": f"question {index}", "answer": "#### 1"}).encode() + b"\n"
        for index in range(65)
    )
    with pytest.raises(ConfigurationError, match="64 rows"):
        _prepare(train=many)
    monkeypatch.setattr(preparation, "MAX_OUTPUT_BYTES", 1)
    with pytest.raises(ConfigurationError, match="output byte limit"):
        _prepare()


def test_public_score_preflight_rejects_invalid_context_without_provider_calls(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid-context.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "gsm8k-0001",
                "format": "gsm8k",
                "prompt": "Question?",
                "answer": "#### 5",
                "few_shot_context": "x" * (32 * 1024 + 1),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    weak_provider, strong_provider = _FakeProvider("#### 5"), _FakeProvider("#### 6")
    weak = PublicScoreModel(AsyncProviderTarget("weak", "weak-upstream", weak_provider), "r1")
    strong = PublicScoreModel(
        AsyncProviderTarget("strong", "strong-upstream", strong_provider), "r1"
    )
    with pytest.raises(ConfigurationError, match="few_shot_context"):
        asyncio.run(
            run_public_scores(
                source,
                tmp_path / "checkpoint.json",
                provenance=PublicScoreProvenance(
                    "fixture:invalid", "MIT", hashlib.sha256(source.read_bytes()).hexdigest()
                ),
                weak=weak,
                strong=strong,
                max_calls=2,
            )
        )
    assert weak_provider.calls == strong_provider.calls == []
    assert not (tmp_path / "checkpoint.json").exists()


def test_cli_create_only_and_hardlink_alias(tmp_path: Path, capfd: Any) -> None:
    train, test, output = tmp_path / "train.jsonl", tmp_path / "test.jsonl", tmp_path / "out.jsonl"
    train.write_bytes(TRAIN)
    test.write_bytes(TEST)
    args = [
        "--train",
        str(train),
        "--test",
        str(test),
        "--train-sha256",
        hashlib.sha256(TRAIN).hexdigest(),
        "--test-sha256",
        hashlib.sha256(TEST).hexdigest(),
        "--train-source-uri",
        "fixture:train",
        "--test-source-uri",
        "fixture:test",
        "--license",
        "MIT",
        "--shots",
        "2",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    evidence = capfd.readouterr().out.encode()
    assert verify_gsm8k_jsonl_preparation(output.read_bytes(), evidence, TRAIN, TEST, _plan())
    assert main(args) == 2
    assert "already exists" in capfd.readouterr().err
    alias = tmp_path / "same-train.jsonl"
    os.link(train, alias)
    alias_args = args.copy()
    alias_args[3] = str(alias)
    assert main(alias_args) == 2
    assert "must differ" in capfd.readouterr().err


def test_cli_source_drift_suffix_and_create_only_race(
    tmp_path: Path, capfd: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    train.write_bytes(TRAIN)
    test.write_bytes(TEST)
    args = [
        "--train",
        str(train),
        "--test",
        str(test),
        "--train-sha256",
        hashlib.sha256(TRAIN).hexdigest(),
        "--test-sha256",
        hashlib.sha256(TEST).hexdigest(),
        "--train-source-uri",
        "fixture:train",
        "--test-source-uri",
        "fixture:test",
        "--license",
        "MIT",
        "--shots",
        "2",
        "--output",
        str(tmp_path / "out.jsonl"),
    ]
    wrong_suffix = args.copy()
    wrong_suffix[-1] = str(tmp_path / "out.txt")
    assert main(wrong_suffix) == 2
    assert "suffix" in capfd.readouterr().err
    original_snapshot = preparation_cli._snapshot
    calls = 0

    def drifting(path: Path) -> bytes:
        nonlocal calls
        calls += 1
        raw = original_snapshot(path)
        return raw if calls <= 2 else raw + b" "

    monkeypatch.setattr(preparation_cli, "_snapshot", drifting)
    assert main(args) == 2
    assert "changed" in capfd.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()
    monkeypatch.setattr(preparation_cli, "_snapshot", original_snapshot)
    output = tmp_path / "race.jsonl"
    payload = _prepare().benchmark_jsonl()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_create_only, output, payload) for _ in range(2)]
        winners = 0
        for future in futures:
            try:
                future.result()
                winners += 1
            except FacetRouteError as error:
                assert "already exists" in str(error)
    assert winners == 1 and output.read_bytes() == payload
