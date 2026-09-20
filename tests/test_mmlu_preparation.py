from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import AsyncIterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

import facetroute.mmlu_preparation as preparation
from facetroute.async_client import AsyncProviderTarget
from facetroute.benchmark_formats import BenchmarkFormat, load_benchmark_examples_bytes
from facetroute.errors import ConfigurationError, FacetRouteError
from facetroute.mmlu_preparation import (
    MAX_CSV_BYTES,
    MMLUCSVPreparationPlan,
    prepare_mmlu_csv,
    verify_mmlu_csv_preparation,
)
from facetroute.mmlu_preparation_cli import _create_only, main
from facetroute.public_scores import PublicScoreModel, PublicScoreProvenance, run_public_scores

DEV = (
    b"What is 2+2?,three,four,five,six,B\n"
    b"Which shape has three sides?,square,triangle,circle,hexagon,B\n"
)
TEST = b"What is 3+3?,five,seven,six,eight,C\n"


def _plan(
    dev: bytes = DEV, test: bytes = TEST, *, shots: int = 2, budget: int = 8192
) -> MMLUCSVPreparationPlan:
    return MMLUCSVPreparationPlan(
        subject="demo_math",
        dev=PublicScoreProvenance("fixture:demo-dev", "MIT", hashlib.sha256(dev).hexdigest()),
        test=PublicScoreProvenance("fixture:demo-test", "MIT", hashlib.sha256(test).hexdigest()),
        requested_shots=shots,
        max_prompt_bytes=budget,
    )


def _prepare(dev: bytes = DEV, test: bytes = TEST, *, shots: int = 2, budget: int = 8192):
    return prepare_mmlu_csv(
        dev,
        test,
        _plan(dev, test, shots=shots, budget=budget),
        dev_name="demo_math_dev.csv",
        test_name="demo_math_test.csv",
    )


def test_hand_written_fewshot_prompt_and_provenance_oracle() -> None:
    prepared = _prepare()
    assert (prepared.dev_rows, prepared.test_rows, prepared.selected_shots) == (2, 1, 2)
    source = prepared.benchmark_jsonl()
    example = load_benchmark_examples_bytes(source)[0]
    assert example.format is BenchmarkFormat.MMLU
    assert example.answer == 2 and example.choices == ("five", "seven", "six", "eight")
    assert example.prompt == "What is 3+3?"
    expected_context = (
        "Examples about demo math (answers shown for examples only):\n\n"
        "Question: What is 2+2?\nA. three\nB. four\nC. five\nD. six\nAnswer: B\n\n"
        "Question: Which shape has three sides?\nA. square\nB. triangle\n"
        "C. circle\nD. hexagon\nAnswer: B"
    )
    assert example.few_shot_context == expected_context
    assert example.to_dict() == json.loads(source)
    request = example.to_request()
    assert request.query == "What is 3+3?"
    assert request.metadata["benchmark_choices"] == ["five", "seven", "six", "eight"]
    assert "few_shot_context" not in request.metadata
    evidence = json.loads(prepared.evidence_json())
    assert evidence["protocol"] == "facet-mmlu-csv-fewshot-v1"
    assert evidence["plan"]["dev"]["sha256"] == hashlib.sha256(DEV).hexdigest()
    assert evidence["plan"]["test"]["sha256"] == hashlib.sha256(TEST).hexdigest()
    assert evidence["artifact_sha256"] == hashlib.sha256(source).hexdigest()
    assert "What is" not in prepared.evidence_json().decode()
    assert verify_mmlu_csv_preparation(
        source,
        prepared.evidence_json(),
        DEV,
        TEST,
        _plan(),
        dev_name="demo_math_dev.csv",
        test_name="demo_math_test.csv",
    )


def test_byte_budget_backoff_is_global_and_zero_shot_can_fail() -> None:
    full = _prepare()
    from facetroute.public_scores import _prompt

    full_bytes = len(_prompt(full.examples[0])[0].encode())
    one = _prepare(shots=1)
    one_bytes = len(_prompt(one.examples[0])[0].encode())
    zero = _prepare(shots=0)
    zero_bytes = len(_prompt(zero.examples[0])[0].encode())
    assert full_bytes > one_bytes > zero_bytes
    assert _prepare(budget=one_bytes).selected_shots == 1
    assert _prepare(budget=zero_bytes).selected_shots == 0
    with pytest.raises(ConfigurationError, match="zero shots"):
        _prepare(budget=zero_bytes - 1)


def test_byte_budget_backoff_uses_longest_test_prompt_for_every_row() -> None:
    extended = TEST + b"What is a longer second question?,one,two,three,four,C\n"
    full = _prepare(test=extended)
    from facetroute.public_scores import _prompt

    longest_one_shot = max(
        len(_prompt(example)[0].encode()) for example in _prepare(test=extended, shots=1).examples
    )
    assert full.selected_shots == 2
    chosen = _prepare(test=extended, budget=longest_one_shot)
    assert chosen.selected_shots == 1
    assert len(chosen.examples) == 2
    assert all(len(_prompt(example)[0].encode()) <= longest_one_shot for example in chosen.examples)


def test_test_gold_perturbation_changes_only_labels_not_provider_prompts() -> None:
    changed = TEST.replace(b",C\n", b",A\n")
    original = _prepare()
    modified = _prepare(test=changed)
    from facetroute.public_scores import _prompt

    assert original.examples[0].answer != modified.examples[0].answer
    assert _prompt(original.examples[0]) == _prompt(modified.examples[0])
    assert original.examples[0].few_shot_context == modified.examples[0].few_shot_context
    assert original.benchmark_jsonl() != modified.benchmark_jsonl()
    assert (
        json.loads(original.evidence_json())["provider_prompt_sha256"]
        == json.loads(modified.evidence_json())["provider_prompt_sha256"]
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


def test_csv_to_jsonl_to_existing_public_scores_injected_provider(tmp_path: Path) -> None:
    async def run_case(test_source: bytes, name: str):
        prepared = _prepare(test=test_source)
        source = tmp_path / f"{name}.jsonl"
        source.write_bytes(prepared.benchmark_jsonl())
        weak_provider, strong_provider = _FakeProvider("C"), _FakeProvider("A")
        weak = PublicScoreModel(AsyncProviderTarget("weak", "weak-upstream", weak_provider), "r1")
        strong = PublicScoreModel(
            AsyncProviderTarget("strong", "strong-upstream", strong_provider), "r1"
        )
        progress = await run_public_scores(
            source,
            tmp_path / f"{name}-checkpoint.json",
            provenance=PublicScoreProvenance(
                "fixture:prepared-mmlu", "MIT", hashlib.sha256(source.read_bytes()).hexdigest()
            ),
            weak=weak,
            strong=strong,
            max_calls=2,
        )
        return progress, weak_provider.calls, strong_provider.calls

    original, weak_calls, strong_calls = asyncio.run(run_case(TEST, "original"))
    altered, altered_weak, altered_strong = asyncio.run(
        run_case(TEST.replace(b",C\n", b",A\n"), "altered")
    )
    assert original.finished and altered.finished
    assert (original.weak_correct, original.strong_correct) == (1, 0)
    assert (altered.weak_correct, altered.strong_correct) == (0, 1)
    assert weak_calls == altered_weak and strong_calls == altered_strong
    assert weak_calls[0]["messages"][0]["content"].endswith(
        "Question: What is 3+3?\nA. five\nB. seven\nC. six\nD. eight\nAnswer with one option letter."
    )
    assert "Answer: B" in weak_calls[0]["messages"][0]["content"]


@pytest.mark.parametrize(
    "changed_dev,changed_test,match",
    [
        (DEV + b"Bad,1,2,3,4,X\n", TEST, "answer"),
        (DEV + b"What is 2+2?,1,2,3,4,A\n", TEST, "duplicate"),
        (DEV, TEST.replace(b"What is 3+3?", b"WHAT  IS  2+2?"), "overlap"),
        (DEV.replace(b",three,four,five,six,", b",three,THREE,five,six,"), TEST, "distinct"),
        (DEV.replace(b",B\n", b",answer\n"), TEST, "answer"),
        (DEV.replace(b"What is 2+2?", b'"What\n is 2+2?"'), TEST, "unsafe"),
        (DEV + b"header\n", TEST, "six columns"),
    ],
)
def test_malformed_or_leaky_csv_rejected(
    changed_dev: bytes, changed_test: bytes, match: str
) -> None:
    with pytest.raises(ConfigurationError, match=match):
        _prepare(changed_dev, changed_test)


def test_source_subject_hash_type_and_size_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError, match="SHA-256"):
        prepare_mmlu_csv(
            DEV + b" ", TEST, _plan(), dev_name="demo_math_dev.csv", test_name="demo_math_test.csv"
        )
    with pytest.raises(ConfigurationError, match="basenames"):
        prepare_mmlu_csv(
            DEV, TEST, _plan(), dev_name="other_dev.csv", test_name="demo_math_test.csv"
        )
    with pytest.raises(ConfigurationError, match="byte snapshots"):
        prepare_mmlu_csv(
            "not bytes", TEST, _plan(), dev_name="demo_math_dev.csv", test_name="demo_math_test.csv"
        )
    with pytest.raises(ConfigurationError):
        MMLUCSVPreparationPlan("Demo Math", _plan().dev, _plan().test)
    with pytest.raises(ConfigurationError):
        MMLUCSVPreparationPlan("demo_math", _plan().dev, _plan().test, requested_shots=True)
    with pytest.raises(ConfigurationError):
        MMLUCSVPreparationPlan("demo_math", _plan().dev, _plan().test, max_prompt_bytes=63)
    too_big = DEV + b" " * MAX_CSV_BYTES
    with pytest.raises(ConfigurationError, match="byte snapshot"):
        _prepare(dev=too_big)
    monkeypatch.setattr(preparation, "MAX_OUTPUT_BYTES", 1)
    with pytest.raises(ConfigurationError, match="output byte limit"):
        _prepare()


def test_strict_csv_utf8_bom_empty_row_and_shot_count() -> None:
    for raw in (b"\xff", b"\xef\xbb\xbf" + DEV, b"", b"q,a,b,c,d,A\n\n"):
        with pytest.raises(ConfigurationError):
            _prepare(dev=raw)
    with pytest.raises(ConfigurationError, match="available dev"):
        _prepare(shots=3)


def test_legacy_mmlu_subject_category_and_fewshot_validation() -> None:
    legacy = b'{"id":"x","question":"Q?","choices":["a","b"],"answer":0,"subject":"math"}\n'
    current = b'{"id":"x","format":"mmlu","prompt":"Q?","choices":["a","b"],"answer":0,"category":"math"}\n'
    assert load_benchmark_examples_bytes(legacy)[0].category == "math"
    assert load_benchmark_examples_bytes(current)[0].category == "math"
    conflict = current.replace(b'"category":"math"', b'"subject":"science","category":"math"')
    with pytest.raises(ConfigurationError, match="disagree"):
        load_benchmark_examples_bytes(conflict)
    invalid = (
        current.removesuffix(b"\n").replace(b'"category":"math"', b'"few_shot_context":"   "')
        + b"\n"
    )
    with pytest.raises(ConfigurationError, match="few_shot_context"):
        load_benchmark_examples_bytes(invalid)


def test_verifier_detects_artifact_or_evidence_mutation() -> None:
    prepared = _prepare()
    artifact, evidence = prepared.benchmark_jsonl(), prepared.evidence_json()
    assert not verify_mmlu_csv_preparation(
        artifact + b" ",
        evidence,
        DEV,
        TEST,
        _plan(),
        dev_name="demo_math_dev.csv",
        test_name="demo_math_test.csv",
    )
    assert not verify_mmlu_csv_preparation(
        artifact,
        evidence + b" ",
        DEV,
        TEST,
        _plan(),
        dev_name="demo_math_dev.csv",
        test_name="demo_math_test.csv",
    )


def test_cli_create_only_and_private_source_roundtrip(tmp_path: Path, capfd: Any) -> None:
    dev = tmp_path / "demo_math_dev.csv"
    test = tmp_path / "demo_math_test.csv"
    output = tmp_path / "prepared.jsonl"
    dev.write_bytes(DEV)
    test.write_bytes(TEST)
    args = [
        "--dev",
        str(dev),
        "--test",
        str(test),
        "--subject",
        "demo_math",
        "--dev-sha256",
        hashlib.sha256(DEV).hexdigest(),
        "--test-sha256",
        hashlib.sha256(TEST).hexdigest(),
        "--dev-source-uri",
        "fixture:demo-dev",
        "--test-source-uri",
        "fixture:demo-test",
        "--license",
        "MIT",
        "--shots",
        "2",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    evidence = capfd.readouterr().out.encode()
    assert verify_mmlu_csv_preparation(
        output.read_bytes(), evidence, DEV, TEST, _plan(), dev_name=dev.name, test_name=test.name
    )
    assert main(args) == 2
    assert "already exists" in capfd.readouterr().err
    assert output.read_bytes() == _prepare().benchmark_jsonl()
    alias_args = [*args[:-1], str(dev)]
    assert main(alias_args) == 2


def test_create_only_race_has_one_complete_winner(tmp_path: Path) -> None:
    output = tmp_path / "race.jsonl"
    payload = _prepare().benchmark_jsonl()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_create_only, output, payload) for _ in range(2)]
        winners = 0
        for future in futures:
            with suppress(FacetRouteError):
                future.result()
                winners += 1
    assert winners == 1 and output.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory fsync semantics")
def test_post_publication_directory_fsync_failure_keeps_complete_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "prepared.jsonl"
    payload = _prepare().benchmark_jsonl()
    original = os.fsync
    count = 0

    def fail_on_directory(descriptor: int) -> None:
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("directory fsync unavailable")
        original(descriptor)

    monkeypatch.setattr(os, "fsync", fail_on_directory)
    with pytest.raises(FacetRouteError, match="cannot create"):
        _create_only(output, payload)
    assert output.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [output]
