from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest

import facetroute.mmlu_suite as suite
import facetroute.mmlu_suite_cli as suite_cli
from facetroute.async_client import AsyncProviderTarget
from facetroute.benchmark_formats import load_benchmark_examples_bytes
from facetroute.errors import ConfigurationError
from facetroute.mmlu_preparation import MMLUCSVPreparationPlan
from facetroute.mmlu_suite import (
    MMLUSuiteSource,
    PreparedMMLUSuite,
    prepare_mmlu_suite,
    verify_mmlu_suite,
)
from facetroute.public_scores import (
    PublicScoreModel,
    PublicScoreProvenance,
    _prompt,
    run_public_scores,
)


def _source(subject: str, dev: bytes, test: bytes) -> MMLUSuiteSource:
    return MMLUSuiteSource(
        MMLUCSVPreparationPlan(
            subject,
            PublicScoreProvenance(f"fixture:{subject}:dev", "MIT", hashlib.sha256(dev).hexdigest()),
            PublicScoreProvenance(
                f"fixture:{subject}:test", "MIT", hashlib.sha256(test).hexdigest()
            ),
            requested_shots=1,
        ),
        dev,
        test,
    )


MATH = _source("math", b"Two plus two?,3,4,5,6,B\n", b"Three plus three?,5,6,7,8,B\n")
HISTORY = _source(
    "history",
    b"First US president?,Lincoln,Washington,Adams,Jefferson,B\n",
    b"US independence year?,1775,1776,1777,1778,B\n",
)


def test_two_subject_suite_exact_order_prompt_isolation_and_replay() -> None:
    prepared = prepare_mmlu_suite((MATH, HISTORY))
    raw = prepared.benchmark_jsonl()
    examples = load_benchmark_examples_bytes(raw)
    assert [item.example_id for item in examples] == ["history-0001", "math-0001"]
    assert [item.category for item in examples] == ["history", "math"]
    assert "First US president?" in _prompt(examples[0])[0]
    assert "Two plus two?" not in _prompt(examples[0])[0]
    assert "Two plus two?" in _prompt(examples[1])[0]
    assert "First US president?" not in _prompt(examples[1])[0]
    assert "US independence year?" not in _prompt(examples[0])[0].split("Question:")[0]
    report = json.loads(prepared.evidence_json())
    assert report["protocol"] == "facet-mmlu-suite-v1"
    assert report["subjects"] == ["history", "math"]
    assert report["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["records"] == 2
    assert "US independence year?" not in prepared.evidence_json().decode()
    assert verify_mmlu_suite(raw, prepared.evidence_json(), (HISTORY, MATH))
    assert not verify_mmlu_suite(raw + b" ", prepared.evidence_json(), (MATH, HISTORY))


def test_suite_input_order_is_canonical_and_gold_flip_changes_only_scoring() -> None:
    forward = prepare_mmlu_suite((MATH, HISTORY))
    reverse = prepare_mmlu_suite((HISTORY, MATH))
    assert forward.benchmark_jsonl() == reverse.benchmark_jsonl()
    assert forward.evidence_json() == reverse.evidence_json()
    flipped = _source("math", MATH.dev_source, MATH.test_source.replace(b",B\n", b",A\n"))
    changed = prepare_mmlu_suite((flipped, HISTORY))
    assert changed.benchmark_jsonl() != forward.benchmark_jsonl()
    before = load_benchmark_examples_bytes(forward.benchmark_jsonl())
    after = load_benchmark_examples_bytes(changed.benchmark_jsonl())
    assert [_prompt(item) for item in before] == [_prompt(item) for item in after]


def test_suite_rejects_duplicate_subject_cross_subject_leak_and_oversize() -> None:
    with pytest.raises(ConfigurationError, match="duplicate subject"):
        prepare_mmlu_suite((MATH, MATH))
    leaked = _source("history", b"Three plus three?,1,2,3,4,A\n", HISTORY.test_source)
    with pytest.raises(ConfigurationError, match="cross-subject dev/test"):
        prepare_mmlu_suite((MATH, leaked))
    with pytest.raises(ConfigurationError, match="at most 64"):
        prepare_mmlu_suite(tuple(MATH for _ in range(65)))
    with pytest.raises(ConfigurationError, match="source SHA-256"):
        prepare_mmlu_suite((MMLUSuiteSource(MATH.plan, b"changed", MATH.test_source),))


def test_suite_supports_57_subject_shape_without_data_or_license_claim() -> None:
    sources = tuple(
        _source(
            f"subject_{number:02d}",
            f"Dev question {number}?,a,b,c,d,A\n".encode(),
            f"Test question {number}?,a,b,c,d,B\n".encode(),
        )
        for number in range(57)
    )
    result = prepare_mmlu_suite(sources)
    assert len(load_benchmark_examples_bytes(result.benchmark_jsonl())) == 57
    assert len(json.loads(result.evidence_json())["subject_evidence"]) == 57


def test_suite_rejects_normalized_cross_subject_stem_overlap() -> None:
    changed = _source(
        "history",
        "  \uff34\uff28\uff32\uff25\uff25 plus THREE?  ,1,2,3,4,A\n".encode(),
        HISTORY.test_source,
    )
    with pytest.raises(ConfigurationError, match="cross-subject dev/test"):
        prepare_mmlu_suite((MATH, changed))


def test_suite_preflights_types_total_source_and_output_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ConfigurationError, match="1 to at most"):
        prepare_mmlu_suite(())
    with pytest.raises(ConfigurationError, match="entries must"):
        prepare_mmlu_suite((object(),))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="plans and immutable"):
        prepare_mmlu_suite((MMLUSuiteSource("wrong", MATH.dev_source, MATH.test_source),))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="plans and immutable"):
        prepare_mmlu_suite(
            (MMLUSuiteSource(MATH.plan, bytearray(MATH.dev_source), MATH.test_source),)
        )  # type: ignore[arg-type]
    monkeypatch.setattr(suite, "MAX_TOTAL_SOURCE_BYTES", 1)
    with pytest.raises(ConfigurationError, match="total source byte"):
        prepare_mmlu_suite((MATH,))
    monkeypatch.setattr(suite, "MAX_TOTAL_SOURCE_BYTES", 64 * 1024 * 1024)
    prepared = prepare_mmlu_suite((MATH,))
    monkeypatch.setattr(suite, "MAX_SUITE_BYTES", 1)
    with pytest.raises(ConfigurationError, match="output byte"):
        prepared.benchmark_jsonl()
    with pytest.raises(ConfigurationError, match="byte snapshots"):
        verify_mmlu_suite("not bytes", b"", (MATH,))  # type: ignore[arg-type]


def test_public_prepared_suite_explicit_bytes_match_individual_subject() -> None:
    from facetroute.mmlu_preparation import prepare_mmlu_csv

    single = prepare_mmlu_csv(
        MATH.dev_source,
        MATH.test_source,
        MATH.plan,
        dev_name="math_dev.csv",
        test_name="math_test.csv",
    )
    assert PreparedMMLUSuite((single,)).benchmark_jsonl() == single.benchmark_jsonl()


class _AnswerProvider:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    async def complete(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> Mapping[str, Any]:
        self.calls.append(payload)
        return {
            "id": "suite-fixture",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "B"}}],
        }

    async def stream(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> AsyncIterator[Mapping[str, Any]]:
        if False:  # pragma: no cover - protocol-only fake
            yield {}


def test_suite_jsonl_flows_through_existing_injected_provider_scoring(tmp_path: Path) -> None:
    raw = prepare_mmlu_suite((MATH, HISTORY)).benchmark_jsonl()
    source = tmp_path / "suite.jsonl"
    source.write_bytes(raw)
    weak_provider, strong_provider = _AnswerProvider(), _AnswerProvider()
    weak = PublicScoreModel(AsyncProviderTarget("weak", "weak-upstream", weak_provider), "r1")
    strong = PublicScoreModel(
        AsyncProviderTarget("strong", "strong-upstream", strong_provider), "r1"
    )
    progress = asyncio.run(
        run_public_scores(
            source,
            tmp_path / "checkpoint.json",
            provenance=PublicScoreProvenance(
                "fixture:mmlu-suite", "MIT", hashlib.sha256(raw).hexdigest()
            ),
            weak=weak,
            strong=strong,
            max_calls=4,
        )
    )
    assert progress.finished
    assert (progress.weak_correct, progress.strong_correct) == (2, 2)
    assert len(weak_provider.calls) == len(strong_provider.calls) == 2
    prompt_history = weak_provider.calls[0]["messages"][0]["content"]
    prompt_math = weak_provider.calls[1]["messages"][0]["content"]
    assert "First US president?" in prompt_history and "Two plus two?" not in prompt_history
    assert "Two plus two?" in prompt_math and "First US president?" not in prompt_math


def test_suite_cli_manifest_is_strict_pinned_create_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from facetroute.mmlu_suite_cli import main

    for source in (MATH, HISTORY):
        (tmp_path / f"{source.plan.subject}_dev.csv").write_bytes(source.dev_source)
        (tmp_path / f"{source.plan.subject}_test.csv").write_bytes(source.test_source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "subjects": [
                    {
                        "subject": source.plan.subject,
                        "dev": f"{source.plan.subject}_dev.csv",
                        "test": f"{source.plan.subject}_test.csv",
                        "dev_sha256": source.plan.dev.source_sha256,
                        "test_sha256": source.plan.test.source_sha256,
                        "dev_source_uri": source.plan.dev.source_uri,
                        "test_source_uri": source.plan.test.source_uri,
                        "license": "MIT",
                        "shots": 1,
                        "max_prompt_bytes": 8192,
                    }
                    for source in (MATH, HISTORY)
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "suite.jsonl"
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 0
    assert output.read_bytes() == prepare_mmlu_suite((MATH, HISTORY)).benchmark_jsonl()
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert evidence["suite"]["artifact_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 2
    assert output.read_bytes() == prepare_mmlu_suite((MATH, HISTORY)).benchmark_jsonl()
    assert "already exists" in capsys.readouterr().err


def test_suite_cli_rejects_bad_manifest_without_creating_output(tmp_path: Path) -> None:
    from facetroute.mmlu_suite_cli import main

    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"subjects":[],"subjects":[]}', encoding="utf-8")
    output = tmp_path / "output.jsonl"
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 2
    assert not output.exists()


def test_suite_cli_rejects_escape_and_mismatched_source_pin(tmp_path: Path) -> None:
    from facetroute.mmlu_suite_cli import main

    manifest = tmp_path / "manifest.json"
    output = tmp_path / "output.jsonl"
    item = {
        "subject": MATH.plan.subject,
        "dev": "../math_dev.csv",
        "test": "math_test.csv",
        "dev_sha256": MATH.plan.dev.source_sha256,
        "test_sha256": MATH.plan.test.source_sha256,
        "dev_source_uri": MATH.plan.dev.source_uri,
        "test_source_uri": MATH.plan.test.source_uri,
        "license": "MIT",
        "shots": 1,
        "max_prompt_bytes": 8192,
    }
    manifest.write_text(json.dumps({"subjects": [item]}), encoding="utf-8")
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 2
    assert not output.exists()
    item["dev"] = "math_dev.csv"
    (tmp_path / "math_dev.csv").write_bytes(b"tampered")
    (tmp_path / "math_test.csv").write_bytes(MATH.test_source)
    manifest.write_text(json.dumps({"subjects": [item]}), encoding="utf-8")
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 2
    assert not output.exists()


def test_suite_cli_manifest_schema_paths_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    output = root / "out.jsonl"
    manifest = root / "manifest.json"
    (root / "math_dev.csv").write_bytes(MATH.dev_source)
    (root / "math_test.csv").write_bytes(MATH.test_source)
    item = {
        "subject": "math",
        "dev": "math_dev.csv",
        "test": "math_test.csv",
        "dev_sha256": MATH.plan.dev.source_sha256,
        "test_sha256": MATH.plan.test.source_sha256,
        "dev_source_uri": MATH.plan.dev.source_uri,
        "test_source_uri": MATH.plan.test.source_uri,
        "license": "MIT",
        "shots": 1,
        "max_prompt_bytes": 8192,
    }
    with pytest.raises(ConfigurationError, match="bounded non-empty"):
        suite_cli._text("", "field")
    with pytest.raises(ConfigurationError, match="subject basename"):
        suite_cli._source_path(root, "math_dev_bad.csv", "math", "dev")
    for bad in (b"{}", b'{"subjects":[]}', b'{"subjects":[{}]}'):
        with pytest.raises(ConfigurationError):
            suite_cli._manifest_sources(bad, root, output, manifest)
    with pytest.raises(ConfigurationError, match="distinct"):
        suite_cli._manifest_sources(
            json.dumps({"subjects": [item, item]}).encode(), root, output, manifest
        )
    assert suite_cli.main(["--manifest", str(manifest), "--output", str(root / "out.txt")]) == 2
    assert suite_cli.main(["--manifest", str(root / "missing.json"), "--output", str(output)]) == 2
    manifest.write_text(json.dumps({"subjects": [item]}), encoding="utf-8")
    monkeypatch.setattr(suite_cli, "MAX_MANIFEST_BYTES", 1)
    assert suite_cli.main(["--manifest", str(manifest), "--output", str(output)]) == 2
    assert not output.exists()
