from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from facetroute.async_client import AsyncProviderTarget
from facetroute.benchmark_formats import BenchmarkFormat, load_benchmark_examples_bytes
from facetroute.errors import ConfigurationError
from facetroute.providers import ProviderError, ProviderFailure
from facetroute.public_scores import (
    PublicScoreLimits,
    PublicScoreModel,
    PublicScoreProvenance,
    run_public_scores,
)


def _completion(text: str, model: str) -> dict[str, Any]:
    return {
        "id": "local",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
    }


class FakeProvider:
    def __init__(self, outcomes: list[str | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    async def complete(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> Mapping[str, Any]:
        self.calls.append((model, dict(payload), timeout_seconds))
        await asyncio.sleep(0)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _completion(outcome, model)

    async def stream(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> AsyncIterator[Mapping[str, Any]]:
        if False:  # pragma: no cover - protocol-only fake
            yield _completion("A", model)


class BlockingProvider(FakeProvider):
    def __init__(self, outcome: str) -> None:
        super().__init__([outcome])
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> Mapping[str, Any]:
        self.entered.set()
        await self.release.wait()
        return await super().complete(payload, model=model, timeout_seconds=timeout_seconds)


def _models(
    weak: FakeProvider, strong: FakeProvider, *, revision: str = "fixture-r1"
) -> tuple[PublicScoreModel, PublicScoreModel]:
    return (
        PublicScoreModel(AsyncProviderTarget("weak", "weak-upstream", weak), revision),
        PublicScoreModel(AsyncProviderTarget("strong", "strong-upstream", strong), revision),
    )


def _dataset(tmp_path: Path, records: list[dict[str, Any]]) -> tuple[Path, PublicScoreProvenance]:
    source = tmp_path / "public.jsonl"
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode()
    source.write_bytes(raw)
    return source, PublicScoreProvenance(
        "fixture:hand-computed-public-score", "CC-BY-4.0", hashlib.sha256(raw).hexdigest()
    )


def _mmlu() -> list[dict[str, Any]]:
    return [
        {
            "id": "q1",
            "question": "Choose the second planet.",
            "choices": ["Venus", "Mercury", "Mars", "Earth"],
            "answer": 1,
            "subject": "astronomy",
        },
        {
            "id": "q2",
            "question": "Choose the first word.",
            "choices": ["alpha", "beta"],
            "answer": "alpha",
            "subject": "vocabulary",
        },
    ]


def test_mmlu_hand_oracle_partial_resume_and_answer_free_requests(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu())
    checkpoint = tmp_path / "checkpoint.json"
    weak_provider = FakeProvider(["A", "A"])
    strong_provider = FakeProvider(["B.", "Answer: A"])
    weak, strong = _models(weak_provider, strong_provider)

    async def scenario() -> None:
        partial = await run_public_scores(
            source, checkpoint, provenance=provenance, weak=weak, strong=strong, max_calls=1
        )
        assert (partial.completed_calls, partial.total_calls, partial.finished) == (1, 4, False)
        assert partial.weak_accuracy is None and partial.strong_accuracy is None
        finished = await run_public_scores(
            source, checkpoint, provenance=provenance, weak=weak, strong=strong
        )
        assert finished.benchmark_format is BenchmarkFormat.MMLU
        assert finished.completed_calls == 4 and finished.finished
        assert (finished.weak_correct, finished.strong_correct) == (1, 2)
        assert (finished.weak_accuracy, finished.strong_accuracy) == (0.5, 1.0)
        assert finished.license_id == "CC-BY-4.0"
        assert finished.source_file_sha256 == provenance.source_sha256
        again = await run_public_scores(
            source, checkpoint, provenance=provenance, weak=weak, strong=strong
        )
        assert again == finished

    asyncio.run(scenario())
    assert len(weak_provider.calls) == len(strong_provider.calls) == 2
    assert [row["role"] for row in json.loads(checkpoint.read_text())["manifest"]["models"]] == [
        "weak",
        "strong",
    ]
    assert [row["correct"] for row in json.loads(checkpoint.read_text())["results"]] == [
        False,
        True,
        True,
        True,
    ]
    for model, payload, timeout in weak_provider.calls + strong_provider.calls:
        assert model in {"weak-upstream", "strong-upstream"}
        assert timeout == 30
        assert "answer" not in payload
        assert "answer" not in payload["messages"][0]
        assert "A." in payload["messages"][0]["content"]
        assert "B." in payload["messages"][0]["content"]
        assert "####" not in payload["messages"][0]["content"]


def test_gsm8k_decimal_oracle_and_missing_prediction(tmp_path: Path) -> None:
    source, provenance = _dataset(
        tmp_path,
        [
            {"id": "g1", "question": "Compute a signed half.", "answer": "work #### -1.5"},
            {"id": "g2", "question": "Compute twelve hundred.", "answer": "#### 1,200"},
        ],
    )
    weak_provider = FakeProvider(["steps #### -1.5", "no number"])
    strong_provider = FakeProvider(["#### -1.50", "result 1,200"])
    weak, strong = _models(weak_provider, strong_provider)
    result = asyncio.run(
        run_public_scores(
            source, tmp_path / "gsm.json", provenance=provenance, weak=weak, strong=strong
        )
    )
    assert result.benchmark_format is BenchmarkFormat.GSM8K
    assert (result.weak_correct, result.strong_correct) == (1, 2)
    assert (result.weak_accuracy, result.strong_accuracy) == (0.5, 1.0)
    for _, payload, _ in weak_provider.calls + strong_provider.calls:
        assert "-1.5" not in payload["messages"][0]["content"]
        assert "1,200" not in payload["messages"][0]["content"]


def test_source_license_and_revision_drift_refuse_before_provider_call(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu())
    checkpoint = tmp_path / "state.json"
    weak_provider = FakeProvider(["A"])
    strong_provider = FakeProvider([])
    weak, strong = _models(weak_provider, strong_provider)
    asyncio.run(
        run_public_scores(
            source, checkpoint, provenance=provenance, weak=weak, strong=strong, max_calls=1
        )
    )
    assert len(weak_provider.calls) == 1

    changed_license = PublicScoreProvenance(provenance.source_uri, "MIT", provenance.source_sha256)
    with pytest.raises(ConfigurationError, match="manifest mismatch"):
        asyncio.run(
            run_public_scores(
                source, checkpoint, provenance=changed_license, weak=weak, strong=strong
            )
        )
    changed_weak, changed_strong = _models(weak_provider, strong_provider, revision="r2")
    with pytest.raises(ConfigurationError, match="manifest mismatch"):
        asyncio.run(
            run_public_scores(
                source,
                checkpoint,
                provenance=provenance,
                weak=changed_weak,
                strong=changed_strong,
            )
        )
    with pytest.raises(ConfigurationError, match="manifest mismatch"):
        asyncio.run(
            run_public_scores(
                source,
                checkpoint,
                provenance=provenance,
                weak=weak,
                strong=strong,
                limits=PublicScoreLimits(timeout_seconds=31),
            )
        )
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ConfigurationError, match="SHA-256"):
        asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda state: state.update(results="not-a-list"), "result sequence"),
        (lambda state: state["results"].extend([state["results"][0]] * 3), "result sequence"),
        (lambda state: state["results"].__setitem__(0, []), "result is invalid"),
        (lambda state: state["results"][0].update(response=7), "result is invalid"),
        (lambda state: state["results"][0].update(response="A" * 9000), "response exceeds"),
        (lambda state: state["results"][0].update(attempt=True), "attempt is invalid"),
        (lambda state: state["results"][0].update(attempt=0), "attempt is invalid"),
        (lambda state: state.update(pending=[]), "pending record"),
        (
            lambda state: state.update(pending={"key": "wrong", "attempt": 1, "extra": 1}),
            "pending record",
        ),
        (lambda state: state.update(pending={"key": "wrong", "attempt": 1}), "pending record"),
        (
            lambda state: state.update(
                pending={
                    "key": hashlib.sha256(b'{"id":"q1","index":0,"role":"strong"}').hexdigest(),
                    "attempt": True,
                }
            ),
            "pending record",
        ),
        (
            lambda state: state.update(
                pending={
                    "key": hashlib.sha256(b'{"id":"q1","index":0,"role":"strong"}').hexdigest(),
                    "attempt": 0,
                }
            ),
            "pending record",
        ),
    ],
)
def test_resealed_checkpoint_still_rejects_inconsistent_structure(
    tmp_path: Path, change: Any, message: str
) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    weak_provider = FakeProvider(["A"])
    weak, strong = _models(weak_provider, FakeProvider([]))
    asyncio.run(
        run_public_scores(
            source, checkpoint, provenance=provenance, weak=weak, strong=strong, max_calls=1
        )
    )
    state = json.loads(checkpoint.read_text())
    change(state)
    content = {key: value for key, value in state.items() if key != "sha256"}
    state["sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    checkpoint.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )
    assert len(weak_provider.calls) == 1


def test_invalid_models_limits_and_call_options_fail_preflight(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    weak, strong = _models(FakeProvider([]), FakeProvider([]))
    with pytest.raises(ConfigurationError, match="AsyncProviderTarget"):
        PublicScoreModel(None, "r1")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="revision"):
        PublicScoreModel(weak.target, "")
    with pytest.raises(ConfigurationError, match="max_records"):
        PublicScoreLimits(max_records=True)
    with pytest.raises(ConfigurationError, match="timeout_seconds"):
        PublicScoreLimits(timeout_seconds=float("inf"))
    with pytest.raises(ConfigurationError, match="source_uri"):
        PublicScoreProvenance("fixture:x\ud800", "MIT", provenance.source_sha256)
    with pytest.raises(ConfigurationError, match="source_uri"):
        PublicScoreProvenance("https://[broken/v1", "MIT", provenance.source_sha256)
    same = PublicScoreModel(weak.target, "r2")
    cases: list[tuple[dict[str, Any], str]] = [
        ({"provenance": None}, "provenance"),
        ({"weak": None}, "PublicScoreModel"),
        ({"strong": None}, "PublicScoreModel"),
        ({"limits": "invalid"}, "PublicScoreLimits"),
        ({"strong": same}, "IDs must differ"),
        ({"max_calls": True}, "max_calls"),
        ({"allow_ambiguous_retry": "yes"}, "boolean"),
    ]
    for override, message in cases:
        kwargs: dict[str, Any] = {"provenance": provenance, "weak": weak, "strong": strong}
        kwargs.update(override)
        with pytest.raises(ConfigurationError, match=message):
            asyncio.run(run_public_scores(source, checkpoint, **kwargs))
        assert not checkpoint.exists()


@pytest.mark.parametrize(
    "value",
    [0, -1, 32 * 1024 * 1024 + 1, 1.5],
)
def test_source_byte_limit_requires_positive_bounded_integer(value: Any) -> None:
    with pytest.raises(ConfigurationError, match="max_source_bytes"):
        PublicScoreLimits(max_source_bytes=value)


@pytest.mark.parametrize("value", [True, "30", 0, 601])
def test_timeout_rejects_invalid_types_and_range(value: Any) -> None:
    with pytest.raises(ConfigurationError, match="timeout_seconds"):
        PublicScoreLimits(timeout_seconds=value)


@pytest.mark.parametrize(
    "uri",
    [
        "mailto:private@example.com",
        "http:/missing-host",
        "https://example.org/path#fragment",
        "https://:secret@example.org/path",
    ],
)
def test_provenance_rejects_nonpublic_or_ambiguous_uris(uri: str) -> None:
    with pytest.raises(ConfigurationError, match="source_uri"):
        PublicScoreProvenance(uri, "MIT", "0" * 64)


@pytest.mark.parametrize(
    "record",
    [
        {"id": "duplicate", "question": "Q", "choices": ["same", "same"], "answer": 0},
        {
            "id": "too-many",
            "question": "Q",
            "choices": [str(index) for index in range(27)],
            "answer": 0,
        },
        {"id": "not-number", "question": "Q", "answer": "no result"},
    ],
)
def test_unscorable_inputs_fail_before_checkpoint(tmp_path: Path, record: dict[str, Any]) -> None:
    source, provenance = _dataset(tmp_path, [record])
    checkpoint = tmp_path / "state.json"
    weak, strong = _models(FakeProvider([]), FakeProvider([]))
    with pytest.raises(ConfigurationError):
        asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )
    assert not checkpoint.exists()


def test_nontext_completion_is_malformed_and_pending(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"

    class NonTextProvider(FakeProvider):
        async def complete(
            self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
        ) -> Mapping[str, Any]:
            self.calls.append((model, dict(payload), timeout_seconds))
            malformed = _completion("A", model)
            malformed["choices"][0]["message"]["content"] = 7
            return malformed

    weak, strong = _models(NonTextProvider([]), FakeProvider([]))
    with pytest.raises(ProviderError) as raised:
        asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )
    assert raised.value.failure is ProviderFailure.MALFORMED
    assert json.loads(checkpoint.read_text())["pending"] is not None


@pytest.mark.parametrize(
    "response",
    [
        {"model": "weak-upstream", "choices": []},
        {"model": "weak-upstream", "choices": [{}]},
        _completion(" ", "weak-upstream"),
        _completion("\ud800", "weak-upstream"),
        _completion("é" * 5, "weak-upstream"),
    ],
)
def test_malformed_completion_shapes_leave_ambiguous_pending(
    tmp_path: Path, response: dict[str, Any]
) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"

    class RawProvider(FakeProvider):
        async def complete(
            self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
        ) -> Mapping[str, Any]:
            self.calls.append((model, dict(payload), timeout_seconds))
            return response

    weak_provider = RawProvider([])
    weak, strong = _models(weak_provider, FakeProvider([]))
    with pytest.raises(ProviderError) as raised:
        asyncio.run(
            run_public_scores(
                source,
                checkpoint,
                provenance=provenance,
                weak=weak,
                strong=strong,
                limits=PublicScoreLimits(max_response_bytes=5),
            )
        )
    assert raised.value.failure is ProviderFailure.MALFORMED
    assert len(weak_provider.calls) == 1
    assert json.loads(checkpoint.read_text())["pending"] is not None


def test_mismatched_reported_model_is_not_misattributed(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"

    class WrongModelProvider(FakeProvider):
        async def complete(
            self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
        ) -> Mapping[str, Any]:
            return _completion("B", "different-upstream")

    weak, strong = _models(WrongModelProvider([]), FakeProvider([]))
    with pytest.raises(ProviderError) as raised:
        asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )
    assert raised.value.failure is ProviderFailure.MALFORMED
    assert json.loads(checkpoint.read_text())["pending"] is not None


def test_provider_mutation_cannot_change_later_payload_or_digest(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"

    class MutatingProvider(FakeProvider):
        async def complete(
            self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
        ) -> Mapping[str, Any]:
            payload["messages"][0]["content"] = "tampered by provider"
            return await super().complete(payload, model=model, timeout_seconds=timeout_seconds)

    weak_provider = MutatingProvider(["B"])
    strong_provider = FakeProvider(["B"])
    weak, strong = _models(weak_provider, strong_provider)
    result = asyncio.run(
        run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
    )
    assert result.finished
    assert "Choose the second planet" in strong_provider.calls[0][1]["messages"][0]["content"]
    assert "tampered" not in checkpoint.read_text()


def test_checkpoint_schema_json_and_size_fail_before_resend(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    weak, strong = _models(FakeProvider([]), FakeProvider([]))
    for raw, message in (
        (b"not-json", "strict JSON"),
        (b"{}", "schema mismatch"),
        ('{"manifest":{},"results":[],"pending":null,"sha256":"é"}'.encode(), "checksum"),
        (b'{"manifest":{},"results":[],"pending":null,"sha256":7}', "checksum"),
        (
            b'{"manifest":{},"results":[],"pending":null,"sha256":"' + b"0" * 64 + b'"}',
            "checksum",
        ),
    ):
        checkpoint.write_bytes(raw)
        with pytest.raises(ConfigurationError, match=message):
            asyncio.run(
                run_public_scores(
                    source, checkpoint, provenance=provenance, weak=weak, strong=strong
                )
            )
    checkpoint.write_bytes(b"x" * 100)
    with pytest.raises(ConfigurationError, match="exceeds byte limit"):
        asyncio.run(
            run_public_scores(
                source,
                checkpoint,
                provenance=provenance,
                weak=weak,
                strong=strong,
                limits=PublicScoreLimits(max_checkpoint_bytes=10),
            )
        )


def test_checkpoint_checksum_and_consistent_score_are_verified(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu())
    checkpoint = tmp_path / "state.json"
    weak_provider = FakeProvider(["A"])
    strong_provider = FakeProvider([])
    weak, strong = _models(weak_provider, strong_provider)
    asyncio.run(
        run_public_scores(
            source, checkpoint, provenance=provenance, weak=weak, strong=strong, max_calls=1
        )
    )
    original = json.loads(checkpoint.read_text())
    changed = {**original, "results": [{**original["results"][0], "correct": True}]}
    checkpoint.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="checksum"):
        asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )
    content = {key: value for key, value in changed.items() if key != "sha256"}
    changed["sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    checkpoint.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="inconsistent"):
        asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )
    assert len(weak_provider.calls) == 1


@pytest.mark.parametrize("retry_safe", [False, True])
def test_presend_and_ambiguous_failures_have_different_resume_rules(
    tmp_path: Path, retry_safe: bool
) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    weak_provider = FakeProvider(
        [ProviderError(ProviderFailure.UNAVAILABLE, retry_safe=retry_safe), "B"]
    )
    strong_provider = FakeProvider(["B"])
    weak, strong = _models(weak_provider, strong_provider)
    with pytest.raises(ProviderError):
        asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )
    checkpoint_state = json.loads(checkpoint.read_text())
    assert (checkpoint_state["pending"] is None) is retry_safe
    if not retry_safe:
        with pytest.raises(ConfigurationError, match="ambiguous pending"):
            asyncio.run(
                run_public_scores(
                    source, checkpoint, provenance=provenance, weak=weak, strong=strong
                )
            )
        assert len(weak_provider.calls) == 1
    finished = asyncio.run(
        run_public_scores(
            source,
            checkpoint,
            provenance=provenance,
            weak=weak,
            strong=strong,
            allow_ambiguous_retry=not retry_safe,
        )
    )
    assert finished.finished and finished.weak_correct == finished.strong_correct == 1
    assert json.loads(checkpoint.read_text())["results"][0]["attempt"] == (1 if retry_safe else 2)
    assert len(weak_provider.calls) == 2


def test_cancelled_call_remains_pending_and_concurrent_runner_is_excluded(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    blocked = BlockingProvider("B")
    weak, strong = _models(blocked, FakeProvider(["B"]))

    async def scenario() -> None:
        first = asyncio.create_task(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )
        await blocked.entered.wait()
        with pytest.raises(ConfigurationError, match="already in use"):
            await run_public_scores(
                source, checkpoint, provenance=provenance, weak=weak, strong=strong
            )
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

    asyncio.run(scenario())
    assert json.loads(checkpoint.read_text())["pending"] is not None
    with pytest.raises(ConfigurationError, match="ambiguous pending"):
        asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )


def test_lock_file_aliases_are_rejected_without_mutating_their_targets(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    lock_path = tmp_path / "state.json.lock"
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_bytes(b"")
    weak, strong = _models(FakeProvider([]), FakeProvider([]))
    os.link(unrelated, lock_path)
    with pytest.raises(ConfigurationError, match="lock file is invalid"):
        asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )
    assert unrelated.read_bytes() == b""
    assert not checkpoint.exists()
    lock_path.unlink()
    try:
        lock_path.symlink_to(unrelated)
    except OSError:
        return  # Windows installations may not permit unprivileged symlinks.
    with pytest.raises(ConfigurationError, match="lock file is invalid"):
        asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )
    assert unrelated.read_bytes() == b""
    assert not checkpoint.exists()


def test_zero_byte_lock_file_fails_closed(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    lock_path = tmp_path / "state.json.lock"
    lock_path.write_bytes(b"")
    weak_provider = FakeProvider([])
    weak, strong = _models(weak_provider, FakeProvider([]))
    with pytest.raises(ConfigurationError, match="lock file is invalid"):
        asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )
    assert lock_path.read_bytes() == b""
    assert not checkpoint.exists()
    assert not weak_provider.calls


def test_short_lock_initialization_is_rejected_and_closes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    weak, strong = _models(FakeProvider([]), FakeProvider([]))
    descriptors: list[int] = []

    def short_write(descriptor: int, data: bytes) -> int:
        descriptors.append(descriptor)
        return 0

    with monkeypatch.context() as patch:
        patch.setattr(os, "write", short_write)
        with pytest.raises(ConfigurationError, match="initialize public-score checkpoint lock"):
            asyncio.run(
                run_public_scores(
                    source, checkpoint, provenance=provenance, weak=weak, strong=strong
                )
            )
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    assert not checkpoint.exists()
    assert not (tmp_path / "state.json.lock").exists()


def test_failed_lock_initialization_does_not_remove_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX can replace a file while its descriptor is open")
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    lock_path = tmp_path / "state.json.lock"
    weak, strong = _models(FakeProvider([]), FakeProvider([]))

    def replace_during_initialization(descriptor: int, data: bytes) -> int:
        lock_path.unlink()
        lock_path.write_bytes(b"replacement")
        return 0

    with monkeypatch.context() as patch:
        patch.setattr(os, "write", replace_during_initialization)
        with pytest.raises(ConfigurationError, match="initialize public-score checkpoint lock"):
            asyncio.run(
                run_public_scores(
                    source, checkpoint, provenance=provenance, weak=weak, strong=strong
                )
            )
    assert lock_path.read_bytes() == b"replacement"
    assert not checkpoint.exists()


def test_checkpoint_is_private_and_parent_directory_is_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX permission and directory-fsync semantics")
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    weak, strong = _models(FakeProvider(["B"]), FakeProvider(["B"]))
    synced_modes: list[int] = []
    original_fsync = os.fsync

    def recorded_fsync(descriptor: int) -> None:
        synced_modes.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", recorded_fsync)
        finished = asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )
    assert finished.finished
    assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600
    assert any(stat.S_ISREG(mode) for mode in synced_modes)
    assert any(stat.S_ISDIR(mode) for mode in synced_modes)


def test_directory_sync_failure_prevents_provider_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("POSIX directory-fsync semantics")
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    weak_provider = FakeProvider([])
    weak, strong = _models(weak_provider, FakeProvider([]))
    original_fsync = os.fsync

    def fail_directory_sync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory sync unavailable")
        original_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_directory_sync)
        with pytest.raises(ConfigurationError, match="cannot write public-score checkpoint"):
            asyncio.run(
                run_public_scores(
                    source, checkpoint, provenance=provenance, weak=weak, strong=strong
                )
            )
    assert not weak_provider.calls
    assert json.loads(checkpoint.read_text())["pending"] is None


def test_preexisting_temporary_name_is_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import facetroute.public_scores as public_scores

    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    temporary = tmp_path / ".state.json.fixed.tmp"
    temporary.write_bytes(b"unrelated")
    weak_provider = FakeProvider([])
    weak, strong = _models(weak_provider, FakeProvider([]))
    with monkeypatch.context() as patch:
        patch.setattr(public_scores.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed"))
        with pytest.raises(ConfigurationError, match="cannot write public-score checkpoint"):
            asyncio.run(
                run_public_scores(
                    source, checkpoint, provenance=provenance, weak=weak, strong=strong
                )
            )
    assert temporary.read_bytes() == b"unrelated"
    assert not checkpoint.exists()
    assert not weak_provider.calls


def test_unlock_failure_still_closes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "nt":
        pytest.skip("Windows msvcrt lock lifecycle")
    import msvcrt

    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    weak, strong = _models(FakeProvider([]), FakeProvider([]))
    original_locking = msvcrt.locking
    descriptors: list[int] = []

    def fail_unlock(descriptor: int, mode: int, count: int) -> None:
        if mode == msvcrt.LK_UNLCK:
            descriptors.append(descriptor)
            raise OSError("fixture unlock failure")
        original_locking(descriptor, mode, count)

    with monkeypatch.context() as patch:
        patch.setattr(msvcrt, "locking", fail_unlock)
        with pytest.raises(OSError, match="fixture unlock failure"):
            asyncio.run(
                run_public_scores(
                    source,
                    checkpoint,
                    provenance=provenance,
                    weak=weak,
                    strong=strong,
                    max_calls=0,
                )
            )
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_injected_provider_must_obey_total_deadline(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    blocked = BlockingProvider("B")
    weak, strong = _models(blocked, FakeProvider([]))
    with pytest.raises(ProviderError) as raised:
        asyncio.run(
            run_public_scores(
                source,
                checkpoint,
                provenance=provenance,
                weak=weak,
                strong=strong,
                limits=PublicScoreLimits(timeout_seconds=0.1),
            )
        )
    assert raised.value.failure is ProviderFailure.TIMEOUT
    assert not raised.value.retry_safe
    assert blocked.entered.is_set()
    assert json.loads(checkpoint.read_text())["pending"] is not None


def test_preflight_bounds_reject_without_provider_or_checkpoint(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu())
    checkpoint = tmp_path / "state.json"
    weak_provider = FakeProvider([])
    strong_provider = FakeProvider([])
    weak, strong = _models(weak_provider, strong_provider)
    cases = [
        PublicScoreLimits(max_source_bytes=5),
        PublicScoreLimits(max_records=1),
        PublicScoreLimits(max_prompt_bytes=10),
        PublicScoreLimits(max_checkpoint_bytes=10),
    ]
    for limits in cases:
        with pytest.raises(ConfigurationError):
            asyncio.run(
                run_public_scores(
                    source,
                    checkpoint,
                    provenance=provenance,
                    weak=weak,
                    strong=strong,
                    limits=limits,
                )
            )
        assert not checkpoint.exists()
    assert not weak_provider.calls and not strong_provider.calls


def test_response_bound_leaves_ambiguous_pending(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    checkpoint = tmp_path / "state.json"
    weak_provider = FakeProvider(["B" * 20])
    weak, strong = _models(weak_provider, FakeProvider([]))
    with pytest.raises(ProviderError) as raised:
        asyncio.run(
            run_public_scores(
                source,
                checkpoint,
                provenance=provenance,
                weak=weak,
                strong=strong,
                limits=PublicScoreLimits(max_response_bytes=5),
            )
        )
    assert raised.value.failure is ProviderFailure.MALFORMED
    assert json.loads(checkpoint.read_text())["pending"] is not None


def test_unsupported_format_paths_and_provenance_are_rejected(tmp_path: Path) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    weak, strong = _models(FakeProvider([]), FakeProvider([]))
    with pytest.raises(ConfigurationError, match="paths must differ"):
        asyncio.run(
            run_public_scores(source, source, provenance=provenance, weak=weak, strong=strong)
        )
    for uri in ("https://user:password@example.com/a", "https://example.com/a?token=x"):
        with pytest.raises(ConfigurationError, match="source_uri"):
            PublicScoreProvenance(uri, "MIT", provenance.source_sha256)
    with pytest.raises(ConfigurationError, match="source_sha256"):
        PublicScoreProvenance("fixture:one", "MIT", "not-a-digest")
    mt_source, mt_provenance = _dataset(
        tmp_path, [{"question_id": "mt1", "turns": ["Give a title."]}]
    )
    with pytest.raises(ConfigurationError, match="only MMLU and GSM8K"):
        asyncio.run(
            run_public_scores(
                mt_source,
                tmp_path / "mt.json",
                provenance=mt_provenance,
                weak=weak,
                strong=strong,
            )
        )


def test_one_byte_snapshot_parser_and_source_change_race(tmp_path: Path, monkeypatch: Any) -> None:
    source, provenance = _dataset(tmp_path, _mmlu()[:1])
    raw = source.read_bytes()
    assert load_benchmark_examples_bytes(raw) == load_benchmark_examples_bytes(raw)
    with pytest.raises(ConfigurationError, match="byte snapshot"):
        load_benchmark_examples_bytes("not bytes")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="exceeds"):
        load_benchmark_examples_bytes(raw, max_bytes=1)

    import facetroute.public_scores as public_scores

    original = public_scores.load_benchmark_examples_bytes

    def change_source(snapshot: bytes, **kwargs: Any) -> Any:
        source.write_bytes(b"changed after bounded read")
        return original(snapshot, **kwargs)

    monkeypatch.setattr(public_scores, "load_benchmark_examples_bytes", change_source)
    weak, strong = _models(FakeProvider(["B"]), FakeProvider(["B"]))
    checkpoint = tmp_path / "race.json"
    completed = asyncio.run(
        run_public_scores(
            source, checkpoint, provenance=provenance, weak=weak, strong=strong, max_calls=0
        )
    )
    assert completed.source_file_sha256 == hashlib.sha256(raw).hexdigest()
    with pytest.raises(ConfigurationError, match="SHA-256"):
        asyncio.run(
            run_public_scores(source, checkpoint, provenance=provenance, weak=weak, strong=strong)
        )
