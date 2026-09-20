"""Opt-in, resumable public-score generation through injected async providers.

This is an original, conservative MMLU/GSM8K workflow. It neither fetches
datasets nor claims to reproduce any benchmark's official prompt protocol.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import stat
import sys
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ._json import loads_strict
from .async_client import AsyncProviderRegistry, AsyncProviderTarget
from .benchmark_formats import BenchmarkExample, BenchmarkFormat, load_benchmark_examples_bytes
from .errors import ConfigurationError
from .providers import ProviderError, ProviderFailure

_PROTOCOL = "facet-public-score-v1"
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MMLU_LETTER = re.compile(r"\s*(?:Answer:\s*)?\(?([A-Z])\)?(?=$|[\s.:])", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")


def _bounded_text(value: object, name: str, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ConfigurationError(f"{name} must be a bounded non-empty string")
    try:
        value.encode("utf-8", "strict")
    except UnicodeError:
        raise ConfigurationError(f"{name} must be valid UTF-8") from None
    return value


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ConfigurationError("public-score state is not strict UTF-8 JSON") from None


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class PublicScoreProvenance:
    """Caller-declared license/source, pinned to the exact local input bytes."""

    source_uri: str
    license_id: str
    source_sha256: str

    def __post_init__(self) -> None:
        uri = _bounded_text(self.source_uri, "source_uri", 2048)
        try:
            parsed = urlsplit(uri)
        except ValueError:
            raise ConfigurationError("source_uri is invalid") from None
        if (
            parsed.scheme not in {"https", "http", "repository", "fixture"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or (parsed.scheme in {"https", "http"} and not parsed.hostname)
        ):
            raise ConfigurationError("source_uri must be a public URI without credentials or query")
        _bounded_text(self.license_id, "license_id", 128)
        if type(self.source_sha256) is not str or not _HEX_SHA256.fullmatch(self.source_sha256):
            raise ConfigurationError("source_sha256 must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class PublicScoreModel:
    """One explicit model binding plus a non-secret deployment revision."""

    target: AsyncProviderTarget
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.target, AsyncProviderTarget):
            raise ConfigurationError("public-score model requires an AsyncProviderTarget")
        _bounded_text(self.revision, "model revision", 256)


@dataclass(frozen=True, slots=True)
class PublicScoreLimits:
    max_source_bytes: int = 8 * 1024 * 1024
    max_records: int = 1000
    max_prompt_bytes: int = 32 * 1024
    max_response_bytes: int = 8 * 1024
    max_checkpoint_bytes: int = 32 * 1024 * 1024
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        ceilings = (
            ("max_source_bytes", self.max_source_bytes, 32 * 1024 * 1024),
            ("max_records", self.max_records, 10_000),
            ("max_prompt_bytes", self.max_prompt_bytes, 64 * 1024),
            ("max_response_bytes", self.max_response_bytes, 64 * 1024),
            ("max_checkpoint_bytes", self.max_checkpoint_bytes, 256 * 1024 * 1024),
        )
        for name, value, maximum in ceilings:
            if type(value) is not int or not 1 <= value <= maximum:
                raise ConfigurationError(f"{name} must be a positive bounded integer")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 600
        ):
            raise ConfigurationError("timeout_seconds must be within (0, 600]")


@dataclass(frozen=True, slots=True)
class PublicScoreProgress:
    manifest_sha256: str
    source_uri: str
    license_id: str
    source_file_sha256: str
    source_canonical_sha256: str
    benchmark_format: BenchmarkFormat
    completed_calls: int
    total_calls: int
    finished: bool
    weak_correct: int | None
    strong_correct: int | None
    weak_accuracy: float | None
    strong_accuracy: float | None


@dataclass(frozen=True, slots=True)
class _Task:
    key: str
    request_sha256: str
    example: BenchmarkExample
    role: str
    model_id: str
    upstream_model: str
    payload: dict[str, Any]


def _number(value: str) -> Decimal | None:
    text = value.rsplit("####", 1)[-1]
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None


def _score(example: BenchmarkExample, response: str) -> tuple[str, bool]:
    if example.format is BenchmarkFormat.MMLU:
        match = _MMLU_LETTER.match(response)
        prediction = match.group(1).upper() if match else ""
        truth = (
            example.answer
            if isinstance(example.answer, int)
            else example.choices.index(example.answer)
        )
        return prediction, prediction == chr(ord("A") + truth)
    prediction_number = _number(response)
    truth_number = _number(str(example.answer))
    prediction = str(prediction_number) if prediction_number is not None else ""
    return prediction, prediction_number is not None and prediction_number == truth_number


def _prompt(example: BenchmarkExample) -> tuple[str, int]:
    if example.format is BenchmarkFormat.MMLU:
        if len(example.choices) > 26 or len(set(example.choices)) != len(example.choices):
            raise ConfigurationError("MMLU choices must be unique and contain at most 26 options")
        options = "\n".join(
            f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(example.choices)
        )
        question = f"Question: {example.prompt}\n{options}\nAnswer with one option letter."
        if example.few_shot_context is not None:
            question = f"{example.few_shot_context}\n\n{question}"
        return question, 8
    if example.format is BenchmarkFormat.GSM8K:
        if _number(str(example.answer)) is None:
            raise ConfigurationError("GSM8K reference answer must contain a finite decimal number")
        question = f"Question: {example.prompt}\nGive the final number after ####."
        if example.few_shot_context is not None:
            question = f"{example.few_shot_context}\n\n{question}"
        return question, 256
    raise ConfigurationError("public-score generation supports only MMLU and GSM8K")


def _tasks(
    examples: tuple[BenchmarkExample, ...],
    weak: PublicScoreModel,
    strong: PublicScoreModel,
    limits: PublicScoreLimits,
) -> tuple[_Task, ...]:
    result: list[_Task] = []
    for index, example in enumerate(examples):
        prompt, max_tokens = _prompt(example)
        if len(prompt.encode("utf-8", "strict")) > limits.max_prompt_bytes:
            raise ConfigurationError("public-score prompt exceeds byte limit")
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        for role, model in (("weak", weak), ("strong", strong)):
            key = _digest({"index": index, "role": role, "id": example.example_id})
            request = {**payload, "model": model.target.upstream_model, "stream": False}
            result.append(
                _Task(
                    key,
                    _digest(request),
                    example,
                    role,
                    model.target.model_id,
                    model.target.upstream_model,
                    payload,
                )
            )
    return tuple(result)


def _completion_text(response: Mapping[str, Any], expected_model: str, limit: int) -> str:
    if response.get("model") != expected_model:
        raise ProviderError(ProviderFailure.MALFORMED)
    try:
        choice = response["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ProviderError(ProviderFailure.MALFORMED) from None
    if type(text) is not str:
        raise ProviderError(ProviderFailure.MALFORMED)
    if len(text) > limit or not text.strip():
        raise ProviderError(ProviderFailure.MALFORMED)
    try:
        encoded = text.encode("utf-8", "strict")
    except UnicodeError:
        raise ProviderError(ProviderFailure.MALFORMED) from None
    if len(encoded) > limit:
        raise ProviderError(ProviderFailure.MALFORMED)
    return text


def _manifest(
    examples: tuple[BenchmarkExample, ...],
    provenance: PublicScoreProvenance,
    weak: PublicScoreModel,
    strong: PublicScoreModel,
    limits: PublicScoreLimits,
) -> dict[str, Any]:
    return {
        "protocol": _PROTOCOL,
        "source": {
            "uri": provenance.source_uri,
            "license": provenance.license_id,
            "file_sha256": provenance.source_sha256,
            "canonical_sha256": _digest([example.to_dict() for example in examples]),
        },
        "format": examples[0].format.value,
        "records": len(examples),
        "models": [
            {
                "role": role,
                "model_id": model.target.model_id,
                "upstream_model": model.target.upstream_model,
                "revision": model.revision,
            }
            for role, model in (("weak", weak), ("strong", strong))
        ],
        "limits": {
            "max_source_bytes": limits.max_source_bytes,
            "max_records": limits.max_records,
            "max_prompt_bytes": limits.max_prompt_bytes,
            "max_response_bytes": limits.max_response_bytes,
            "max_checkpoint_bytes": limits.max_checkpoint_bytes,
            "timeout_seconds": limits.timeout_seconds,
        },
    }


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Non-blocking inter-process lock; the harmless lock file may persist."""

    lock_path = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    except FileExistsError:
        try:
            before = lock_path.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != 1:
                raise ConfigurationError("public-score checkpoint lock file is invalid")
            flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(lock_path, flags)
            try:
                opened = os.fstat(descriptor)
                after = lock_path.lstat()
                if (
                    not stat.S_ISREG(after.st_mode)
                    or after.st_nlink != 1
                    or after.st_size != 1
                    or (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
                    or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                ):
                    raise ConfigurationError("public-score checkpoint lock file is invalid")
            except BaseException:
                os.close(descriptor)
                raise
        except OSError:
            raise ConfigurationError("cannot open public-score checkpoint lock") from None
    except OSError:
        raise ConfigurationError("cannot open public-score checkpoint lock") from None
    else:
        try:
            if os.write(descriptor, b"0") != 1:
                raise OSError("short lock initialization")
        except OSError:
            created = os.fstat(descriptor)
            os.close(descriptor)
            try:
                current = lock_path.lstat()
                if (
                    stat.S_ISREG(current.st_mode)
                    and current.st_nlink == 1
                    and (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino)
                ):
                    lock_path.unlink()
            except OSError:
                pass  # Preserve the original error; a leftover lock fails closed.
            raise ConfigurationError("cannot initialize public-score checkpoint lock") from None
    acquired = False
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            raise ConfigurationError("public-score checkpoint is already in use") from None
        yield
    finally:
        try:
            if acquired:
                if sys.platform == "win32":
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _sealed(state: dict[str, Any]) -> bytes:
    content = {key: value for key, value in state.items() if key != "sha256"}
    return _canonical({**content, "sha256": _digest(content)}) + b"\n"


def _write(path: Path, state: dict[str, Any], limit: int) -> None:
    raw = _sealed(state)
    if len(raw) > limit:
        raise ConfigurationError("public-score checkpoint exceeds byte limit")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    created = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except OSError:
        raise ConfigurationError("cannot write public-score checkpoint") from None
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def _read(path: Path, limit: int) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
    except OSError:
        raise ConfigurationError("cannot read public-score checkpoint") from None
    if len(raw) > limit:
        raise ConfigurationError("public-score checkpoint exceeds byte limit")
    try:
        state = loads_strict(raw)
    except ValueError:
        raise ConfigurationError("public-score checkpoint is not strict JSON") from None
    if not isinstance(state, dict) or set(state) != {"manifest", "results", "pending", "sha256"}:
        raise ConfigurationError("public-score checkpoint schema mismatch")
    content = {key: value for key, value in state.items() if key != "sha256"}
    if type(state["sha256"]) is not str or not _HEX_SHA256.fullmatch(state["sha256"]):
        raise ConfigurationError("public-score checkpoint checksum mismatch")
    if not hmac.compare_digest(state["sha256"], _digest(content)):
        raise ConfigurationError("public-score checkpoint checksum mismatch")
    return state


def _validated_state(
    state: dict[str, Any], manifest: dict[str, Any], tasks: tuple[_Task, ...], limit: int
) -> list[dict[str, Any]]:
    if _canonical(state["manifest"]) != _canonical(manifest):
        raise ConfigurationError("public-score checkpoint manifest mismatch")
    records = state["results"]
    if not isinstance(records, list) or len(records) > len(tasks):
        raise ConfigurationError("public-score checkpoint result sequence is invalid")
    for index, record in enumerate(records):
        task = tasks[index]
        if not isinstance(record, dict) or type(record.get("response")) is not str:
            raise ConfigurationError("public-score checkpoint result is invalid")
        response = record["response"]
        if len(response.encode("utf-8", "strict")) > limit:
            raise ConfigurationError("public-score checkpoint response exceeds byte limit")
        attempt = record.get("attempt")
        if type(attempt) is not int or attempt < 1:
            raise ConfigurationError("public-score checkpoint attempt is invalid")
        prediction, correct = _score(task.example, response)
        expected = {
            "key": task.key,
            "request_sha256": task.request_sha256,
            "reported_model": task.upstream_model,
            "response": response,
            "prediction": prediction,
            "correct": correct,
            "attempt": attempt,
        }
        if _canonical(record) != _canonical(expected):
            raise ConfigurationError("public-score checkpoint result is inconsistent")
    pending = state["pending"]
    if pending is not None:
        if len(records) == len(tasks) or not isinstance(pending, dict):
            raise ConfigurationError("public-score checkpoint pending record is invalid")
        attempt = pending.get("attempt")
        if (
            set(pending) != {"key", "attempt"}
            or pending.get("key") != tasks[len(records)].key
            or type(attempt) is not int
            or attempt < 1
        ):
            raise ConfigurationError("public-score checkpoint pending record is invalid")
    return records


def _progress(
    manifest: dict[str, Any], records: list[dict[str, Any]], tasks: tuple[_Task, ...]
) -> PublicScoreProgress:
    finished = len(records) == len(tasks)
    weak_correct = sum(bool(row["correct"]) for row in records[::2]) if finished else None
    strong_correct = sum(bool(row["correct"]) for row in records[1::2]) if finished else None
    count = len(tasks) // 2
    return PublicScoreProgress(
        manifest_sha256=_digest(manifest),
        source_uri=manifest["source"]["uri"],
        license_id=manifest["source"]["license"],
        source_file_sha256=manifest["source"]["file_sha256"],
        source_canonical_sha256=manifest["source"]["canonical_sha256"],
        benchmark_format=BenchmarkFormat(manifest["format"]),
        completed_calls=len(records),
        total_calls=len(tasks),
        finished=finished,
        weak_correct=weak_correct,
        strong_correct=strong_correct,
        weak_accuracy=weak_correct / count if weak_correct is not None else None,
        strong_accuracy=strong_correct / count if strong_correct is not None else None,
    )


async def run_public_scores(
    source_path: str | Path,
    checkpoint_path: str | Path,
    *,
    provenance: PublicScoreProvenance,
    weak: PublicScoreModel,
    strong: PublicScoreModel,
    limits: PublicScoreLimits | None = None,
    max_calls: int = 2000,
    allow_ambiguous_retry: bool = False,
) -> PublicScoreProgress:
    """Generate bounded weak/strong outcomes, safely resuming prior results.

    A pending request may have reached its provider. Resending it requires an
    explicit `allow_ambiguous_retry=True` for this invocation.
    """

    if not isinstance(provenance, PublicScoreProvenance):
        raise ConfigurationError("provenance must be a PublicScoreProvenance")
    if not isinstance(weak, PublicScoreModel) or not isinstance(strong, PublicScoreModel):
        raise ConfigurationError("weak and strong must be PublicScoreModel instances")
    if limits is None:
        limits = PublicScoreLimits()
    if not isinstance(limits, PublicScoreLimits):
        raise ConfigurationError("limits must be PublicScoreLimits")
    if weak.target.model_id == strong.target.model_id:
        raise ConfigurationError("weak and strong model IDs must differ")
    if type(max_calls) is not int or not 0 <= max_calls <= 20_000:
        raise ConfigurationError("max_calls must be between 0 and 20000")
    if type(allow_ambiguous_retry) is not bool:
        raise ConfigurationError("allow_ambiguous_retry must be a boolean")
    source = Path(source_path)
    checkpoint = Path(checkpoint_path)
    lock_path = checkpoint.with_name(checkpoint.name + ".lock")
    if source.resolve() in {checkpoint.resolve(), lock_path.resolve()}:
        raise ConfigurationError("public-score input and checkpoint paths must differ")
    try:
        with source.open("rb") as handle:
            raw = handle.read(limits.max_source_bytes + 1)
    except OSError:
        raise ConfigurationError("cannot read public-score input") from None
    if len(raw) > limits.max_source_bytes:
        raise ConfigurationError("public-score input exceeds byte limit")
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != provenance.source_sha256:
        raise ConfigurationError("public-score input SHA-256 does not match provenance")
    examples = load_benchmark_examples_bytes(
        raw, max_bytes=limits.max_source_bytes, max_records=limits.max_records
    )
    if examples[0].format not in {BenchmarkFormat.MMLU, BenchmarkFormat.GSM8K}:
        raise ConfigurationError("public-score generation supports only MMLU and GSM8K")
    tasks = _tasks(examples, weak, strong, limits)
    manifest = _manifest(examples, provenance, weak, strong, limits)
    registry = AsyncProviderRegistry((weak.target, strong.target))
    try:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ConfigurationError("cannot create public-score checkpoint directory") from None
    with _locked(checkpoint):
        if checkpoint.exists():
            state = _read(checkpoint, limits.max_checkpoint_bytes)
        else:
            state = {"manifest": manifest, "results": [], "pending": None}
            _write(checkpoint, state, limits.max_checkpoint_bytes)
        records = _validated_state(state, manifest, tasks, limits.max_response_bytes)
        if state["pending"] is not None and not allow_ambiguous_retry:
            raise ConfigurationError("public-score checkpoint has an ambiguous pending call")
        calls = 0
        while len(records) < len(tasks) and calls < max_calls:
            task = tasks[len(records)]
            pending = state["pending"]
            attempt = pending["attempt"] + 1 if pending is not None else 1
            marked = {**state, "pending": {"key": task.key, "attempt": attempt}}
            _write(checkpoint, marked, limits.max_checkpoint_bytes)
            state = marked
            try:
                async with asyncio.timeout(limits.timeout_seconds):
                    response = await registry.complete(
                        task.model_id,
                        deepcopy(task.payload),
                        timeout_seconds=limits.timeout_seconds,
                    )
                expected_model = task.upstream_model
                text = _completion_text(response, expected_model, limits.max_response_bytes)
                prediction, correct = _score(task.example, text)
            except ProviderError as error:
                if error.retry_safe:
                    cleared = {**state, "pending": None}
                    _write(checkpoint, cleared, limits.max_checkpoint_bytes)
                    state = cleared
                raise
            except TimeoutError:
                raise ProviderError(ProviderFailure.TIMEOUT) from None
            committed = {
                **state,
                "results": [
                    *records,
                    {
                        "key": task.key,
                        "request_sha256": task.request_sha256,
                        "reported_model": expected_model,
                        "response": text,
                        "prediction": prediction,
                        "correct": correct,
                        "attempt": attempt,
                    },
                ],
                "pending": None,
            }
            _write(checkpoint, committed, limits.max_checkpoint_bytes)
            state = committed
            records = state["results"]
            calls += 1
        return _progress(manifest, records, tasks)
