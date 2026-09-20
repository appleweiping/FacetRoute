"""Read-only, provenance-pinned analysis of completed public-score checkpoints.

The output deliberately contains no prompts, gold answers, or model responses.
It is an independent local protocol, not an official benchmark submission.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._json import loads_strict
from .async_client import AsyncProviderTarget
from .benchmark_formats import BenchmarkExample, load_benchmark_examples_bytes
from .errors import ConfigurationError
from .public_scores import (
    PublicScoreLimits,
    PublicScoreModel,
    PublicScoreProvenance,
    _digest,
    _manifest,
    _tasks,
    _validated_state,
)

_MAX_SOURCE_BYTES = 32 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
_MAX_SCORE_BYTES = 64 * 1024 * 1024
_MAX_SCORE_LINE_BYTES = _MAX_SOURCE_BYTES + 1024
_MAX_EXCLUSION_BYTES = 2 * 1024 * 1024
_MAX_EXCLUSION_LINE_BYTES = 256
_HEX = frozenset("0123456789abcdef")
AUDIT_SCHEMA = "facet-public-score-audit-v1"


class _OfflineProvider:
    """Identity placeholder; analysis never asks it to execute a request."""

    async def complete(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> Mapping[str, Any]:
        raise ConfigurationError("public-score audit cannot contact a provider")

    def stream(
        self, payload: Mapping[str, Any], *, model: str, timeout_seconds: float
    ) -> AsyncIterator[Mapping[str, Any]]:
        raise ConfigurationError("public-score audit cannot contact a provider")


def _snapshot(path: str | Path, limit: int, label: str) -> bytes:
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(limit + 1)
    except OSError:
        raise ConfigurationError(f"cannot read {label}") from None
    if len(raw) > limit:
        raise ConfigurationError(f"{label} exceeds byte limit")
    return raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _models(manifest: dict[str, Any]) -> tuple[PublicScoreModel, PublicScoreModel]:
    raw = manifest.get("models")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ConfigurationError("public-score audit model manifest is invalid")
    result = []
    for role, item in zip(("weak", "strong"), raw, strict=True):
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "model_id", "upstream_model", "revision"}
            or item["role"] != role
        ):
            raise ConfigurationError("public-score audit model manifest is invalid")
        result.append(
            PublicScoreModel(
                AsyncProviderTarget(item["model_id"], item["upstream_model"], _OfflineProvider()),
                item["revision"],
            )
        )
    if result[0].target.model_id == result[1].target.model_id:
        raise ConfigurationError("public-score audit model IDs must differ")
    return result[0], result[1]


def _archive(
    source_path: str | Path, checkpoint_path: str | Path
) -> tuple[tuple[BenchmarkExample, ...], tuple[tuple[bool, bool], ...], dict[str, Any]]:
    if Path(source_path).resolve() == Path(checkpoint_path).resolve():
        raise ConfigurationError("public-score audit source and checkpoint must differ")
    raw_source = _snapshot(source_path, _MAX_SOURCE_BYTES, "public-score source")
    raw_checkpoint = _snapshot(checkpoint_path, _MAX_CHECKPOINT_BYTES, "public-score checkpoint")
    # The core reader validates strict JSON and the checksum. The files were
    # snapshotted above; validate those exact bytes even if a writer replaces a
    # path between the reads.
    state = _read_bytes(raw_checkpoint)
    manifest = state["manifest"]
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"protocol", "source", "format", "records", "models", "limits"}
        or manifest["protocol"] != "facet-public-score-v1"
    ):
        raise ConfigurationError("public-score audit manifest is invalid")
    source = manifest["source"]
    bounds = manifest["limits"]
    if (
        not isinstance(source, dict)
        or set(source) != {"uri", "license", "file_sha256", "canonical_sha256"}
        or not isinstance(bounds, dict)
        or set(bounds)
        != {
            "max_source_bytes",
            "max_records",
            "max_prompt_bytes",
            "max_response_bytes",
            "max_checkpoint_bytes",
            "timeout_seconds",
        }
    ):
        raise ConfigurationError("public-score audit manifest is invalid")
    provenance = PublicScoreProvenance(source["uri"], source["license"], source["file_sha256"])
    if _sha(raw_source) != provenance.source_sha256:
        raise ConfigurationError("public-score audit source SHA-256 mismatch")
    limits = PublicScoreLimits(**bounds)
    if (
        len(raw_source) > limits.max_source_bytes
        or len(raw_checkpoint) > limits.max_checkpoint_bytes
    ):
        raise ConfigurationError("public-score audit declared limit exceeded")
    examples = load_benchmark_examples_bytes(
        raw_source, max_bytes=limits.max_source_bytes, max_records=limits.max_records
    )
    weak, strong = _models(manifest)
    expected = _manifest(examples, provenance, weak, strong, limits)
    if _digest(expected) != _digest(manifest):
        raise ConfigurationError("public-score audit manifest mismatch")
    tasks = _tasks(examples, weak, strong, limits)
    records = _validated_state(state, manifest, tasks, limits.max_response_bytes)
    if state["pending"] is not None or len(records) != len(tasks):
        raise ConfigurationError("public-score audit requires a completed checkpoint")
    outcomes = tuple(
        (records[index]["correct"], records[index + 1]["correct"])
        for index in range(0, len(records), 2)
    )
    return (
        examples,
        outcomes,
        {
            "source_uri": provenance.source_uri,
            "license_id": provenance.license_id,
            "source_file_sha256": provenance.source_sha256,
            "source_canonical_sha256": source["canonical_sha256"],
            "checkpoint_file_sha256": _sha(raw_checkpoint),
            "checkpoint_manifest_sha256": _digest(manifest),
            "format": manifest["format"],
            "models": manifest["models"],
        },
    )


def _read_bytes(raw: bytes) -> dict[str, Any]:
    # Reuse the core validator without reopening the mutable path.
    try:
        state = loads_strict(raw)
    except ValueError:
        raise ConfigurationError("public-score checkpoint is not strict JSON") from None
    if not isinstance(state, dict) or set(state) != {"manifest", "results", "pending", "sha256"}:
        raise ConfigurationError("public-score checkpoint schema mismatch")
    seal = state["sha256"]
    if type(seal) is not str or len(seal) != 64 or any(char not in _HEX for char in seal):
        raise ConfigurationError("public-score checkpoint checksum mismatch")
    from hmac import compare_digest

    if not compare_digest(
        seal, _digest({key: value for key, value in state.items() if key != "sha256"})
    ):
        raise ConfigurationError("public-score checkpoint checksum mismatch")
    return state


def _jsonl(raw: bytes, label: str, maximum: int, max_line_bytes: int) -> tuple[dict[str, Any], ...]:
    if not raw or not raw.endswith(b"\n"):
        raise ConfigurationError(f"{label} must be non-empty newline-terminated JSONL")
    lines = raw.splitlines()
    if len(lines) > maximum:
        raise ConfigurationError(f"{label} exceeds record limit")
    parsed = []
    for line in lines:
        if not line or len(line) > max_line_bytes:
            raise ConfigurationError(f"{label} contains an empty or oversized line")
        try:
            item = loads_strict(line)
        except ValueError:
            raise ConfigurationError(f"{label} contains invalid strict JSON") from None
        if not isinstance(item, dict):
            raise ConfigurationError(f"{label} rows must be objects")
        parsed.append(item)
    return tuple(parsed)


def _scores(raw: bytes, examples: tuple[BenchmarkExample, ...]) -> dict[str, float]:
    parsed = _jsonl(raw, "public-score route scores", len(examples), _MAX_SCORE_LINE_BYTES)
    scores = {}
    for item in parsed:
        if set(item) != {"id", "strong_win_rate"} or type(item["id"]) is not str:
            raise ConfigurationError("route score rows require id and strong_win_rate")
        value = item["strong_win_rate"]
        if type(value) not in {int, float} or not 0 <= value <= 1 or not math.isfinite(value):
            raise ConfigurationError("strong_win_rate must be finite in [0, 1]")
        if item["id"] in scores:
            raise ConfigurationError("route score IDs must be unique")
        scores[item["id"]] = float(value)
    if set(scores) != {example.example_id for example in examples}:
        raise ConfigurationError("route score IDs must match all source records exactly")
    return scores


def _exclusions(raw: bytes, examples: tuple[BenchmarkExample, ...]) -> set[str]:
    parsed = _jsonl(raw, "public-score exclusions", len(examples), _MAX_EXCLUSION_LINE_BYTES)
    hashes = set()
    for item in parsed:
        value = item.get("prompt_sha256")
        if (
            set(item) != {"prompt_sha256"}
            or type(value) is not str
            or len(value) != 64
            or any(char not in _HEX for char in value)
        ):
            raise ConfigurationError("exclusions require lowercase prompt_sha256")
        if value in hashes:
            raise ConfigurationError("exclusion prompt digests must be unique")
        hashes.add(value)
    found = {_sha(example.prompt.encode("utf-8")) for example in examples}
    if not hashes <= found:
        raise ConfigurationError("exclusion digest does not match the source")
    return hashes


@dataclass(frozen=True, slots=True)
class PublicScoreAudit:
    manifest: dict[str, Any]
    baselines: dict[str, float]
    domains: dict[str, dict[str, float | int]]
    points: tuple[dict[str, float | int | None], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AUDIT_SCHEMA,
            "manifest": self.manifest,
            "baselines": self.baselines,
            "domains": self.domains,
            "points": list(self.points),
        }


def audit_public_scores(
    source_path: str | Path,
    checkpoint_path: str | Path,
    route_scores_path: str | Path,
    *,
    exclusions_path: str | Path | None = None,
) -> PublicScoreAudit:
    """Analyze complete weak/strong outcomes against a pinned local score file.

    A weak-only endpoint and every unique observed threshold are evaluated.
    The call-budget oracle uses realized outcomes, not a deployable router.
    """

    paths = [Path(source_path), Path(checkpoint_path), Path(route_scores_path)]
    if exclusions_path is not None:
        paths.append(Path(exclusions_path))
    if len({path.resolve() for path in paths}) != len(paths):
        raise ConfigurationError("public-score audit input paths must differ")
    examples, outcomes, provenance = _archive(source_path, checkpoint_path)
    score_raw = _snapshot(route_scores_path, _MAX_SCORE_BYTES, "public-score route scores")
    scores = _scores(score_raw, examples)
    exclusion_raw = (
        _snapshot(exclusions_path, _MAX_EXCLUSION_BYTES, "public-score exclusions")
        if exclusions_path is not None
        else None
    )
    excluded = _exclusions(exclusion_raw, examples) if exclusion_raw is not None else set()
    kept = [
        (example, weak, strong, scores[example.example_id])
        for example, (weak, strong) in zip(examples, outcomes, strict=True)
        if _sha(example.prompt.encode("utf-8")) not in excluded
    ]
    if not kept:
        raise ConfigurationError("public-score audit excludes every source record")
    n = len(kept)
    weak_total = sum(weak for _, weak, _, _ in kept)
    strong_total = sum(strong for _, _, strong, _ in kept)
    domain_groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for example, weak, strong, _ in kept:
        domain_groups[example.category or "(unspecified)"].append((weak, strong))
    domains = {
        name: {
            "records": len(rows),
            "weak_accuracy": sum(weak for weak, _ in rows) / len(rows),
            "strong_accuracy": sum(strong for _, strong in rows) / len(rows),
        }
        for name, rows in sorted(domain_groups.items())
    }
    gains = sorted((int(strong) - int(weak) for _, weak, strong, _ in kept), reverse=True)
    oracle_correct = [weak_total]
    for gain in gains:
        oracle_correct.append(oracle_correct[-1] + gain)
    by_score: dict[float, list[tuple[bool, bool]]] = defaultdict(list)
    for _, weak, strong, score in kept:
        by_score[score].append((weak, strong))

    def point(threshold: float | None, calls: int, correct: int) -> dict[str, float | int | None]:
        return {
            "threshold": threshold,
            "strong_calls": calls,
            "strong_fraction": calls / n,
            "accuracy": correct / n,
            "oracle_accuracy_at_same_calls": oracle_correct[calls] / n,
            "regret_to_oracle": (oracle_correct[calls] - correct) / n,
        }

    descending = []
    calls = 0
    correct = weak_total
    for threshold in sorted(by_score, reverse=True):
        group = by_score[threshold]
        calls += len(group)
        correct += sum(int(strong) - int(weak) for weak, strong in group)
        descending.append(point(threshold, calls, correct))
    points = [point(None, 0, weak_total), *reversed(descending)]
    manifest = {
        **provenance,
        "route_scores_file_sha256": _sha(score_raw),
        "exclusions_file_sha256": _sha(exclusion_raw) if exclusion_raw is not None else None,
        "source_records": len(examples),
        "excluded_records": len(examples) - n,
        "evaluated_records": n,
        "threshold_rule": "strong iff score >= threshold; null means weak-only",
        "score_meaning": "caller-supplied strong_win_rate; no calibration asserted",
        "official_benchmark_parity": False,
    }
    return PublicScoreAudit(
        manifest,
        {"weak_accuracy": weak_total / n, "strong_accuracy": strong_total / n},
        domains,
        tuple(points),
    )


def audit_json(audit: PublicScoreAudit) -> bytes:
    """Portable, deterministic aggregate-only JSON output."""

    return (
        json.dumps(audit.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
