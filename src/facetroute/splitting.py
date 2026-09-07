"""Deterministic, group-aware partitioning for route-trace experiments."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .traces import (
    RouteTrace,
    _load_traces_with_sha256,
    file_sha256,
    traces_sha256,
    write_traces,
)

_PARTITIONS = ("train", "calibration", "test")


@dataclass(frozen=True, slots=True)
class TracePartitions:
    """Three disjoint trace sets and their auditable split metadata."""

    train: tuple[RouteTrace, ...]
    calibration: tuple[RouteTrace, ...]
    test: tuple[RouteTrace, ...]
    seed: int
    group_by: str
    train_fraction: float
    calibration_fraction: float

    def __post_init__(self) -> None:
        _validate_trace_partitions(self)

    def traces(self, name: str) -> tuple[RouteTrace, ...]:
        if name == "train":
            return self.train
        if name == "calibration":
            return self.calibration
        if name == "test":
            return self.test
        raise ConfigurationError(f"unknown trace partition: {name}")


def _fraction(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a finite number in (0, 1)")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ConfigurationError(f"{name} must be a finite number in (0, 1)") from exc
    if not math.isfinite(number) or not 0 < number < 1:
        raise ConfigurationError(f"{name} must be a finite number in (0, 1)")
    return number


def _group_key(trace: RouteTrace, group_by: str) -> str:
    if group_by == "request_id":
        return trace.request.request_id
    if group_by == "user_id":
        if trace.request.user_id is None:
            raise ConfigurationError("group_by user_id requires every trace to have a user_id")
        return trace.request.user_id
    prefix = "metadata:"
    if group_by.startswith(prefix) and len(group_by) > len(prefix):
        key = group_by[len(prefix) :]
        value = trace.request.metadata.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigurationError(
                f"group_by {group_by} requires a non-empty string on every trace"
            )
        return value
    raise ConfigurationError("group_by must be request_id, user_id, or metadata:<field>")


def _validate_trace_partitions(partitions: TracePartitions) -> None:
    named = {name: partitions.traces(name) for name in _PARTITIONS}
    if any(not isinstance(partition, tuple) for partition in named.values()):
        raise ConfigurationError("trace partitions must be tuples")
    if any(not partition for partition in named.values()):
        raise ConfigurationError("train, calibration, and test partitions must be non-empty")
    if isinstance(partitions.seed, bool) or not isinstance(partitions.seed, int):
        raise ConfigurationError("split seed must be an integer")
    train_ratio = _fraction("train_fraction", partitions.train_fraction)
    calibration_ratio = _fraction("calibration_fraction", partitions.calibration_fraction)
    if train_ratio + calibration_ratio >= 1:
        raise ConfigurationError("train and calibration fractions must sum to less than 1")
    if not isinstance(partitions.group_by, str):
        raise ConfigurationError("group_by must be request_id, user_id, or metadata:<field>")

    seen_ids: set[str] = set()
    group_partitions: dict[str, str] = {}
    for name, partition in named.items():
        for trace in partition:
            if not isinstance(trace, RouteTrace):
                raise ConfigurationError("trace partitions must contain RouteTrace values")
            request_id = trace.request.request_id
            if request_id in seen_ids:
                raise ConfigurationError("trace partitions must have disjoint request ids")
            seen_ids.add(request_id)
            group = _group_key(trace, partitions.group_by)
            previous = group_partitions.setdefault(group, name)
            if previous != name:
                raise ConfigurationError(
                    f"group {group!r} occurs in both {previous} and {name} partitions"
                )

    object.__setattr__(partitions, "train_fraction", train_ratio)
    object.__setattr__(partitions, "calibration_fraction", calibration_ratio)


def split_traces(
    traces: tuple[RouteTrace, ...],
    *,
    seed: int = 17,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
    group_by: str = "request_id",
) -> TracePartitions:
    """Split traces without allowing a declared group to cross partitions.

    Groups are considered largest first, with a SHA-256 digest of the seed and
    group key providing a deterministic order among equal sizes. Each group is
    assigned where it minimizes squared record-count error while reserving
    enough groups to keep every partition non-empty. Source order is kept within
    each resulting file.
    """

    if not traces:
        raise ConfigurationError("trace splitting requires at least one trace")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError("split seed must be an integer")
    if not isinstance(group_by, str):
        raise ConfigurationError("group_by must be request_id, user_id, or metadata:<field>")
    train_ratio = _fraction("train_fraction", train_fraction)
    calibration_ratio = _fraction("calibration_fraction", calibration_fraction)
    if train_ratio + calibration_ratio >= 1:
        raise ConfigurationError("train and calibration fractions must sum to less than 1")
    groups: dict[str, list[RouteTrace]] = defaultdict(list)
    for trace in traces:
        groups[_group_key(trace, group_by)].append(trace)
    if len(groups) < 3:
        raise ConfigurationError("trace splitting requires at least three distinct groups")
    ordered_groups = sorted(
        groups,
        key=lambda key: (
            -len(groups[key]),
            hashlib.sha256(f"{seed}\0{key}".encode()).digest(),
            key,
        ),
    )
    targets = {
        "train": len(traces) * train_ratio,
        "calibration": len(traces) * calibration_ratio,
        "test": len(traces) * (1 - train_ratio - calibration_ratio),
    }
    assigned_groups: dict[str, str] = {}
    counts = dict.fromkeys(_PARTITIONS, 0)
    assigned_group_counts = dict.fromkeys(_PARTITIONS, 0)
    for index, group in enumerate(ordered_groups):
        empty = tuple(name for name in _PARTITIONS if assigned_group_counts[name] == 0)
        groups_after = len(ordered_groups) - index - 1
        candidates = empty if groups_after < len(empty) else _PARTITIONS
        size = len(groups[group])

        def allocation_key(candidate: str, group_size: int = size) -> tuple[float, float, int]:
            error = sum(
                (counts[name] + (group_size if name == candidate else 0) - targets[name]) ** 2
                for name in _PARTITIONS
            )
            remaining = targets[candidate] - counts[candidate]
            return error, -remaining, _PARTITIONS.index(candidate)

        name = min(candidates, key=allocation_key)
        assigned_groups[group] = name
        counts[name] += size
        assigned_group_counts[name] += 1
    members: dict[str, list[RouteTrace]] = {name: [] for name in _PARTITIONS}
    for trace in traces:
        members[assigned_groups[_group_key(trace, group_by)]].append(trace)
    return TracePartitions(
        train=tuple(members["train"]),
        calibration=tuple(members["calibration"]),
        test=tuple(members["test"]),
        seed=seed,
        group_by=group_by,
        train_fraction=train_ratio,
        calibration_fraction=calibration_ratio,
    )


def _manifest_partition(path: Path, traces: tuple[RouteTrace, ...]) -> dict[str, object]:
    return {
        "file": path.name,
        "records": len(traces),
        "canonical_sha256": traces_sha256(traces),
        "file_sha256": file_sha256(path),
    }


def _paths_collide(source: Path, output: Path) -> bool:
    try:
        if source.resolve() == output.resolve():
            return True
        return source.exists() and output.exists() and source.samefile(output)
    except (OSError, RuntimeError):
        return False


def _split_statistics(partitions: TracePartitions) -> dict[str, dict[str, float] | dict[str, int]]:
    total = sum(len(partitions.traces(name)) for name in _PARTITIONS)
    requested = {
        "train": partitions.train_fraction,
        "calibration": partitions.calibration_fraction,
        "test": 1 - partitions.train_fraction - partitions.calibration_fraction,
    }
    actual = {name: len(partitions.traces(name)) / total for name in _PARTITIONS}
    errors = {name: abs(actual[name] - requested[name]) for name in _PARTITIONS}
    group_counts = {
        name: len({_group_key(trace, partitions.group_by) for trace in partitions.traces(name)})
        for name in _PARTITIONS
    }
    return {
        "requested_fractions": requested,
        "actual_fractions": actual,
        "absolute_fraction_error": errors,
        "group_counts": group_counts,
    }


def write_trace_partitions(
    directory: str | Path,
    partitions: TracePartitions,
    *,
    source_path: str | Path,
    dataset_name: str,
    source_uri: str,
    license_name: str,
) -> dict[str, Any]:
    """Write three canonical JSONL files and a versioned provenance manifest."""

    text_fields = {
        "dataset_name": dataset_name,
        "source_uri": source_uri,
        "license": license_name,
    }
    for name, value in text_fields.items():
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"{name} must be a non-empty string")
    target = Path(directory)
    source = Path(source_path)
    paths = {name: target / f"{name}.jsonl" for name in _PARTITIONS}
    manifest_path = target / "manifest.json"
    declared_paths = [("source", source), *paths.items(), ("manifest", manifest_path)]
    for index, (left_name, left_path) in enumerate(declared_paths):
        for right_name, right_path in declared_paths[index + 1 :]:
            if _paths_collide(left_path, right_path):
                raise ConfigurationError(
                    "trace partition path collides: "
                    f"{left_name}={left_path} and {right_name}={right_path}"
                )

    source_traces, source_sha256 = _load_traces_with_sha256(source)
    expected = split_traces(
        source_traces,
        seed=partitions.seed,
        train_fraction=partitions.train_fraction,
        calibration_fraction=partitions.calibration_fraction,
        group_by=partitions.group_by,
    )
    mismatched = [
        name
        for name in _PARTITIONS
        if traces_sha256(partitions.traces(name)) != traces_sha256(expected.traces(name))
    ]
    if mismatched:
        raise ConfigurationError(
            "trace partitions do not match the declared source and deterministic split "
            f"configuration: {', '.join(mismatched)}"
        )

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(f"cannot create split directory {target}: {exc}") from exc
    for name, path in paths.items():
        write_traces(path, partitions.traces(name))
    partition_metadata: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        metadata = _manifest_partition(path, partitions.traces(name))
        if metadata["file_sha256"] != metadata["canonical_sha256"]:
            raise ConfigurationError(
                f"written {name} partition does not match its canonical trace records"
            )
        partition_metadata[name] = metadata
    statistics = _split_statistics(partitions)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "dataset": {
            **text_fields,
            "input_file_sha256": source_sha256,
            "canonical_sha256": traces_sha256(source_traces),
            "records": sum(len(partitions.traces(name)) for name in _PARTITIONS),
        },
        "split": {
            "algorithm": "sha256-size-aware-deficit-v2",
            "seed": partitions.seed,
            "group_by": partitions.group_by,
            **statistics,
        },
        "partitions": partition_metadata,
    }
    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError as exc:
        raise ConfigurationError(f"cannot write split manifest: {exc}") from exc
    return manifest
