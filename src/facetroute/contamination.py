"""Offline cosine-similarity audit for declared train/evaluation embeddings.

This does not generate embeddings, read prompts, or contact a provider.  Both
sets must declare the same model identity and vector dimension.  The output
contains only record identifiers and aggregate similarity evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._json import loads_strict
from .errors import ConfigurationError

EMBEDDING_SCHEMA = "facet-embedding-set-v1"
AUDIT_SCHEMA = "facet-contamination-audit-v1"
_MAX_FILE_BYTES = 128 * 1024 * 1024
_MAX_RECORDS = 20_000
_MAX_DIMENSION = 2_048
_MAX_COORDINATES = 8_000_000
_MAX_WORK = 100_000_000
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class ContaminationHit:
    evaluation_id: str
    training_id: str
    cosine_similarity: float


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    schema: str
    model_id: str
    model_revision: str
    dimension: int
    threshold: float
    training_sha256: str
    evaluation_sha256: str
    training_count: int
    evaluation_count: int
    coordinate_work: int
    hits: tuple[ContaminationHit, ...]


@dataclass(frozen=True, slots=True)
class _EmbeddingSet:
    model_id: str
    model_revision: str
    dimension: int
    file_sha256: str
    records: tuple[tuple[str, tuple[float, ...]], ...]


def _snapshot(path: str | Path, label: str) -> bytes:
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(_MAX_FILE_BYTES + 1)
    except OSError:
        raise ConfigurationError(f"cannot read {label} embedding set") from None
    if len(raw) > _MAX_FILE_BYTES:
        raise ConfigurationError(f"{label} embedding set exceeds byte limit")
    return raw


def _embedding_set(raw: bytes, label: str) -> _EmbeddingSet:
    try:
        data = loads_strict(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} embedding set is not strict JSON: {exc}") from exc
    if (
        not isinstance(data, dict)
        or set(data) != {"schema", "model_id", "model_revision", "dimension", "records"}
        or data["schema"] != EMBEDDING_SCHEMA
    ):
        raise ConfigurationError(f"{label} embedding set schema is invalid")
    model_id = data["model_id"]
    revision = data["model_revision"]
    dimension = data["dimension"]
    records = data["records"]
    if (
        not isinstance(model_id, str)
        or not _ID.fullmatch(model_id)
        or not isinstance(revision, str)
        or not _ID.fullmatch(revision)
        or isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or not 1 <= dimension <= _MAX_DIMENSION
        or not isinstance(records, list)
        or not 1 <= len(records) <= _MAX_RECORDS
        or len(records) * dimension > _MAX_COORDINATES
    ):
        raise ConfigurationError(f"{label} embedding set metadata or size is invalid")
    seen: set[str] = set()
    normalized: list[tuple[str, tuple[float, ...]]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"id", "embedding"}:
            raise ConfigurationError(f"{label} embedding record schema is invalid")
        record_id = record["id"]
        vector = record["embedding"]
        if (
            not isinstance(record_id, str)
            or not _ID.fullmatch(record_id)
            or record_id in seen
            or not isinstance(vector, list)
            or len(vector) != dimension
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector
            )
        ):
            raise ConfigurationError(f"{label} embedding record is invalid")
        try:
            values = tuple(float(value) for value in vector)
        except (OverflowError, ValueError) as exc:
            raise ConfigurationError(f"{label} embedding coordinate is invalid") from exc
        if any(not math.isfinite(value) for value in values):
            raise ConfigurationError(f"{label} embedding coordinate is not finite")
        # Scale before taking the norm: finite large coordinates can make a
        # direct hypot overflow, while tiny subnormal vectors remain valid.
        scale = max(abs(value) for value in values)
        if scale == 0:
            raise ConfigurationError(f"{label} embedding vector has invalid norm")
        scaled = tuple(value / scale for value in values)
        norm = math.hypot(*scaled)
        normalized.append((record_id, tuple(value / norm for value in scaled)))
        seen.add(record_id)
    return _EmbeddingSet(
        model_id, revision, dimension, hashlib.sha256(raw).hexdigest(), tuple(normalized)
    )


def audit_contamination(
    training_path: str | Path,
    evaluation_path: str | Path,
    *,
    threshold: float = 0.95,
) -> ContaminationReport:
    """Flag each evaluation vector whose nearest training vector meets threshold.

    Ties select the lexicographically smallest training identifier.  The work
    bound is checked before any pairwise comparisons.  A hit is *similarity
    evidence*, not proof of train-set contamination or benchmark leakage.
    """

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ConfigurationError("threshold must be a finite number between -1 and 1")
    try:
        cutoff = float(threshold)
    except OverflowError:
        raise ConfigurationError("threshold must be a finite number between -1 and 1") from None
    if not math.isfinite(cutoff) or not -1 <= cutoff <= 1:
        raise ConfigurationError("threshold must be a finite number between -1 and 1")
    try:
        same_path = Path(training_path).resolve() == Path(evaluation_path).resolve()
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError("cannot resolve embedding paths") from exc
    if same_path:
        raise ConfigurationError("training and evaluation paths must differ")
    training = _embedding_set(_snapshot(training_path, "training"), "training")
    evaluation = _embedding_set(_snapshot(evaluation_path, "evaluation"), "evaluation")
    if (
        training.model_id != evaluation.model_id
        or training.model_revision != evaluation.model_revision
        or training.dimension != evaluation.dimension
    ):
        raise ConfigurationError("embedding model identity or dimension differs")
    work = len(training.records) * len(evaluation.records) * training.dimension
    if work > _MAX_WORK:
        raise ConfigurationError("pairwise similarity work limit exceeded")
    hits: list[ContaminationHit] = []
    ordered_training = sorted(training.records)
    for evaluation_id, vector in evaluation.records:
        best_id = ""
        best_similarity = -math.inf
        for training_id, candidate in ordered_training:
            # The dot product of identical normalized vectors can round just
            # below one (for example [1, 1]). Preserve the inclusive
            # threshold=1 contract without admitting merely near neighbors.
            similarity = (
                1.0
                if vector == candidate
                else math.fsum(left * right for left, right in zip(vector, candidate, strict=True))
            )
            # Numerical dot error may exceed the mathematical [-1, 1] range
            # by a few ulps even though both operands were normalized.
            similarity = max(-1.0, min(1.0, similarity))
            if similarity > best_similarity:
                best_id, best_similarity = training_id, similarity
        if best_similarity >= cutoff:
            hits.append(ContaminationHit(evaluation_id, best_id, best_similarity))
    return ContaminationReport(
        AUDIT_SCHEMA,
        training.model_id,
        training.model_revision,
        training.dimension,
        cutoff,
        training.file_sha256,
        evaluation.file_sha256,
        len(training.records),
        len(evaluation.records),
        work,
        tuple(hits),
    )


def contamination_json(report: ContaminationReport) -> bytes:
    """Encode an aggregate, strict, versioned report with no vector values."""

    payload: dict[str, Any] = {
        "schema": report.schema,
        "model_id": report.model_id,
        "model_revision": report.model_revision,
        "dimension": report.dimension,
        "threshold": report.threshold,
        "training_sha256": report.training_sha256,
        "evaluation_sha256": report.evaluation_sha256,
        "training_count": report.training_count,
        "evaluation_count": report.evaluation_count,
        "coordinate_work": report.coordinate_work,
        "hits": [
            {
                "evaluation_id": hit.evaluation_id,
                "training_id": hit.training_id,
                "cosine_similarity": hit.cosine_similarity,
            }
            for hit in report.hits
        ],
    }
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
