"""Join a declared embedding audit to benchmark IDs and prompt exclusions.

This is an offline screening workflow, not evidence that vectors really encode
the named prompts or that a model's training data contained those prompts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark_formats import BenchmarkFormat, load_benchmark_examples_bytes
from .contamination import (
    ContaminationReport,
    _embedding_set,
    _snapshot,
    audit_contamination,
)
from .errors import ConfigurationError

SCHEMA = "facet-contamination-exclusions-v1"
_MAX_SOURCE_BYTES = 32 * 1024 * 1024
_MAX_SOURCE_RECORDS = 10_000


@dataclass(frozen=True, slots=True)
class ContaminationExclusionPlan:
    source_sha256: str
    source_records: int
    matched_embedding_records: int
    excluded_source_records: int
    prompt_sha256: tuple[str, ...]
    audit: ContaminationReport

    def exclusions_jsonl(self) -> bytes:
        """Return strict JSONL accepted by the public-score audit."""

        return b"".join(
            (json.dumps({"prompt_sha256": digest}, sort_keys=True) + "\n").encode("ascii")
            for digest in self.prompt_sha256
        )

    def evidence_json(self) -> bytes:
        """Return aggregate linkage evidence without benchmark text or vectors."""

        exclusions = self.exclusions_jsonl()
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "source_sha256": self.source_sha256,
            "source_records": self.source_records,
            "embedding_model_id": self.audit.model_id,
            "embedding_model_revision": self.audit.model_revision,
            "embedding_dimension": self.audit.dimension,
            "threshold": self.audit.threshold,
            "training_sha256": self.audit.training_sha256,
            "evaluation_sha256": self.audit.evaluation_sha256,
            "matched_embedding_records": self.matched_embedding_records,
            "excluded_source_records": self.excluded_source_records,
            "exclusion_digests": len(self.prompt_sha256),
            "exclusions_sha256": hashlib.sha256(exclusions).hexdigest(),
            "coordinate_work": self.audit.coordinate_work,
        }
        return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def plan_contamination_exclusions(
    source_path: str | Path,
    training_path: str | Path,
    evaluation_path: str | Path,
    *,
    threshold: float = 0.95,
) -> ContaminationExclusionPlan:
    """Bind declared evaluation embeddings to every normalized benchmark ID.

    An exact one-to-one ID join prevents omissions but cannot authenticate the
    external encoder or prove that each vector represents its named prompt.
    """

    try:
        paths = [Path(path).resolve() for path in (source_path, training_path, evaluation_path)]
    except (OSError, RuntimeError) as exc:
        raise ConfigurationError("cannot resolve contamination linkage paths") from exc
    if len(set(paths)) != len(paths):
        raise ConfigurationError("contamination linkage inputs must differ")
    try:
        with Path(source_path).open("rb") as handle:
            source = handle.read(_MAX_SOURCE_BYTES + 1)
    except OSError:
        raise ConfigurationError("cannot read contamination benchmark source") from None
    if len(source) > _MAX_SOURCE_BYTES:
        raise ConfigurationError("contamination benchmark source exceeds byte limit")
    examples = load_benchmark_examples_bytes(
        source, max_bytes=_MAX_SOURCE_BYTES, max_records=_MAX_SOURCE_RECORDS
    )
    if examples[0].format not in {BenchmarkFormat.MMLU, BenchmarkFormat.GSM8K}:
        raise ConfigurationError("contamination linkage supports only MMLU and GSM8K")

    audit = audit_contamination(training_path, evaluation_path, threshold=threshold)
    # The audit took its own bounded snapshot. Re-read and require the same
    # digest before using IDs, so a path replacement cannot create a mixed join.
    evaluation = _embedding_set(_snapshot(evaluation_path, "evaluation"), "evaluation")
    if evaluation.file_sha256 != audit.evaluation_sha256:
        raise ConfigurationError("evaluation embeddings changed during contamination linkage")
    source_ids = {example.example_id for example in examples}
    if {record_id for record_id, _ in evaluation.records} != source_ids:
        raise ConfigurationError("evaluation embedding IDs must match all source records exactly")

    hits = {hit.evaluation_id for hit in audit.hits}
    digests = {
        hashlib.sha256(example.prompt.encode("utf-8")).hexdigest()
        for example in examples
        if example.example_id in hits
    }
    excluded = sum(
        hashlib.sha256(example.prompt.encode("utf-8")).hexdigest() in digests
        for example in examples
    )
    return ContaminationExclusionPlan(
        hashlib.sha256(source).hexdigest(),
        len(examples),
        len(audit.hits),
        excluded,
        tuple(sorted(digests)),
        audit,
    )
