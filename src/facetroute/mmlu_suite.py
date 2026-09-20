"""Deterministic, bounded composition of pinned MMLU-shaped subject snapshots.

Each subject is prepared by the existing single-subject protocol. This layer
binds their ordering, cross-subject split safety, and one aggregate artifact;
it neither downloads MMLU nor reproduces an official evaluation result.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from .errors import ConfigurationError
from .mmlu_preparation import (
    MAX_DEV_ROWS,
    MAX_TEST_ROWS,
    MMLUCSVPreparationPlan,
    PreparedMMLUCSV,
    _csv_rows,
    _json,
    _sha,
    _stem_key,
    prepare_mmlu_csv,
)

PROTOCOL = "facet-mmlu-suite-v1"
MAX_SUBJECTS = 64
MAX_TOTAL_SOURCE_BYTES = 64 * 1024 * 1024
MAX_SUITE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MMLUSuiteSource:
    """One subject's exact caller-owned byte snapshots and declared hashes."""

    plan: MMLUCSVPreparationPlan
    dev_source: bytes
    test_source: bytes


@dataclass(frozen=True, slots=True)
class PreparedMMLUSuite:
    """Canonical multi-subject public-score input and aggregate provenance."""

    subjects: tuple[PreparedMMLUCSV, ...]

    def benchmark_jsonl(self) -> bytes:
        """Return combined private-gold JSONL in alphabetical subject order."""

        chunks: list[bytes] = []
        size = 0
        for subject in self.subjects:
            chunk = subject.benchmark_jsonl()
            size += len(chunk)
            if size > MAX_SUITE_BYTES:
                raise ConfigurationError("prepared MMLU suite exceeds output byte limit")
            chunks.append(chunk)
        return b"".join(chunks)

    def evidence_json(self) -> bytes:
        """Return source and prompt hashes, never prompt text or gold answers."""

        artifact = self.benchmark_jsonl()
        return (
            _json(
                {
                    "protocol": PROTOCOL,
                    "subjects": [subject.plan.subject for subject in self.subjects],
                    "records": sum(subject.test_rows for subject in self.subjects),
                    "artifact_sha256": _sha(artifact),
                    "subject_evidence": [
                        json.loads(subject.evidence_json()) for subject in self.subjects
                    ],
                }
            )
            + b"\n"
        )


def prepare_mmlu_suite(sources: Sequence[MMLUSuiteSource]) -> PreparedMMLUSuite:
    """Compose distinct pinned subjects and reject cross-subject dev/test leakage."""

    if not 1 <= len(sources) <= MAX_SUBJECTS:
        raise ConfigurationError("MMLU suite requires 1 to at most 64 subjects")
    if any(not isinstance(source, MMLUSuiteSource) for source in sources):
        raise ConfigurationError("MMLU suite entries must be MMLUSuiteSource values")
    if any(
        not isinstance(source.plan, MMLUCSVPreparationPlan)
        or type(source.dev_source) is not bytes
        or type(source.test_source) is not bytes
        for source in sources
    ):
        raise ConfigurationError("MMLU suite requires plans and immutable byte snapshots")
    if (
        sum(len(source.dev_source) + len(source.test_source) for source in sources)
        > MAX_TOTAL_SOURCE_BYTES
    ):
        raise ConfigurationError("MMLU suite exceeds total source byte limit")
    names = [source.plan.subject for source in sources]
    if len(names) != len(set(names)):
        raise ConfigurationError("MMLU suite contains a duplicate subject")

    prepared: list[PreparedMMLUCSV] = []
    dev_stems: dict[str, set[str]] = {}
    test_stems: dict[str, set[str]] = {}
    for source in sorted(sources, key=lambda item: item.plan.subject):
        name = source.plan.subject
        prepared.append(
            prepare_mmlu_csv(
                source.dev_source,
                source.test_source,
                source.plan,
                dev_name=f"{name}_dev.csv",
                test_name=f"{name}_test.csv",
            )
        )
        dev_stems[name] = {
            _stem_key(row)
            for row in _csv_rows(source.dev_source, label=f"{name} dev", maximum=MAX_DEV_ROWS)
        }
        test_stems[name] = {
            _stem_key(row)
            for row in _csv_rows(source.test_source, label=f"{name} test", maximum=MAX_TEST_ROWS)
        }
    all_dev: dict[str, set[str]] = {}
    for subject, stems in dev_stems.items():
        for stem in stems:
            all_dev.setdefault(stem, set()).add(subject)
    for subject, stems in test_stems.items():
        if any(all_dev.get(stem, set()) - {subject} for stem in stems):
            raise ConfigurationError("MMLU suite has cross-subject dev/test question overlap")

    result = PreparedMMLUSuite(tuple(prepared))
    result.benchmark_jsonl()
    return result


def verify_mmlu_suite(artifact: bytes, evidence: bytes, sources: Sequence[MMLUSuiteSource]) -> bool:
    """Replay all exact snapshots and compare aggregate bytes and evidence."""

    if type(artifact) is not bytes or type(evidence) is not bytes:
        raise ConfigurationError("MMLU suite artifact and evidence must be byte snapshots")
    expected = prepare_mmlu_suite(sources)
    return artifact == expected.benchmark_jsonl() and evidence == expected.evidence_json()
