"""Offline, create-only composition of caller-pinned MMLU-shaped subjects."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._json import loads_strict
from .errors import ConfigurationError, FacetRouteError
from .mmlu_preparation import MMLUCSVPreparationPlan, _json
from .mmlu_preparation_cli import _create_only, _snapshot
from .mmlu_suite import MAX_SUBJECTS, MMLUSuiteSource, prepare_mmlu_suite
from .public_scores import PublicScoreProvenance

MAX_MANIFEST_BYTES = 1024 * 1024
_FIELDS = {
    "subject",
    "dev",
    "test",
    "dev_sha256",
    "test_sha256",
    "dev_source_uri",
    "test_source_uri",
    "license",
    "shots",
    "max_prompt_bytes",
}


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value or len(value) > 2048:
        raise ConfigurationError(f"MMLU suite {label} must be bounded non-empty text")
    return value


def _source_path(root: Path, value: Any, subject: str, split: str) -> Path:
    name = _text(value, f"{split} path")
    supplied = Path(name)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise ConfigurationError("MMLU suite source paths must be relative to manifest directory")
    path = (root / supplied).resolve()
    if not path.is_relative_to(root) or path.name != f"{subject}_{split}.csv":
        raise ConfigurationError("MMLU suite source path escapes directory or subject basename")
    return path


def _manifest_sources(
    raw: bytes, root: Path, output: Path, manifest: Path
) -> tuple[MMLUSuiteSource, ...]:
    try:
        parsed = loads_strict(raw)
    except ValueError as error:
        raise ConfigurationError(f"MMLU suite manifest is not strict JSON: {error}") from error
    if type(parsed) is not dict or set(parsed) != {"subjects"}:
        raise ConfigurationError("MMLU suite manifest requires only subjects")
    entries = parsed["subjects"]
    if type(entries) is not list or not 1 <= len(entries) <= MAX_SUBJECTS:
        raise ConfigurationError("MMLU suite requires 1 to at most 64 subjects")
    sources: list[MMLUSuiteSource] = []
    used_paths = {output.resolve(), manifest.resolve()}
    for item in entries:
        if type(item) is not dict or set(item) != _FIELDS:
            raise ConfigurationError("MMLU suite subject manifest fields are invalid")
        subject = _text(item["subject"], "subject")
        dev = _source_path(root, item["dev"], subject, "dev")
        test = _source_path(root, item["test"], subject, "test")
        if dev in used_paths or test in used_paths or dev == test:
            raise ConfigurationError("MMLU suite input and output paths must be distinct")
        used_paths.update((dev, test))
        license_id = _text(item["license"], "license")
        plan = MMLUCSVPreparationPlan(
            subject,
            PublicScoreProvenance(
                _text(item["dev_source_uri"], "dev source URI"),
                license_id,
                _text(item["dev_sha256"], "dev SHA-256"),
            ),
            PublicScoreProvenance(
                _text(item["test_source_uri"], "test source URI"),
                license_id,
                _text(item["test_sha256"], "test SHA-256"),
            ),
            requested_shots=item["shots"],
            max_prompt_bytes=item["max_prompt_bytes"],
        )
        sources.append(MMLUSuiteSource(plan, _snapshot(dev), _snapshot(test)))
    return tuple(sources)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="facetroute-mmlu-suite",
        description="Combine caller-pinned MMLU-shaped CSV subjects into private-gold JSONL",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.suffix != ".jsonl":
            raise ConfigurationError("MMLU suite output must have .jsonl suffix")
        manifest = args.manifest.resolve()
        try:
            with manifest.open("rb") as handle:
                raw = handle.read(MAX_MANIFEST_BYTES + 1)
        except OSError:
            raise ConfigurationError("cannot read MMLU suite manifest") from None
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ConfigurationError("MMLU suite manifest exceeds byte limit")
        sources = _manifest_sources(raw, manifest.parent, args.output, manifest)
        prepared = prepare_mmlu_suite(sources)
        artifact = prepared.benchmark_jsonl()
        evidence = {
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "suite": loads_strict(prepared.evidence_json()),
        }
        _create_only(args.output, artifact)
        sys.stdout.buffer.write(_json(evidence) + b"\n")
    except FacetRouteError as error:
        print(f"MMLU suite: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
