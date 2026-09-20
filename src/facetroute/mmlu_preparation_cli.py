"""Offline, create-only MMLU CSV few-shot preparation command."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from .errors import FacetRouteError
from .mmlu_preparation import MAX_CSV_BYTES, MMLUCSVPreparationPlan, prepare_mmlu_csv
from .public_scores import PublicScoreProvenance


def _snapshot(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_CSV_BYTES + 1)
    except OSError:
        raise FacetRouteError(f"cannot read MMLU CSV source {path}") from None
    if len(raw) > MAX_CSV_BYTES:
        raise FacetRouteError("MMLU CSV source exceeds byte limit")
    return raw


def _create_only(path: Path, raw: bytes) -> None:
    if not path.parent.is_dir():
        raise FacetRouteError("output parent directory does not exist")
    descriptor = -1
    staged: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        staged = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            if handle.write(raw) != len(raw):
                raise OSError("short output write")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staged, path)
        if os.name == "posix":
            parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
    except FileExistsError:
        raise FacetRouteError("MMLU output already exists") from None
    except OSError:
        raise FacetRouteError("cannot create MMLU output") from None
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        if staged is not None:
            with suppress(OSError):
                staged.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="facetroute-mmlu-prepare",
        description="Prepare caller-pinned MMLU CSV dev/test into local few-shot JSONL",
    )
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--dev-sha256", required=True)
    parser.add_argument("--test-sha256", required=True)
    parser.add_argument("--dev-source-uri", required=True)
    parser.add_argument("--test-source-uri", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--max-prompt-bytes", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.suffix != ".jsonl":
            raise FacetRouteError("output must have .jsonl suffix")
        paths = (args.dev.resolve(), args.test.resolve(), args.output.resolve())
        if len(set(paths)) != 3:
            raise FacetRouteError("dev, test, and output paths must differ")
        dev_source = _snapshot(args.dev)
        test_source = _snapshot(args.test)
        plan = MMLUCSVPreparationPlan(
            subject=args.subject,
            dev=PublicScoreProvenance(args.dev_source_uri, args.license, args.dev_sha256),
            test=PublicScoreProvenance(args.test_source_uri, args.license, args.test_sha256),
            requested_shots=args.shots,
            max_prompt_bytes=args.max_prompt_bytes,
        )
        prepared = prepare_mmlu_csv(
            dev_source,
            test_source,
            plan,
            dev_name=args.dev.name,
            test_name=args.test.name,
        )
        _create_only(args.output, prepared.benchmark_jsonl())
        sys.stdout.buffer.write(prepared.evidence_json())
    except FacetRouteError as error:
        print(f"MMLU preparation: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
