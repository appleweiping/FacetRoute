"""Create-only local GSM8K-shaped few-shot preparation command."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from .errors import FacetRouteError
from .gsm8k_preparation import (
    MAX_SOURCE_BYTES,
    GSM8KJSONLPreparationPlan,
    prepare_gsm8k_jsonl,
)
from .public_scores import PublicScoreProvenance


def _snapshot(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_SOURCE_BYTES + 1)
    except OSError:
        raise FacetRouteError(f"cannot read GSM8K source {path}") from None
    if len(raw) > MAX_SOURCE_BYTES:
        raise FacetRouteError("GSM8K source exceeds byte limit")
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
        raise FacetRouteError("GSM8K output already exists") from None
    except OSError:
        raise FacetRouteError("cannot create GSM8K output") from None
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        if staged is not None:
            with suppress(OSError):
                staged.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="facetroute-gsm8k-prepare",
        description="Prepare caller-pinned GSM8K-shaped train/test JSONL into local few-shot JSONL",
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--test-sha256", required=True)
    parser.add_argument("--train-source-uri", required=True)
    parser.add_argument("--test-source-uri", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--max-prompt-bytes", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.suffix != ".jsonl":
            raise FacetRouteError("output must have .jsonl suffix")
        paths = (args.train.resolve(), args.test.resolve(), args.output.resolve())
        if len(set(paths)) != 3:
            raise FacetRouteError("train, test, and output paths must differ")
        try:
            if os.path.samefile(args.train, args.test):
                raise FacetRouteError("train and test paths must differ, including hard links")
        except OSError:
            raise FacetRouteError("cannot inspect GSM8K source paths") from None
        train_source = _snapshot(args.train)
        test_source = _snapshot(args.test)
        plan = GSM8KJSONLPreparationPlan(
            train=PublicScoreProvenance(args.train_source_uri, args.license, args.train_sha256),
            test=PublicScoreProvenance(args.test_source_uri, args.license, args.test_sha256),
            requested_shots=args.shots,
            max_prompt_bytes=args.max_prompt_bytes,
        )
        prepared = prepare_gsm8k_jsonl(train_source, test_source, plan)
        if _snapshot(args.train) != train_source or _snapshot(args.test) != test_source:
            raise FacetRouteError("GSM8K source changed before publication")
        _create_only(args.output, prepared.benchmark_jsonl())
        sys.stdout.buffer.write(prepared.evidence_json())
    except FacetRouteError as error:
        print(f"GSM8K preparation: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
