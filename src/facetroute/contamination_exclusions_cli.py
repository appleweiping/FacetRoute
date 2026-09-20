"""Create a prompt-digest exclusion file from an offline similarity screen."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .contamination_exclusions import plan_contamination_exclusions
from .errors import ConfigurationError, FacetRouteError


def _create_only(path: Path, content: bytes) -> None:
    """Stage complete private bytes and atomically link only a vacant name."""

    if not path.parent.is_dir():
        raise ConfigurationError("exclusion output parent directory does not exist")
    temporary: Path | None = None
    descriptor = -1
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            if handle.write(content) != len(content):
                raise OSError("short exclusion write")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise ConfigurationError("exclusion output already exists") from None
    except OSError:
        raise ConfigurationError("cannot create exclusion output") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="facetroute-contamination-exclusions",
        description="Offline ID-joined embedding screen to prompt-digest exclusions",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = plan_contamination_exclusions(
            args.source, args.training, args.evaluation, threshold=args.threshold
        )
        _create_only(args.output, plan.exclusions_jsonl())
    except FacetRouteError as error:
        print(f"contamination exclusions: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(plan.evidence_json())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
