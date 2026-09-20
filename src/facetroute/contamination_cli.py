"""Offline prompt-embedding similarity audit command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .contamination import audit_contamination, contamination_json
from .errors import FacetRouteError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="facetroute-contamination-audit",
        description="Offline cosine audit of declared training/evaluation embedding sets",
    )
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args(argv)
    try:
        report = audit_contamination(args.training, args.evaluation, threshold=args.threshold)
    except FacetRouteError as error:
        print(f"contamination audit: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(contamination_json(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
