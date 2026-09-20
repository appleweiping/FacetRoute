"""Aggregate-only command for local public-score checkpoint analysis."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .errors import FacetRouteError
from .public_score_audit import audit_json, audit_public_scores


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="facetroute-public-score-audit",
        description="Offline, aggregate-only audit of a completed public-score checkpoint",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--route-scores", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_public_scores(
            args.source, args.checkpoint, args.route_scores, exclusions_path=args.exclusions
        )
    except FacetRouteError as error:
        print(f"public-score audit: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(audit_json(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
