"""Enforce the branch-only coverage floor on a pytest-cov JSON report."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: Path) -> int:
    try:
        totals = json.loads(path.read_text(encoding="utf-8"))["totals"]
        covered = totals["covered_branches"]
        branches = totals["num_branches"]
        if (
            isinstance(covered, bool)
            or isinstance(branches, bool)
            or not isinstance(covered, int)
            or not isinstance(branches, int)
            or branches <= 0
            or not 0 <= covered <= branches
        ):
            raise ValueError("invalid branch totals")
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Cannot verify branch coverage: {exc}", file=sys.stderr)
        return 2

    percentage = 100 * covered / branches
    print(f"Pure branch coverage: {covered}/{branches} = {percentage:.2f}% (minimum 90%)")
    return 0 if 10 * covered >= 9 * branches else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_branch_coverage.py coverage.json", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
