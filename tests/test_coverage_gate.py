from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_branch_coverage.py"


@pytest.mark.parametrize(
    ("covered", "branches", "expected"),
    [(9, 10, 0), (90, 100, 0), (899, 1000, 1), (0, 0, 2), (11, 10, 2)],
)
def test_branch_gate_uses_branch_counts_not_combined_percentage(
    tmp_path: Path, covered: int, branches: int, expected: int
) -> None:
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps(
            {
                "totals": {
                    "covered_branches": covered,
                    "num_branches": branches,
                    "percent_covered": 100.0,
                }
            }
        ),
        encoding="utf-8",
    )
    main = runpy.run_path(str(SCRIPT))["main"]
    assert main(report) == expected


def test_branch_gate_rejects_missing_report_and_cli_arguments(tmp_path: Path) -> None:
    main = runpy.run_path(str(SCRIPT))["main"]
    assert main(tmp_path / "missing.json") == 2
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "Usage" in result.stderr
