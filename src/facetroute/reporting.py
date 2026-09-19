"""Dependency-free JSON, CSV, and standalone HTML experiment reports."""

from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkReport, IntervalEstimate
from .calibration import CalibrationReport
from .persistence import _atomic_write_bytes_bundle, _json_bytes


def _atomic_text(path: str | Path, content: str) -> None:
    _atomic_write_bytes_bundle({Path(path): content.encode("utf-8")})


def write_json(path: str | Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
    )


def write_calibration_csv(path: str | Path, report: CalibrationReport) -> None:
    output = io.StringIO(newline="")
    fieldnames = list(report.points[0].to_dict())
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for point in report.points:
        writer.writerow(point.to_dict())
    _atomic_text(path, output.getvalue())


def _estimate_columns(prefix: str, value: IntervalEstimate) -> dict[str, float | None]:
    return {
        prefix: value.estimate,
        f"{prefix}_ci_lower": value.lower,
        f"{prefix}_ci_upper": value.upper,
    }


def benchmark_rows(report: BenchmarkReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, metrics in sorted(report.policies.items()):
        row: dict[str, Any] = {
            "policy": policy,
            "requests": metrics.requests,
            "routed": metrics.routed,
        }
        for name in (
            "average_quality",
            "average_cost_usd",
            "average_latency_ms",
            "p95_latency_ms",
            "success_rate",
            "average_quality_regret",
            "failure_rate",
            "constraint_violation_rate",
        ):
            row.update(_estimate_columns(name, getattr(metrics, name)))
        rows.append(row)
    return rows


def write_benchmark_csv(path: str | Path, report: BenchmarkReport) -> None:
    _atomic_text(path, _benchmark_csv(report))


def _benchmark_csv(report: BenchmarkReport) -> str:
    rows = benchmark_rows(report)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _benchmark_html(report: BenchmarkReport) -> str:
    rows = benchmark_rows(report)
    columns = (
        "policy",
        "routed",
        "average_quality",
        "average_cost_usd",
        "p95_latency_ms",
        "success_rate",
        "average_quality_regret",
        "failure_rate",
        "constraint_violation_rate",
    )
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(_format(row[column]))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    manifest = report.manifest
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>FacetRoute benchmark</title>
<style>
body{{font:15px system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#18212f}}
table{{border-collapse:collapse;width:100%;overflow:auto}}th,td{{padding:9px;border:1px solid #d7dee8;text-align:right}}
th:first-child,td:first-child{{text-align:left}}th{{background:#eef4ff}}code{{word-break:break-all}}.note{{color:#526176}}
</style></head><body>
<h1>FacetRoute offline benchmark</h1>
<p class="note">Counterfactual trace evaluation; no provider calls were made. Confidence intervals use
{manifest.bootstrap_samples} seeded bootstrap samples at {manifest.confidence_level:.1%} confidence.</p>
<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
<h2>Reproducibility manifest</h2>
<p>records: {manifest.records}; seed: {manifest.seed}</p>
<p>dataset canonical SHA-256: <code>{html.escape(manifest.dataset_canonical_sha256)}</code></p>
<p>dataset file SHA-256: <code>{html.escape(manifest.dataset_file_sha256 or "not supplied")}</code></p>
<p>catalog SHA-256: <code>{html.escape(manifest.catalog_sha256)}</code></p>
<p>input SHA-256: <code>{html.escape(json.dumps(dict(manifest.input_sha256), sort_keys=True))}</code></p>
</body></html>
"""
    return page


def write_benchmark_html(path: str | Path, report: BenchmarkReport) -> None:
    _atomic_text(path, _benchmark_html(report))


def write_benchmark_bundle(directory: str | Path, report: BenchmarkReport) -> None:
    """Render every benchmark artifact before committing the recoverable bundle."""

    destination = Path(directory)
    _atomic_write_bytes_bundle(
        {
            destination / "benchmark.json": _json_bytes(report.to_dict()),
            destination / "benchmark.csv": _benchmark_csv(report).encode("utf-8"),
            destination / "benchmark.html": _benchmark_html(report).encode("utf-8"),
        }
    )
