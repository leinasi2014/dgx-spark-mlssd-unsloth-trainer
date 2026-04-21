from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import dump_json


def aggregate_run(runs_root: Path, run_id: str) -> dict[str, Any]:
    run_dir = runs_root / run_id
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(run_dir.glob("*/summary.json")):
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    aggregate = {
        "run_id": run_id,
        "benchmarks": summaries,
    }
    dump_json(aggregate, run_dir / "aggregate.json")
    (run_dir / "aggregate.md").write_text(render_markdown(aggregate), encoding="utf-8")
    return aggregate


def render_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        f"# Agent Benchmark Aggregate: {aggregate['run_id']}",
        "",
        "| Benchmark | Profile | Status | Primary Metric | Value |",
        "|---|---|---|---|---|",
    ]
    for item in aggregate["benchmarks"]:
        metric_name = item.get("primary_metric_name") or "-"
        metric_value = item.get("primary_metric_value")
        metric_text = "-" if metric_value is None else str(metric_value)
        lines.append(
            f"| {item.get('display_name', item['benchmark'])} | {item.get('profile', '-')} | {item.get('status', '-')} | {metric_name} | {metric_text} |"
        )
    lines.append("")
    return "\n".join(lines)
