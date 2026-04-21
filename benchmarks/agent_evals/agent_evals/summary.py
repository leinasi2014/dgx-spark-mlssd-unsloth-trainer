from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "pass_rate",
    "pass@1",
    "pass@k",
    "resolved_rate",
    "resolve_rate",
    "success_rate",
    "score",
    "accuracy",
)

COUNT_KEYS = (
    ("resolved_instances", "total_instances"),
    ("resolved", "total"),
    ("resolved_count", "total_count"),
    ("passed", "total"),
    ("successes", "total"),
)


def _walk_numeric_metrics(data: Any, prefix: str = "") -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (int, float)):
                found.append((path, float(value)))
            else:
                found.extend(_walk_numeric_metrics(value, path))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            found.extend(_walk_numeric_metrics(value, f"{prefix}[{index}]"))
    return found


def infer_primary_metric(data: Any) -> tuple[str | None, float | None]:
    if isinstance(data, dict):
        for key in METRIC_KEYS:
            if key in data and isinstance(data[key], (int, float)):
                return key, float(data[key])
        for resolved_key, total_key in COUNT_KEYS:
            if resolved_key in data and total_key in data:
                resolved = data[resolved_key]
                total = data[total_key]
                if isinstance(resolved, (int, float)) and isinstance(total, (int, float)) and total:
                    return f"{resolved_key}/{total_key}", round(float(resolved) / float(total), 4)
        for value in data.values():
            metric_name, metric_value = infer_primary_metric(value)
            if metric_name is not None:
                return metric_name, metric_value
    return None, None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_candidates(candidates: list[Path], raw_dir: Path) -> list[str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        destination = raw_dir / candidate.name
        if destination.resolve() != candidate.resolve():
            shutil.copy2(candidate, destination)
        copied.append(str(destination))
    return copied


def find_candidate_files(base_dirs: list[Path], patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for pattern in patterns:
            for match in base_dir.glob(pattern):
                resolved = match.resolve()
                if resolved in seen or not match.is_file():
                    continue
                seen.add(resolved)
                found.append(match)
    return found


def summarize_candidate_files(candidates: list[Path]) -> tuple[str | None, float | None, str | None]:
    for candidate in candidates:
        if candidate.suffix.lower() != ".json":
            continue
        try:
            data = load_json(candidate)
        except Exception:
            continue
        metric_name, metric_value = infer_primary_metric(data)
        if metric_name is not None:
            return metric_name, metric_value, str(candidate)
    return None, None, None
