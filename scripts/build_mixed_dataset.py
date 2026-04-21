#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from common import append_run_note, ensure_path_is_new, ensure_run_dirs, load_config, load_jsonl, logger, write_jsonl
from source_adapters import resolve_named_dataset_path


def expand_rows(rows: list[dict[str, Any]], target: int, source_name: str) -> list[dict[str, Any]]:
    if not rows or target <= 0:
        return []
    out = []
    idx = 0
    while len(out) < target:
        item = dict(rows[idx % len(rows)])
        base_id = str(item.get('id') or f'{source_name}-row-{idx}')
        item['id'] = f'{base_id}__mix{len(out):05d}'
        item['mix_source'] = source_name
        out.append(item)
        idx += 1
    return out


def normalized_mix_targets(source_rows: dict[str, list[dict[str, Any]]], source_weights: dict[str, float]) -> dict[str, int]:
    total_weight = sum(source_weights.values())
    if total_weight <= 0:
        raise ValueError('At least one mixed source weight must be positive.')
    for source_name, weight in source_weights.items():
        if weight < 0:
            raise ValueError(f'Mixed source weight must be non-negative: {source_name}')
        if weight > 0 and not source_rows.get(source_name):
            raise ValueError(f'Mixed source {source_name} has positive weight but its dataset is empty.')

    positive_weights = {source_name: weight / total_weight for source_name, weight in source_weights.items() if weight > 0}
    required_totals = [
        len(source_rows[source_name]) / normalized_weight
        for source_name, normalized_weight in positive_weights.items()
        if source_rows[source_name]
    ]
    total = max(1, int(math.ceil(max(required_totals, default=1.0))))
    raw_targets = {source_name: total * positive_weights[source_name] for source_name in positive_weights}
    targets: dict[str, int] = {}
    allocated = 0
    remainders: list[tuple[float, str]] = []
    for source_name, weight in source_weights.items():
        if weight <= 0:
            targets[source_name] = 0
            continue
        raw_target = raw_targets[source_name]
        target = max(1, int(math.floor(raw_target)))
        targets[source_name] = target
        allocated += target
        remainders.append((raw_target - math.floor(raw_target), source_name))

    if allocated < total:
        for _fraction, source_name in sorted(remainders, reverse=True):
            if allocated >= total:
                break
            targets[source_name] += 1
            allocated += 1
    return targets


def resolve_mixed_sources(cfg: dict[str, Any], run_dir: Path) -> list[tuple[str, Path, float]]:
    mixed_cfg = cfg.get('training_plan', {}).get('mixed', {})
    configured_sources = mixed_cfg.get('sources')
    if isinstance(configured_sources, list) and configured_sources:
        sources: list[tuple[str, Path, float]] = []
        seen_source_names: set[str] = set()
        for item in configured_sources:
            if not isinstance(item, dict):
                raise TypeError('training_plan.mixed.sources entries must be objects.')
            source_name = str(item.get('source') or '').strip()
            if not source_name:
                raise ValueError('training_plan.mixed.sources requires a non-empty source field.')
            if source_name in seen_source_names:
                raise ValueError(f'training_plan.mixed.sources contains duplicate source entries: {source_name}')
            seen_source_names.add(source_name)
            weight = float(item.get('weight', 0.0))
            sources.append((source_name, resolve_named_dataset_path(cfg, run_dir, source_name), weight))
        return sources

    code_weight = float(cfg.get('skill0', {}).get('mixed_weights', {}).get('code', 0.7))
    skill_weight = float(cfg.get('skill0', {}).get('mixed_weights', {}).get('skill0', 0.3))
    return [
        ('code', resolve_named_dataset_path(cfg, run_dir, 'code'), code_weight),
        ('skill0', resolve_named_dataset_path(cfg, run_dir, 'skill0'), skill_weight),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a mixed training dataset from prepared sources.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    source_specs = resolve_mixed_sources(cfg, run_dir)

    source_rows: dict[str, list[dict[str, Any]]] = {}
    source_weights: dict[str, float] = {}
    for source_name, path, weight in source_specs:
        if weight <= 0:
            source_rows[source_name] = []
            source_weights[source_name] = weight
            continue
        if not path.exists():
            raise FileNotFoundError(f'Missing mixed-source dataset {source_name}: {path}')
        source_rows[source_name] = load_jsonl(path)
        source_weights[source_name] = weight

    targets = normalized_mix_targets(source_rows, source_weights)
    mixed_rows: list[dict[str, Any]] = []
    for source_name, rows in source_rows.items():
        mixed_rows.extend(expand_rows(rows, targets.get(source_name, 0), source_name))

    out_path = Path(cfg['paths']['mixed_train_jsonl'])
    ensure_path_is_new(out_path, 'mixed training dataset')
    write_jsonl(out_path, mixed_rows)
    append_run_note(run_dir, [f'Built mixed dataset: {out_path.name} ({len(mixed_rows)} rows) from {", ".join(source_rows)}.'])
    logger.info('Wrote mixed dataset to %s (%d rows)', out_path, len(mixed_rows))


if __name__ == '__main__':
    main()
