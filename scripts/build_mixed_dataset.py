#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import append_run_note, ensure_run_dirs, load_config, load_jsonl, logger, write_jsonl


def expand_rows(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    if not rows or target <= 0:
        return []
    out = []
    idx = 0
    while len(out) < target:
        item = dict(rows[idx % len(rows)])
        base_id = str(item.get('id') or f'row-{idx}')
        item['id'] = f'{base_id}__mix{len(out):05d}'
        out.append(item)
        idx += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a mixed training dataset from code and skill0 rows.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    code_path = Path(cfg['paths']['ssd_train_jsonl'])
    skill_path = Path(cfg['paths']['skill0_train_jsonl'])
    if not code_path.exists():
        raise FileNotFoundError(f'Missing code training dataset: {code_path}. Generate and convert SSD outputs first.')
    if not skill_path.exists():
        raise FileNotFoundError(f'Missing skill0 training dataset: {skill_path}. Build skill views and skill0 dataset first.')
    code_rows = load_jsonl(code_path)
    skill_rows = load_jsonl(skill_path)

    code_weight = float(cfg.get('skill0', {}).get('mixed_weights', {}).get('code', 0.7))
    skill_weight = float(cfg.get('skill0', {}).get('mixed_weights', {}).get('skill0', 0.3))
    total = max(len(code_rows), len(skill_rows), 1)
    code_target = max(1, int(round(total * code_weight)))
    skill_target = max(1, int(round(total * skill_weight)))
    mixed_rows = expand_rows(code_rows, code_target) + expand_rows(skill_rows, skill_target)

    out_path = Path(cfg['paths']['mixed_train_jsonl'])
    write_jsonl(out_path, mixed_rows)
    append_run_note(run_dir, [f'Built mixed dataset: {out_path.name} ({len(mixed_rows)} rows).'])
    logger.info('Wrote mixed dataset to %s (%d rows)', out_path, len(mixed_rows))


if __name__ == '__main__':
    main()
