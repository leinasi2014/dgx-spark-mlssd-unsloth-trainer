#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import append_run_note, ensure_run_dirs, load_config
from source_adapters import ensure_prepared_source, format_agent_trajectory_rows, get_source_config, source_limit


def default_agent_source(cfg: dict) -> str:
    return str(cfg.get('training_plan', {}).get('agent', {}).get('source') or 'coderforge_preview')


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare agent-trajectory data from configured upstream datasets.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--source-name', default=None)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    source_name = str(args.source_name or default_agent_source(cfg)).strip()
    source_cfg = get_source_config(cfg, source_name)
    if str(source_cfg.get('family') or '').strip() != 'agent_trajectory':
        raise ValueError(f'Source {source_name} must use family=agent_trajectory for prepare_agent_data.py')
    rows = format_agent_trajectory_rows(source_name, source_cfg, limit=source_limit(source_cfg, args.limit))
    dataset_path = ensure_prepared_source(
        run_dir,
        source_name,
        rows,
        {
            'source_name': source_name,
            'family': source_cfg.get('family'),
            'adapter': source_cfg.get('adapter'),
            'dataset': source_cfg.get('dataset'),
            'n_rows': len(rows),
        },
    )
    append_run_note(run_dir, [f'Prepared agent trajectory dataset {dataset_path.name} from source {source_name}.'])


if __name__ == '__main__':
    main()
