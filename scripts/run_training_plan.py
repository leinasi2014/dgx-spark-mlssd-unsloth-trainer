#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import append_run_note, ensure_run_dirs, load_config, logger


def sequential_commands(cfg_path: str, run_name: str, cfg: dict) -> list[str]:
    seq = cfg['training_plan']['sequential']
    return [
        f'python scripts/build_skill_views.py --config {cfg_path}',
        f'python scripts/build_skill0_dataset.py --config {cfg_path}',
        f'python scripts/train_unsloth_lora.py --config {cfg_path} --dataset-path runs/{run_name}/ssd_train.jsonl --output-subdir {seq["code_output_subdir"]}',
        f'python scripts/train_unsloth_lora.py --config {cfg_path} --dataset-path runs/{run_name}/skill0_train.jsonl --output-subdir {seq["skill_output_subdir"]} --init-adapter runs/{run_name}/{seq["code_output_subdir"]}',
        f'python scripts/squeeze_lora.py --config {cfg_path} --source-subdir {seq["skill_output_subdir"]} --output-subdir {seq["squeezed_output_subdir"]}',
        f'python scripts/recover_after_squeeze.py --config {cfg_path} --dataset-path runs/{run_name}/skill0_train.jsonl --source-subdir {seq["squeezed_output_subdir"]} --output-subdir {seq["recovered_output_subdir"]}',
        f'python scripts/evaluate_codegen.py --config {cfg_path} --adapter-subdir {seq["recovered_output_subdir"]}',
    ]


def mixed_commands(cfg_path: str, run_name: str, cfg: dict) -> list[str]:
    mixed = cfg['training_plan']['mixed']
    return [
        f'python scripts/build_skill_views.py --config {cfg_path}',
        f'python scripts/build_skill0_dataset.py --config {cfg_path}',
        f'python scripts/build_mixed_dataset.py --config {cfg_path}',
        f'python scripts/train_unsloth_lora.py --config {cfg_path} --dataset-path runs/{run_name}/mixed_train.jsonl --output-subdir {mixed["output_subdir"]}',
        f'python scripts/squeeze_lora.py --config {cfg_path} --source-subdir {mixed["output_subdir"]} --output-subdir {mixed["squeezed_output_subdir"]}',
        f'python scripts/recover_after_squeeze.py --config {cfg_path} --dataset-path runs/{run_name}/mixed_train.jsonl --source-subdir {mixed["squeezed_output_subdir"]} --output-subdir {mixed["recovered_output_subdir"]}',
        f'python scripts/evaluate_codegen.py --config {cfg_path} --adapter-subdir {mixed["recovered_output_subdir"]}',
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare sequential or mixed training artifacts and print the recommended command plan.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['sequential', 'mixed'], default=None)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    mode = args.mode or cfg.get('training_plan', {}).get('mode', 'sequential')

    # always materialize shared skill assets first
    from build_skill_views import main as build_skill_views_main  # type: ignore
    from build_skill0_dataset import main as build_skill0_dataset_main  # type: ignore

    # lightweight direct calls via subprocess semantics are avoided to keep it simple
    import sys
    old_argv = sys.argv[:]
    try:
        sys.argv = ['build_skill_views.py', '--config', args.config]
        build_skill_views_main()
        sys.argv = ['build_skill0_dataset.py', '--config', args.config]
        build_skill0_dataset_main()
        if mode == 'mixed':
            mixed_source = Path(cfg['paths']['ssd_train_jsonl'])
            if mixed_source.exists():
                from build_mixed_dataset import main as build_mixed_dataset_main  # type: ignore
                sys.argv = ['build_mixed_dataset.py', '--config', args.config]
                build_mixed_dataset_main()
            else:
                logger.warning('Skipping mixed dataset build because %s does not exist yet. Generate and convert SSD outputs first.', mixed_source)
    finally:
        sys.argv = old_argv

    commands = sequential_commands(args.config, cfg['run_name'], cfg) if mode == 'sequential' else mixed_commands(args.config, cfg['run_name'], cfg)
    plan_path = run_dir / f'{mode}_plan.sh'
    plan_path.write_text('#!/usr/bin/env bash\nset -euo pipefail\n\n' + '\n'.join(commands) + '\n', encoding='utf-8')
    append_run_note(run_dir, [f'Prepared {mode} training plan at {plan_path.name}.'])
    logger.info('Prepared %s training plan at %s', mode, plan_path)
    print('\n'.join(commands))


if __name__ == '__main__':
    main()
