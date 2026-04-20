#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

REQUIRED = [
    'AGENTS.md',
    'configs/train.example.yaml',
    'scripts/bootstrap_dgx_spark.sh',
    'scripts/prepare_ssd_data.py',
    'scripts/train_unsloth_lora.py',
    'scripts/build_skill_views.py',
    'scripts/build_skill0_dataset.py',
    'scripts/build_mixed_dataset.py',
    'scripts/run_training_plan.py',
    'scripts/squeeze_lora.py',
    'scripts/recover_after_squeeze.py',
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    missing = [rel for rel in REQUIRED if not (repo_root / rel).exists()]
    if missing:
        print('[missing] required repo files:')
        for item in missing:
            print('-', item)
        raise SystemExit(1)
    print('[ok] repo layout looks valid')


if __name__ == '__main__':
    main()
