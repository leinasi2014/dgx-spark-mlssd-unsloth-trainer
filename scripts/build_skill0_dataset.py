#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import append_run_note, ensure_run_dirs, load_config, load_jsonl, logger, write_jsonl

LEVELS = ('full', 'summary', 'tool_only', 'zero')


def stage_sequence(cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for stage in cfg.get('skill0', {}).get('stages', []):
        mixture = stage.get('mixture', {})
        total = 0
        for level in LEVELS:
            total += int(round(float(mixture.get(level, 0.0)) * 10))
            out.extend([level] * int(round(float(mixture.get(level, 0.0)) * 10)))
        if total == 0:
            out.append('zero')
    return out or ['full', 'summary', 'tool_only', 'zero']


def make_system_prompt(view_text: str, level: str) -> str:
    if level == 'zero':
        return 'Follow the repository conventions if you already know them, but do not assume extra runtime skill files are available.'
    return f'Use this skill context level={level}. Internalize the procedure rather than quoting it verbatim.\n\n{view_text.strip()}'


def main() -> None:
    parser = argparse.ArgumentParser(description='Build skill0 supervised training rows from task prompts and compiled skill views.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    task_rows = load_jsonl(Path(cfg['paths']['skill_task_dataset']))
    skill_dir = run_dir / 'skill_views'
    views = {
        'full': (skill_dir / 'full.md').read_text(encoding='utf-8') if (skill_dir / 'full.md').exists() else '',
        'summary': (skill_dir / 'summary.md').read_text(encoding='utf-8') if (skill_dir / 'summary.md').exists() else '',
        'tool_only': (skill_dir / 'tool_only.md').read_text(encoding='utf-8') if (skill_dir / 'tool_only.md').exists() else '',
        'zero': '',
    }
    levels = stage_sequence(cfg)
    rows = []
    for idx, task in enumerate(task_rows):
        level = levels[idx % len(levels)]
        user = str(task.get('user') or '').strip()
        if not user:
            continue
        system = make_system_prompt(views[level], level)
        assistant = (
            f"Plan: solve task {task.get('task_id', idx)} in category={task.get('category', 'unknown')} using repository-specific steps. "
            f"Prefer concrete commands, files, and checks. Do not restate long background text."
        )
        rows.append({
            'id': str(task.get('task_id') or f'skill0-{idx:04d}'),
            'skill_level': level,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': assistant},
            ],
        })

    out_path = Path(cfg['paths']['skill0_train_jsonl'])
    write_jsonl(out_path, rows)
    append_run_note(run_dir, [f'Built skill0 dataset: {out_path.name} ({len(rows)} rows).'])
    logger.info('Wrote skill0 dataset to %s (%d rows)', out_path, len(rows))


if __name__ == '__main__':
    main()
