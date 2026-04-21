#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import append_run_note, ensure_path_is_new, ensure_run_dirs, load_config, load_jsonl, logger, write_jsonl

LEVELS = ('full', 'summary', 'tool_only', 'zero')


def expand_stage_pattern(stage: dict[str, Any]) -> list[str]:
    mixture = stage.get('mixture', {})
    out: list[str] = []
    total = 0
    for level in LEVELS:
        count = int(round(float(mixture.get(level, 0.0)) * 10))
        total += count
        out.extend([level] * count)
    return out if total > 0 else ['zero']


def stage_assignments(cfg: dict[str, Any], n_tasks: int) -> list[tuple[str, str]]:
    stages = list(cfg.get('skill0', {}).get('stages', []))
    if not stages:
        fallback = ['full', 'summary', 'tool_only', 'zero']
        return [('default', fallback[idx % len(fallback)]) for idx in range(n_tasks)]
    assignments: list[tuple[str, str]] = []
    n_stages = len(stages)
    for idx in range(n_tasks):
        stage_idx = min((idx * n_stages) // max(n_tasks, 1), n_stages - 1)
        stage = stages[stage_idx]
        pattern = expand_stage_pattern(stage)
        local_idx = idx - ((stage_idx * n_tasks) // n_stages)
        assignments.append((str(stage.get('name') or f'stage_{stage_idx}'), pattern[local_idx % len(pattern)]))
    return assignments


def make_system_prompt(view_text: str, level: str) -> str:
    if level == 'zero':
        return 'Follow the repository conventions if you already know them, but do not assume extra runtime skill files are available.'
    return f'Use this skill context level={level}. Internalize the procedure rather than quoting it verbatim.\n\n{view_text.strip()}'


def load_skill_views(skill_dir: Path, required_levels: set[str]) -> dict[str, str]:
    views = {'zero': ''}
    for level in ['full', 'summary', 'tool_only']:
        view_path = skill_dir / f'{level}.md'
        if level in required_levels:
            if not view_path.exists():
                raise FileNotFoundError(f'Missing required skill view for level={level}: {view_path}')
            text = view_path.read_text(encoding='utf-8').strip()
            if not text:
                raise ValueError(f'Required skill view is empty for level={level}: {view_path}')
            views[level] = text
        elif view_path.exists():
            views[level] = view_path.read_text(encoding='utf-8').strip()
        else:
            views[level] = ''
    return views


def main() -> None:
    parser = argparse.ArgumentParser(description='Build skill0 supervised training rows from task prompts and compiled skill views.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    task_rows = load_jsonl(Path(cfg['paths']['skill_task_dataset']))
    skill_dir = run_dir / 'skill_views'
    assignments = stage_assignments(cfg, len(task_rows))
    views = load_skill_views(skill_dir, {level for _, level in assignments if level != 'zero'})
    rows = []
    for idx, task in enumerate(task_rows):
        stage_name, level = assignments[idx]
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
            'skill_stage': stage_name,
            'skill_level': level,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': assistant},
            ],
        })

    out_path = Path(cfg['paths']['skill0_train_jsonl'])
    ensure_path_is_new(out_path, 'skill0 training dataset')
    write_jsonl(out_path, rows)
    append_run_note(run_dir, [f'Built skill0 dataset: {out_path.name} ({len(rows)} rows).'])
    logger.info('Wrote skill0 dataset to %s (%d rows)', out_path, len(rows))


if __name__ == '__main__':
    main()
