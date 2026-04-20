#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import append_run_note, ensure_run_dirs, load_config, logger


def compact_lines(text: str, limit: int) -> str:
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith('```'):
            continue
        out.append(line)
        if len(out) >= limit:
            break
    return '\n'.join(out).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description='Build full, summary, and tool-only skill views from repo docs.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    root = Path(cfg['paths']['project_root'])

    agents_md = (root / 'AGENTS.md').read_text(encoding='utf-8')
    skill_md = (root / '.codex' / 'skills' / 'dgx-spark-code-trainer' / 'SKILL.md').read_text(encoding='utf-8')
    refs_md = (root / 'docs' / 'references.md').read_text(encoding='utf-8')

    full = '\n\n'.join([
        '# Skill bundle (full)',
        '## AGENTS.md',
        agents_md.strip(),
        '## Codex skill',
        skill_md.strip(),
        '## References',
        refs_md.strip(),
    ])
    summary = '\n'.join([
        '# Skill bundle (summary)',
        compact_lines(agents_md, int(cfg['skill0'].get('summary_max_lines', 14))),
        compact_lines(skill_md, int(cfg['skill0'].get('summary_max_lines', 14))),
        'See docs/references.md for the canonical upstream paper and repository list.',
    ]).strip()
    tool_only = '\n'.join([
        '# Skill bundle (tool-only)',
        'Use these repository tools and constraints:',
        compact_lines(skill_md, int(cfg['skill0'].get('tool_only_max_lines', 10))),
    ]).strip()

    out_dir = run_dir / 'skill_views'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'full.md').write_text(full + '\n', encoding='utf-8')
    (out_dir / 'summary.md').write_text(summary + '\n', encoding='utf-8')
    (out_dir / 'tool_only.md').write_text(tool_only + '\n', encoding='utf-8')
    append_run_note(run_dir, ['Built skill views: full.md, summary.md, tool_only.md.'])
    logger.info('Wrote skill views to %s', out_dir)


if __name__ == '__main__':
    main()
