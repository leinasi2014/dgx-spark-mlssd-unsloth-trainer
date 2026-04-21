#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from common import append_run_note, ensure_path_is_new, ensure_run_dirs, load_config, logger

DEFAULT_STAGE_CONFIGS = {
    'code': {
        'source': 'rstar_coder_seed_sft',
        'output_subdir': 'adapter_code_high_rank',
        'squeezed_output_subdir': 'adapter_code_squeezed',
        'recovered_output_subdir': 'adapter_code_recovered',
    },
    'skill0': {
        'output_subdir': 'adapter_skill0_high_rank',
        'squeezed_output_subdir': 'adapter_skill0_squeezed',
        'recovered_output_subdir': 'adapter_skill0_recovered',
    },
    'agent': {
        'source': 'coderforge_preview',
        'output_subdir': 'adapter_agent_high_rank',
        'squeezed_output_subdir': 'adapter_agent_squeezed',
        'recovered_output_subdir': 'adapter_agent_recovered',
    },
    'mixed': {
        'output_subdir': 'adapter_mixed_high_rank',
        'squeezed_output_subdir': 'adapter_mixed_squeezed',
        'recovered_output_subdir': 'adapter_mixed_recovered',
        'sources': [
            {'source': 'code', 'weight': 0.7},
            {'source': 'skill0', 'weight': 0.3},
        ],
    },
}


def shell_join(parts: list[str]) -> str:
    return ' '.join(shlex.quote(part) for part in parts)


def script_command(python_bin: str, script_path: Path, *args: str) -> str:
    return shell_join([python_bin, str(script_path), *args])


def resolve_plan(raw_plan: str | None, cfg: dict | None = None) -> str:
    if raw_plan is None and cfg is not None:
        raw_plan = cfg.get('training_plan', {}).get('plan')
        if raw_plan is None:
            legacy_mode = str(cfg.get('training_plan', {}).get('mode', 'sequential')).strip().lower()
            raw_plan = 'mixed_sources' if legacy_mode == 'mixed' else 'code_then_skill0'
    plan = str(raw_plan or 'code_then_skill0').strip().lower()
    if plan == 'sequential':
        return 'code_then_skill0'
    if plan == 'mixed':
        return 'mixed_sources'
    supported = {'code_only', 'code_then_skill0', 'code_then_agent', 'mixed_sources'}
    if plan not in supported:
        raise ValueError(f'Unsupported training_plan.plan: {raw_plan!r}')
    return plan


def resolve_plan_python(cfg: dict) -> str:
    project_root = Path(cfg['paths']['project_root'])
    venv_python = project_root / '.venv' / 'bin' / 'python'
    return str(venv_python) if venv_python.exists() else sys.executable


def public_eval_command(cfg_path: str, python_bin: str, project_root: Path, adapter_subdir: str) -> str:
    return script_command(python_bin, project_root / 'scripts' / 'evaluate_livecodebench.py', '--config', cfg_path, '--adapter-subdir', adapter_subdir)


def training_plan_section(cfg: dict, section_name: str) -> dict:
    defaults = dict(DEFAULT_STAGE_CONFIGS[section_name])
    configured = cfg.get('training_plan', {}).get(section_name)
    if not isinstance(configured, dict):
        return defaults
    merged = dict(defaults)
    merged.update(configured)
    return merged


def postprocess_commands(
    cfg_path: str,
    cfg: dict,
    python_bin: str,
    project_root: Path,
    dataset_path: str,
    output_subdir: str,
    squeezed_output_subdir: str,
    recovered_output_subdir: str,
) -> list[str]:
    commands: list[str] = []
    final_adapter_subdir = output_subdir
    squeeze_enabled = bool(cfg.get('lora_squeeze', {}).get('enabled', True))
    recovery_enabled = bool(cfg.get('recovery', {}).get('enabled', True))
    if recovery_enabled and not squeeze_enabled:
        raise ValueError('recovery.enabled=true requires lora_squeeze.enabled=true when materializing a training plan.')
    if squeeze_enabled:
        commands.append(
            script_command(
                python_bin,
                project_root / 'scripts' / 'squeeze_lora.py',
                '--config',
                cfg_path,
                '--source-subdir',
                output_subdir,
                '--output-subdir',
                squeezed_output_subdir,
            )
        )
        final_adapter_subdir = squeezed_output_subdir
    if recovery_enabled:
        commands.append(
            script_command(
                python_bin,
                project_root / 'scripts' / 'recover_after_squeeze.py',
                '--config',
                cfg_path,
                '--dataset-path',
                dataset_path,
                '--source-subdir',
                final_adapter_subdir,
                '--output-subdir',
                recovered_output_subdir,
            )
        )
        final_adapter_subdir = recovered_output_subdir
    commands.append(public_eval_command(cfg_path, python_bin, project_root, final_adapter_subdir))
    return commands


def code_only_commands(cfg_path: str, cfg: dict, python_bin: str) -> list[str]:
    project_root = Path(cfg['paths']['project_root'])
    code = training_plan_section(cfg, 'code')
    return [
        script_command(python_bin, project_root / 'scripts' / 'train_unsloth_lora.py', '--config', cfg_path, '--dataset-path', cfg['paths']['ssd_train_jsonl'], '--output-subdir', code['output_subdir']),
    ] + postprocess_commands(
        cfg_path,
        cfg,
        python_bin,
        project_root,
        cfg['paths']['ssd_train_jsonl'],
        code['output_subdir'],
        code['squeezed_output_subdir'],
        code['recovered_output_subdir'],
    )


def code_then_skill0_commands(cfg_path: str, cfg: dict, run_dir: Path, python_bin: str) -> list[str]:
    project_root = Path(cfg['paths']['project_root'])
    code = training_plan_section(cfg, 'code')
    skill = training_plan_section(cfg, 'skill0')
    return [
        script_command(python_bin, project_root / 'scripts' / 'train_unsloth_lora.py', '--config', cfg_path, '--dataset-path', cfg['paths']['ssd_train_jsonl'], '--output-subdir', code['output_subdir']),
        script_command(python_bin, project_root / 'scripts' / 'train_unsloth_lora.py', '--config', cfg_path, '--dataset-path', cfg['paths']['skill0_train_jsonl'], '--output-subdir', skill['output_subdir'], '--init-adapter', str(run_dir / code['output_subdir'])),
    ] + postprocess_commands(
        cfg_path,
        cfg,
        python_bin,
        project_root,
        cfg['paths']['skill0_train_jsonl'],
        skill['output_subdir'],
        skill['squeezed_output_subdir'],
        skill['recovered_output_subdir'],
    )


def code_then_agent_commands(cfg_path: str, cfg: dict, run_dir: Path, python_bin: str) -> list[str]:
    project_root = Path(cfg['paths']['project_root'])
    code = training_plan_section(cfg, 'code')
    agent = training_plan_section(cfg, 'agent')
    agent_dataset = run_dir / 'prepared_sources' / f"{agent['source']}.jsonl"
    return [
        script_command(python_bin, project_root / 'scripts' / 'train_unsloth_lora.py', '--config', cfg_path, '--dataset-path', cfg['paths']['ssd_train_jsonl'], '--output-subdir', code['output_subdir']),
        script_command(python_bin, project_root / 'scripts' / 'train_unsloth_lora.py', '--config', cfg_path, '--dataset-path', str(agent_dataset), '--output-subdir', agent['output_subdir'], '--init-adapter', str(run_dir / code['output_subdir'])),
    ] + postprocess_commands(
        cfg_path,
        cfg,
        python_bin,
        project_root,
        str(agent_dataset),
        agent['output_subdir'],
        agent['squeezed_output_subdir'],
        agent['recovered_output_subdir'],
    )


def mixed_sources_commands(cfg_path: str, cfg: dict, python_bin: str) -> list[str]:
    project_root = Path(cfg['paths']['project_root'])
    mixed = training_plan_section(cfg, 'mixed')
    return [
        script_command(python_bin, project_root / 'scripts' / 'train_unsloth_lora.py', '--config', cfg_path, '--dataset-path', cfg['paths']['mixed_train_jsonl'], '--output-subdir', mixed['output_subdir']),
    ] + postprocess_commands(
        cfg_path,
        cfg,
        python_bin,
        project_root,
        cfg['paths']['mixed_train_jsonl'],
        mixed['output_subdir'],
        mixed['squeezed_output_subdir'],
        mixed['recovered_output_subdir'],
    )


def prepare_skill_assets(cfg_path: str) -> None:
    from build_skill_views import main as build_skill_views_main  # type: ignore
    from build_skill0_dataset import main as build_skill0_dataset_main  # type: ignore

    old_argv = sys.argv[:]
    try:
        sys.argv = ['build_skill_views.py', '--config', cfg_path]
        build_skill_views_main()
        sys.argv = ['build_skill0_dataset.py', '--config', cfg_path]
        build_skill0_dataset_main()
    finally:
        sys.argv = old_argv


def prepare_agent_source(cfg_path: str, source_name: str) -> None:
    from prepare_agent_data import main as prepare_agent_data_main  # type: ignore

    old_argv = sys.argv[:]
    try:
        sys.argv = ['prepare_agent_data.py', '--config', cfg_path, '--source-name', source_name]
        prepare_agent_data_main()
    finally:
        sys.argv = old_argv


def referenced_mixed_external_sources(cfg: dict) -> list[str]:
    mixed = training_plan_section(cfg, 'mixed')
    configured_sources = mixed.get('sources')
    if not isinstance(configured_sources, list):
        return []
    source_names: list[str] = []
    for item in configured_sources:
        if not isinstance(item, dict):
            continue
        source_name = str(item.get('source') or '').strip()
        if not source_name or source_name in {'code', 'skill0', 'mixed'}:
            continue
        if float(item.get('weight', 0.0)) <= 0:
            continue
        if source_name not in source_names:
            source_names.append(source_name)
    return source_names


def prepare_external_mixed_sources(cfg_path: str, cfg: dict) -> None:
    from source_adapters import get_source_config  # type: ignore

    code_source = training_plan_section(cfg, 'code').get('source')
    for source_name in referenced_mixed_external_sources(cfg):
        if source_name == code_source:
            continue
        source_cfg = get_source_config(cfg, source_name)
        family = str(source_cfg.get('family') or '').strip()
        if family == 'agent_trajectory':
            prepare_agent_source(cfg_path, source_name)
            continue
        raise ValueError(
            f'Mixed source {source_name} uses family={family!r}. '
            'Only code/skill0 aliases and prepared agent_trajectory sources are currently supported in mixed_sources.'
        )


def prepare_mixed_dataset(cfg_path: str) -> None:
    from build_mixed_dataset import main as build_mixed_dataset_main  # type: ignore

    old_argv = sys.argv[:]
    try:
        sys.argv = ['build_mixed_dataset.py', '--config', cfg_path]
        build_mixed_dataset_main()
    finally:
        sys.argv = old_argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prepare and materialize stage-based training plans.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--plan', choices=['code_only', 'code_then_skill0', 'code_then_agent', 'mixed_sources', 'sequential', 'mixed'], default=None)
    parser.add_argument('--mode', choices=['sequential', 'mixed'], default=None, help='Legacy alias kept for compatibility with the original scaffold commands.')
    parser.add_argument('--prepare-only', action='store_true')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg_path = str(Path(args.config).resolve())
    cfg = load_config(cfg_path)
    run_dir = ensure_run_dirs(cfg)
    requested_plan = args.plan if args.plan is not None else args.mode
    plan = resolve_plan(requested_plan, cfg)

    if plan in {'code_then_skill0', 'mixed_sources'}:
        prepare_skill_assets(cfg_path)
    if plan == 'code_then_agent':
        prepare_agent_source(cfg_path, str(training_plan_section(cfg, 'agent')['source']))
    if plan == 'mixed_sources':
        prepare_external_mixed_sources(cfg_path, cfg)
        prepare_mixed_dataset(cfg_path)

    plan_python = resolve_plan_python(cfg)
    if plan == 'code_only':
        commands = code_only_commands(cfg_path, cfg, plan_python)
    elif plan == 'code_then_skill0':
        commands = code_then_skill0_commands(cfg_path, cfg, run_dir, plan_python)
    elif plan == 'code_then_agent':
        commands = code_then_agent_commands(cfg_path, cfg, run_dir, plan_python)
    else:
        commands = mixed_sources_commands(cfg_path, cfg, plan_python)

    plan_path = run_dir / f'{plan}_plan.sh'
    ensure_path_is_new(plan_path, 'training plan file')
    plan_path.write_text('#!/usr/bin/env bash\nset -euo pipefail\n\n' + '\n'.join(commands) + '\n', encoding='utf-8')
    append_run_note(run_dir, [f'Prepared {plan} training plan at {plan_path.name}.'])
    logger.info('Prepared %s training plan at %s', plan, plan_path)
    print('\n'.join(commands))


if __name__ == '__main__':
    main()
