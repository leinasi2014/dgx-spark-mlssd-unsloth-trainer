#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import append_run_note, dump_yaml, ensure_path_is_new, ensure_run_dirs, load_config, load_jsonl, logger, write_jsonl
from source_adapters import format_problem_code_rows, get_source_config, source_limit

MIN_RESPONSE_CHARS = 12
MAX_RESPONSE_CHARS = 12000


def raw_prompt_text(item: dict[str, Any]) -> str:
    prompt = str(item.get('prompt') or '').strip()
    if prompt:
        return prompt
    messages = item.get('messages')
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if str(message.get('role') or '').strip().lower() != 'user':
                continue
            content = str(message.get('content') or '').strip()
            if content:
                return content
    raise ValueError(f'Raw SSD row is missing a usable prompt/messages payload: {item.get("prompt_id", "sample")}')


def build_generated_train_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_rows: list[dict[str, Any]] = []
    seen_answers: set[tuple[str, str]] = set()
    for item in raw_rows:
        prompt = raw_prompt_text(item)
        raw_outputs = item.get('raw_outputs') or []
        prompt_id = str(item.get('prompt_id') or 'sample')
        for idx, text in enumerate(raw_outputs):
            if not isinstance(text, str):
                continue
            answer = text.strip()
            if not answer or len(answer) < MIN_RESPONSE_CHARS or len(answer) > MAX_RESPONSE_CHARS:
                continue
            dedupe_key = (prompt_id, answer)
            if dedupe_key in seen_answers:
                continue
            seen_answers.add(dedupe_key)
            train_rows.append({
                'id': f'{prompt_id}-{idx}',
                'source_name': item.get('source_name', 'code'),
                'messages': [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': answer},
                ],
            })
    return train_rows


def code_source_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get('training_plan', {}).get('code', {}).get('source') or 'rstar_coder_seed_sft')


def ml_ssd_output_dir(run_dir: Path) -> Path:
    return run_dir / 'ml_ssd_like_output'


def ml_ssd_templates_dir(cfg: dict[str, Any]) -> Path:
    configured = cfg.get('ssd', {}).get('templates', {})
    if isinstance(configured, dict) and configured.get('template_root'):
        return Path(str(configured['template_root']))
    return Path(cfg['paths']['project_root']) / 'scripts' / 'ml_ssd_templates'


def prepare_problem_rows(cfg: dict[str, Any], limit: int = 0) -> tuple[str, list[dict[str, Any]]]:
    source_name = code_source_name(cfg)
    source_cfg = get_source_config(cfg, source_name)
    if str(source_cfg.get('family') or '').strip() != 'problem_code':
        raise ValueError(f'Code source must use family=problem_code, got {source_cfg.get("family")!r} for {source_name}')
    rows = format_problem_code_rows(
        source_name,
        source_cfg,
        ml_ssd_templates_dir(cfg),
        limit=source_limit(source_cfg, limit),
    )
    return source_name, rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare and convert ml-ssd-aligned problem-code data.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--prepare-prompts', action='store_true', help='Load the configured problem-code dataset and render ml-ssd prompts.')
    parser.add_argument('--write-ssd-config', action='store_true', help='Write a reference ml-ssd config aligned to the prepared prompts.')
    parser.add_argument('--convert-raw', help='Path to raw SSD outputs JSONL from generate_ssd_local.py')
    parser.add_argument('--limit', type=int, default=0, help='Optional row cap applied during prompt preparation.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    prompt_path = run_dir / 'ssd_prompts.jsonl'

    if args.prepare_prompts or args.write_ssd_config:
        source_name, prompt_rows = prepare_problem_rows(cfg, limit=args.limit)
        ensure_path_is_new(prompt_path, 'ml-ssd prompt file')
        write_jsonl(prompt_path, prompt_rows)
        logger.info('Wrote ml-ssd prompt file: %s (%d rows)', prompt_path, len(prompt_rows))
        append_run_note(run_dir, [f'Prepared {len(prompt_rows)} problem-code prompts from source {source_name}.'])

        if args.write_ssd_config:
            source_cfg = get_source_config(cfg, source_name)
            dataset_cfg = source_cfg.get('dataset', {})
            output_dir = ml_ssd_output_dir(run_dir)
            ml_ssd_cfg: dict[str, Any] = {
                'model': {
                    'name': cfg['ssd']['model_for_generation'],
                    'max_model_len': cfg['ssd'].get('max_model_len', cfg['model']['max_seq_length']),
                    'tensor_parallel_size': cfg['ssd']['tensor_parallel_size'],
                    'gpu_memory_utilization': cfg['ssd'].get('gpu_memory_utilization', 0.85),
                },
                'dataset': {
                    'name': dataset_cfg.get('name'),
                    'config': dataset_cfg.get('config'),
                    'split': dataset_cfg.get('split', 'train'),
                },
                'output': {'path': str(output_dir)},
                'generation': {
                    'temperature': cfg['ssd']['temperature'],
                    'top_p': cfg['ssd']['top_p'],
                    'top_k': cfg['ssd']['top_k'],
                    'repetition_penalty': cfg['ssd'].get('repetition_penalty', 1.0),
                    'max_tokens': cfg['ssd']['max_new_tokens'],
                },
                'post_process': {'filter_shortest_percent': cfg['ssd'].get('filter_shortest_percent', 10)},
                'notes': {
                    'local_wrapper': 'scripts/prepare_ssd_data.py + scripts/generate_ssd_local.py',
                    'prepared_prompt_path': str(prompt_path),
                    'code_source': source_name,
                },
            }
            out_path = run_dir / 'ml_ssd_config.generated.yaml'
            ensure_path_is_new(out_path, 'ml-ssd generated config')
            dump_yaml(out_path, ml_ssd_cfg)
            logger.info('Wrote ml-ssd reference config: %s', out_path)

    if args.convert_raw:
        raw_rows = load_jsonl(Path(args.convert_raw))
        train_rows = build_generated_train_rows(raw_rows)
        train_path = Path(cfg['paths']['ssd_train_jsonl'])
        ensure_path_is_new(train_path, 'SSD training dataset')
        write_jsonl(train_path, train_rows)
        logger.info('Wrote training jsonl: %s (%d rows)', train_path, len(train_rows))
        append_run_note(run_dir, [f'Converted raw SSD outputs into {len(train_rows)} SFT rows at {train_path.name}.'])


if __name__ == '__main__':
    main()
