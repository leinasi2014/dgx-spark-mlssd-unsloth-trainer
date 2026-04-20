#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import append_run_note, dump_yaml, ensure_run_dirs, load_config, load_jsonl, logger, write_jsonl

MIN_RESPONSE_CHARS = 12
MAX_RESPONSE_CHARS = 12000


def normalize_prompt_rows(prompt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in prompt_rows:
        prompt_id = str(item.get('prompt_id') or '').strip()
        messages = item.get('messages')
        if not prompt_id or prompt_id in seen:
            continue
        if not isinstance(messages, list) or not messages:
            continue
        normalized_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get('role') or '').strip()
            content = msg.get('content')
            if role not in {'system', 'user', 'assistant'}:
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            normalized_messages.append({'role': role, 'content': content.strip()})
        if not normalized_messages:
            continue
        seen.add(prompt_id)
        out.append({'prompt_id': prompt_id, 'messages': normalized_messages})
    return out


def build_generated_train_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_rows: list[dict[str, Any]] = []
    seen_answers: set[tuple[str, str]] = set()
    for item in raw_rows:
        prompt_messages = item.get('messages') or []
        raw_outputs = item.get('raw_outputs') or []
        for idx, text in enumerate(raw_outputs):
            if not isinstance(text, str):
                continue
            answer = text.strip()
            if not answer or len(answer) < MIN_RESPONSE_CHARS or len(answer) > MAX_RESPONSE_CHARS:
                continue
            dedupe_key = (str(item.get('prompt_id', 'sample')), answer)
            if dedupe_key in seen_answers:
                continue
            seen_answers.add(dedupe_key)
            train_rows.append({'id': f"{item.get('prompt_id', 'sample')}-{idx}", 'messages': [*prompt_messages, {'role': 'assistant', 'content': answer}]})
    return train_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--prepare-prompts', action='store_true')
    parser.add_argument('--write-ssd-config', action='store_true')
    parser.add_argument('--convert-raw', help='Path to raw SSD outputs JSONL from generate_ssd_local.py')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    prompt_path = Path(cfg['paths']['prompt_dataset'])
    prompt_rows = normalize_prompt_rows(load_jsonl(prompt_path))
    local_prompts_path = run_dir / 'ssd_prompts.jsonl'

    if args.prepare_prompts or args.write_ssd_config:
        write_jsonl(local_prompts_path, prompt_rows)
        logger.info('Wrote normalized prompt file: %s (%d rows)', local_prompts_path, len(prompt_rows))
        append_run_note(run_dir, [f'Prepared {len(prompt_rows)} normalized prompts from {prompt_path.name}.'])

    if args.write_ssd_config:
        output_dir = run_dir / 'ml_ssd_like_output'
        ml_ssd_cfg: dict[str, Any] = {
            'model': {'name': cfg['ssd']['model_for_generation'], 'tensor_parallel_size': cfg['ssd']['tensor_parallel_size'], 'gpu_memory_utilization': cfg['ssd'].get('gpu_memory_utilization', 0.85)},
            'dataset': {'name': 'local_prompts_via_wrapper', 'config': 'messages_jsonl', 'split': 'train', 'path': str(local_prompts_path)},
            'output': {'path': str(output_dir)},
            'generation': {'n': cfg['ssd']['samples_per_prompt'], 'temperature': cfg['ssd']['temperature'], 'top_p': cfg['ssd']['top_p'], 'top_k': cfg['ssd']['top_k'], 'repetition_penalty': cfg['ssd'].get('repetition_penalty', 1.0), 'max_tokens': cfg['ssd']['max_new_tokens']},
            'post_process': {'filter_shortest_percent': cfg['ssd'].get('filter_shortest_percent', 10)},
            'notes': {'upstream_generator_expects_hf_dataset': True, 'local_jsonl_generation_is_done_by': 'scripts/generate_ssd_local.py'},
        }
        out_path = Path(cfg['paths']['ml_ssd_repo']) / 'data_generation' / 'config.generated.yaml'
        dump_yaml(out_path, ml_ssd_cfg)
        logger.info('Wrote ml-ssd reference config: %s', out_path)

    if args.convert_raw:
        raw_rows = load_jsonl(Path(args.convert_raw))
        train_rows = build_generated_train_rows(raw_rows)
        train_path = Path(cfg['paths']['ssd_train_jsonl'])
        write_jsonl(train_path, train_rows)
        logger.info('Wrote training jsonl: %s (%d rows)', train_path, len(train_rows))
        append_run_note(run_dir, [f'Converted raw SSD outputs into {len(train_rows)} SFT rows.'])


if __name__ == '__main__':
    main()
