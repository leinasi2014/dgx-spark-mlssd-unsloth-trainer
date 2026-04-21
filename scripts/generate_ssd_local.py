#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from common import append_run_note, ensure_path_is_new, ensure_run_dirs, load_config, load_jsonl, logger, write_jsonl
from prepare_ssd_data import ml_ssd_output_dir


def maybe_write_parquet(path: Path, rows: list[dict]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        logger.warning('Skipping ml-ssd parquet export because pyarrow is unavailable: %s', exc)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def prompt_token_budgets(tokenizer: AutoTokenizer, prompts: list[str], max_context: int, requested_max_tokens: int) -> list[int]:
    budgets: list[int] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = encoded.get('input_ids', [])
        prompt_length = len(prompt_tokens)
        remaining = max_context - prompt_length
        if remaining <= 0:
            raise ValueError(
                f'Prompt length {prompt_length} tokens leaves no room for generation within max_seq_length={max_context}. '
                'Shorten the prompt template or increase model.max_seq_length.'
            )
        budgets.append(min(requested_max_tokens, remaining))
    return budgets


def ssd_max_model_len(cfg: dict) -> int:
    return int(cfg.get('ssd', {}).get('max_model_len') or cfg['model']['max_seq_length'])


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate ml-ssd-aligned local samples with vLLM.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    prompt_path = run_dir / 'ssd_prompts.jsonl'
    if not prompt_path.exists():
        raise FileNotFoundError(f'Missing prepared prompt file: {prompt_path}. Run prepare_ssd_data.py --prepare-prompts first.')

    rows = load_jsonl(prompt_path)
    if args.limit > 0:
        rows = rows[:args.limit]

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        cfg['ssd']['model_for_generation'],
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
    )
    prompts = [str(row['prompt']) for row in rows]

    llm = LLM(
        model=cfg['ssd']['model_for_generation'],
        tensor_parallel_size=cfg['ssd']['tensor_parallel_size'],
        gpu_memory_utilization=cfg['ssd'].get('gpu_memory_utilization', 0.85),
        max_model_len=ssd_max_model_len(cfg),
        dtype='bfloat16',
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
        enforce_eager=False,
    )
    max_context = ssd_max_model_len(cfg)
    requested_max_tokens = int(cfg['ssd']['max_new_tokens'])
    budgets = prompt_token_budgets(tokenizer, prompts, max_context, requested_max_tokens)
    if budgets and min(budgets) < requested_max_tokens:
        logger.warning(
            'Capped per-prompt SSD generation tokens to fit context window: requested=%d min_effective=%d max_effective=%d max_seq_length=%d',
            requested_max_tokens,
            min(budgets),
            max(budgets),
            max_context,
        )
    sampling_params_list = [
        SamplingParams(
            temperature=cfg['ssd']['temperature'],
            top_p=cfg['ssd']['top_p'],
            top_k=cfg['ssd']['top_k'],
            repetition_penalty=cfg['ssd'].get('repetition_penalty', 1.0),
            max_tokens=max_tokens,
            skip_special_tokens=True,
            stop=['<|im_end|>', '<|endoftext|>'],
        )
        for max_tokens in budgets
    ]

    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params_list, use_tqdm=True)

    ml_ssd_rows = []
    raw_rows = []
    for row, output in zip(rows, outputs):
        answer = output.outputs[0].text.strip() if output.outputs else ''
        ml_ssd_rows.append({
            'question_id': row['prompt_id'],
            'question': row.get('question', ''),
            'starter_code': row.get('starter_code', ''),
            'problem_type': row.get('problem_type', 'stdin'),
            'prompt': row['prompt'],
            'output': answer,
            'source_name': row.get('source_name', 'code'),
        })
        raw_rows.append({
            'prompt_id': row['prompt_id'],
            'prompt': row['prompt'],
            'question': row.get('question', ''),
            'starter_code': row.get('starter_code', ''),
            'problem_type': row.get('problem_type', 'stdin'),
            'source_name': row.get('source_name', 'code'),
            'raw_outputs': [candidate.text.strip() for candidate in output.outputs],
        })

    raw_path = run_dir / 'raw_ssd_outputs.jsonl'
    ensure_path_is_new(raw_path, 'raw SSD outputs file')
    write_jsonl(raw_path, raw_rows)
    logger.info('Wrote raw SSD outputs: %s (%d prompts)', raw_path, len(raw_rows))

    out_dir = ml_ssd_output_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maybe_write_parquet(out_dir / 'train.parquet', ml_ssd_rows)
    append_run_note(run_dir, [f'Generated raw SSD samples for {len(raw_rows)} prompts aligned to ml-ssd output semantics.'])


if __name__ == '__main__':
    main()
