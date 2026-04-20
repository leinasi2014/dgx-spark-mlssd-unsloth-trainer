#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from common import append_run_note, ensure_run_dirs, load_config, load_jsonl, logger, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate local SSD samples with vLLM using a messages JSONL prompt set.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    prompt_path = run_dir / 'ssd_prompts.jsonl'
    if not prompt_path.exists():
        raise FileNotFoundError(f'Missing normalized prompts file: {prompt_path}. Run prepare_ssd_data.py --prepare-prompts first.')

    rows = load_jsonl(prompt_path)
    if args.limit > 0:
        rows = rows[:args.limit]

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(cfg['ssd']['model_for_generation'], trust_remote_code=cfg['model'].get('trust_remote_code', True))
    prompts = [tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True) for row in rows]

    llm = LLM(
        model=cfg['ssd']['model_for_generation'],
        tensor_parallel_size=cfg['ssd']['tensor_parallel_size'],
        gpu_memory_utilization=cfg['ssd'].get('gpu_memory_utilization', 0.85),
        max_model_len=cfg['ssd']['max_new_tokens'],
        dtype='bfloat16',
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
        enforce_eager=False,
    )
    sampling_params = SamplingParams(
        n=cfg['ssd']['samples_per_prompt'],
        temperature=cfg['ssd']['temperature'],
        top_p=cfg['ssd']['top_p'],
        top_k=cfg['ssd']['top_k'],
        repetition_penalty=cfg['ssd'].get('repetition_penalty', 1.0),
        max_tokens=cfg['ssd']['max_new_tokens'],
        skip_special_tokens=True,
        stop=['<|im_end|>', '<|endoftext|>'],
    )

    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=True)
    raw_rows = [{'prompt_id': row['prompt_id'], 'messages': row['messages'], 'raw_outputs': [candidate.text.strip() for candidate in output.outputs]} for row, output in zip(rows, outputs)]
    raw_path = run_dir / 'raw_ssd_outputs.jsonl'
    write_jsonl(raw_path, raw_rows)
    logger.info('Wrote raw SSD outputs: %s (%d prompts)', raw_path, len(raw_rows))
    append_run_note(run_dir, [f'Generated raw SSD samples for {len(raw_rows)} prompts with n={cfg["ssd"]["samples_per_prompt"]}.'])


if __name__ == '__main__':
    main()
