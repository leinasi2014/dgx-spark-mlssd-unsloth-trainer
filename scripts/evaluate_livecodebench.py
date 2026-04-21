#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from transformers import set_seed

from common import ensure_path_is_new, ensure_run_dirs, load_config, logger
from evaluate_codegen import load_model_and_tokenizer
from livecodebench_utils import compute_metrics_from_results, lcb_run, map_to_example, post_process_code, translate_private_test_cases

FENCE_RE = re.compile(r'```(?:python)?\n(.*?)```', re.DOTALL | re.IGNORECASE)

LCB_PROMPT_WITHOUT_STARTER_CODE = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: {problem_description}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
```python
# YOUR CODE HERE
```"""

LCB_PROMPT_WITH_STARTER_CODE = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: {problem_description}

You will use the following starter code to write the solution to the problem and enclose your code within delimiters.
```python
{entry_point}
```"""


def extract_python(text: str) -> str:
    matches = FENCE_RE.findall(text)
    return matches[-1].strip() if matches else text.strip()


def filter_by_contest_month(example: dict[str, Any], allowed_months: set[str]) -> bool:
    contest_date = str(example.get('contest_date') or '')
    return contest_date[:7] in allowed_months


def load_livecodebench_examples(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from datasets import load_dataset

    eval_cfg = cfg.get('evaluation', {}).get('public', {})
    dataset_name = str(eval_cfg.get('dataset_name') or 'livecodebench/code_generation_lite')
    version_tag = str(eval_cfg.get('version_tag') or '').strip()
    dataset_split = str(eval_cfg.get('split') or 'test')
    allowed_months = set(eval_cfg.get('contest_months') or ['2025-02', '2025-03', '2025-04', '2025-05'])
    load_kwargs: dict[str, Any] = {'split': dataset_split, 'trust_remote_code': True}
    if version_tag:
        load_kwargs['version_tag'] = version_tag
    ds = load_dataset(dataset_name, **load_kwargs)
    examples: list[dict[str, Any]] = []
    limit = int(eval_cfg.get('limit', 0) or 0)
    for row in ds:
        row_dict = dict(row)
        if not filter_by_contest_month(row_dict, allowed_months):
            continue
        row_dict['private_test_cases'] = translate_private_test_cases(row_dict['private_test_cases'])
        examples.append(map_to_example(row_dict))
        if limit > 0 and len(examples) >= limit:
            break
    return examples


def ensure_nonempty_examples(examples: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    if examples:
        return
    eval_cfg = cfg.get('evaluation', {}).get('public', {})
    version_tag = str(eval_cfg.get('version_tag') or 'unset')
    contest_months = list(eval_cfg.get('contest_months') or [])
    raise ValueError(
        'LiveCodeBench evaluation resolved to zero examples after filtering. '
        f'Check evaluation.public.version_tag={version_tag!r}, contest_months={contest_months!r}, and limit.'
    )


def prompt_for_example(tokenizer: Any, example: dict[str, Any]) -> str:
    if example['is_stdin']:
        prompt_text = LCB_PROMPT_WITHOUT_STARTER_CODE.format(problem_description=example['prompt'])
    else:
        prompt_text = LCB_PROMPT_WITH_STARTER_CODE.format(problem_description=example['prompt'], entry_point=example['entry_point'])
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt_text}], tokenize=False, add_generation_prompt=True)


def generate_completion(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int, temperature: float, top_p: float, seed_offset: int) -> str:
    import torch

    device = getattr(model, 'device', None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_example(example: dict[str, Any], completions: list[str]) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    for completion in completions:
        code = post_process_code(extract_python(completion))
        result_list = lcb_run(example, code, timeout=6, is_extracted=not example['is_stdin'])
        test_results = [1 if item[0] else 0 for item in result_list]
        per_run.append({
            'code': code,
            'correct': bool(test_results and all(test_results)),
            'test_results': test_results,
            'raw_result_count': len(result_list),
        })
    return {
        'task_id': example['task_id'],
        'difficulty': example['difficulty'],
        'runs': per_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a model on LiveCodeBench with ml-ssd-aligned prompts.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--adapter-subdir', default=None)
    parser.add_argument('--output-subdir', default='eval_livecodebench')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    out_dir = run_dir / args.output_subdir
    ensure_path_is_new(out_dir, 'LiveCodeBench evaluation output directory')
    out_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)

    examples = load_livecodebench_examples(cfg)
    ensure_nonempty_examples(examples, cfg)
    eval_cfg = cfg.get('evaluation', {}).get('public', {})
    n_repeat = int(eval_cfg.get('n_repeat', 1))
    k_list = list(eval_cfg.get('pass_k') or [1])

    if args.dry_run:
        summary = {
            'run_name': cfg['run_name'],
            'status': 'dry-run',
            'dataset_name': eval_cfg.get('dataset_name', 'livecodebench/code_generation_lite'),
            'version_tag': eval_cfg.get('version_tag', ''),
            'n_examples': len(examples),
            'n_repeat': n_repeat,
            'pass_k': k_list,
        }
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        logger.info('Wrote dry-run LiveCodeBench summary to %s', out_dir / 'summary.json')
        return

    model, tokenizer, eval_target = load_model_and_tokenizer(cfg, run_dir, args.adapter_subdir)
    prompts = {example['task_id']: prompt_for_example(tokenizer, example) for example in examples}
    temperature = float(eval_cfg.get('temperature', 0.6))
    top_p = float(eval_cfg.get('top_p', 0.95))
    max_new_tokens = int(eval_cfg.get('max_tokens', 32768))

    example_results: list[dict[str, Any]] = []
    pass_inputs: dict[str, list[list[int]]] = {}
    for example in examples:
        completions = []
        for repeat_idx in range(n_repeat):
            completions.append(
                generate_completion(
                    model,
                    tokenizer,
                    prompts[example['task_id']],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed_offset=repeat_idx,
                )
            )
        result = evaluate_example(example, completions)
        example_results.append(result)
        pass_inputs[result['task_id']] = [run['test_results'] for run in result['runs']]

    summary_metrics = compute_metrics_from_results(pass_inputs, k_list=k_list)
    summary = {
        'run_name': cfg['run_name'],
        'eval_target': eval_target,
        'dataset_name': eval_cfg.get('dataset_name', 'livecodebench/code_generation_lite'),
        'version_tag': eval_cfg.get('version_tag', ''),
        'n_examples': len(examples),
        'n_repeat': n_repeat,
        'metrics': summary_metrics,
    }

    (out_dir / 'results.json').write_text(json.dumps(example_results, indent=2, ensure_ascii=False), encoding='utf-8')
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info('Wrote LiveCodeBench summary to %s', out_dir / 'summary.json')


if __name__ == '__main__':
    main()
