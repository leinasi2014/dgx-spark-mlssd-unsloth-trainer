#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import set_seed

from common import choose_unsloth_loader, ensure_path_is_new, ensure_run_dirs, load_config, load_jsonl, logger

FENCE_RE = re.compile(r'```(?:python)?\n(.*?)```', re.DOTALL | re.IGNORECASE)


def resolve_eval_prompt(row: dict[str, Any], tokenizer: Any) -> str:
    messages = row.get('messages')
    if isinstance(messages, list) and messages:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = row.get('prompt')
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    raise ValueError(f'Row is missing prompt/messages: {row.get("id", "unknown")}')


def extract_python(text: str) -> str:
    matches = FENCE_RE.findall(text)
    return matches[0].strip() if matches else text.strip()


def preferred_eval_family(cfg: dict[str, Any]) -> str:
    training_plan = cfg.get('training_plan', {})
    plan = str(training_plan.get('plan') or '').strip().lower()
    legacy_mode = str(training_plan.get('mode', 'sequential')).strip().lower()
    if plan == 'mixed_sources':
        return 'mixed'
    if plan == 'code_then_agent':
        return 'agent'
    if plan == 'code_only':
        return 'code'
    if plan == 'code_then_skill0':
        return 'skill0'
    if legacy_mode == 'mixed':
        return 'mixed'
    return 'skill0'


def resolve_adapter(run_dir: Path, explicit: str | None, preferred_family: str | None = None) -> tuple[Path | None, str]:
    if explicit:
        adapter_dir = run_dir / explicit
        if not (adapter_dir / 'adapter_config.json').exists():
            raise FileNotFoundError(f'Explicit adapter was not found or is incomplete: {adapter_dir}')
        return adapter_dir, explicit
    candidate_groups = {
        'mixed': ['adapter_mixed_recovered', 'adapter_mixed_squeezed', 'adapter_mixed_high_rank'],
        'agent': ['adapter_agent_recovered', 'adapter_agent_squeezed', 'adapter_agent_high_rank'],
        'skill0': ['adapter_skill0_recovered', 'adapter_skill0_squeezed', 'adapter_skill0_high_rank'],
        'code': ['adapter_code_recovered', 'adapter_code_squeezed', 'adapter_code_high_rank'],
        'generic': ['adapter_recovered', 'adapter_squeezed', 'adapter_high_rank'],
    }
    order = {
        'mixed': ['mixed', 'skill0', 'agent', 'code', 'generic'],
        'agent': ['agent', 'skill0', 'mixed', 'code', 'generic'],
        'code': ['code', 'skill0', 'agent', 'mixed', 'generic'],
        'skill0': ['skill0', 'mixed', 'agent', 'code', 'generic'],
    }.get(preferred_family or 'skill0', ['skill0', 'mixed', 'agent', 'code', 'generic'])
    candidates = [candidate for family in order for candidate in candidate_groups[family]]
    for candidate in candidates:
        adapter_dir = run_dir / candidate
        if (adapter_dir / 'adapter_config.json').exists():
            return adapter_dir, candidate
    return None, 'base_model'


def load_model_and_tokenizer(cfg: dict[str, Any], run_dir: Path, adapter_subdir: str | None) -> tuple[Any, Any, str]:
    loader, _loader_name = choose_unsloth_loader(cfg['model']['base_model'], cfg['model'].get('unsloth_loader', 'auto'))
    model, tokenizer = loader.from_pretrained(
        model_name=cfg['model']['base_model'],
        max_seq_length=cfg['model']['max_seq_length'],
        load_in_4bit=cfg['model']['load_in_4bit'],
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
    )
    adapter_dir, tag = resolve_adapter(run_dir, adapter_subdir, preferred_eval_family(cfg))
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    return model.eval(), tokenizer, tag


def model_input_device(model: Any) -> Any:
    device = getattr(model, 'device', None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_completion(model: Any, tokenizer: Any, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
    device = model_input_device(model)
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
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


def local_smoke_dataset_path(cfg: dict[str, Any]) -> Path:
    configured = cfg.get('evaluation', {}).get('local_smoke', {}).get('dataset_path')
    return Path(str(configured or cfg['paths']['eval_dataset']))


def run_tests(code: str, tests: list[str], timeout_seconds: int = 10) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix='eval_codegen_') as tmpdir:
        test_file = Path(tmpdir) / 'candidate_test.py'
        payload = code.rstrip() + '\n\n' + '\n'.join(tests) + '\n'
        test_file.write_text(payload, encoding='utf-8')
        try:
            proc = subprocess.run([sys.executable, str(test_file)], capture_output=True, text=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or '') + (exc.stderr or '') + f'\nTimed out after {timeout_seconds} seconds.'
            return False, output[-4000:]
        output = (proc.stdout or '') + (proc.stderr or '')
        return proc.returncode == 0, output[-4000:]


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a coding adapter with executable Python tests.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--adapter-subdir', default=None)
    parser.add_argument('--output-subdir', default='eval')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    set_seed(args.seed)
    eval_rows = load_jsonl(local_smoke_dataset_path(cfg))
    if args.limit > 0:
        eval_rows = eval_rows[:args.limit]

    out_dir = run_dir / args.output_subdir
    ensure_path_is_new(out_dir, 'evaluation output directory')
    out_dir.mkdir(parents=True, exist_ok=False)

    if args.dry_run:
        summary = {
            'run_name': cfg['run_name'],
            'temperatures': cfg['inference']['temperature_sweep'],
            'n_tasks': len(eval_rows),
            'adapter_subdir': args.adapter_subdir,
            'seed': args.seed,
            'status': 'dry-run',
        }
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        logger.info('Wrote dry-run eval summary to %s', out_dir / 'summary.json')
        return

    model, tokenizer, eval_target = load_model_and_tokenizer(cfg, run_dir, args.adapter_subdir)
    aggregate: dict[str, Any] = {'run_name': cfg['run_name'], 'eval_target': eval_target, 'seed': args.seed, 'temperatures': {}}

    for temperature in cfg['inference']['temperature_sweep']:
        passed = 0
        total = 0
        per_temp_rows = []
        for row in eval_rows:
            prompt = resolve_eval_prompt(row, tokenizer)
            completion = generate_completion(
                model,
                tokenizer,
                prompt,
                temperature=float(temperature),
                top_p=float(cfg['inference']['top_p']),
                max_new_tokens=int(cfg['inference']['max_new_tokens']),
            )
            code = extract_python(completion)
            ok, output = run_tests(code, row.get('tests') or [])
            passed += int(ok)
            total += 1
            per_temp_rows.append({'id': row.get('id'), 'temperature': temperature, 'passed': ok, 'code': code, 'test_output': output})
        aggregate['temperatures'][str(temperature)] = {'passed': passed, 'total': total, 'pass_rate': round(passed / max(total, 1), 4)}
        write_path = out_dir / f'results_t{str(temperature).replace(".", "_")}.jsonl'
        write_path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in per_temp_rows) + '\n', encoding='utf-8')

    (out_dir / 'summary.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    logger.info('Wrote eval summary to %s', out_dir / 'summary.json')


if __name__ == '__main__':
    main()
