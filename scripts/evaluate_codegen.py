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

from common import choose_unsloth_loader, ensure_run_dirs, load_config, load_jsonl, logger

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


def resolve_adapter(run_dir: Path, explicit: str | None) -> tuple[Path | None, str]:
    if explicit:
        adapter_dir = run_dir / explicit
        return (adapter_dir if (adapter_dir / 'adapter_config.json').exists() else None), explicit
    for candidate in [
        'adapter_skill0_recovered',
        'adapter_mixed_recovered',
        'adapter_recovered',
        'adapter_skill0_squeezed',
        'adapter_mixed_squeezed',
        'adapter_squeezed',
        'adapter_skill0_high_rank',
        'adapter_mixed_high_rank',
        'adapter_code_high_rank',
        'adapter_high_rank',
    ]:
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
    adapter_dir, tag = resolve_adapter(run_dir, adapter_subdir)
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    return model.eval(), tokenizer, tag


def generate_completion(model: Any, tokenizer: Any, prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if hasattr(model, 'to'):
        model = model.to(device)
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


def run_tests(code: str, tests: list[str], timeout_seconds: int = 10) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix='eval_codegen_') as tmpdir:
        test_file = Path(tmpdir) / 'candidate_test.py'
        payload = code.rstrip() + '\n\n' + '\n'.join(tests) + '\n'
        test_file.write_text(payload, encoding='utf-8')
        proc = subprocess.run([sys.executable, str(test_file)], capture_output=True, text=True, timeout=timeout_seconds)
        output = (proc.stdout or '') + (proc.stderr or '')
        return proc.returncode == 0, output[-4000:]


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a coding adapter with executable Python tests.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--adapter-subdir', default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    eval_rows = load_jsonl(Path(cfg['paths']['eval_dataset']))
    if args.limit > 0:
        eval_rows = eval_rows[:args.limit]

    out_dir = run_dir / 'eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        summary = {
            'run_name': cfg['run_name'],
            'temperatures': cfg['inference']['temperature_sweep'],
            'n_tasks': len(eval_rows),
            'adapter_subdir': args.adapter_subdir,
            'status': 'dry-run',
        }
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        logger.info('Wrote dry-run eval summary to %s', out_dir / 'summary.json')
        return

    model, tokenizer, eval_target = load_model_and_tokenizer(cfg, run_dir, args.adapter_subdir)
    aggregate: dict[str, Any] = {'run_name': cfg['run_name'], 'eval_target': eval_target, 'temperatures': {}}

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
