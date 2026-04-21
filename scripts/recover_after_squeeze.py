#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import PeftModel
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from common import append_run_note, choose_unsloth_loader, effective_optimizer_steps, ensure_path_is_new, ensure_run_dirs, load_config, load_jsonl, logger, maybe_enable_response_only


def render_chat(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    text = tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
    return {'text': text}


def default_recovery_dataset(cfg: dict[str, Any], source_subdir: str) -> Path:
    paths = cfg['paths']
    plan = str(cfg.get('training_plan', {}).get('plan') or '').strip().lower()
    legacy_mode = str(cfg.get('training_plan', {}).get('mode', 'sequential')).strip().lower()
    lowered = source_subdir.lower()
    if 'mixed' in lowered:
        return Path(paths['mixed_train_jsonl'])
    if 'skill0' in lowered:
        return Path(paths['skill0_train_jsonl'])
    if 'agent' in lowered:
        agent_source = str(cfg.get('training_plan', {}).get('agent', {}).get('source') or 'coderforge_preview').strip()
        if agent_source:
            return Path(paths['output_root']) / cfg['run_name'] / 'prepared_sources' / f'{agent_source}.jsonl'
    if 'code' in lowered:
        return Path(paths['ssd_train_jsonl'])
    if plan == 'code_only':
        return Path(paths['ssd_train_jsonl'])
    if plan == 'code_then_agent':
        agent_source = str(cfg.get('training_plan', {}).get('agent', {}).get('source') or 'coderforge_preview').strip()
        return Path(paths['output_root']) / cfg['run_name'] / 'prepared_sources' / f'{agent_source}.jsonl'
    if plan == 'mixed_sources' or legacy_mode == 'mixed':
        return Path(paths['mixed_train_jsonl'])
    if plan == 'code_then_skill0':
        return Path(paths['skill0_train_jsonl'])
    return Path(paths['skill0_train_jsonl'])


def validate_recovery_adapter_guardrails(adapter_dir: Path) -> None:
    config_path = adapter_dir / 'adapter_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Missing adapter config for recovery source: {config_path}')
    adapter_cfg = json.loads(config_path.read_text(encoding='utf-8'))
    raw_target_modules = adapter_cfg.get('target_modules', [])
    if isinstance(raw_target_modules, str):
        target_modules = [raw_target_modules]
    else:
        target_modules = [str(module) for module in raw_target_modules]
    forbidden = [module for module in target_modules if 'router' in module.lower()]
    if forbidden:
        raise ValueError(f'Recovery source adapter targets forbidden MoE router modules: {forbidden}')


def validate_recovery_training_guardrails(cfg: dict[str, Any], adapter_dir: Path) -> None:
    if not cfg.get('training', {}).get('response_only', True):
        raise ValueError('response_only masking must remain enabled for recovery training.')
    validate_recovery_adapter_guardrails(adapter_dir)


def recovery_max_steps(cfg: dict[str, Any], n_examples: int, world_size: int) -> tuple[int, int]:
    steps_per_epoch = effective_optimizer_steps(
        n_examples,
        cfg['training']['per_device_train_batch_size'],
        cfg['training']['gradient_accumulation_steps'],
        world_size=world_size,
    )
    total_steps = max(1, int(math.ceil(steps_per_epoch * float(cfg['training'].get('num_train_epochs', 1.0)))))
    max_steps = max(1, int(total_steps * float(cfg['recovery']['max_steps_ratio'])))
    return steps_per_epoch, max_steps


def main() -> None:
    parser = argparse.ArgumentParser(description='Run short recovery training from a squeezed adapter.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset-path', default=None)
    parser.add_argument('--source-subdir', default='adapter_squeezed')
    parser.add_argument('--output-subdir', default='adapter_recovered')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg['recovery']['enabled']:
        logger.info('Recovery disabled')
        return

    run_dir = ensure_run_dirs(cfg)
    squeezed_dir = run_dir / args.source_subdir
    out_dir = run_dir / args.output_subdir
    ensure_path_is_new(out_dir, 'recovery output directory')
    validate_recovery_training_guardrails(cfg, squeezed_dir)
    set_seed(args.seed)

    loader, loader_name = choose_unsloth_loader(cfg['model']['base_model'], cfg['model'].get('unsloth_loader', 'auto'))
    model, tokenizer = loader.from_pretrained(
        model_name=cfg['model']['base_model'],
        max_seq_length=cfg['model']['max_seq_length'],
        load_in_4bit=cfg['model']['load_in_4bit'],
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
    )
    model = PeftModel.from_pretrained(model, str(squeezed_dir), is_trainable=True)

    dataset_path = Path(args.dataset_path) if args.dataset_path else default_recovery_dataset(cfg, args.source_subdir)
    train_rows = load_jsonl(dataset_path)
    ds = Dataset.from_list(train_rows)
    ds = ds.map(lambda x: render_chat(x, tokenizer), remove_columns=ds.column_names)

    world_size = int(os.environ.get('WORLD_SIZE', '1') or '1')
    steps_per_epoch, max_steps = recovery_max_steps(cfg, len(ds), world_size)
    total_steps = max(1, int(math.ceil(steps_per_epoch * float(cfg['training'].get('num_train_epochs', 1.0)))))
    logger.info(
        'Recovery optimizer steps: per_epoch=%d total=%d ratio=%s max_steps=%d world_size=%d',
        steps_per_epoch,
        total_steps,
        cfg['recovery']['max_steps_ratio'],
        max_steps,
        world_size,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(out_dir),
            max_seq_length=cfg['model']['max_seq_length'],
            dataset_text_field='text',
            packing=False,
            learning_rate=cfg['recovery']['learning_rate'],
            per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
            logging_steps=cfg['training']['logging_steps'],
            save_steps=cfg['training']['save_steps'],
            max_steps=max_steps,
            warmup_ratio=cfg['recovery'].get('warmup_ratio', 0.05),
            weight_decay=cfg['recovery'].get('weight_decay', 0.01),
            bf16=cfg['model']['bf16'],
            report_to=[],
        ),
    )
    trainer = maybe_enable_response_only(trainer, tokenizer, cfg)
    append_run_note(run_dir, [
        f'Starting recovery finetune with loader={loader_name} dataset={dataset_path.name} source={args.source_subdir} output={args.output_subdir} max_steps={max_steps}.',
    ])
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info('Wrote recovered adapter to %s', out_dir)


if __name__ == '__main__':
    main()
