#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import PeftModel
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from common import append_run_note, choose_unsloth_loader, effective_optimizer_steps, ensure_run_dirs, load_config, load_jsonl, logger, maybe_enable_response_only


def render_chat(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    text = tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
    return {'text': text}


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
    set_seed(args.seed)

    loader, loader_name = choose_unsloth_loader(cfg['model']['base_model'], cfg['model'].get('unsloth_loader', 'auto'))
    model, tokenizer = loader.from_pretrained(
        model_name=cfg['model']['base_model'],
        max_seq_length=cfg['model']['max_seq_length'],
        load_in_4bit=cfg['model']['load_in_4bit'],
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
    )
    model = PeftModel.from_pretrained(model, str(squeezed_dir), is_trainable=True)

    dataset_path = Path(args.dataset_path or cfg['paths']['ssd_train_jsonl'])
    train_rows = load_jsonl(dataset_path)
    ds = Dataset.from_list(train_rows)
    ds = ds.map(lambda x: render_chat(x, tokenizer), remove_columns=ds.column_names)

    total_steps = effective_optimizer_steps(len(ds), cfg['training']['per_device_train_batch_size'], cfg['training']['gradient_accumulation_steps'])
    max_steps = max(1, int(total_steps * float(cfg['recovery']['max_steps_ratio'])))
    logger.info('Recovery optimizer steps: total=%d ratio=%s max_steps=%d', total_steps, cfg['recovery']['max_steps_ratio'], max_steps)

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
