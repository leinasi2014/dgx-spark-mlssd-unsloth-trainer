#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from common import append_run_note, choose_unsloth_loader, ensure_run_dirs, load_config, load_jsonl, logger, maybe_enable_response_only


def render_chat(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    text = tokenizer.apply_chat_template(example['messages'], tokenize=False, add_generation_prompt=False)
    return {'text': text}


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a high-rank Unsloth adapter on a specified chat JSONL dataset.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset-path', default=None, help='Override training dataset path. Defaults to paths.ssd_train_jsonl.')
    parser.add_argument('--output-subdir', default='adapter_high_rank', help='Run-relative output subdirectory for the adapter.')
    parser.add_argument('--init-adapter', default=None, help='Optional existing adapter path to continue training from.')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = ensure_run_dirs(cfg)
    set_seed(args.seed)

    loader, loader_name = choose_unsloth_loader(cfg['model']['base_model'], cfg['model'].get('unsloth_loader', 'auto'))
    logger.info('Using Unsloth loader: %s', loader_name)
    model, tokenizer = loader.from_pretrained(
        model_name=cfg['model']['base_model'],
        max_seq_length=cfg['model']['max_seq_length'],
        load_in_4bit=cfg['model']['load_in_4bit'],
        trust_remote_code=cfg['model'].get('trust_remote_code', True),
    )

    init_adapter = args.init_adapter
    if init_adapter:
        from peft import PeftModel
        logger.info('Continuing from adapter: %s', init_adapter)
        model = PeftModel.from_pretrained(model, str(Path(init_adapter).resolve()), is_trainable=True)
    else:
        target_modules = list(cfg['training']['target_modules']) + list(cfg['training'].get('extra_target_modules', []))
        model = loader.get_peft_model(
            model,
            r=cfg['training']['source_rank'],
            target_modules=target_modules,
            lora_alpha=cfg['training']['lora_alpha'],
            lora_dropout=cfg['training']['lora_dropout'],
            bias='none',
            use_gradient_checkpointing='unsloth',
            random_state=args.seed,
        )

    dataset_path = Path(args.dataset_path or cfg['paths']['ssd_train_jsonl'])
    train_rows = load_jsonl(dataset_path)
    ds = Dataset.from_list(train_rows)
    ds = ds.map(lambda x: render_chat(x, tokenizer), remove_columns=ds.column_names)

    out_dir = run_dir / args.output_subdir
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(out_dir),
            max_seq_length=cfg['model']['max_seq_length'],
            dataset_text_field='text',
            packing=False,
            learning_rate=cfg['training']['learning_rate'],
            per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
            warmup_ratio=cfg['training']['warmup_ratio'],
            num_train_epochs=cfg['training']['num_train_epochs'],
            weight_decay=cfg['training']['weight_decay'],
            logging_steps=cfg['training']['logging_steps'],
            save_steps=cfg['training']['save_steps'],
            save_total_limit=cfg['training']['save_total_limit'],
            bf16=cfg['model']['bf16'],
            report_to=[],
        ),
    )
    trainer = maybe_enable_response_only(trainer, tokenizer, cfg)
    append_run_note(run_dir, [
        f'Starting high-rank SFT with loader={loader_name} dataset={dataset_path.name} output={args.output_subdir}.',
        f'Continuation adapter: {init_adapter if init_adapter else "fresh"}.',
    ])
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info('Saved adapter to %s', out_dir)


if __name__ == '__main__':
    main()
