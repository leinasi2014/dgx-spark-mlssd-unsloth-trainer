---
name: dgx-spark-code-trainer
description: orchestrate dgx spark training for qwen3.6 moe style models using apple simple self-distillation, unsloth, a skill0-inspired skill-internalization stage, and lora-squeeze compression. use when an agent needs to bootstrap the repo, prepare code data, build skill-conditioned data, choose between sequential or mixed training, compress adapters, recover after compression, evaluate code quality, or debug this training scaffold.
---

# Dgx Spark Code Trainer

Use this skill when operating inside this repository to run or update the training workflow.

## Read first

1. `AGENTS.md`
2. `configs/train.example.yaml`
3. `docs/references.md`
4. `references/runbook.md`
5. `references/modes.md`

## Supported modes

### Sequential

Use when the goal is to first improve raw code generation and only then internalize repository skills.

1. Prepare SSD-backed code data.
2. Train `adapter_code_high_rank`.
3. Build skill views and the skill0 dataset.
4. Continue training from `adapter_code_high_rank` into `adapter_skill0_high_rank`.
5. Squeeze and recover the final adapter.
6. Evaluate the recovered final adapter.

### Mixed

Use when the goal is to co-train code and skill internalization in one adapter.

1. Prepare SSD-backed code data.
2. Build skill views and the skill0 dataset.
3. Build `mixed_train.jsonl`.
4. Train `adapter_mixed_high_rank`.
5. Squeeze and recover the mixed adapter.
6. Evaluate the recovered mixed adapter.

## Commands

Use `python scripts/run_training_plan.py --config <cfg> --mode sequential --prepare-only` to materialize the sequential plan.

Use `python scripts/run_training_plan.py --config <cfg> --mode mixed --prepare-only` to materialize the mixed plan.
