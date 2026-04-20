# DGX Spark Code Training Scaffold

A deployable training scaffold for improving the code generation and skill-internalization ability of `samuelcardillo/Carnice-Qwen3.6-MoE-35B-A3B` on DGX Spark.

This revision supports **two training modes**:

1. **Sequential mode**: train code capability first with SSD, then continue finetuning with a SKILL0-inspired skill-internalization stage.
2. **Mixed mode**: combine code-training rows and skill0 rows into one mixed curriculum and train them together.

The project combines:

- Apple Simple Self-Distillation (`apple/ml-ssd`) for the code-first data-generation recipe
- Unsloth for efficient Qwen3 / Qwen3-MoE LoRA and QLoRA training
- A SKILL0-inspired skill-conditioning stage that gradually reduces explicit skill context
- LoRA-Squeeze-style post-hoc adapter compression and short recovery training

> All papers and repository URLs used by this project are listed in `docs/references.md`.

## What was fixed and added in this revision

- Kept the previous bug fixes: LoRA scaling preservation, correct recovery-step calculation, working code evaluation, explicit env templating, and MoE-safe loader selection.
- Added **sequential** and **mixed** training plans.
- Added a **skill0 dataset builder** and **skill-view compiler**.
- Added a **training-plan orchestrator** to materialize datasets and print the exact commands for each mode.
- Added a dedicated `docs/references.md` file with every paper and upstream repository used by the project.
- Updated `AGENTS.md`, the Codex skill, and the frontend design document to describe both modes.

## Repository layout

- `AGENTS.md` - operator guide for autonomous coding agents
- `docs/frontend-design.md` - Markdown web-console design doc
- `docs/references.md` - papers, repositories, and URLs used by this project
- `configs/train.example.yaml` - defaults for code, skill0, and mixed training
- `scripts/bootstrap_dgx_spark.sh` - environment bootstrap and version-pinned `ml-ssd` checkout
- `scripts/prepare_ssd_data.py` - normalize code prompts and convert raw SSD outputs into chat JSONL
- `scripts/generate_ssd_local.py` - local vLLM SSD generation for `messages` JSONL prompt pools
- `scripts/build_skill_views.py` - compile full / summary / tool-only skill views from repo docs
- `scripts/build_skill0_dataset.py` - construct SKILL0-style supervised rows from task prompts and skill views
- `scripts/build_mixed_dataset.py` - merge code and skill0 datasets with configurable weights
- `scripts/run_training_plan.py` - orchestrate sequential or mixed plan preparation and print next commands
- `scripts/train_unsloth_lora.py` - high-rank Unsloth training entrypoint with optional adapter continuation
- `scripts/squeeze_lora.py` - LoRA-Squeeze-style post-hoc compression via randomized SVD
- `scripts/recover_after_squeeze.py` - short recovery training from compressed adapters
- `scripts/evaluate_codegen.py` - executable code-eval harness with temperature sweep
- `.codex/skills/dgx-spark-code-trainer/` - reusable skill for agents working inside this repo

## Training modes

### 1) Sequential mode

Use this when the first priority is raw code generation quality.

Flow:

1. Prepare and generate the SSD-backed code dataset.
2. Train a **code high-rank adapter**.
3. Build a skill0 dataset from repo skills and operator docs.
4. Continue finetuning from the code adapter on the skill0 dataset.
5. Squeeze and recover the **final** adapter.
6. Evaluate the final adapter.

This mode maps best to the intent of “先训练 code 编码相关的，然后再训练 skill 0”.

### 2) Mixed mode

Use this when you want the model to co-learn code behavior and skill internalization in a single training stage.

Flow:

1. Prepare SSD-backed code data.
2. Build skill0 rows.
3. Mix code rows and skill0 rows according to configured weights.
4. Train one **mixed high-rank adapter**.
5. Squeeze and recover the mixed adapter.
6. Evaluate the final adapter.

This mode maps best to “code 编码训练 和 skill 0 训练一起混合”.

## Quick start

### Sequential mode

```bash
bash scripts/bootstrap_dgx_spark.sh
cp configs/train.example.yaml configs/train.local.yaml
python scripts/prepare_ssd_data.py --config configs/train.local.yaml --prepare-prompts --write-ssd-config
python scripts/generate_ssd_local.py --config configs/train.local.yaml
python scripts/prepare_ssd_data.py --config configs/train.local.yaml --convert-raw runs/<run_name>/raw_ssd_outputs.jsonl
python scripts/run_training_plan.py --config configs/train.local.yaml --mode sequential --prepare-only
python scripts/train_unsloth_lora.py --config configs/train.local.yaml --dataset-path runs/<run_name>/ssd_train.jsonl --output-subdir adapter_code_high_rank
python scripts/train_unsloth_lora.py --config configs/train.local.yaml --dataset-path runs/<run_name>/skill0_train.jsonl --output-subdir adapter_skill0_high_rank --init-adapter runs/<run_name>/adapter_code_high_rank
python scripts/squeeze_lora.py --config configs/train.local.yaml --source-subdir adapter_skill0_high_rank --output-subdir adapter_skill0_squeezed
python scripts/recover_after_squeeze.py --config configs/train.local.yaml --dataset-path runs/<run_name>/skill0_train.jsonl --source-subdir adapter_skill0_squeezed --output-subdir adapter_skill0_recovered
python scripts/evaluate_codegen.py --config configs/train.local.yaml --adapter-subdir adapter_skill0_recovered
```

### Mixed mode

```bash
bash scripts/bootstrap_dgx_spark.sh
cp configs/train.example.yaml configs/train.local.yaml
python scripts/prepare_ssd_data.py --config configs/train.local.yaml --prepare-prompts --write-ssd-config
python scripts/generate_ssd_local.py --config configs/train.local.yaml
python scripts/prepare_ssd_data.py --config configs/train.local.yaml --convert-raw runs/<run_name>/raw_ssd_outputs.jsonl
python scripts/run_training_plan.py --config configs/train.local.yaml --mode mixed --prepare-only
python scripts/train_unsloth_lora.py --config configs/train.local.yaml --dataset-path runs/<run_name>/mixed_train.jsonl --output-subdir adapter_mixed_high_rank
python scripts/squeeze_lora.py --config configs/train.local.yaml --source-subdir adapter_mixed_high_rank --output-subdir adapter_mixed_squeezed
python scripts/recover_after_squeeze.py --config configs/train.local.yaml --dataset-path runs/<run_name>/mixed_train.jsonl --source-subdir adapter_mixed_squeezed --output-subdir adapter_mixed_recovered
python scripts/evaluate_codegen.py --config configs/train.local.yaml --adapter-subdir adapter_mixed_recovered
```

## Notes on scope

- The **code stage** follows Apple SSD’s “sample → SFT on raw outputs → tune decoding temperature separately” recipe.
- The **skill0 stage** is intentionally simpler than the full SKILL0 RL framework: it uses supervised skill-conditioned curriculum data instead of importing the full RL stack.
- The **compression stage** is a practical LoRA-Squeeze-style implementation focused on post-hoc RSVD compression and short recovery training.
