# DGX Spark Code Training Scaffold

A deployable training scaffold for improving the code generation and skill-internalization ability of `samuelcardillo/Carnice-Qwen3.6-MoE-35B-A3B` on DGX Spark.

The current default code-data path is intentionally aligned with `apple/ml-ssd`:

- load a Hugging Face problem dataset, defaulting to `microsoft/rStar-Coder`
- render the upstream-style `stdin` / `function` self-distillation prompts
- generate raw completions locally with vLLM
- convert those completions into response-only SFT rows for the code stage

This revision supports **four stage plans**:

1. **`code_only`**: train only the ml-ssd-aligned code stage.
2. **`code_then_skill0`**: train code capability first, then continue finetuning with a SKILL0-inspired skill-internalization stage.
3. **`code_then_agent`**: train code capability first, then continue on a prepared agent-trajectory dataset such as `CoderForge`.
4. **`mixed_sources`**: combine multiple prepared datasets into one mixed curriculum and train them together.

Legacy `sequential` / `mixed` CLI names are still accepted by `scripts/run_training_plan.py` for compatibility.

The project combines:

- Apple Simple Self-Distillation (`apple/ml-ssd`) for the code-first data-generation recipe
- Unsloth for efficient Qwen3 / Qwen3-MoE LoRA and QLoRA training
- A SKILL0-inspired skill-conditioning stage that gradually reduces explicit skill context
- LoRA-Squeeze-style post-hoc adapter compression and short recovery training

> All papers and repository URLs used by this project are listed in `docs/references.md`.

## What was fixed and added in this revision

- Kept the previous bug fixes: LoRA scaling preservation, correct recovery-step calculation, working code evaluation, explicit env templating, and MoE-safe loader selection.
- Added stage-based training plans with legacy `sequential` / `mixed` aliases.
- Added a **skill0 dataset builder** and **skill-view compiler**.
- Added a **training-plan orchestrator** to materialize datasets and print the exact commands for each mode.
- Added a dedicated `docs/references.md` file with every paper and upstream repository used by the project.
- Updated `AGENTS.md`, the Codex skill, and the frontend design document to describe both modes.

## Repository layout

- `AGENTS.md` - operator guide for autonomous coding agents
- `docs/frontend-design.md` - Markdown web-console design doc
- `docs/references.md` - papers, repositories, and URLs used by this project
- `configs/train.example.yaml` - defaults for code, skill0, and mixed training
- `scripts/bootstrap_dgx_spark.sh` - environment bootstrap using the repo-local virtualenv, with optional pinned `ml-ssd` checkout
- `scripts/source_adapters.py` - adapters for upstream Hugging Face datasets and source-family normalization
- `scripts/ml_ssd_templates/` - upstream-aligned `apple/ml-ssd` prompt templates
- `scripts/prepare_ssd_data.py` - render ml-ssd prompts from the configured problem-code dataset and convert raw SSD outputs into chat JSONL
- `scripts/generate_ssd_local.py` - local vLLM SSD generation for ml-ssd prompt rows
- `scripts/prepare_agent_data.py` - normalize agent-trajectory datasets such as `CoderForge` into trainable chat JSONL
- `scripts/build_skill_views.py` - compile full / summary / tool-only skill views from repo docs
- `scripts/build_skill0_dataset.py` - construct SKILL0-style supervised rows from task prompts and skill views
- `scripts/build_mixed_dataset.py` - merge any prepared datasets with configurable weights
- `scripts/run_training_plan.py` - orchestrate stage plans and print next commands
- `scripts/train_unsloth_lora.py` - high-rank Unsloth training entrypoint with optional adapter continuation
- `scripts/squeeze_lora.py` - LoRA-Squeeze-style post-hoc compression via randomized SVD
- `scripts/recover_after_squeeze.py` - short recovery training from compressed adapters
- `scripts/evaluate_livecodebench.py` - public LiveCodeBench-style code baseline aligned to the upstream `ml-ssd` evaluation direction
- `scripts/evaluate_codegen.py` - local executable smoke baseline for quick functional regression checks
- `.codex/skills/dgx-spark-code-trainer/` - reusable skill for agents working inside this repo

## Stage plans

### 1) `code_then_skill0`

Use this when the first priority is raw code generation quality.

Flow:

1. Prepare and generate the SSD-backed code dataset.
2. Train a **code high-rank adapter**.
3. Build a skill0 dataset from repo skills and operator docs.
4. Continue finetuning from the code adapter on the skill0 dataset.
5. Squeeze and recover the **final** adapter.
6. Evaluate the final adapter.

This plan maps best to the intent of “先训练 code 编码相关的，然后再训练 skill 0”.

### 2) `code_then_agent`

Use this when you want a clean second stage on top of the code adapter for agent-style data such as `CoderForge`.

Flow:

1. Prepare and generate the SSD-backed code dataset.
2. Train a **code high-rank adapter**.
3. Normalize the configured agent dataset into `runs/your-run-name/prepared_sources/<source>.jsonl`.
4. Continue finetuning from the code adapter on that agent dataset.
5. Squeeze and recover the **final** adapter.
6. Evaluate the final adapter.

### 3) `mixed_sources`

Use this when you want the model to co-learn code behavior and skill internalization in a single training stage.

Flow:

1. Prepare SSD-backed code data.
2. Build skill0 rows.
3. Mix code rows and skill0 rows according to configured weights.
4. Train one **mixed high-rank adapter**.
5. Squeeze and recover the mixed adapter.
6. Evaluate the final adapter.

By default this mixes `code` and `skill0`. If you add an external source such as `coderforge_preview`, keep it enabled in `data_sources` and list it in `training_plan.mixed.sources`; plan preparation will materialize that agent dataset automatically.

### 4) `code_only`

Use this when you only want the upstream-aligned code stage and public code baseline before exploring skill or agent data.

## Quick start

### `code_only`

```bash
bash scripts/bootstrap_dgx_spark.sh
cp configs/train.example.yaml configs/train.local.yaml
# edit configs/train.local.yaml and set a unique run_name before continuing
.venv/bin/python scripts/prepare_ssd_data.py --config configs/train.local.yaml --prepare-prompts --write-ssd-config
.venv/bin/python scripts/generate_ssd_local.py --config configs/train.local.yaml
.venv/bin/python scripts/prepare_ssd_data.py --config configs/train.local.yaml --convert-raw runs/your-run-name/raw_ssd_outputs.jsonl
.venv/bin/python scripts/run_training_plan.py --config configs/train.local.yaml --plan code_only --prepare-only
bash runs/your-run-name/code_only_plan.sh
```

### `code_then_skill0`

```bash
bash scripts/bootstrap_dgx_spark.sh
# optional upstream checkout for comparison/debugging:
# INSTALL_ML_SSD=1 bash scripts/bootstrap_dgx_spark.sh
cp configs/train.example.yaml configs/train.local.yaml
# edit configs/train.local.yaml and set a unique run_name before continuing
.venv/bin/python scripts/prepare_ssd_data.py --config configs/train.local.yaml --prepare-prompts --write-ssd-config
.venv/bin/python scripts/generate_ssd_local.py --config configs/train.local.yaml
.venv/bin/python scripts/prepare_ssd_data.py --config configs/train.local.yaml --convert-raw runs/your-run-name/raw_ssd_outputs.jsonl
.venv/bin/python scripts/run_training_plan.py --config configs/train.local.yaml --plan code_then_skill0 --prepare-only
bash runs/your-run-name/code_then_skill0_plan.sh
```

### `mixed_sources`

```bash
bash scripts/bootstrap_dgx_spark.sh
# optional upstream checkout for comparison/debugging:
# INSTALL_ML_SSD=1 bash scripts/bootstrap_dgx_spark.sh
cp configs/train.example.yaml configs/train.local.yaml
# edit configs/train.local.yaml and set a unique run_name before continuing
.venv/bin/python scripts/prepare_ssd_data.py --config configs/train.local.yaml --prepare-prompts --write-ssd-config
.venv/bin/python scripts/generate_ssd_local.py --config configs/train.local.yaml
.venv/bin/python scripts/prepare_ssd_data.py --config configs/train.local.yaml --convert-raw runs/your-run-name/raw_ssd_outputs.jsonl
.venv/bin/python scripts/run_training_plan.py --config configs/train.local.yaml --plan mixed_sources --prepare-only
bash runs/your-run-name/mixed_sources_plan.sh
```

### `code_then_agent`

```bash
bash scripts/bootstrap_dgx_spark.sh
cp configs/train.example.yaml configs/train.local.yaml
# set data_sources.<your_source>.enabled=true and point training_plan.agent.source to it first
.venv/bin/python scripts/prepare_ssd_data.py --config configs/train.local.yaml --prepare-prompts --write-ssd-config
.venv/bin/python scripts/generate_ssd_local.py --config configs/train.local.yaml
.venv/bin/python scripts/prepare_ssd_data.py --config configs/train.local.yaml --convert-raw runs/your-run-name/raw_ssd_outputs.jsonl
.venv/bin/python scripts/run_training_plan.py --config configs/train.local.yaml --plan code_then_agent --prepare-only
bash runs/your-run-name/code_then_agent_plan.sh
```

## Notes on scope

- The **code stage** follows Apple SSD’s “problem dataset -> self-distillation prompt -> raw sample -> SFT on raw outputs -> tune decoding temperature separately” recipe.
- The **skill0 stage** is intentionally simpler than the full SKILL0 RL framework: it uses supervised skill-conditioned curriculum data instead of importing the full RL stack.
- `LiteCoder-Terminal-RL-preview` is not treated as direct SFT data in this scaffold. The repository fails fast there until a dedicated adapter exists.
- The **compression stage** is a practical LoRA-Squeeze-style implementation focused on post-hoc RSVD compression and short recovery training.

## Baselines

- Public code baseline: `scripts/evaluate_livecodebench.py` with an explicit `evaluation.public.version_tag`
- Local smoke baseline: `scripts/evaluate_codegen.py`
- Public agent benchmarks: `benchmarks/agent_evals/`

## External benchmark baselines

If you need a reproducible public benchmark baseline before training, use the independent subproject in [benchmarks/agent_evals](benchmarks/agent_evals/README.md).

It wraps four public coding-agent benchmarks with pinned upstream refs:

- `SWE-bench Verified`
- `SWE-bench Multilingual`
- `Terminal-Bench 2.0`
- `MCPMark`

The subproject has its own virtualenv, cache, external checkouts, and `runs/` directory so it does not interfere with the main training scaffold. All upstream benchmark references are listed in [docs/references.md](docs/references.md).
