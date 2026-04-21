# AGENTS.md

This repository is designed for an autonomous coding agent operating on DGX Spark to improve the code programming ability and skill internalization behavior of `samuelcardillo/Carnice-Qwen3.6-MoE-35B-A3B`.

## Primary objective

Operate one of the supported training plans safely and reproducibly:

1. **Code only**
   - prepare SSD-backed code data
   - train the code adapter
   - optionally squeeze and recover it if those stages are enabled
   - evaluate the resulting code adapter

2. **Code then skill0**
   - prepare SSD-backed code data
   - train code adapter first
   - build skill0 curriculum data
   - continue finetuning from the code adapter on skill0 data
   - squeeze, recover, and evaluate the final adapter

3. **Code then agent**
   - prepare SSD-backed code data
   - train code adapter first
   - prepare one agent-trajectory dataset from a configured upstream source
   - continue finetuning from the code adapter on that agent dataset
   - squeeze, recover, and evaluate the final adapter

4. **Mixed sources**
   - prepare SSD-backed code data
   - build skill0 curriculum data
   - optionally prepare extra upstream agent datasets
   - mix the selected datasets by configured weights
   - train one mixed adapter
   - squeeze, recover, and evaluate the final adapter

## Guardrails

- Do not fine-tune the MoE router layer.
- Prefer the default target modules from `configs/train.example.yaml`, which cover gated-attention, DeltaNet projections, and shared-expert MLP layers.
- Keep response-only masking enabled for training.
- Do not add verifier-based filtering, reward models, or RL into the SSD stage.
- Keep the default code-stage source aligned with `apple/ml-ssd`: Hugging Face problem dataset input plus upstream-style `stdin` / `function` prompt templates.
- Treat the skill0 stage as **skill-conditioned supervised finetuning**, not as a claim of reproducing the full SKILL0 RL stack.
- Do not overwrite artifacts in `runs/` unless explicitly asked.

## Canonical preparation command

```bash
bash scripts/bootstrap_dgx_spark.sh
# optional upstream checkout for comparison/debugging:
# INSTALL_ML_SSD=1 bash scripts/bootstrap_dgx_spark.sh
cp configs/train.example.yaml configs/train.local.yaml
# edit configs/train.local.yaml and set a unique run_name before continuing
.venv/bin/python scripts/prepare_ssd_data.py --config configs/train.local.yaml --prepare-prompts --write-ssd-config
.venv/bin/python scripts/generate_ssd_local.py --config configs/train.local.yaml
.venv/bin/python scripts/prepare_ssd_data.py --config configs/train.local.yaml --convert-raw runs/your-run-name/raw_ssd_outputs.jsonl
```

Then choose exactly one plan-specific prepare command from the next section for the same `run_name`.

## Mode-specific commands

### Code only

```bash
.venv/bin/python scripts/run_training_plan.py --config configs/train.local.yaml --plan code_only --prepare-only
bash runs/your-run-name/code_only_plan.sh
```

### Code then skill0

```bash
.venv/bin/python scripts/run_training_plan.py --config configs/train.local.yaml --plan code_then_skill0 --prepare-only
bash runs/your-run-name/code_then_skill0_plan.sh
```

### Code then agent

```bash
# enable data_sources.<your_source>.enabled=true and set training_plan.agent.source first
.venv/bin/python scripts/run_training_plan.py --config configs/train.local.yaml --plan code_then_agent --prepare-only
bash runs/your-run-name/code_then_agent_plan.sh
```

### Mixed sources

```bash
.venv/bin/python scripts/run_training_plan.py --config configs/train.local.yaml --plan mixed_sources --prepare-only
bash runs/your-run-name/mixed_sources_plan.sh
```

## Reference policy

Every operator-facing explanation that mentions upstream methods should point to `docs/references.md`, which contains the papers and upstream repository URLs where available for Apple SSD, Unsloth, LoRA-Squeeze, SKILL0, and the default benchmark sources used by this repository.
