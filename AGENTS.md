# AGENTS.md

This repository is designed for an autonomous coding agent operating on DGX Spark to improve the code programming ability and skill internalization behavior of `samuelcardillo/Carnice-Qwen3.6-MoE-35B-A3B`.

## Primary objective

Operate one of two supported training plans safely and reproducibly:

1. **Sequential mode**
   - prepare SSD-backed code data
   - train code adapter first
   - build skill0 curriculum data
   - continue finetuning from the code adapter on skill0 data
   - squeeze, recover, and evaluate the final adapter

2. **Mixed mode**
   - prepare SSD-backed code data
   - build skill0 curriculum data
   - mix the two datasets by configured weights
   - train one mixed adapter
   - squeeze, recover, and evaluate the final adapter

## Guardrails

- Do not fine-tune the MoE router layer.
- Prefer the default target modules from `configs/train.example.yaml`, which cover gated-attention, DeltaNet projections, and shared-expert MLP layers.
- Keep response-only masking enabled for the code stage.
- Do not add verifier-based filtering, reward models, or RL into the SSD stage.
- Treat the skill0 stage as **skill-conditioned supervised finetuning**, not as a claim of reproducing the full SKILL0 RL stack.
- Do not overwrite artifacts in `runs/` unless explicitly asked.

## Canonical preparation command

```bash
bash scripts/bootstrap_dgx_spark.sh
cp configs/train.example.yaml configs/train.local.yaml
python scripts/prepare_ssd_data.py --config configs/train.local.yaml --prepare-prompts --write-ssd-config
python scripts/generate_ssd_local.py --config configs/train.local.yaml
python scripts/prepare_ssd_data.py --config configs/train.local.yaml --convert-raw runs/<run_name>/raw_ssd_outputs.jsonl
python scripts/run_training_plan.py --config configs/train.local.yaml --mode sequential --prepare-only
```

## Mode-specific commands

### Sequential mode

```bash
python scripts/train_unsloth_lora.py --config configs/train.local.yaml --dataset-path runs/<run_name>/ssd_train.jsonl --output-subdir adapter_code_high_rank
python scripts/train_unsloth_lora.py --config configs/train.local.yaml --dataset-path runs/<run_name>/skill0_train.jsonl --output-subdir adapter_skill0_high_rank --init-adapter runs/<run_name>/adapter_code_high_rank
python scripts/squeeze_lora.py --config configs/train.local.yaml --source-subdir adapter_skill0_high_rank --output-subdir adapter_skill0_squeezed
python scripts/recover_after_squeeze.py --config configs/train.local.yaml --dataset-path runs/<run_name>/skill0_train.jsonl --source-subdir adapter_skill0_squeezed --output-subdir adapter_skill0_recovered
python scripts/evaluate_codegen.py --config configs/train.local.yaml --adapter-subdir adapter_skill0_recovered
```

### Mixed mode

```bash
python scripts/run_training_plan.py --config configs/train.local.yaml --mode mixed --prepare-only
python scripts/train_unsloth_lora.py --config configs/train.local.yaml --dataset-path runs/<run_name>/mixed_train.jsonl --output-subdir adapter_mixed_high_rank
python scripts/squeeze_lora.py --config configs/train.local.yaml --source-subdir adapter_mixed_high_rank --output-subdir adapter_mixed_squeezed
python scripts/recover_after_squeeze.py --config configs/train.local.yaml --dataset-path runs/<run_name>/mixed_train.jsonl --source-subdir adapter_mixed_squeezed --output-subdir adapter_mixed_recovered
python scripts/evaluate_codegen.py --config configs/train.local.yaml --adapter-subdir adapter_mixed_recovered
```

## Reference policy

Every operator-facing explanation that mentions upstream methods should point to `docs/references.md`, which contains the papers and repository URLs for Apple SSD, Unsloth, LoRA-Squeeze, and SKILL0.
