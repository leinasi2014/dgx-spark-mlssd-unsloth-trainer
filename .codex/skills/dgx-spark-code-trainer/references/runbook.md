# Runbook

## Shared preparation

```bash
bash scripts/bootstrap_dgx_spark.sh
cp configs/train.example.yaml configs/train.local.yaml
python scripts/prepare_ssd_data.py --config configs/train.local.yaml --prepare-prompts --write-ssd-config
python scripts/generate_ssd_local.py --config configs/train.local.yaml
python scripts/prepare_ssd_data.py --config configs/train.local.yaml --convert-raw runs/<run_name>/raw_ssd_outputs.jsonl
```

## Sequential mode

```bash
python scripts/run_training_plan.py --config configs/train.local.yaml --mode sequential --prepare-only
python scripts/train_unsloth_lora.py --config configs/train.local.yaml --dataset-path runs/<run_name>/ssd_train.jsonl --output-subdir adapter_code_high_rank
python scripts/train_unsloth_lora.py --config configs/train.local.yaml --dataset-path runs/<run_name>/skill0_train.jsonl --output-subdir adapter_skill0_high_rank --init-adapter runs/<run_name>/adapter_code_high_rank
python scripts/squeeze_lora.py --config configs/train.local.yaml --source-subdir adapter_skill0_high_rank --output-subdir adapter_skill0_squeezed
python scripts/recover_after_squeeze.py --config configs/train.local.yaml --dataset-path runs/<run_name>/skill0_train.jsonl --source-subdir adapter_skill0_squeezed --output-subdir adapter_skill0_recovered
python scripts/evaluate_codegen.py --config configs/train.local.yaml --adapter-subdir adapter_skill0_recovered
```

## Mixed mode

```bash
python scripts/run_training_plan.py --config configs/train.local.yaml --mode mixed --prepare-only
python scripts/train_unsloth_lora.py --config configs/train.local.yaml --dataset-path runs/<run_name>/mixed_train.jsonl --output-subdir adapter_mixed_high_rank
python scripts/squeeze_lora.py --config configs/train.local.yaml --source-subdir adapter_mixed_high_rank --output-subdir adapter_mixed_squeezed
python scripts/recover_after_squeeze.py --config configs/train.local.yaml --dataset-path runs/<run_name>/mixed_train.jsonl --source-subdir adapter_mixed_squeezed --output-subdir adapter_mixed_recovered
python scripts/evaluate_codegen.py --config configs/train.local.yaml --adapter-subdir adapter_mixed_recovered
```
