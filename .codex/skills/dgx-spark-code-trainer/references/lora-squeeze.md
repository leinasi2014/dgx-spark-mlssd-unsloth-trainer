# LoRA-Squeeze Policy

- Source rank: `128`
- Target rank: `64`
- Recovery budget: `5-10%` of optimizer steps
- Preserve original scaling by adjusting `lora_alpha` when rank changes
- Save `compression_report.json` with Frobenius error and energy-kept metrics
