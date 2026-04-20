# References

This project intentionally combines a small set of papers and upstream repositories. Use this file as the single source of truth for citations, upstream code, and the rationale for each dependency.

## Papers

1. **Embarrassingly Simple Self-Distillation Improves Code Generation**
   - Purpose in this project: defines the SSD recipe used for the code-first stage.
   - Paper: https://arxiv.org/abs/2604.01193
   - PDF: https://arxiv.org/pdf/2604.01193
   - Upstream repository: https://github.com/apple/ml-ssd

2. **LoRA-Squeeze: Simple and Effective Post-Tuning and In-Tuning Compression of LoRA Modules**
   - Purpose in this project: motivates high-rank training followed by RSVD compression and short recovery training.
   - Paper: https://arxiv.org/abs/2602.10993
   - PDF: https://arxiv.org/pdf/2602.10993
   - Project note: this scaffold implements a lightweight post-hoc compression path; it does not claim to reproduce every experiment from the paper.

3. **SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization**
   - Purpose in this project: inspires the skill-conditioned curriculum, staged skill withdrawal, and zero-skill target behavior used in the skill0 stage.
   - Paper: https://arxiv.org/abs/2604.02268
   - PDF: https://arxiv.org/pdf/2604.02268
   - Official code: https://github.com/ZJU-REAL/SkillZero
   - Project note: this scaffold borrows the training idea (skill internalization via curriculum) without importing the full RL stack.

## Repositories and documentation

1. **apple/ml-ssd**
   - URL: https://github.com/apple/ml-ssd
   - Role here: upstream SSD reference implementation and method description.

2. **Unsloth documentation for Qwen3 / Qwen3 MoE**
   - Docs: https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune
   - Product docs mirror: https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune
   - Role here: loader choice, MoE caveats, and training guidance for Qwen3-family models.

3. **Unsloth GitHub repository**
   - URL: https://github.com/unslothai/unsloth
   - Role here: training/runtime framework used by the LoRA and QLoRA stages.

4. **SkillZero official repository**
   - URL: https://github.com/ZJU-REAL/SkillZero
   - Role here: reference for curriculum design, skill organization, and zero-skill target behavior.

## Usage inside this repository

- `scripts/generate_ssd_local.py` and `scripts/prepare_ssd_data.py` implement the SSD-oriented code-first stage inspired by Apple SSD.
- `scripts/build_skill_views.py`, `scripts/build_skill0_dataset.py`, and `scripts/run_training_plan.py` implement the skill-internalization stage inspired by SKILL0.
- `scripts/squeeze_lora.py` and `scripts/recover_after_squeeze.py` implement the LoRA-Squeeze-style post-hoc compression path.
- `scripts/train_unsloth_lora.py` uses Unsloth loaders and PEFT continuation for both sequential and mixed training plans.

## Important scope note

This repository is a practical training scaffold, not a claim of full-paper reproduction. It intentionally simplifies:

- Apple SSD data generation by supporting local `messages` JSONL prompts.
- SKILL0 by using supervised skill-conditioned curriculum instead of the full RL stack.
- LoRA-Squeeze by providing a practical RSVD-based post-hoc compression path with recovery training.
