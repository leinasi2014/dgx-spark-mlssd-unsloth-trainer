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
   - Repository status: no official public repository URL is cited by the paper as of 2026-04-21.
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

5. **SWE-bench**
   - URL: https://github.com/SWE-bench/SWE-bench
   - Role here: official evaluation harness and dataset reference for `SWE-bench Verified`.

6. **Multi-SWE-bench**
   - URL: https://github.com/multi-swe-bench/multi-swe-bench
   - Role here: official multilingual issue-resolution benchmark and evaluation harness.

7. **Terminal-Bench**
   - URL: https://github.com/harbor-framework/terminal-bench
   - Docs: https://www.tbench.ai/docs
   - Role here: public terminal-agent benchmark used by the optional external baseline subproject.

8. **MCPMark**
   - URL: https://github.com/eval-sys/mcpmark
   - Docs: https://mcpmark.ai/docs
   - Role here: public MCP-tool benchmark used by the optional external baseline subproject.

9. **LiveCodeBench**
   - URL: https://github.com/LiveCodeBench/LiveCodeBench
   - Dataset: https://huggingface.co/datasets/livecodebench/code_generation_lite
   - Versioning note: this repository pins `evaluation.public.version_tag` in config so the public code baseline is reproducible across reruns.
   - Role here: default public code-evaluation source used by `scripts/evaluate_livecodebench.py`.

## Usage inside this repository

- `scripts/prepare_ssd_data.py`, `scripts/generate_ssd_local.py`, and `scripts/ml_ssd_templates/` implement the SSD-oriented code-first stage by mirroring the upstream `apple/ml-ssd` structure:
  - load a Hugging Face problem dataset
  - render the upstream-style `stdin` / `function` self-distillation templates
  - sample raw code completions locally with vLLM
  - convert raw completions into response-only chat SFT rows
- `scripts/evaluate_livecodebench.py` and `scripts/livecodebench_utils.py` implement the public code baseline in the same direction as the upstream `ml-ssd` LiveCodeBench evaluation flow.
- `scripts/build_skill_views.py`, `scripts/build_skill0_dataset.py`, and `scripts/run_training_plan.py` implement the skill-internalization stage inspired by SKILL0.
- `scripts/prepare_agent_data.py` and `scripts/source_adapters.py` normalize optional upstream agent-trajectory datasets without pretending that they are part of the `ml-ssd` problem-to-code path.
- `scripts/squeeze_lora.py` and `scripts/recover_after_squeeze.py` implement the LoRA-Squeeze-style post-hoc compression path.
- `scripts/train_unsloth_lora.py` uses Unsloth loaders and PEFT continuation for both sequential and mixed training plans.
- `benchmarks/agent_evals/` wraps public benchmark harnesses for external pre-train / post-train baselines without changing the main training scaffold.

## Important scope note

This repository is a practical training scaffold, not a claim of full-paper reproduction. It intentionally simplifies:

- Apple SSD data generation by keeping the same overall dataset -> template -> sample -> SFT shape while still using local wrapper scripts instead of the upstream repo entrypoints.
- SKILL0 by using supervised skill-conditioned curriculum instead of the full RL stack.
- LoRA-Squeeze by providing a practical RSVD-based post-hoc compression path with recovery training.
