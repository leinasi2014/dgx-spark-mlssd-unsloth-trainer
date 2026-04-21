# Agent Evals

`benchmarks/agent_evals/` 是这个仓库里的独立测试子项目，用来在训练前和训练后跑公开 coding-agent benchmark，并把结果统一汇总成一份 baseline 报告。

当前覆盖：

- `SWE-bench Verified`
- `SWE-bench Multilingual`
- `Terminal-Bench 2.0`
- `MCPMark`

这个子项目默认面向 **Linux / DGX**，并且假设你已经有一个 **OpenAI 兼容模型端点**，例如 vLLM、sglang 或 TGI。

## 设计边界

- 不复用主项目 `.venv`
- 不把产物写入主项目 `runs/`
- 外部 benchmark 全部固定到 `configs/benchmarks.yaml` 里的 pinned upstream ref
- 只做统一编排层，不重写官方 benchmark 逻辑

对于四个 benchmark，支持程度分两类：

- `Terminal-Bench 2.0`、`MCPMark`
  - 可以直接通过官方 CLI/harness 跑
- `SWE-bench Verified`、`SWE-bench Multilingual`
  - 官方 harness 负责验证 patch
  - patch 生成仍然外置，所以运行时需要传 `--predictions-path`

这不是缺失实现，而是有意保持边界清晰：SWE-bench 类 benchmark 的“生成 patch”本身就是一个独立 agent 系统，不应该被硬编码进这个仓库。

## 快速开始

```bash
cd benchmarks/agent_evals
bash scripts/bootstrap.sh
.venv/bin/agent-evals doctor
.venv/bin/agent-evals fetch --benchmark all
```

### 训练前 baseline

先启动你的模型服务，例如：

- `OPENAI_BASE_URL=http://127.0.0.1:8000/v1`
- `OPENAI_API_KEY=dummy`

然后：

```bash
.venv/bin/agent-evals run \
  --benchmark terminal_bench \
  --profile smoke \
  --run-id baseline-pretrain-qwen \
  --model-name openai/carnice-qwen-base \
  --base-url "$OPENAI_BASE_URL"

.venv/bin/agent-evals run \
  --benchmark mcpmark \
  --profile smoke \
  --run-id baseline-pretrain-qwen \
  --model-name openai/carnice-qwen-base \
  --base-url "$OPENAI_BASE_URL"
```

如果你已经有 SWE-bench patch 文件：

```bash
.venv/bin/agent-evals run \
  --benchmark swebench_verified \
  --profile smoke \
  --run-id baseline-pretrain-qwen \
  --model-name openai/carnice-qwen-base \
  --base-url "$OPENAI_BASE_URL" \
  --predictions-path /path/to/swebench_predictions.jsonl
```

最后聚合：

```bash
.venv/bin/agent-evals aggregate --run-id baseline-pretrain-qwen
```

产物位置：

- 单 benchmark 结果：`runs/<run_id>/<benchmark>/summary.json`
- 聚合结果：`runs/<run_id>/aggregate.json`
- 可读摘要：`runs/<run_id>/aggregate.md`

## 命令

### `agent-evals doctor`

检查 Linux、`git`、`docker`、基础环境变量是否就绪。

### `agent-evals fetch --benchmark <name|all>`

拉取 pinned upstream repo，并在各自 checkout 里创建 `.venv` 后执行安装命令。

### `agent-evals run`

通用参数：

- `--benchmark`
- `--profile smoke|full`
- `--run-id`
- `--model-name`
- `--base-url`
- `--max-workers`
- `--dry-run`

SWE-bench 类额外需要：

- `--predictions-path`

### `agent-evals aggregate`

按 `run_id` 汇总所有已完成 benchmark 的 `summary.json`。

## 配置

主配置在 [configs/benchmarks.yaml](configs/benchmarks.yaml)。

你通常只需要改这些项：

- benchmark 的 `install_commands`
- `run_command` / `evaluation_command`
- `summary_candidates`
- `full` profile 的并发和任务规模

默认配置已经 pin 了当前验证过的 upstream ref：

- `SWE-bench`: `f7bbbb2ccdf479001d6467c9e34af59e44a840f9`
- `multi-swe-bench`: `24f493f8a103e72312ded4f6b9c89f081d69cb09`
- `terminal-bench`: `1a6ffa9674b571da0ed040c470cb40c4d85f9b9b`
- `mcpmark`: `adc5e6558f05c4c9a4d5ebc58062da0b2391dc30`

## 说明

- `Terminal-Bench 2.0` 和 `MCPMark` 的 CLI 仍在快速演进。这个子项目把上游命令模板放进 YAML，就是为了将来只改配置，不必改 Python wrapper。
- `SWE-bench Verified` 和 `SWE-bench Multilingual` 这里默认只做官方 harness 验证，不在本仓库硬塞一个 patch generation agent。
- 如果你要把训练前和训练后结果放在一起比，直接复用同一个 benchmark 子项目，只换 `run_id` 和模型端点即可。
