# DGX Spark Training Console Frontend Design

## 1. Purpose

This document describes a web console for operating the `dgx-spark-mlssd-unsloth-trainer` project on DGX Spark.

The console is built around the real stages in this repository:

1. bootstrap environment
2. prepare prompts
3. generate SSD samples
4. convert raw outputs into SFT JSONL
5. build skill views
6. build skill0 dataset
7. train adapters in sequential or mixed mode
8. squeeze adapter
9. recover adapter
10. evaluate code quality

## 2. Product goals

- Make long-running training operations easy to start, monitor, resume, and inspect.
- Reduce the chance of running stages in the wrong order.
- Make **sequential mode** and **mixed mode** explicit choices in the UI.
- Surface artifacts, logs, and metrics without forcing shell access for routine actions.
- Preserve an escape hatch for advanced users who still need terminal-level control.

## 3. Core screens

### Dashboard

Show active runs, failed runs, latest evaluation pass rates, free disk, training mode, and recent artifacts.

### Run list

A table with run name, stage, mode, status, base model, latest pass rate, and quick actions.

### New run wizard

Step-by-step creation flow covering config, dataset, SSD settings, training mode, squeeze/recovery, and launch.

### Training-mode selector

A dedicated panel that explains:

- **Sequential** = code first, then skill0 continuation
- **Mixed** = code and skill0 co-trained in one adapter

### Run detail

Show header summary, stage timeline, logs, artifacts, metrics, config YAML, and notes.

### Dataset page

Preview code prompts, skill0 tasks, mixed datasets, validate rows, detect duplicates, and export cleaned copies.

### Artifact page

Provide download links for prompts, raw SSD outputs, skill views, skill0 dataset, mixed dataset, adapters, compression reports, and eval summaries.

### System page

Expose Python version, Torch version, CUDA availability, architecture, free disk, and the checked out `ml-ssd` ref.

### References page

Render `docs/references.md` directly so operators can see every upstream paper and repository URL used in the project.

## 4. Suggested backend API

- `GET /api/runs`
- `POST /api/runs`
- `GET /api/runs/{run_name}`
- `POST /api/runs/{run_name}/stage/{stage_name}`
- `GET /api/runs/{run_name}/logs?stage=train`
- `GET /api/runs/{run_name}/artifacts`
- `GET /api/runs/{run_name}/artifacts/{artifact_name}`
- `POST /api/datasets/validate`
- `POST /api/datasets/normalize`
- `POST /api/datasets/build-skill0`
- `POST /api/datasets/build-mixed`
- `GET /api/system`
- `GET /api/references`

## 5. UX rules

- Never allow `squeeze` before the selected training stage succeeds.
- Never allow `recover` before `squeeze` succeeds.
- Evaluation should default to the best available adapter in this order: recovered, squeezed, high-rank, base.
- The UI must display whether the active run is **sequential** or **mixed**.
- Show shell commands and make them copyable from every stage card.
- Use streaming logs via SSE or WebSocket.

## 6. Recommended stack

### Frontend

- React
- TypeScript
- Vite
- TanStack Query
- React Router
- Monaco editor
- Recharts

### Backend

- FastAPI
- Pydantic
- asyncio subprocess orchestration
- SSE or WebSocket for logs/events

## 7. Source references

The canonical list of papers and repository URLs lives in `docs/references.md`.
