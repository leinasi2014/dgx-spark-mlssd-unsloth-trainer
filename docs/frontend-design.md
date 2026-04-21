# DGX Spark Training Console Frontend Design

## 1. Purpose

A web console for operating the `dgx-spark-mlssd-unsloth-trainer` project on DGX Spark.
The console is built around the real stages in this repository and provides a GUI for
starting, monitoring, resuming, and inspecting long-running ML training operations.

## 2. Product Goals

- Make long-running training operations easy to start, monitor, resume, and inspect.
- Reduce the chance of running stages in the wrong order.
- Make **sequential mode** and **mixed mode** explicit choices in the UI.
- Surface artifacts, logs, and metrics without forcing shell access for routine actions.
- Preserve an escape hatch for advanced users who still need terminal-level control.
- Provide proactive notifications for key events (stage failure, training completion, low disk).

## 3. Training Pipeline Stages

### 3.1 Global Setup (not run-scoped)

| Stage | Description | Scope |
|-------|-------------|-------|
| bootstrap | Initialize venv, install dependencies, verify CUDA/GPU | Global, one-time |

### 3.2 Run-Scoped Stages

The pipeline has two modes. Stage numbering is shared; skipped stages are marked accordingly.

| # | Stage | Sequential | Mixed | Depends On |
|---|-------|-----------|-------|------------|
| 1 | prepare-prompts | yes | yes | bootstrap |
| 2 | generate-ssd | yes | yes | prepare-prompts |
| 3 | convert-ssd | yes | yes | generate-ssd |
| 4 | build-datasets | yes (skill-views + skill0) | yes (skill-views + skill0 + mixed) | convert-ssd |
| 5a | train-code-adapter | yes | — | build-datasets |
| 5b | train-adapter | — | yes (mixed adapter) | build-datasets |
| 5c | train-skill0-adapter | yes (continues from code adapter) | — | train-code-adapter |
| 6 | squeeze | yes | yes | train-* (any completed) |
| 7 | recover | yes | yes | squeeze |
| 8 | evaluate | yes | yes | train-* (best available adapter) |

**Stage statuses:** `pending` | `running` | `succeeded` | `failed` | `skipped` | `cancelled`

### 3.3 Adapter Priority Chain (for evaluation)

```
recovered > squeezed > high-rank > base
```

Within each tier, sequential mode prefers skill0 variants over mixed variants:
```
adapter_skill0_recovered > adapter_mixed_recovered > adapter_recovered
```

## 4. Core Screens

### 4.1 Dashboard

The landing page. Shows system health and active run status at a glance.

**Sections:**
- **System health strip**: Python/Torch version, CUDA status, GPU memory, free disk (GB + %), ml-ssd ref
- **Active runs**: Cards showing currently running stages with live progress
- **Recent failures**: Failed runs needing attention
- **Quick actions**: "New Run" button, "Bootstrap Environment" (if not bootstrapped)
- **Latest evaluation**: Best pass rate across recent runs

### 4.2 Run List

A filterable, sortable table of all runs.

**Columns:** Run name | Mode badge (Sequential/Mixed) | Current stage | Status | Base model | Best pass rate | Created at

**Quick actions** (context-dependent):
- Running → View Logs / Cancel
- Failed → Retry / View Error / Delete
- Completed → View Results / Evaluate Again / Clone to New Run

**Features:** Search by name, filter by mode/status, sort by date/pass rate, pagination (cursor-based).

### 4.3 New Run Wizard

Step-by-step creation flow. Training-mode choice is integrated here (not a separate page).

**Steps:**
1. **Config** — Load from template or custom YAML. Monaco editor with schema validation.
2. **Dataset** — Review prompt count, SSD settings (limit, tensor_parallel_size).
3. **Training Mode** — Explicit choice: Sequential (code → skill0) vs Mixed (co-trained). Visual flow diagram showing the different pipelines.
4. **Post-Training** — Squeeze settings (rank, enabled toggle), Recovery settings (enabled toggle).
5. **Review & Launch** — Summary of all settings, estimated command preview, launch button.

**Post-launch behavior:** Immediately redirect to Run Detail, auto-focus on first running stage.

### 4.4 Run Detail

The primary monitoring page. Uses a fixed header + sidebar timeline + tabbed content layout.

**Layout:**
- **Fixed header**: Run name, mode badge, status, base model, created/finished timestamps
- **Left sidebar — Stage Timeline**: Vertical timeline of all stages. Each node is clickable. Status indicated by icon/color. Clicking a stage switches the content area to that stage's context.
- **Right content area — Tabs:**
  - **Overview**: Key metrics summary (loss curve thumbnail, pass rate, disk usage), stage duration chart
  - **Logs**: Streaming log viewer (SSE). Filter by stage. Search within logs. Virtual scrolling for performance.
  - **Metrics**: Interactive charts (loss over steps, eval scores). Recharts with sliding window (last 500 points) + zoom for history.
  - **Artifacts**: File list with name, type, size, created date. Download buttons. Grouped by stage.
  - **Config**: Read-only YAML view (Monaco, syntax highlighted)
  - **Notes**: Editable markdown notes (persisted via API). Auto-populated by scripts.

**Stage card states:**
- `pending` — Gray, inactive, dependencies not met
- `running` — Animated indicator, progress bar (% steps), elapsed time, latest 3 log lines preview, Cancel button
- `succeeded` — Green check, duration, key metric (e.g. final loss), copyable shell command
- `failed` — Red, error summary, exit code, full error log link, Retry button (with auto-cleanup option)
- `skipped` — Dashed border, gray, explanation (e.g. "squeeze disabled in config")
- `cancelled` — Orange, timestamp of cancellation

### 4.5 Dataset Page

Tabbed interface for dataset inspection and preparation.

**Tabs:**
- **Preview**: JSONL viewer (first 50 rows), row count, column info, file size
- **Validate**: Run validation, show results (missing fields, type errors, format issues)
- **Clean**: Detect duplicates, show duplicate count, one-click dedup, quality score
- **Export**: Format selection (JSONL, CSV), filtered export, download

### 4.6 Artifact Browser

Cross-run artifact browser. Distinct from Run Detail's artifacts tab (which is run-scoped).

**Features:**
- Browse all artifacts across runs
- Filter by type (adapter, dataset, log, eval result, config)
- Sort by date, size, run
- Download with Range Request support for large files

### 4.7 System Page

Environment information and global controls.

**Sections:**
- **Environment**: Python version, Torch version, CUDA version, GPU model, GPU memory (total/free)
- **Storage**: Disk usage (free/total GB), per-run disk breakdown
- **ml-ssd**: Checked out ref, last updated
- **Actions**: Bootstrap/Re-bootstrap Environment, Disk Cleanup
- **Resources**: Links to `docs/references.md` content (rendered inline)

## 5. Suggested Backend API

### 5.1 Runs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/runs?cursor={c}&limit=20&status={s}&mode={m}` | List runs (cursor-paginated, filterable) |
| POST | `/api/runs` | Create a new run (accepts `{name, config, mode}`) |
| GET | `/api/runs/{run_name}` | Run detail with all stage statuses |
| DELETE | `/api/runs/{run_name}?confirm=true` | Delete run + all artifacts (requires `confirm=true`) |

**Pagination format:**
```
Request:  ?cursor=abc123&limit=20
Response: { "items": [...], "next_cursor": "def456", "has_more": true }
```
Cursor is an opaque string. Pass `next_cursor` to fetch the next page.

### 5.2 Stage Actions

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/runs/{run_name}/actions/prepare-prompts` | Execute prepare-prompts stage |
| POST | `/api/runs/{run_name}/actions/generate-ssd` | Execute generate-ssd stage |
| POST | `/api/runs/{run_name}/actions/convert-ssd` | Execute convert-ssd stage |
| POST | `/api/runs/{run_name}/actions/build-datasets` | Execute build-datasets stage |
| POST | `/api/runs/{run_name}/actions/train-code` | Train code adapter (sequential) |
| POST | `/api/runs/{run_name}/actions/train` | Train mixed adapter |
| POST | `/api/runs/{run_name}/actions/train-skill0` | Train skill0 adapter (sequential) |
| POST | `/api/runs/{run_name}/actions/squeeze` | Execute squeeze |
| POST | `/api/runs/{run_name}/actions/recover` | Execute recovery |
| POST | `/api/runs/{run_name}/actions/evaluate?adapter_subdir={dir}` | Run evaluation (optional adapter override) |
| POST | `/api/runs/{run_name}/cancel` | Cancel running stage |
| POST | `/api/runs/{run_name}/stage/{stage}/retry` | Retry failed stage (with cleanup) |

### 5.3 Data & Observability

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/runs/{run_name}/logs?stage={stage}&after={line_id}&limit=1000` | Historical logs (line-offset paginated) |
| GET | `/api/runs/{run_name}/logs/stream?stage={stage}&after={last_event_id}` | SSE log stream (resumes from `last_event_id`) |
| GET | `/api/runs/{run_name}/metrics?stage={stage}&from_step=0&to_step=500` | Training metrics (step-range paginated) |
| GET | `/api/runs/{run_name}/artifacts` | Artifact list with metadata (name, type, size, created_at) |
| GET | `/api/runs/{run_name}/artifacts/{path}` | Download artifact (supports Range requests) |
| GET | `/api/runs/{run_name}/config` | Parsed run config |
| GET | `/api/runs/{run_name}/commands` | Copyable shell commands per stage |
| GET | `/api/runs/{run_name}/notes` | Run notes (markdown) |
| PUT | `/api/runs/{run_name}/notes` | Update run notes |

### 5.4 Datasets

> **Note:** `build-skill0` and `build-mixed` endpoints are for the Dataset page's manual/standalone operations.
> The run-scoped `/actions/build-datasets` calls these internally as part of the pipeline.
> Both can coexist: standalone for data exploration, pipeline for automated execution.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/datasets/preview?path={path}&limit=50` | Preview JSONL rows (path sandboxed to runs directory, no `..` allowed) |
| POST | `/api/datasets/validate` | Validate dataset |
| POST | `/api/datasets/normalize` | Normalize dataset |
| POST | `/api/datasets/build-skill0` | Build skill0 dataset |
| POST | `/api/datasets/build-mixed` | Build mixed dataset |

### 5.5 System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/system` | System info (Python, Torch, CUDA, GPU, disk, ml-ssd ref) |
| POST | `/api/system/bootstrap` | Run environment bootstrap |
| GET | `/api/config/schema` | JSON Schema for train config YAML |
| GET | `/api/references` | Rendered references content |

### 5.6 SSE Event Formats

Every SSE event includes an `id` field (monotonic sequence number) for reconnect-based resume
and a `retry:` field (recommended 3000ms).

**Log line event:**
```
id: 42
retry: 3000
event: log
data: {"stream": "stdout", "text": "...", "timestamp": "2025-01-01T00:00:00Z"}
```

**Progress event:**
```
id: 43
retry: 3000
event: progress
data: {"stage": "train", "step": 100, "total": 500, "loss": 0.45, "elapsed_seconds": 1200}
```

**Stage status event:**
```
id: 44
retry: 3000
event: stage-status
data: {"stage": "train", "status": "succeeded", "duration_seconds": 3600}
```

### 5.7 Error Response Format

```json
{
  "error": {
    "code": "RUN_NOT_FOUND",
    "message": "Run 'foo' does not exist",
    "details": {}
  }
}
```

**Status code mapping:**
| Code | Meaning | When |
|------|---------|------|
| 400 | Bad Request | Malformed parameters |
| 404 | Not Found | Run/stage/artifact does not exist |
| 409 | Conflict | Stage already running, run name exists, path conflict |
| 422 | Validation Error | Invalid config, missing required fields |
| 500 | Internal Error | Unexpected server failure |

**Stage action idempotency:** `POST /actions/{stage}` returns 409 if the stage is already `running`.
`POST /cancel` returns 409 if no stage is running. `POST /retry` cleans up failed artifacts before re-executing.

## 6. Artifact Map

| Stage | Produced Artifacts |
|-------|--------------------|
| prepare-prompts | `ssd_prompts.jsonl`, `ml_ssd_config.generated.yaml` |
| generate-ssd | `raw_ssd_outputs.jsonl` |
| convert-ssd | `ssd_train.jsonl` |
| build-datasets | `skill_views/`, `skill0_train.jsonl`, `mixed_train.jsonl` (mixed mode only) |
| train-code | `adapter_code_high_rank/` |
| train (mixed) | `adapter_mixed_high_rank/` |
| train-skill0 | `adapter_skill0_high_rank/` |
| squeeze | `adapter_squeezed/`, `compression_report.json` |
| recover | `adapter_recovered/` |
| evaluate | `eval/results_t{temp}.jsonl`, `eval/summary.json` |
| plan | `{mode}_plan.sh` |
| global | `notes.md`, `config.yaml` (saved by backend on run creation) |

## 7. UX Rules

### 7.1 Stage Guards

- Never allow `squeeze` before the selected training stage succeeds.
- Never allow `recover` before `squeeze` succeeds.
- Evaluation defaults to the best available adapter (recovered > squeezed > high-rank > base).
- The UI must display whether the active run is **sequential** or **mixed** with a prominent badge.
- `build-datasets` includes `build-mixed` only in mixed mode; sequential mode skips it.

### 7.2 Visual Design Language

**Style direction:** Industrial/utilitarian — dense information, minimal decoration, high signal-to-noise.
Think "mission control" not "landing page."

**Dark-first color system:**

| Token | Light | Dark | Usage |
|-------|-------|------|-------|
| `bg-primary` | `#FFFFFF` | `#0F1117` | Page background |
| `bg-surface` | `#F5F5F5` | `#1A1B23` | Cards, panels |
| `bg-elevated` | `#FFFFFF` | `#242631` | Modals, popovers |
| `text-primary` | `#1A1A2E` | `#E4E5F0` | Body text (contrast ≥ 4.5:1) |
| `text-secondary` | `#555770` | `#9CA0B0` | Labels, descriptions |
| `accent` | `#3B7DD8` | `#5B9CF5` | Primary actions, links |
| `status-success` | `#16A34A` | `#22C55E` | Succeeded stages |
| `status-error` | `#DC2626` | `#EF4444` | Failed stages |
| `status-warning` | `#D97706` | `#F59E0B` | Cancelled, low disk |
| `status-info` | `#2563EB` | `#60A5FA` | Running indicator |

**Typography:**
- UI font: `Geist Sans` (clean, tech-oriented) — fallback: `Inter, system-ui, sans-serif`
- Code font: `JetBrains Mono` — fallback: `Menlo, monospace`
- Scale: 8pt grid. Base 14px. Line-height 1.5. Small 12px (labels), Large 20px (headings).

**Spacing:** 8pt grid system (4, 8, 12, 16, 24, 32, 48, 64).

**Icon strategy:** Ant Design Icons as base + custom SVG for ML-specific concepts (GPU, loss curve, adapter).
No emoji as icons.

**Stage status colors** must be paired with an icon + text label (never color alone):
- succeeded → green circle + check icon + "Succeeded"
- failed → red circle + x icon + "Failed"
- running → blue pulsing circle + spinner + "Running"
- pending → gray circle + clock icon + "Pending"
- skipped → dashed gray circle + skip icon + "Skipped"
- cancelled → amber circle + stop icon + "Cancelled"

- Show shell commands and make them copyable from every stage card.
- Use SSE for streaming logs (not WebSocket). Client: `@microsoft/fetch-event-source`.
- Running stages show progress (step/total), elapsed time, and latest log preview.
- Failed stages show error summary, exit code, and Retry button with auto-cleanup.
- Launch from New Run Wizard redirects immediately to Run Detail.
- Empty states (no runs, no datasets) show helpful CTA to guide first action.
- Destructive actions (Delete Run, Cancel Stage, Re-bootstrap) require confirmation dialog with clear description of consequences. Confirm button uses `status-error` color.
- Time display: relative for recent ("3 minutes ago" with ISO tooltip), duration in `Xh Ym Zs` format.

### 7.3 Notification System

**Three-layer notification strategy:**

| Layer | Tool | Triggers | Dismiss |
|-------|------|----------|---------|
| Browser push | Notification API | Stage failed, training completed, disk <10% | Auto or manual |
| Dashboard bell | In-app event list | All browser push events + stage transitions | Manual clear |
| Toast | sonner | Action feedback (started, cancelled, saved) | Auto 4s |

**Browser notification permission:** Request on first stage launch (not on page load).
**Notification click:** Deep-links to the relevant Run Detail page, auto-scrolls to the triggering stage.
**SSE reconnect:** On disconnect, show toast "Connection lost. Reconnecting…" + auto-retry (via `retry: 3000`). On reconnect, resume from last `id`.

### 7.4 Loading & Error States

**Loading strategy (three tiers):**

| Tier | Trigger | Pattern |
|------|---------|---------|
| Route transition | Page navigation | Ant Design `Spin` overlay (300ms debounce) |
| Data fetch | TanStack Query loading | Skeleton placeholders for tables/cards |
| Long operation | Stage running | Progress bar + elapsed time + live log preview |

**Error handling UI:**

| Scenario | Frontend behavior |
|----------|-------------------|
| API 4xx | Toast with error message. Form: inline field errors. |
| API 5xx | Full-page error boundary with "Retry" button. |
| Network offline | Banner: "Connection lost. Retrying…" |
| SSE disconnect | Toast + auto-reconnect. If >30s: "Connection lost. Reconnecting…" |
| Zod parse failure | Error boundary + console error (API contract violation) |
| Empty state (no runs) | Title "No runs yet" + description + CTA button "Create Your First Run" |
| Empty state (no datasets) | Title "No datasets prepared" + CTA "Start with Prepare Prompts" |

**TanStack Query error config:** Network errors retry 3x (1s backoff). Business errors (4xx) no retry.

### 7.5 Keyboard & Accessibility

- All interactive elements reachable via Tab navigation. Tab order follows visual layout (left→right, top→bottom).
- Focus ring: 2px solid `accent` color, 2px offset. Always visible, never `outline: none`.
- Enter to activate buttons, Escape to close modals/dialogs. Focus returns to trigger element on close.
- ARIA labels on icon-only buttons (e.g., `aria-label="Cancel stage"`).
- `prefers-reduced-motion` respected: disable pulse animations, reduce transitions to 0ms.
- Run Detail keyboard trap: Tab cycles within sidebar timeline → tab content. `F6` to jump between sidebar and content.
- Monaco editor: accessible via Tab. Escape exits editor focus back to page flow.
- Desktop-first; mobile is not a target for V1. Minimum viewport width 1024px. Below 1024px, sidebar collapses to hamburger menu.
- Charts (Recharts): include `aria-label` describing data. Provide data table alternative for screen readers.
- Min interactive element size: 32×32px (desktop). Touch targets for icon buttons padded to meet this.

## 8. Recommended Stack

### 8.1 Frontend

| Category | Choice | Rationale |
|----------|--------|-----------|
| Framework | React 19 + TypeScript 5 | Mature ecosystem, strong typing |
| Build | Vite 6 | Fast HMR, native TS support |
| Server State | TanStack Query v5 | Caching, refetch, SSE integration |
| Client State | Zustand | Lightweight, wizard/UI state |
| Routing | TanStack Router | Type-safe, same ecosystem as TanStack Query |
| Forms | React Hook Form + Zod | Multi-step wizard, runtime validation aligned with backend Pydantic |
| UI Components | Ant Design 5 | Production Table/Form/Steps/Timeline/Notification; customize via design tokens |
| Charts | Recharts | Loss curves, eval scores; sliding window for real-time |
| Code Editor | Monaco Editor | Config YAML editing with syntax highlighting + validation (NOT for logs) |
| Log Viewer | `@tanstack/react-virtual` + `prism-react-renderer` | Virtual scrolling for 100k+ lines, lightweight syntax highlight |
| Realtime | `@microsoft/fetch-event-source` | SSE client (POST support, auto-reconnect) |
| Notifications | sonner | Toast notifications, lightweight |
| Markdown | react-markdown | Notes tab, references page |

**Styling strategy:** Ant Design's design token system for all theming. CSS-in-JS via Ant Design's built-in
emotional styling. No Tailwind — avoid class conflicts with Ant Design's style system.

### 8.2 Backend

| Category | Choice | Rationale |
|----------|--------|-----------|
| Framework | FastAPI | Async, Pydantic, Python-native |
| Validation | Pydantic v2 | Auto-generates JSON Schema for frontend |
| Subprocess | asyncio subprocess | Non-blocking stage orchestration |
| Realtime | SSE | Unidirectional push (logs, progress); WebSocket only if bidirectional needed later |
| File Serving | FastAPI StaticFiles + Range | Large artifact downloads with resume |

### 8.3 Developer Tooling

| Category | Choice |
|----------|--------|
| Testing | Vitest + React Testing Library + MSW |
| Linting | ESLint + Prettier |
| Type Checking | TypeScript strict mode |

### 8.4 Type System & Schema Alignment

**Source of truth:** Backend Pydantic models → `GET /api/config/schema` returns JSON Schema.

**Frontend strategy:**
1. Pydantic models auto-generate JSON Schema via `model_json_schema()`
2. Frontend uses `json-schema-to-zod` to convert JSON Schema → Zod schemas
3. Zod schemas used for: API response validation, form validation, type inference
4. CI check: `GET /api/config/schema` output diff against committed Zod schemas

**Stage status as discriminated union:**
```typescript
type StageStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'skipped' | 'cancelled';

type Stage = {
  name: string;
  status: StageStatus;
} & (
  | { status: 'pending' }
  | { status: 'running'; progress: number; total: number; elapsed_seconds: number }
  | { status: 'succeeded'; duration_seconds: number; metric?: number }
  | { status: 'failed'; error_message: string; exit_code: number }
  | { status: 'skipped'; reason: string }
  | { status: 'cancelled'; cancelled_at: string }
);
```

**Error boundary placement:** One per page route + one wrapping Run Detail tab content.

## 9. Navigation Architecture

```
Sidebar (5-6 items):
├── Dashboard (/)
├── Runs (/runs)
│   └── Run Detail (/runs/:runName) ← sub-route, not top-level nav
├── Datasets (/datasets)
├── Artifacts (/artifacts)
└── System (/system)
```

Training-mode selector is NOT a separate page — it is Step 3 in the New Run Wizard.
References content is rendered within the System page, not a standalone page.

## 10. Source References

The canonical list of papers and repository URLs lives in `docs/references.md`.
