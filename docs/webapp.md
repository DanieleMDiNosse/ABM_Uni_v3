---
title: Webapp
nav_order: 9
---

# Live simulation webapp (single user)

This repository includes a Dash webapp that runs `scripts.run.simulate()` in a separate process and streams live diagnostics:

- CEX + DEX prices with no-arbitrage fee band and return distributions
- Per-block and cumulative PnL panels for smart/noise/arb/LP/Jiter cohorts
- Fee controller telemetry (signal alignment + fee distribution)
- LP decomposition (hedged vs unhedged), fee value, and cumulative LVR
- Normalized LVR diagnostics from streamed per-step deltas
- Execution activity and smart-router DEX routing-share analytics
- Live tail of the simulation log file

This page explains the webapp construction from a software-engineering perspective: architecture, module boundaries, process model, data contracts, reliability behavior, and extensibility.

## Engineering goals

The webapp is intentionally built around a few conservative goals:

- Keep simulation execution isolated from UI rendering.
- Preserve reproducibility by persisting full scenario YAML + streamed metrics.
- Use standard, low-friction components (Dash, stdlib `multiprocessing`, stdlib `sqlite3`, plain JS SSE client).
- Prioritize robustness and debuggability over distributed scalability.

Non-goal: multi-tenant deployment. The current design is explicitly single-user, one active run at a time.

## System architecture

The design separates responsibilities into four layers:

1. **Presentation layer** (`abm_webapp/app.py`, `abm_webapp/assets/theme.css`):
   - Dash layout, Plotly figure builders, run controls, callbacks.
   - Local Flask route for SSE (`/stream/run/<run_id>`).
2. **Worker layer** (`abm_webapp/worker.py`):
   - Process entry point that validates/normalizes runtime parameters and executes `scripts.run.simulate(...)`.
3. **Persistence layer** (`abm_webapp/storage.py`):
   - SQLite schema, writer sink (`SQLiteLiveSink`), read APIs (`read_metrics`, `read_status`), log tail helper.
4. **Simulation engine** (`scripts/run.py`):
   - Core ABM loop with optional live hooks (`live_sink`, `live_every`, `stop_event`, `log_flush_every`).

This yields a clear control plane / data plane split:

- **Control plane**: start/stop/reset actions and run lifecycle state (`_RunController` in `app.py`).
- **Data plane**: metric rows + status + logs written by the worker and consumed by UI/SSE readers.

## Runtime lifecycle (end-to-end)

1. User loads/edits YAML in Dash and clicks **Run**.
2. `app.py` callback validates config against `scripts.run.simulate` signature using `core.utils.load_simulation_parameters`.
3. `_RunController.start(...)` creates a unique run folder:
   - `abm_results/web_runs/<run_id>/scenario.yml`
   - `abm_results/web_runs/<run_id>/live.db`
   - `abm_results/web_runs/<run_id>/logs/...`
4. A separate `multiprocessing.Process` starts `run_simulation_process(...)`.
5. Worker calls `scripts.run.simulate(...)` with live hooks enabled.
6. Simulation loop periodically writes compact metric rows to SQLite and flushes logs.
7. Browser-side `EventSource` connects to `/stream/run/<run_id>`.
8. SSE stream emits events (`snapshot`, `metrics_delta`, `status_change`, `heartbeat`, `end`).
9. JS bridge increments a hidden Dash input (`stream-event-seq`) to trigger callbacks.
10. Dash callbacks pull fresh data from SQLite/log files, rebuild selected figures, and update UI state.

## Module-level construction details

### `abm_webapp/worker.py`: process boundary and run normalization

- Defines top-level `run_simulation_process(...)` so the target is picklable for process spawning.
- Uses `_normalize_params_for_webapp(...)` to enforce interactive-safe defaults:
  - `visualize=False` (avoid heavy artifact generation during live runs)
  - `liquidity_for_gif=False`
  - `verbose=False` (no tqdm/console noise; file logs still enabled)
  - `light_mode=False` (keep logging + needed series available)
- Persists scenario YAML into run folder for reproducibility.
- Handles exceptions centrally:
  - Writes traceback to `error.log`
  - Marks status as `error` in SQLite.
- On normal completion, maps stop-request vs completion to `stopped` or `finished`.

### `abm_webapp/storage.py`: SQLite-backed live telemetry

Key design choice: **SQLite in WAL mode** for one writer (worker) + concurrent readers (Dash/SSE).

- Connection setup:
  - `PRAGMA journal_mode=WAL`
  - `PRAGMA synchronous=NORMAL`
  - `PRAGMA foreign_keys=ON`
- Schema:
  - `run_meta`: immutable run context (`run_id`, `params_yaml`, `results_root`, `T`, timestamp).
  - `run_status`: mutable lifecycle record (`state`, `t_last`, `message`, `log_path`, timestamp).
  - `metrics`: time-indexed (`t` primary key) live telemetry columns for price, PnL, LP/LVR, routing, fee-controller signals.
- Backward compatibility:
  - `_ensure_schema(...)` includes additive column migrations via guarded `ALTER TABLE ... ADD COLUMN`.
- Writer object (`SQLiteLiveSink`):
  - Buffers rows and commits every `commit_every` for amortized write overhead.
  - Exposes `set_log_path`, `record_step`, `set_status`, `flush`, `close`.
- Reader APIs:
  - `read_status(...)` returns a typed `RunStatus` dataclass.
  - `read_metrics(...)` supports incremental reads (`since_t`) and capped latest-window reads (`limit`).
  - `tail_text_file(...)` supports byte-tailing of large logs.

### `scripts/run.py`: non-invasive live hooks inside simulation loop

`simulate(...)` accepts optional webapp hooks:

- `live_sink`
- `live_every`
- `stop_event`
- `log_flush_every`

Engineering choice: hooks are **duck-typed and best-effort**. Failures in live reporting should not break the simulation.

- Every loop iteration:
  - checks `stop_event.is_set()` and exits gracefully if requested.
  - flushes buffered log text every `log_flush_every` blocks.
  - emits one compact metrics row every `live_every` blocks via `live_sink.record_step(...)`.
- On startup:
  - creates scenario-aware log path and calls `live_sink.set_log_path(...)` when available.

This keeps web instrumentation low-coupled with the simulation core.

### `abm_webapp/app.py`: orchestration, SSE, callback graph, visualization

`app.py` is the control center, with four important engineering mechanisms:

1. **Run orchestration**
   - `_RunController` stores active `run_id`, `run_root`, `Process`, and `Event`.
   - `start` validates input and starts worker process.
   - `stop` signals event only (cooperative cancellation).
   - `reset_all` terminates any active process if needed and clears `abm_results/web_runs/`.

2. **SSE endpoint**
   - Flask route `/stream/run/<run_id>` polls SQLite and emits structured events.
   - Event types:
     - `snapshot`: initial hydrate metadata
     - `metrics_delta`: new rows detected
     - `status_change`: lifecycle transition/message changes
     - `heartbeat`: keepalive when data is quiet
     - `end`: terminal signal once status is terminal and no pending rows remain
   - Adds response headers for streaming friendliness (`no-cache`, `keep-alive`, `X-Accel-Buffering: no`).

3. **Tiered callback execution (performance governor)**
   - Core figures update every stream tick:
     - price, per-block PnL, cumulative PnL, fee panel, summary cards/status.
   - Medium-cost figures update every `MEDIUM_FIG_UPDATE_EVERY` deltas.
   - Heavy-cost figures update every `HEAVY_FIG_UPDATE_EVERY` deltas.
   - Uses Dash `no_update` aggressively when data did not change.
   - Applies point-capping/downsampling constants (`MAX_TIMESERIES_POINTS`, `MAX_DISTRIBUTION_POINTS`, `LIVE_METRICS_LIMIT`) to bound browser load.

4. **In-memory caching**
   - Metrics cache by run root with `last_t` tracking for incremental hydration.
   - Log text cache and byte-offset cache for incremental tail rendering.
   - Fee mode cache (avoid reparsing scenario YAML each tick).
   - Update-counter cache to coordinate tiered figure cadence.
   - All caches are guarded by thread locks.

### Frontend stream bridge: `abm_webapp/assets/stream_sse.js`

Dash callbacks are not directly bound to EventSource, so the JS bridge:

- Watches hidden input `stream-run-id`.
- Opens/closes `EventSource` per active run.
- Implements reconnect with exponential backoff.
- Throttles dispatch frequency (`MIN_DISPATCH_MS`) for noisy event types.
- Pushes synthetic sequence increments into hidden numeric input (`stream-event-seq`) to trigger Dash callbacks.

This pattern keeps the app event-driven without relying on polling intervals in Dash itself.

### Presentation layer styling: `abm_webapp/assets/theme.css`

- Uses CSS custom properties for palette and spacing tokens.
- Customizes Dash/React-select controls for visual consistency.
- Includes responsive breakpoints (`1200px`, `760px`) for desktop/laptop use.
- Uses light animation (`rise`) and controlled graph heights for readability in long dashboards.

## Data and artifact layout

For each run:

- `abm_results/web_runs/<run_id>/scenario.yml`: immutable run config snapshot.
- `abm_results/web_runs/<run_id>/live.db`: streaming DB (status + metrics).
- `abm_results/web_runs/<run_id>/logs/*.txt`: verbose simulation logs.
- `abm_results/web_runs/<run_id>/error.log`: traceback if worker fails.

This folder structure enables deterministic post-mortem debugging and reproducibility.

## Reliability and failure handling

Main reliability patterns used:

- Preflight validation of YAML against actual `simulate(...)` signature before process start.
- Worker isolation from UI process prevents UI crash when simulation errors.
- Terminal statuses (`finished`, `stopped`, `error`) are persisted in DB and emitted through SSE.
- Heartbeat events keep stream liveness explicit.
- Graceful stop uses `Event`; forced terminate only as reset fallback.
- Streaming/log hooks in `scripts/run.py` are wrapped to avoid destabilizing the simulation engine.

## Performance and scalability trade-offs

The app is optimized for a single researcher workflow, not horizontal scale:

- One active run controller in-process.
- Local SQLite file per run (excellent simplicity, limited write throughput ceiling).
- Event-driven UI updates with throttled callback triggering.
- Tiered figure cadence and downsampling avoid expensive redraw storms.

If multi-user or high-throughput execution is needed later, natural upgrade points are:

- Replace in-process controller with job queue + worker supervisor.
- Replace SQLite with external time-series/store.
- Replace local SSE polling loop with pub/sub stream.

## Test coverage and quality gates

Webapp-specific tests are focused on correctness of the engineering contract:

- `tests/test_webapp_storage.py`:
  - sink writes/readbacks, status integrity, and schema-level roundtrip fields.
- `tests/test_webapp_sse_helpers.py`:
  - SSE frame format, stream snapshot emission, text-delta behavior, callback wiring assumptions.
- `tests/test_webapp_app_sampling.py`:
  - downsampling bounds and figure trace point caps.

These tests target the fragile boundaries: serialization, streaming semantics, and frontend performance constraints.

## Install

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Run

```bash
amm-abm-web
```

Or with custom host/port:

```bash
amm-abm-web --host 0.0.0.0 --port 9000
```

Open the local URL shown in terminal (typically `http://127.0.0.1:8050`).
