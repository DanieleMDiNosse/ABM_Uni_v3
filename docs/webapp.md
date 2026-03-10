---
title: Webapp
nav_order: 9
---

# Live Simulation Webapp

This repository ships a single-user Dash webapp that runs `scripts.run.simulate()` in a separate process and streams live diagnostics:

- CEX and DEX prices with the no-arbitrage band
- per-block and cumulative PnL for smart router, noise trader, arbitrageur, LP cohorts, and JIT LP
- fee-controller telemetry
- LP hedged vs unhedged decomposition and cumulative LVR
- execution counts and smart-router DEX share
- live tail of the simulation log

The design target is pragmatic local research use: one user, one active run at a time, with strong reproducibility and easy crash recovery.

## User-Facing Runtime Controls

The UI exposes two cadence knobs that directly affect streaming behavior:

- `live_every`: write one metrics row to SQLite every `N` simulation blocks. Lower values improve plot responsiveness but increase write and redraw overhead.
- `log_flush_every`: flush the buffered simulation log every `N` blocks. Lower values reduce log-tail latency but increase I/O.

The worker also forces interactive-safe defaults before calling `simulate(...)`:

- `visualize=False`
- `liquidity_for_gif=False`
- `verbose=False`
- `light_mode=False`

That combination avoids heavy static artifact generation while keeping the series needed for live panels.

## Architecture

The webapp is split into four layers:

1. **Presentation layer**
   - `abm_webapp/app.py`
   - `abm_webapp/assets/theme.css`
   - `abm_webapp/assets/stream_sse.js`
2. **Worker process**
   - `abm_webapp/worker.py`
3. **Persistence layer**
   - `abm_webapp/storage.py`
4. **Simulation engine**
   - `scripts/run.py`

This keeps the UI process separate from the simulation process and makes failure modes easier to reason about.

## Validation And Run Startup

When the user clicks **Run**, `_RunController.start(...)` in `abm_webapp/app.py` performs two validation passes:

1. strict webapp validation through `abm_webapp.config.validate_scenario_text(...)`
   - safe YAML loading
   - unknown-key rejection
   - type and bounds checks on critical parameters
   - resource caps via `ABM_MAX_T`, `ABM_MAX_N_LP`, and `ABM_MAX_BLOCK_TIME`
2. compatibility validation against the real simulation signature
   - write the validated YAML to a temporary file
   - call `core.utils.load_simulation_parameters(...)` against `scripts.run.simulate`

If validation succeeds, the app canonicalizes the YAML, collects reproducibility metadata, creates `abm_results/web_runs/<run_id>/`, and starts a separate `multiprocessing.Process`.

## Worker Process

`abm_webapp/worker.py` defines the top-level picklable target `run_simulation_process(...)`.

Its responsibilities are:

- persist the canonical scenario YAML as `scenario.yml`
- persist environment/run metadata as `run_meta.json`
- open `live.db` via `SQLiteLiveSink`
- start a heartbeat thread that periodically updates the DB
- normalize parameters for interactive runs
- call `scripts.run.simulate(...)` with:
  - `live_sink`
  - `live_every`
  - `stop_event`
  - `log_flush_every`
- mark the final state as:
  - `finished`
  - `stopped`
  - or `error`

If the simulation raises, the worker writes `error.log` and stores the failure message in SQLite.

## SQLite Storage Model

`abm_webapp/storage.py` uses SQLite in WAL mode so a single writer can coexist with concurrent readers.

### Tables

- `run_meta`
  - `run_id`
  - `created_at`
  - `params_yaml`
  - `results_root`
  - `T`
  - `meta_json`
- `run_status`
  - `run_id`
  - `state`
  - `t_last`
  - `message`
  - `updated_at`
  - `log_path`
  - `pid`
  - `heartbeat_at`
  - `stop_reason`
  - `stopped_at`
- `metrics`
  - one row per streamed time step
  - includes price, band, PnL, LP/LVR, routing, and fee-controller fields

### Lifecycle States

The storage layer currently uses:

- non-terminal: `running`, `stopping`
- terminal: `finished`, `stopped`, `error`, `abandoned`

`abandoned` is used for crash recovery when a previously running worker is no longer alive or has a stale heartbeat.

## Streaming Path

The webapp uses Server-Sent Events rather than polling-only Dash intervals.

1. Browser opens `EventSource` on `/stream/run/<run_id>`
2. The SSE loop polls SQLite and log files
3. It emits:
   - `snapshot`
   - `metrics_delta`
   - `status_change`
   - `heartbeat`
   - `end`
4. `abm_webapp/assets/stream_sse.js` increments a hidden Dash input
5. Dash callbacks rehydrate figures from SQLite and the log tail

To keep the UI usable on longer runs, `app.py` also uses:

- in-memory metrics caches
- in-memory log caches
- per-tier render throttling
- point caps such as `MAX_TIMESERIES_POINTS` and `LIVE_METRICS_LIMIT`

## Per-Run Artifact Layout

Each live run lives under:

```text
abm_results/web_runs/<run_id>/
```

Main files:

- `scenario.yml`: canonical validated scenario snapshot
- `live.db`: SQLite status + metrics + metadata
- `run_meta.json`: environment fingerprint and run metadata
- `logs/*.txt`: simulation logs
- `error.log`: traceback on worker failure

`run_meta.json` is intentionally redundant with `run_meta.meta_json` in SQLite so metadata is easy to inspect without opening the DB.

## Crash Recovery

On startup, `abm_webapp/app.py` calls `scan_and_recover(...)`.

A run is marked `abandoned` when:

- its stored state is `running` or `stopping`, and
- either the recorded worker PID is dead, or the heartbeat is stale

This behavior is designed for local workflows where the app or worker may be interrupted by notebook restarts, terminal closes, or machine sleep.

## Reliability Notes

The current implementation deliberately favors debuggability over distributed scale:

- one active run controller in-process
- one SQLite file per run
- cooperative stop via `multiprocessing.Event`
- best-effort live hooks in `scripts/run.py` so streaming failures do not crash the simulation
- reset deletes only `abm_results/web_runs/`, not scenario-run outputs

## Install And Run

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
amm-abm-web
```

Conda-based alternative:

```bash
conda activate main
pip install -e ".[dev]"
python -m abm_webapp --host 127.0.0.1 --port 8050
```

By default the app listens on `http://127.0.0.1:8050`.

## Test Coverage

The webapp-specific tests currently cover:

- `tests/test_webapp_storage.py`
  - sink writes, schema round-trips, and status/metrics readers
- `tests/test_webapp_sse_helpers.py`
  - SSE formatting, delta logic, and helper behavior
- `tests/test_webapp_app_sampling.py`
  - downsampling and plot point caps
- `tests/test_webapp_lifecycle.py`
  - config validation, canonical YAML generation, metadata capture, schema migration, crash recovery, and abandoned-run handling

These tests cover the brittle boundaries: serialization, lifecycle transitions, and streaming behavior.
