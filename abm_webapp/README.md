# `abm_webapp` – ABM Live Lab

Single-user Dash webapp for running ABM_Uni_v3 simulations with live plots and
diagnostics. It is an exploratory single-run laboratory: use it to quickly inspect
what happens under different fee schedules and parameter settings, then use the
CLI/multi-run scripts for paper-grade results, robustness checks, and confirmatory
claims. Designed for colleagues to install and run on their personal PCs
(Windows / macOS / Linux).

## Quick Start (end user)

```bash
# 1. Clone the repo
git clone <repo-url>
cd ABM_Uni_v3

# 2. Create a virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 3. Install the package (includes all dependencies)
pip install -e ".[dev]"

# 4. Start the webapp
amm-abm-web
```

The webapp opens at **http://127.0.0.1:8050** (local-only by default).

### Options

```
amm-abm-web --host 0.0.0.0 --port 9000   # bind to all interfaces
```

You can also run with `python -m abm_webapp --host 127.0.0.1 --port 8050`.

### Conda alternative

```bash
conda activate main          # or your existing conda env
pip install -e ".[dev]"      # installs the package into the conda env
amm-abm-web
```

## Live Panels

| Tab | Content |
|-----|---------|
| **Price** | CEX/DEX path with no-arbitrage band + return distributions |
| **PnL** | Per-block and cumulative PnL for all cohorts |
| **Fees** | Fee and controller signal alignment + fee distribution |
| **LP & LVR** | Hedged vs unhedged LP decomposition, cumulative fee value/LVR, normalized LVR diagnostics |
| **Activity & Routing** | Execution counts and smart-router DEX routing share |

## Scenario Configuration

Place runnable `.yml` scenario files under `configs/scenarios/`. The
webapp only lists files that validate against the current `scripts.run.simulate()`
contract; sweep/dashboard YAMLs that are not runnable scenarios are ignored.
The **Fee schedule preset** dropdown updates `fee_mode` (top-level and
`simulate.fee_mode`) so quick comparisons keep all other YAML parameters fixed.
For the `linear_asymmetric` preset it also inserts a default
`simulate.asymmetric_fee_slope` when that key is absent, because this schedule
requires a signed DEX/oracle gap gain.
Edit the YAML directly for other exploratory knobs such as `T`, `seed`,
`block_time`, arrival rates, LP shares, JIT settings, and fee-controller gains.
YAML configs are strictly validated before starting a run:

- Unknown top-level keys are rejected
- Critical parameters are type-checked and bounds-checked
- Resource-heavy settings are capped (override via `ABM_MAX_T`, `ABM_MAX_N_LP`,
  `ABM_MAX_BLOCK_TIME` environment variables). The default `ABM_MAX_T` is
  set high enough for the repository's bundled yearly scenarios.
- Live plots keep the full run history by default. Set `ABM_WEBAPP_LIVE_METRICS_LIMIT`
  to a positive integer only if you explicitly want to cap in-memory live history.
- YAML is always parsed with `yaml.safe_load` (no arbitrary object construction)

## Run Metadata & Reproducibility

Every run persists to `abm_results/web_runs/<run_id>/`; these outputs are kept
separate from canonical CLI scenario folders under `abm_results/scenarios/`.

| File | Contents |
|------|----------|
| `scenario.yml` | Validated, canonical config snapshot |
| `live.db` | SQLite with metrics, status, heartbeat, and run metadata |
| `run_meta.json` | App version, git commit, Python version, platform, pip freeze, seed |
| `logs/` | Simulation log files |

## Crash Recovery

On startup the webapp scans existing runs. Any run marked `running` or
`stopping` with a dead worker PID or stale heartbeat (>60 s) is automatically
marked `abandoned` with a diagnostic message.

## Architecture

```
abm_webapp/
├── app.py        – Dash layout, callbacks, SSE endpoint, run controller
├── config.py     – Strict YAML validation, bounds checks, resource caps
├── run_meta.py   – Environment fingerprint collection
├── storage.py    – SQLite schema, WAL, heartbeat, crash recovery, retry
├── worker.py     – Simulation process entry point + heartbeat thread
├── __init__.py   – Version
├── __main__.py   – `python -m abm_webapp` entry
└── assets/       – JS (SSE client) + CSS (dark theme)
```

## For Developers

### Run tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Run specific webapp tests

```bash
pytest tests/test_webapp_lifecycle.py -v
pytest tests/test_webapp_storage.py -v
pytest tests/test_webapp_sse_helpers.py -v
```

### Code style

```bash
ruff check abm_webapp/ tests/
ruff format abm_webapp/ tests/
```

## Migration Notes

### From pre-v0.2.0

- DB schema is auto-migrated: new columns (`pid`, `heartbeat_at`,
  `stop_reason`, `stopped_at`, `meta_json`) and a `schema_info` table
  are added transparently. v0.2.0+ databases also migrate directional fee
  columns (`fee_x_to_y`, `fee_y_to_x`) for the linear-asymmetric schedule.
- Existing runs are not affected; the migration only adds columns.
- The `abandoned` state is new; pre-existing "stuck running" runs will be
  marked abandoned on next startup.
- Config validation is now stricter: scenarios that previously worked may
  need small adjustments if they contained unknown keys.

See `docs/webapp.md` for more details.
