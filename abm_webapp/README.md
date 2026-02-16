# `abm_webapp` – ABM Live Lab

Single-user Dash webapp for running ABM_Uni_v3 simulations with live plots,
diagnostics, and log tailing. Designed for colleagues to install and run on
their personal PCs (Windows / macOS / Linux).

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
| **Logs** | Live tail of the simulation verbose log |

## Scenario Configuration

Place `.yml` scenario files under `abm_results/scenarios/`. The webapp will
list them in a dropdown for quick loading. YAML configs are strictly validated
before starting a run:

- Unknown top-level keys are rejected
- Critical parameters are type-checked and bounds-checked
- Resource-heavy settings are capped (override via `ABM_MAX_T`, `ABM_MAX_N_LP`,
  `ABM_MAX_BLOCK_TIME` environment variables)
- YAML is always parsed with `yaml.safe_load` (no arbitrary object construction)

## Run Metadata & Reproducibility

Every run persists to `abm_results/web_runs/<run_id>/`:

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
  are added transparently.
- Existing runs are not affected; the migration only adds columns.
- The `abandoned` state is new; pre-existing "stuck running" runs will be
  marked abandoned on next startup.
- Config validation is now stricter: scenarios that previously worked may
  need small adjustments if they contained unknown keys.

See `docs/webapp.md` for more details.
