---
title: Webapp
nav_order: 9
---

# Live simulation webapp (single user)

This repository includes a Dash webapp that runs `run.simulate()` in a separate process and streams live diagnostics:

- CEX + DEX prices with no-arbitrage fee band and return distributions
- Per-block and cumulative PnL panels for smart/noise/arb/LP/Jiter cohorts
- Fee controller telemetry (signal alignment + fee distribution)
- LP decomposition (hedged vs unhedged), fee value, and cumulative LVR
- Normalized LVR diagnostics from streamed per-step deltas
- Execution activity and smart-router DEX routing-share analytics
- Live tail of the simulation log file

## Install

From the repo root:

```bash
conda activate main
# Dash is not part of the original environment.yml; install it once:
conda install -c conda-forge dash
```

If `conda activate main` does not work in your shell, you can prefix commands with `conda run -n main ...`.

(Alternative: `pip install dash`.)

## Run

```bash
conda activate main
python -m abm_webapp.app --host 127.0.0.1 --port 8050
```

Alternative (no activation required):
```bash
conda run -n main python -m abm_webapp.app --host 127.0.0.1 --port 8050
```

Open the printed local address in your browser (typically `http://127.0.0.1:8050`).

## How it works (high level)

- The UI lives in `abm_webapp/app.py`.
- Each run executes in a separate `multiprocessing.Process` via `abm_webapp/worker.py`.
- Live data is written to `abm_results/web_runs/<run_id>/live.db` (SQLite WAL mode) using `abm_webapp/storage.py`.
- The simulation log is written under `abm_results/web_runs/<run_id>/logs/…` and flushed periodically for live tailing.
