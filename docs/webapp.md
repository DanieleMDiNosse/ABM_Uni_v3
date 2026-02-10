---
title: Webapp
nav_order: 9
---

# Live simulation webapp (single user)

This repository now includes a small Dash webapp that can run `run.simulate()` in a separate process and stream:

- CEX + DEX prices with the no-arbitrage band
- PnL per block
- Cumulative PnL
- Fee controller signal + fee distribution
- A live tail of the simulation log file

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
