# `abm_webapp`

Single-user Dash webapp for running ABM_Uni_v3 simulations with live plots, diagnostics, and log tailing.

Key live panels:
- `Price`: CEX/DEX path with no-arbitrage band + return distributions.
- `PnL`: per-block and cumulative PnL for all cohorts.
- `Fees`: fee and controller signal alignment + fee distribution.
- `LP & LVR`: hedged vs unhedged LP decomposition, cumulative fee value/LVR, normalized LVR diagnostics.
- `Activity & Routing`: execution counts and smart-router DEX routing share.
- `Logs`: live tail of the simulation verbose log.

Run from repo root:

```bash
conda activate main
conda install -c conda-forge dash
python -m abm_webapp.app
```

See `docs/webapp.md` for details.
