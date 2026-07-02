---
title: Model Overview
nav_order: 2
---

# ABM Uni v3 Simulation

<p align="center">
  <img src="abm_results/cex_dex_price.png" alt="Simulation example" width="500"/>
</p>

`ABM_Uni_v3` is a Python-first research simulator for a Uniswap v3-style pool with concentrated liquidity, block-level mempool execution, external CEX reference pricing, LP accounting, experiment-design tooling, and a single-user live Dash webapp.

The main engine lives in `scripts/run.py` and is configured by scenario YAML files under `configs/scenarios/`. A bundled starting point is `configs/scenarios/section4_microstructure_model0_static.yml`.

## What The Simulator Implements

- **Full Uniswap v3 pool math** in `core/uniswapv3_pool.py`: concentrated-liquidity ranges, tick crossing, sparse liquidity-net bookkeeping, and fee-on-input allocation per traversed span.
- **External reference market** in `core/utils.py`: GBM-style diffusion or Heston-style stochastic volatility, plus immediate permanent CEX impact whenever an action trades against the reference market.
- **Agent roster** in `scripts/run.py` and `core/agents.py`: smart router, noise trader, arbitrageur, passive LPs, active narrow LPs, and an optional one-tick JIT LP searcher.
- **LP accounting**: wallet value, open-position value, cumulative fees, rebalancing benchmark, hedged PnL, unhedged PnL, and LVR diagnostics.
- **Experiment tooling**: multi-seed runners, ND grid sweeps, sampled designs, dashboards, profiling helpers, stylized-facts reports, and a live webapp.

For the code-level execution rules, see [Agent Behaviour Details](agents_spec.md). For LP accounting, see [LP PnL](LP_PnL.md) and [Loss-Versus-Rebalancing](LVR_explanation.md).

## Core Mechanics

- **Block-aware mempool execution**: `block_time` micro-steps per block, with the DEX frozen during intake and the mempool replayed at the block boundary. The current implementation requires `block_time > 1`.
- **Validated snapshots**: smart/noise trader slippage baselines, LP out-of-range logic, and arbitrage targets use the previous block’s validated snapshot (`agent_S_ref`, `agent_tick_ref`, `cex_ref_for_agents`), while the live CEX continues to diffuse during the micro-step loop.
- **Fee controllers**: five supported modes, documented in [Fee Schedules](fee_schedules.md):
  - `static`
  - `volatility_cex`
  - `volatility_dex`
  - `toxicity`
  - `lvr_fee_ewma`
- **LP width rule**: active narrow LPs size width from an EWMA of absolute CEX log returns plus a mean-zero binomial noise term, then snap/clamp to the tick grid. Research extensions are discussed in [LP Width Mint Signals](lp_width_mint_signals.md).
- **Seed liquidity**: the pool is bootstrapped by a sharded binomial hill of synthetic seed LPs (`is_seed=True`), which provide initial background liquidity but are excluded from the active/passive cohort aggregates.

### Reference Market Notes

- Static mode uses a per-micro-step GBM update with `cex_mu` and `cex_sigma`.
- Heston mode uses `cex_heston_kappa`, `cex_heston_theta`, `cex_heston_sigma_v`, `cex_heston_rho`, and optional `cex_heston_v0`.
- If `cex_heston_theta_schedule` is provided, the list of theta values is shuffled once per run and then applied as equal-duration regimes across the horizon. This means the YAML list is treated as a multiset of regime levels, not a fixed chronological script.
- Every CEX-touching action applies permanent impact immediately via `ref.apply_impact_only(Δa)`.

## Simulation Flow
1. **CLI setup**
   - `python -m scripts.run --config <scenario.yml>`
   - Load scenario YAML, seed RNGs, build the pool, create LP cohorts, and initialize the rebalancing benchmarks.
2. **Per block**
   - Freeze the validated snapshot.
   - Apply any committed fee update.
   - Diffuse the CEX through `block_time` micro-steps while trader intents are enqueued.
   - Insert one arbitrage intent, schedule LP actions, optionally wrap one target swap with JIT liquidity, then replay the mempool.
   - Update accounting, controller signals, logs, and the next validated snapshot.
3. **Post processing**
   - Save Plotly HTML figures and, when Kaleido/Chrome is available, PNG exports.
   - Persist run metadata and a machine-readable summary row.
   - Optionally build a liquidity GIF.

## Running A Scenario

```bash
conda activate main
python -m scripts.run --config configs/scenarios/section4_microstructure_model0_static.yml
```

Recommended workflow: copy a file from `configs/scenarios/`, rename it, then edit only the parameters you intend to change. The YAML loader is strict: missing required fields, unknown keys, or conflicting top-level vs `simulate.fee_mode` definitions fail fast.

### Preferred Scenario Knobs

- `fee_mode` at the top level is both the scenario label and the default controller selection.
- Prefer the real-time Poisson knobs:
  - `smart_trades_per_second`
  - `noise_trades_per_second`
  - `narrow_mints_per_second`
  - `passive_mints_per_second`
  - `passive_burns_per_second`
- Prefer `tau_seconds` over legacy `tau` when you want LP review clocks to scale with the micro-step interpretation.
- `block_time` must be greater than 1.
- When `*_per_second` is set, the corresponding legacy `*_per_block` knob is ignored.

Illustrative override snippet:

```yaml
fee_mode: volatility_cex
simulate:
  T: 2_000
  seed: 7
  block_time: 5
  cex_sigma_mode: heston
  cex_sigma: 1.5e-4
  cex_heston_kappa: 1.0
  cex_heston_theta: 1.0e-8
  cex_heston_sigma_v: 0.01
  cex_heston_rho: -0.5
  smart_trades_per_second: 0.4
  noise_trades_per_second: 0.4
  narrow_mints_per_second: 0.5
  passive_mints_per_second: 0.3
  passive_burns_per_second: 0.1
  tau_seconds: 25.0
```

## Output Layout

For a scenario file `configs/scenarios/foo.yml`, the scenario output root is:

```text
abm_results/scenarios/foo/
```

Each CLI run creates a new immutable record under:

```text
abm_results/scenarios/foo/runs/<fee_mode>_seed<seed>_<n>/
```

Important files:

- `latest_run.json`: pointer to the most recent CLI run for that scenario.
- `runs/<run_id>/config_snapshot.yml`: exact scenario snapshot used for the run.
- `runs/<run_id>/metadata.json`: provenance from `core.artifacts.build_run_manifest(...)` plus the effective simulate parameters.
- `runs/<run_id>/summary.csv`: one-row summary with run id, seed, horizon, and final cohort metrics.
- `runs/<run_id>/logs/*.txt`: verbose per-step logs when `light_mode=False`.
- `runs/<run_id>/png/` and `runs/<run_id>/html/`: Plotly exports for price, liquidity, fee, activity, and PnL diagnostics.
- `runs/<run_id>/output_data/*.npy`: saved DEX/CEX series for downstream analysis scripts.

The `simulate(...)` function also returns a Python dict of recorded series. The exact keys are defined near the tail of `scripts/run.py`; LP cohort PnL, rebalancing, fee, and LVR series are documented in [LP PnL](LP_PnL.md) and [Loss-Versus-Rebalancing](LVR_explanation.md).

## Main Entry Points

- `scripts/run.py`: single scenario run with plots and per-run artifacts.
- `scripts/run_multiple.py`: multi-seed scenario runner with cohort mean and dispersion plots.
- `scripts/profile_simulation.py`: cProfile-based profiling run that writes reports under the scenario root.
- `scripts/sigma_calibration.py`: calibrate per-second `cex_sigma` from 1-second ETH/USDC data. CSV and pickle work out of the box; Parquet requires an optional pandas parquet engine such as `pyarrow` or `fastparquet`.
- `scripts/visualize_distributions.py`: export figures for the stochastic primitives implied by a scenario YAML, writing unique run folders under `abm_results/scenarios/<scenario>/distributions/<run_id>/`.
- `scripts/run_experiment_design.py`: sampled or sequential experiment designs with immutable cache folders.
- `scripts/analyze_experiment_design.py`: feature screening, convergence plots, Sobol outputs, and top-point summaries from an experiment cache.
- `scripts/run_parameter_surface_nd_pnl_fee_dashboard.py`: cache-only ND grid sweeps for dashboarding.
- `scripts/build_experiment_design_dashboard.py`: standalone HTML for experiment-design caches.
- `scripts/build_parameter_surface_nd_pnl_fee_dashboard.py`: standalone HTML for ND grid caches.
- `scripts/stylized_facts_report.py`: stylized-facts diagnostics for an input price/returns series.
- `scripts/analysis/run_paper_figures.py`: batch production of the paper’s comparison figures.
- `python -m abm_webapp`: local live Dash webapp for interactive runs.

## Validation And Diagnostics

- Unit tests live under `tests/`.
- `pytest -q` is the main regression suite.
- Pool math, LP accounting, experiment-design helpers, plotting utilities, and webapp storage/lifecycle all have targeted tests.
- When changing simulator behavior, prefer adding a test or an invariant alongside the documentation update.

## Related Docs

- [Agent Behaviour Details](agents_spec.md)
- [Loss-Versus-Rebalancing](LVR_explanation.md)
- [LP PnL](LP_PnL.md)
- [Fee Schedules](fee_schedules.md)
- [Sigma Calibration](sigma_calibration.md)
- [Stress Tests](stress_tests.md)
- [Webapp](webapp.md)
- [nD Sampling Designs](nd_grid_sampling_methods.md)
