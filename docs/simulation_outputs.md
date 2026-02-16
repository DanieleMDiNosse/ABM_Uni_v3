---
title: Simulation Outputs
nav_order: 8
---

# Simulation Outputs (`simulate(...)`)

This page documents the keys returned by `scripts.run.simulate(...)`.

## Two output modes

The simulator has two output modes controlled by `light_mode`:

- `light_mode=True`: returns a small dict (fast, used by grid-search/dashboard scripts).
- `light_mode=False`: returns the full telemetry dict (used by the default plotting suite).

Notes:

- Unless stated otherwise, “series” means a per-block time series of length `T` (the number of blocks).
- Types are intentionally mixed: most keys are JSON-friendly Python lists, but some are returned as NumPy arrays for convenience. If you need to serialize to JSON, call `.tolist()` on NumPy arrays.
- Intra-block micro-step traces are written to the verbose log and used for plotting, but are **not** returned in the output dict.

---

## `light_mode=True` return dict (22 keys)

PnL series (cumulative):

- `smart_router_pnl_cum`: cumulative PnL of the smart router (token1 units).
- `noise_trader_pnl_cum`: cumulative PnL of the noise trader (token1 units).
- `arb_pnl_cum`: cumulative PnL of the arbitrageur (token1 units).
- `lp_pnl_active`: cumulative **hedged** PnL of the active LP cohort (token1 units).
- `lp_pnl_passive`: cumulative **hedged** PnL of the passive LP cohort (token1 units).
- `lp_unhedged_active`: cumulative **unhedged** PnL of the active LP cohort (token1 units).
- `lp_unhedged_passive`: cumulative **unhedged** PnL of the passive LP cohort (token1 units).

Fee controller:

- `fee_series`: per-block fee series (fraction, e.g. `0.003` for 30 bps).

Smart-router routing metrics:

- `smart_router_dex_share_steps`: block indices where the router routed some flow to the DEX.
- `smart_router_dex_share_series`: per-block DEX share values recorded at `smart_router_dex_share_steps`.
- `smart_router_dex_share_overall`: overall DEX share across the run (DEX notional / total notional).
- `smart_router_dex_share_mean`: mean of `smart_router_dex_share_series`.

Execution totals:

- `total_noise_trader_swaps`, `noise_trader_swaps_rejected_slippage`
- `total_smart_router_swaps`, `smart_router_swaps_rejected_slippage`
- `smart_router_swaps_cex_routed`, `smart_router_swaps_dex_routed`
- `total_arb_swaps`, `arb_no_op_in_band`, `arb_swaps_rejected_profitability`
- `total_jit_trades_executed`

---

## `light_mode=False` return dict (full telemetry)

### Prices, bands, and reference volatility

- `DEX_price`: DEX price series `P_t` (NumPy array, token1 per token0).
- `CEX_price`: CEX mid series `M_t` (NumPy array, token1 per token0).
- `band_lo`, `band_hi`: no-arbitrage band targets (NumPy arrays, token1 per token0).
- `cex_sigma_series`: per-block reference volatility series used by the CEX diffusion (Python list).
- `cex_regime_series`: per-block regime labels (Python list; in Heston mode it is a constant label).

### Liquidity state and active-tick reserves

- `L_active_end`: active liquidity at end of each block (NumPy array).
- `L_pre_step`, `L_pre_trader`, `L_pre_arb_eff`: liquidity snapshots around execution (NumPy arrays).
- `x_active_reserves`, `y_active_reserves`: reserves in the active tick at end of each block (NumPy arrays).
- `liq_history`, `tick_history`: recorded liquidity and tick evolution used by the liquidity plots/GIFs.
- `grid_base_s`, `grid_g`: parameters of the tick grid used by the pool (scalars).

### Event times (sparse per-run logs)

These are “event lists” (not length `T`):

- `trader_steps`, `trader_dirs`: executed trader swap steps and directions.
- `arb_steps`, `arb_dirs`: executed arbitrage steps and directions.
- `mint_steps`, `mint_sizes`, `mint_widths`: LP mint step indices, minted liquidity sizes, and widths.
- `burn_steps`, `burn_sizes`: LP burn step indices and burned liquidity sizes.

### Notional flow (token1 units)

- `dex_notional_y`: per-block executed DEX notional (Python list).
- `trader_notional_y`: per-block total trader notional (smart + noise legs).
- `arb_notional_y`: per-block arbitrage notional.
- `smart_router_notional_y`, `noise_trader_notional_y`: per-block smart/noise notional series.

### PnL series (token1 units)

All PnL series are per-block unless specified otherwise:

- `trader_pnl_steps`, `trader_pnl_cum`: trader PnL (smart + noise) steps and cumulative.
- `arb_pnl_steps`, `arb_pnl_cum`: arbitrageur PnL steps and cumulative.
- `smart_router_pnl_steps`, `smart_router_pnl_cum`: smart-router PnL.
- `noise_trader_pnl_steps`, `noise_trader_pnl_cum`: noise-trader PnL.

LP cohort PnL and benchmarks:

- `lp_pnl_total`, `lp_pnl_active`, `lp_pnl_passive`: **hedged** LP PnL (total/active/passive).
- `lp_unhedged_total`, `lp_unhedged_active`, `lp_unhedged_passive`: **unhedged** LP PnL.
- `lp_rebal_total_series`, `lp_rebal_active_series`, `lp_rebal_passive_series`: rebalancer benchmark PnL series.
- `lp_rebal_value_total_series`, `lp_rebal_value_active_series`, `lp_rebal_value_passive_series`: benchmark portfolio values.

Fee/LVR decomposition (identity-based bookkeeping):

- `lp_fee_value_total_series`, `lp_fee_value_active_series`, `lp_fee_value_passive_series`
- `lp_fees0_earned_total_series`, `lp_fees1_earned_total_series` (and active/passive variants)
- `lp_lvr_total_series`, `lp_lvr_active_series`, `lp_lvr_passive_series`

LP inventory diagnostics:

- `lp_wallet_series`, `lp_wallet_active_series`, `lp_wallet_passive_series`
- `lp_wealth_series`, `lp_wealth_active_series`, `lp_wealth_passive_series`

### JIT LP (“Jiter”) telemetry (token1 units)

- `jiter_wallet_series`, `jiter_wealth_series`
- `jiter_fee_value_series`, `jiter_fees0_earned_series`, `jiter_fees1_earned_series`
- `jiter_position_value_series`, `jiter_pnl_series`
- `jiter_flash_fee_paid_series`
- `jiter_activity_cum`: cumulative JIT mint/burn activity (per block).

### Fee controller telemetry

- `fee_series`: per-block fee series.
- `fee_mode`: the fee mode string.
- `f_min`, `f_max`: fee bounds.
- `fee_sigma_series`, `fee_basis_ticks_series`, `fee_imb_series`, `fee_signal_series`: controller signals recorded each block.

### Execution counts and routing diagnostics

Counts are returned as Python lists (per block) unless they are explicitly scalars:

- `trader_exec_count`, `arb_exec_count`
- `smart_router_exec_count`, `noise_trader_exec_count`
- `smart_router_cex_exec_count`, `smart_router_dex_exec_count`
- `smart_router_dex_share_steps`, `smart_router_dex_share_series`, `smart_router_dex_share_overall`, `smart_router_dex_share_mean`

Scalar totals:

- `total_noise_trader_swaps`, `noise_trader_swaps_rejected_slippage`
- `total_smart_router_swaps`, `smart_router_swaps_rejected_slippage`
- `smart_router_swaps_cex_routed`, `smart_router_swaps_dex_routed`
- `total_arb_swaps`, `arb_no_op_in_band`, `arb_swaps_rejected_profitability`
- `total_jit_trades_executed`

Activity series (cumulative per block):

- `smart_router_activity_cum`, `noise_trader_activity_cum`
- `lp_active_activity_cum`, `lp_passive_activity_cum`
- `arb_activity_cum`, `jiter_activity_cum`
