---
title: Simulation Outputs
nav_order: 8
---

# Simulation Outputs from `simulate(...)`

This document describes, series by series, what the main simulation routine
`run.simulate(...)` returns. It is intended to complement:

- `docs/README.md` (model and agent overview),
- `docs/LP_PnL.md` (PnL definitions),
- `docs/LVR_explanation.md` (LVR and rebalancing benchmark),
- `docs/fee_schedules.md` (fee controllers),
- `docs/hedged_pnl_vs_lvr_notes.md` (qualitative behaviour of hedged PnL).

Unless otherwise stated, all series below are one value **per simulation step**
(per *block*), indexed by `t = 0, 1, …, T-1`, where `T` is the `T` argument to
`simulate`. The current implementation requires mempool-style execution with
`block_time > 1`; micro‑steps are internal to a step and not directly exposed.

Most numeric series are returned as Python lists (`list[float]`); the main
per-step state paths are NumPy arrays (e.g. `DEX_price`, `CEX_price`,
`band_*`, and the `L_*` liquidity series). For analysis it is natural to
convert everything to NumPy/Pandas.

When `light_mode=True`, a **reduced** dictionary is returned; see the final
section.

---

## 1. Prices, Volatility, and No‑Arb Bands

- `DEX_price[t]`  
  Mid‑price on the AMM at the **end** of step `t`, in token1 per token0:
  $$
    P_t := S_t^2,
  $$
  where `S_t` is the pool’s active sqrt‑price.

- `CEX_price[t]`  
  Mid‑price on the reference market at the end of step `t`:
  $$
    m_t.
  $$

- `cex_sigma_series[t]`  
  Per‑step volatility of CEX log‑returns used in the diffusion step at `t`
  (see `ReferenceMarket.diffuse_only`). In Heston mode this is
  
  $$
  \sigma_t = \sqrt{v_t}
  $$
  .

- `cex_regime_series[t]`  
  Regime label at step `t`. In regime‑switching mode this is `"L"` or `"H"`;
  in noisy‑sine mode it is `"S"`; in Heston mode it is `"H"` (kept for
  backwards compatibility but not used for logic).

- `band_lo_pre[t]`, `band_hi_pre[t]`  
  No‑arb band at the **start** of step `t`, based on the validated snapshot
  (`agent_S_ref`, `cex_ref_for_agents`) and the fee in force at that point.
  If the current taker fee is 
  $$
  f_t
  $$
   and 
  $$
  r_t = 1 - f_t
  $$
  , then
  $$
    \text{band\_lo\_pre}[t] = m_t\,r_t,\qquad
    \text{band\_hi\_pre}[t] = \frac{m_t}{r_t}.
  $$

- `band_lo_post[t]`, `band_hi_post[t]`  
  No‑arb band at the **end** of step `t`, using the post‑impact CEX price
  `ref.m` and the same fee `f_t`:
  $$
    \text{band\_lo\_post}[t] = m_t^{\text{post}} r_t,\qquad
    \text{band\_hi\_post}[t] = \frac{m_t^{\text{post}}}{r_t}.
  $$

---

## 2. Liquidity and Reserves

- `L_active_end[t]`  
  Active liquidity in the band `[tick_t, tick_t + tick_spacing)` at the end of
  step `t`, equal to the prefix sum of `liquidity_net` up to `tick_t`.

- `L_pre_step[t]`  
  Active liquidity at the **start** of step `t` (after any seed/binomial
  initialization and previous step’s operations).

- `L_pre_trader[t]`  
  Active liquidity immediately **before** trader flow is applied in step `t`
  (after any LP burns/re‑centers in that step).

- `L_pre_arb_eff[t]`  
  Active liquidity immediately before the arbitrageur executes in step `t`
  (after smart + noise trader flow for that step).

- `x_active_reserves[t]`, `y_active_reserves[t]`  
  Token0 and token1 reserves *inside the active band* at the end of step `t`,
  computed as
  $$
    (x_t, y_t) = \text{reserves\_in\_active\_tick}(L_{\text{active},t}, S_t),
  $$
  using the standard Uniswap v3 formulas in sqrt‑price space.

- `liq_history[t]`  
  Snapshot of the sparse `liquidity_net` map at the end of step `t`:
  `{boundary_tick: delta_L_at_boundary}`.

- `tick_history[t]`  
  Active band’s lower tick index at the end of step `t`.

- `grid_base_s`, `grid_g`  
  Grid parameters for the AMM:
  - `base_s` is the base sqrt‑price,
  - `g > 1` is the geometric tick ratio in sqrt‑price.  
  Tick `i` has sqrt‑price interval 
  $$
  [s_i, s_{i+Δ})
  $$
   with
  
  $$
  s_i = \text{base\_s} \cdot g^i
  $$
  , `Δ = tick_spacing`.

### 2.3. LP mint/burn event logs

These are event-level logs (indexed by event `k`, not by step `t`).

- `mint_steps[k]`, `mint_sizes[k]`, `mint_widths[k]`  
  For the `k`-th mint event: the step index, the minted liquidity `L`, and the tick width (`upper - lower`) of the minted position.

- `burn_steps[k]`, `burn_sizes[k]`  
  For the `k`-th burn event: the step index and the burned liquidity `L`.

---

## 3. Trader and Arbitrage Flow

All PnLs here are in token1 units; sign conventions are from the *agent*
perspective (positive = profit).

### 3.1. Per‑step notionals and directions

- `trader_notional_y[t]`  
  Net signed notional of **smart + noise** traders at step `t`, expressed in
  token1 units using the pre‑trade reference price:
  - For `X_to_Y` trades we aggregate 
    $$
    \Delta_y = -P_{\text{ref}} \cdot \Delta x
    $$
    
    (price‑down direction, negative sign),
  - For `Y_to_X` trades we aggregate 
    $$
    \Delta_y = +\Delta y
    $$
    
    (price‑up direction, positive sign),
  and sum across all trader legs (both DEX and CEX) in that step.

- `arb_notional_y[t]`  
  Net signed notional of the arbitrageur at step `t`, measured in token1 with
  the same sign convention: negative when the arb sells token0 against the DEX
  (`X_to_Y`, price down), positive when it buys token0 (`Y_to_X`, price up).

- `trader_steps[k]`, `trader_dirs[k]`  
  Event‑level logs of trader actions:
  - `trader_steps[k]` is the step index of the `k`‑th trader execution,
  - `trader_dirs[k]` is `"down"` for `X_to_Y` and `"up"` for `Y_to_X`.

- `arb_steps[k]`, `arb_dirs[k]`  
  Event‑level logs for arbitrage trades, with the same `"down"`/`"up"`
  convention.

### 3.2. Trader and arbitrage PnL

- `smart_router_pnl_steps[t]`, `noise_trader_pnl_steps[t]`  
  Per‑step PnL increments for smart router and noise trader, settled at the
  end‑of‑step CEX price 
  $$
  m_t^{\text{post}}
  $$
  . Internally these are computed
  from token flows as
  $$
    \text{PnL}_t = (\Delta y_{\text{out}} - \Delta y_{\text{in}})
                  + (\Delta x_{\text{out}} - \Delta x_{\text{in}}) \, m_t^{\text{post}}.
  $$

- `smart_router_pnl_cum[t]`, `noise_trader_pnl_cum[t]`  
  Cumulative sums of the above up to step `t`.

- `smart_router_notional_y[t]`, `noise_trader_notional_y[t]`  
  Net signed token1‑valued notional per step for smart and noise traders
  separately, analogous to `trader_notional_y[t]`.

- `smart_router_exec_count[t]`, `noise_trader_exec_count[t]`  
  Number of executed DEX or CEX legs for smart and noise traders at step `t`.
  (Skipped trades and pure intents that fail slippage/best‑ex checks are not
  counted.)

- `arb_pnl_steps[t]`, `arb_pnl_cum[t]`  
  Arbitrageur PnL per step and cumulative, using the same flow‑based formula
  above and settled at `settlement_m = ref.m` after impact and diffusion. Note
  that while each arb trade is ex‑ante filtered to be profitable at the
  **snapshot** CEX price, realized `arb_pnl_steps[t]` can be slightly negative
  in a given step due to adverse CEX moves between the snapshot and settlement.

- `trader_pnl_steps[t]`, `trader_pnl_cum[t]`  
  Aggregated trader PnL (smart + noise) per step and cumulative:
  $$
    \text{trader\_pnl\_steps}[t]
      = \text{smart\_router\_pnl\_steps}[t]
      + \text{noise\_trader\_pnl\_steps}[t].
  $$

- `trader_exec_count[t]`, `arb_exec_count[t]`  
  Total number of trader executions and arbitrage trades in step `t`.

---

## 4. LP Wealth, PnL, Fees, and LVR

For each LP 
$$
i
$$
 at step 
$$
t
$$
:

- Let 
  $$
  V^{\text{LP},i}_t
  $$
   be its wealth in token1 (wallet + mark‑to‑market of
  open positions), as defined in `LP_PnL.md`.
- Let 
  $$
  V^{\text{reb},i}_t
  $$
   be the value of its rebalancing benchmark
  (self‑financing delta‑hedging portfolio) in token1.
- Let 
  $$
  F^i_t
  $$
   be cumulative fees in token1.
- Let 
  $$
  \text{LVR}^i_t
  $$
   be Loss‑Versus‑Rebalancing.

The simulator enforces, per LP,
$$
  V^{\text{LP},i}_t = V^{\text{reb},i}_t + F^i_t - \text{LVR}^i_t
$$
so that **hedged PnL** is
$$
  \text{PnL}^{\text{hedged},i}_t
    := V^{\text{LP},i}_t - V^{\text{reb},i}_t
    = F^i_t - \text{LVR}^i_t,
$$
and **unhedged PnL** is
$$
  \text{PnL}^{\text{unhedged},i}_t
    := V^{\text{LP},i}_t - V^{\text{LP},i}_0.
$$

Series below aggregate these quantities across LP cohorts.

### 4.1. Hedged and unhedged PnL

- `lp_pnl_total[t]`  
  Total hedged LP PnL across all strategic (non‑seed) LPs:
  $$
    \text{lp\_pnl\_total}[t]
      = \sum_i \left(F^i_t - \text{LVR}^i_t\right).
  $$

- `lp_pnl_active[t]`, `lp_pnl_passive[t]`  
  Hedged PnL aggregated over active narrow (`is_active_narrow=True`) and
  passive (`is_passive=True`) LP cohorts respectively.

- `lp_unhedged_total[t]`, `lp_unhedged_active[t]`,
  `lp_unhedged_passive[t]`  
  Unhedged PnL (wealth change) aggregated over the same cohorts:
  $$
    \text{lp\_unhedged\_total}[t]
      = \sum_i \left(V^{\text{LP},i}_t - V^{\text{LP},i}_0\right),
  $$
  and analogously for active/passive splits.

### 4.2. Rebalancing benchmark and LVR

- `lp_rebal_total_series[t]`, `lp_rebal_active_series[t]`,
  `lp_rebal_passive_series[t]`  
  Benchmark PnL 
  $$
  \sum_i (V^{\text{reb},i}_t - V^{\text{reb},i}_0)
  $$
   for
  total/active/passive cohorts.

- `lp_rebal_value_total_series[t]`,
  `lp_rebal_value_active_series[t]`,
  `lp_rebal_value_passive_series[t]`  
  Benchmark value 
  $$
  \sum_i V^{\text{reb},i}_t
  $$
   for each cohort.

- `lp_fee_value_total_series[t]`,
  `lp_fee_value_active_series[t]`,
  `lp_fee_value_passive_series[t]`  
  Cumulative fees 
  $$
  F_t
  $$
   aggregated per cohort.

- `lp_lvr_total_series[t]`, `lp_lvr_active_series[t]`,
  `lp_lvr_passive_series[t]`  
  Aggregated LVR per cohort, computed by the identity
  $$
    \text{lp\_lvr\_total\_series}[t]
      = \text{lp\_fee\_value\_total\_series}[t]
        - \text{lp\_pnl\_total}[t],
  $$
  and analogously for active/passive.

### 4.3. Wallet and wealth

- `lp_wallet_series[t]`, `lp_wallet_active_series[t]`,
  `lp_wallet_passive_series[t]`  
  Aggregated LP wallet balances in token1 (realized value after mints/burns)
  per cohort.

- `lp_wealth_series[t]`, `lp_wealth_active_series[t]`,
  `lp_wealth_passive_series[t]`  
  Aggregated wealth in token1 (wallet + mark‑to‑market of open positions) per
  cohort:
  $$
    \text{lp\_wealth\_total}[t]
      = \sum_i V^{\text{LP},i}_t.
  $$

### 4.4. Jiter (JIT LP) accounting

When Jiter is enabled (`p_jit > 0`, `N_jit > 0`, `liquidity_perc_jit > 0`), these track the single Jiter agent separately from the strategic LP cohorts. When disabled, these series are identically zero.

- `jiter_wallet_series[t]`, `jiter_wealth_series[t]`  
  Jiter wallet and total wealth in token1 (wealth = wallet + mark-to-market of open positions).

- `jiter_position_value_series[t]`  
  Mark-to-market value (token1) of Jiter’s open positions (principal + uncollected fees).

- `jiter_fee_value_series[t]`  
  Mark-to-market value (token1) of Jiter’s cumulative fees earned (realized + uncollected).

- `jiter_pnl_series[t]`  
  Jiter “hedged” series reported with the opposite sign convention to LP hedged PnL:
  $$
    \text{jiter\_pnl\_series}[t] = V^{\text{reb}}_t - V^{\text{LP}}_t = \text{LVR}_t - F_t.
  $$
  (This matches the default plot label “Jiter hedged (LVR - fees)”.)

Seed LPs (`is_seed=True`) created by `bootstrap_initial_binomial_hill_sharded`
are **excluded** from all strategic LP cohorts.

---

## 5. Fees and Signals

The fee controller is described in detail in `docs/fee_schedules.md`. The
series below expose its internal signals.

- `fee_series[t]`  
  Taker fee 
  $$
  f_t
  $$
   on the AMM at step `t` (fraction, e.g. 0.003 = 30 bps).  
  In mempool execution with `fee_mode: volatility_oracle`, the fee can vary
  within a block at micro-step resolution; `fee_series[t]` records the
  **end-of-step** value of `pool.f` after all intra-block updates.

- `fee_mode`  
  Active fee controller mode: `"static"`, `"volatility"`, `"volatility_oracle"`, or `"toxicity"`.

- `f_min`, `f_max`  
  Hard lower/upper bounds for `fee_series` (same values as the `f_min`,
  `f_max` parameters to `simulate`).

- `fee_sigma_series[t]`  
  Volatility signal used by volatility-based fee modes:
  - in `"volatility"` mode, this is the EWMA of absolute CEX log-returns
    $$
      \hat{\sigma}_t = \text{EWMA}\bigl(|\log m_t - \log m_{t-1}|\bigr),
    $$
    with half-life `fee_half_life`;
  - in `"volatility_oracle"` mode, this is the per-step CEX volatility path
    
    $$
    \sigma_t
    $$
     taken directly from `ReferenceMarket.sigma` (no smoothing).

- `fee_basis_ticks_series[t]`  
  EWMA of fee‑adjusted log basis (in ticks) used in toxicity mode:
  $$
    B_{\text{obs},t}
      = \max\bigl(0,\; |\log P_t - \log m_t| - \log(1/(1-f_t))\bigr),
  $$
  $$
    B_{\hat{t}} = \text{EWMA}(B_{\text{obs},t}),\qquad
    \text{basis\_ticks}_t = \frac{B_{\hat{t}}}{\log(1.0001)}.
  $$

- `fee_imb_series[t]`  
  Imbalance proxy in token1 units within the active band at step `t`, defined
  from active reserves
  
  $$
  (x_t, y_t) = \text{reserves\_in\_active\_tick}(t)
  $$
   as
  $$
    \text{fee\_imb\_series}[t]
      = \frac{y_t - x_t P_t}{\max(10^{-12},\, y_t + x_t P_t)}.
  $$
  (This is tracked for diagnostics; the current controllers do not use it.)

- `fee_signal_series[t]`  
  The controller’s primary signal:
  - volatility signal (`sigma_hat` or 
    $$
    \sigma_t
    $$
    ) in `"volatility"` / `"volatility_oracle"` modes,
  - `basis_ticks` in `"toxicity"` mode,
  - `0` in `"static"` mode.

---

## 6. Agent Activity Indicators

These series summarize directional activity counts per step and their
cumulative sums. They are mainly used for diagnostic plots.

- `smart_router_activity_cum[t]`  
  Cumulative signed count of smart‑router trades up to step `t`:
  +1 for `X_to_Y` (price‑down), −1 for `Y_to_X` (price‑up).

- `noise_trader_activity_cum[t]`  
  Same as above for noise traders.

- `lp_active_activity_cum[t]`, `lp_passive_activity_cum[t]`  
  Cumulative LP activity counts:
  +1 for a mint, −1 for a burn, split by active/passive LP cohorts.

- `arb_activity_cum[t]`  
  Cumulative count of successful arbitrage trades (each successful arb adds +1;
  skipped opportunities do not change the series).

- `jiter_activity_cum[t]`  
  Cumulative count of successful JIT targets (one increment per targeted swap that executed with an active Jiter position).

---

## 7. Light‑Mode Output

When `light_mode=True`, `simulate` returns a reduced dictionary:

- `smart_router_pnl_cum`, `noise_trader_pnl_cum`, `arb_pnl_cum`  
  Cumulative PnL series for smart router, noise trader, and arbitrageur
  (token1 units).

- `lp_pnl_active`, `lp_pnl_passive`  
  Hedged LP PnL series for active and passive cohorts (fees − LVR).

- `lp_unhedged_active`, `lp_unhedged_passive`  
  Unhedged LP PnL series for active and passive cohorts.

- `fee_series`  
  Fee path 
  $$
  f_t
  $$
  .

All other detailed telemetry (price paths, liquidity, fees, activity counts,
etc.) is omitted in light mode to reduce memory and serialization overhead. For
any analysis that needs the full telemetry, run with `light_mode=False`.
