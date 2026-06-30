---
title: Stress Tests
nav_order: 8
---

# Stress Tests (ABM Uni v3)

This note proposes **stress tests** that probe failure modes of the simulator’s dynamics (liquidity crises, volatility spikes, flow shocks), using **scenario YAML changes only**.

The intent is scientific: each test is phrased as a *checkable* hypothesis with concrete knobs and diagnostics.

---

## Assumptions & definitions (as implemented)

- Tokens:
  - `X` = token0 (“ETH-like”)
  - `Y` = token1 (“USDC-like”, also the **numéraire** in reported wealth/PnL)
- Prices:
  - CEX mid: `m_t = ref.m` in `Y per X`
  - DEX state: `S_t = pool.S` (sqrt-price), so `P_t^{DEX} = S_t^2`
- Time:
  - One **block** is one simulation step `t`.
  - Inside each block there are `block_time = B` **micro-steps** where the CEX diffuses.
  - In this repo, **one micro-step corresponds to ~1 second** (so `cex_sigma` is “per second”).

Key dynamic semantics (important for interpreting stress tests):
- During the `B` micro-steps, the **DEX is frozen**; traders only **enqueue intents**.
- At the block boundary, the simulator **replays the mempool**: arbitrage first, then other orders in random order (optional JIT wrapper).
- Many decisions use a **validated snapshot** from the *previous* block (`agent_S_ref`, `cex_ref_for_agents`), so high volatility and/or large `block_time` mechanically produce more dislocation and slippage pressure.

---

## Diagnostics to track (already returned by `simulate()`)

These are especially informative under stress:
- **Price efficiency / dislocation**
  - `DEX_price`, `CEX_price`
  - log-basis (in ticks): `fee_basis_ticks_series` (and the raw `fee_signal_series` if using the toxicity controller)
  - arb “no-arb band” targets: `band_lo`, `band_hi`
- **Liquidity continuity**
  - `L_active_end`, `dex_notional_y`
  - mint/burn events: `mint_steps`, `mint_sizes`, `burn_steps`, `burn_sizes`
- **Market quality / routing**
  - slippage rejects: `noise_trader_swaps_rejected_slippage`, `smart_router_swaps_rejected_slippage`
  - smart router routing: `smart_router_swaps_cex_routed`, `smart_router_swaps_dex_routed`, and `smart_router_dex_share_series`
- **Welfare decomposition**
  - LP hedged PnL (`lp_pnl_active`, `lp_pnl_passive`) vs fees and LVR (`lp_fee_value_*`, `lp_lvr_*`)

---

## Stress tests (YAML-only)

### 1) Volatility jump via Heston schedule (CEX)

**Goal.** Force an abrupt transition from a calm period to a high-volatility period and measure how quickly:
(i) arbitrage restores price efficiency, (ii) DEX volume migrates to CEX, and (iii) LP wealth decomposes into fees vs LVR.

**Why this is a meaningful stress in *this* ABM.**
Because intents are formed off a validated snapshot and executed at the block boundary, a volatility spike creates a “latency wedge”:
the CEX can move far during micro-steps while the DEX is frozen, so the eventual mempool replay has to absorb a larger correction.

**Knobs (scenario YAML).**
- `cex_sigma_mode: heston`
- `cex_heston_theta_schedule` (piecewise variance levels)
- `cex_heston_kappa` (controls how quickly `v_t` tracks schedule changes)
- `cex_heston_sigma_v` (adds stochasticity around the scheduled baseline)

Important implementation detail: the current code shuffles `cex_heston_theta_schedule` once per run and then assigns the shuffled values to equal-duration segments. So the YAML list is a set of regime levels, not a guaranteed low-then-high chronology.

**Suggested “spike” design.**
- Include one or more calm variance levels (`theta_low = sigma_low^2`) and one or more stressed levels (`theta_high` larger by 10–30× in variance).
- Use smaller `kappa` for gradual transitions, larger `kappa` for sharper jumps once the scheduled regime changes.
- If you need a guaranteed chronological low-then-high path, the current implementation does not provide it through YAML alone because of the per-run shuffle.

**Expected signatures (to verify).**
- Dislocation widens rapidly after the first jump to high `theta`:
  - `|log(DEX/CEX)|` increases; `fee_basis_ticks_series` jumps.
- Trade quality deteriorates:
  - higher slippage rejects; higher share of smart-router flow routed to CEX.
- Arbitrage intensity increases *if* arb remains profitable:
  - `total_arb_swaps` rises and dislocation mean-reverts faster.
- LP outcomes:
  - `lp_lvr_*` steepens sharply during high-volatility windows, and whether `fee_series` compensates depends on `fee_mode`.

**Controller robustness variant.**
Repeat the same volatility-jump design under different fee controllers:
`fee_mode ∈ {static, volatility_cex, volatility_dex, toxicity}` and compare:
mean dislocation, LP hedged PnL (`fees − LVR`), and slippage rejects.

Minimal YAML sketch (overrides):
```yaml
fee_mode: volatility_cex
simulate:
  cex_sigma_mode: heston
  cex_sigma: 1.0e-4
  cex_heston_kappa: 1.5
  cex_heston_theta: 1.0e-8
  cex_heston_sigma_v: 0.01
  cex_heston_rho: -0.5
  cex_heston_theta_schedule:
    - 1.0e-8
    - 4.0e-6
```

---

### 2) Vol-of-vol clustering (Heston CEX)

**Goal.** Generate clustered volatility (endogenous “spikes”) and test whether the ABM produces:
fat-tailed dislocation, clustered slippage failures, and drawdown clustering in LP wealth.

**Why this differs from (1).**
Section (1) uses a scheduled variance baseline to induce a macro jump. This section uses Heston’s endogenous variance noise to create clustering, which tends to create:
- clusters of high `cex_sigma_series`,
- more variable “length” and “severity” of stress episodes,
- and (with `cex_heston_rho < 0`) leverage-style co-movement between returns and volatility.

**Knobs (scenario YAML).**
- `cex_sigma_mode: heston`
- `cex_heston_sigma_v` (volatility of variance; main “spikiness” knob)
- `cex_heston_kappa` (mean reversion speed; controls episode duration)
- `cex_heston_theta` (long-run variance floor)
- `cex_heston_rho` (return–vol correlation; negative produces leverage-like behavior)
- optional `cex_heston_v0` (initial variance), otherwise `cex_sigma^2`

**Suggested Heston stress sweep (conceptual).**
Start from a baseline calibrated to “normal” days, then vary only `sigma_v`:
- mild clustering: `cex_heston_sigma_v` small
- violent spikes: increase `cex_heston_sigma_v` by 5–20×

Also vary `kappa`:
- small `kappa`: long-lived high-vol episodes (harder for LPs)
- large `kappa`: sharp but brief spikes (harder for execution/slippage)

**Expected signatures (to verify).**
- `cex_sigma_series` becomes heavy-tailed / clustered.
- `fee_sigma_series` (if using volatility fee) reacts with a lag set by `fee_half_life`.
- Dislocation (`fee_basis_ticks_series`) becomes clustered even if average volatility is unchanged.
- Slippage rejects cluster (bursty failures rather than steady-state rejects).

Minimal YAML sketch (overrides):
```yaml
fee_mode: volatility_cex
simulate:
  cex_sigma_mode: heston
  cex_sigma: 1.5e-4        # sets sqrt(v0) if v0 is omitted
  cex_heston_theta: 1.0e-8
  cex_heston_kappa: 1.0
  cex_heston_sigma_v: 0.01  # stress: raise for spikier variance
  cex_heston_rho: -0.5
```

---

### 3) Latency stress (larger `block_time`)

Increase `block_time` (micro-steps per block). This amplifies “snapshot staleness” and should mechanically widen basis and increase arb/trader slippage pressure.

---

### 4) Arb outage / expensive arbitrage

Raise `flash_loan_fee` until arbitrage frequently fails the profitability filter (`arb_swaps_rejected_profitability` increases). This tests how the DEX behaves with weak price-correction.

---

### 5) Liquidity run (withdrawal shock / “liquidity crisis”)

**Goal.** Produce an endogenous collapse in active liquidity and DEX execution capacity, then characterize the system-level response:
- Does volume migrate to CEX?
- Does the fee controller react (if dynamic)?
- Does the pool get “stuck” in low-liquidity deserts, or does liquidity re-enter?

**Mechanism in this ABM (important).**
- Strategic LPs hold **cash in token1**; minting buys token0 on CEX (immediate impact) and deposits (X,Y) into the pool.
- Burning withdraws (X,Y)+fees, then **immediately sells token0 on CEX** (immediate impact).
So a liquidity run is also a **CEX flow shock** (sell pressure) in this model.

**Knobs (scenario YAML).**
Two main channels:
1) **Forced / frequent burns**
   - preferred: `passive_burns_per_second`
   - legacy: `passive_burns_per_block`
   - For active narrow LPs: lower `theta_SL` (burn earlier on losses), optionally lower `theta_TP` to realize gains sooner (more churn).
2) **Starving new liquidity**
   - preferred: reduce `passive_mints_per_second` and `narrow_mints_per_second`
   - legacy: reduce `passive_mints_per_block` and `narrow_mints_per_block`
   - prefer increasing `tau_seconds` (or legacy `tau`) so LPs review less often and recovery is slower

**Design variants.**
- *Slow run (grinding crisis):* moderate burns + low mints + high `tau`.
- *Sudden run (bank-run style):* very high preferred `passive_burns_per_second` (or legacy `passive_burns_per_block`) and low/zero mints for a sustained window.
  (Without code changes, this is implemented as “run the whole scenario” at run-like settings; later we can add time-scheduled shocks if useful.)

**Expected signatures (to verify).**
- Liquidity collapse:
  - `L_active_end` trends down; mint/burn imbalance visible in `mint_*` vs `burn_*`.
  - `dex_notional_y` drops and/or becomes intermittent (execution failures when `L_active` is tiny).
- Market quality deterioration:
  - slippage rejects increase; smart router routes more to CEX.
- Feedback loops:
  - burn-induced CEX sell pressure can move `CEX_price` and interact with arb profitability and LP PnL (watch `cex_sigma_series` and dislocation).

Minimal YAML sketch (overrides):
```yaml
fee_mode: static
simulate:
  passive_lp_share: 1.0
  passive_burns_per_second: 1.0
  passive_mints_per_second: 0.02
  tau_seconds: 25.0
```

Active-narrow “panic” variant (adds stop-loss sensitivity):
```yaml
simulate:
  passive_lp_share: 0.5
  theta_SL: 0.05
  narrow_mints_per_second: 0.02
```

---

### 6) Concentration fragility

Make liquidity very narrow (`passive_width_pct` small and/or `w_min_ticks` small) and combine with high volatility. Expect frequent out-of-range episodes and faster liquidity depletion under losses.

---

### 7) Flow shock (volume + order-size shock)

**Goal.** Stress execution capacity and LP risk by increasing both:
- trade arrival intensity (more swaps per block), and
- order-size dispersion (fatter tail in log-normal sizes).

This is the “liquidity-demand shock” complement to the “liquidity-supply shock” in (5).

**Mechanism in this ABM (important).**
- Smart-router orders can be **routed to CEX immediately** when DEX is uncompetitive (and those CEX trades apply immediate permanent impact).
- All DEX intents face an execution-time slippage gate computed from the **validated DEX snapshot**, so large within-block price moves create systematic rejects.
So a flow shock can simultaneously:
(i) push more flow to CEX (impacting `ref.m`), and
(ii) stress the DEX at the block boundary (tick crossing, fee allocation, LP exposure changes).

**Knobs (scenario YAML).**
- Arrivals:
  - preferred: `smart_trades_per_second`, `noise_trades_per_second`
  - legacy: `smart_trades_per_block`, `noise_trades_per_block`
- Sizes:
  - `trader_mean` (median notional scale), `trader_sigma` (tail thickness)
- Routing / execution filters:
  - `theta_T` (smart-router “DEX competitiveness” threshold)
  - `slippage_tolerance` (execution-time rejection threshold)

**Suggested stress sweep.**
Start with a baseline scenario and apply multiplicative shocks:
- intensity shock: multiply both preferred `*_trades_per_second` knobs by 5×, 10×, 25×
- size shock: increase `trader_mean` and/or `trader_sigma` (fatter tail tends to create rare “whales”)

Then cross with:
- strict execution: `slippage_tolerance` small (more rejects)
- loose execution: `slippage_tolerance` larger (more execution, more price impact)

**Expected signatures (to verify).**
- `dex_notional_y` increases up to the point where slippage rejects dominate; after that it can *decrease* (the DEX can’t clear trades).
- `smart_router_swaps_cex_routed` rises when DEX becomes uncompetitive.
- Fee revenue rises mechanically with notional, but LP hedged PnL can fall if LVR rises faster (especially under high volatility + long `block_time`).

Minimal YAML sketch (overrides):
```yaml
fee_mode: static
simulate:
  smart_trades_per_second: 2.0
  noise_trades_per_second: 2.0
  trader_mean: 3.2
  trader_sigma: 2.0
  slippage_tolerance: 0.005
  theta_T: 0.98
```

---

### 8) MEV/JIT stress

Enable Jiter (`p_jit > 0`, `liquidity_perc_jit > 0`, `jit_flash_loan_fee` small) and compare outcomes vs. JIT-off, focusing on fee capture, LP hedged PnL, and execution quality.

---

## Reproduction recipe (recommended workflow)

1) Create a new scenario YAML under `abm_results/scenarios/` (do not overwrite existing runs).
2) Run one seed:
```bash
python -m scripts.run --config abm_results/scenarios/<your_scenario>.yml
```
Outputs are written under:
`abm_results/scenarios/<scenario_name>/runs/<fee_mode>_seed<seed>_<n>/`, and the newest CLI run is recorded in `abm_results/scenarios/<scenario_name>/latest_run.json`.

3) Run multiple seeds to stabilize conclusions:
```bash
python -m scripts.run_multiple --config abm_results/scenarios/<your_scenario>.yml --runs 20 --seed-base 1 --max-workers 8
```
This writes mean ± std bands under:
`abm_results/scenarios/<scenario_name>/multi_runs/...`.
