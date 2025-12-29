---
title: Results
nav_order: 9
---

# ABM Uni v3 — Noisy-Sine Fee Experiments (Preliminary Results)

This note summarizes the outcomes of three single-seed scenarios in the **noisy-sine volatility regime** (center `cex_sigma = 0.00015`, amp `0.0001`, period `10,000` steps, `block_time = 5`, `T = 10,000`, seed `7`) using:

- `abm_results/scenarios/sigma_sine_fee_static.yml` (`fee_mode: static`)
- `abm_results/scenarios/sigma_sine_fee_volatility.yml` (`fee_mode: volatility`)
- `abm_results/scenarios/sigma_sine_fee_toxicity.yml` (`fee_mode: toxicity`)

All runs share the same agent configuration: 10 **strategic** LPs with 70% passive (`passive_lp_share = 0.7`), plus a binomial-hill background of seed LPs (`is_seed=True`) that provide initial liquidity but are excluded from the passive/active LP cohorts. There is one smart router and one noise trader on average per block, and an arbitrageur with `flash_loan_fee = 5 bps`. The metrics below are in token1 units (e.g. USDC) and are taken at the final step (`t = 10,000`). “Hedged PnL” is **fees − LVR** for passive LPs.

---

## Passive LP Outcomes by Fee Mode

| Scenario / fee mode                         | Mean fee (bps) | Fee range (bps) | Fees (passive) | LVR (passive) | Hedged PnL (passive) | Unhedged PnL (passive) |
|--------------------------------------------|----------------|-----------------|----------------|---------------|-----------------------|-------------------------|
| `sigma_sine_fee_static` (`static`)         | ~5.0           | 5 → 5           | ~19.3k         | ~43.3k        | **−24.0k**            | −139.5k                 |
| `sigma_sine_fee_volatility` (`volatility`) | ~36.7          | 5 → 120         | ~48.7k         | ~55.5k        | **−6.8k**             | −1.0k                   |
| `sigma_sine_fee_toxicity` (`toxicity`)     | ~176.4         | 5 → 500         | ~47.2k         | ~44.9k        | **+2.3k**             | +94.6k                  |

Numbers are rounded; each row corresponds to a single long run (no averaging over seeds yet). Active LP aggregates are near zero in these configs, so the table focuses on the **passive LP cohort**, which is the main object of interest for “baseline” liquidity provision.

### Where do these numbers come from?

For each YAML scenario above, the simulator was run once with `T = 10_000`, `block_time = 5`, `seed = 7` and `visualize = False`, and the table entries were computed directly from the series returned by `simulate(...)` (all LP series are computed over the non-seed, strategic LP cohort only):

- **Fee statistics**: mean / min / max of `out["fee_series"]` (converted to bps).
- **Passive LP fees**: final value of `out["lp_fee_value_passive_series"]`.
- **Passive LP LVR**: final value of `out["lp_lvr_passive_series"]`.
- **Passive LP hedged PnL**: final value of `out["lp_pnl_passive"]` (which is `fees − LVR` by construction).
- **Passive LP unhedged PnL**: final value of `out["lp_unhedged_passive"]`.

All quantities in the table are taken **from these single-run outputs**, not from the grid-search CSV.

---

## Key Plots per Scenario

Each scenario writes a standard set of Plotly figures under `abm_results/scenarios/<scenario>/png` and `.../html`. The most relevant for this note are the price, PnL, and fee panels:

### `sigma_sine_fee_static` (`fee_mode: static`)

- Price panel: `abm_results/scenarios/sigma_sine_fee_static/png/<pid>_static_1_price_steps10000.png`
- PnL panel: `abm_results/scenarios/sigma_sine_fee_static/png/<pid>_static_6_pnl_steps10000.png`
- Fee panel: `abm_results/scenarios/sigma_sine_fee_static/png/<pid>_static_7_fee_steps10000.png`

### `sigma_sine_fee_volatility` (`fee_mode: volatility`)

- Price panel: `abm_results/scenarios/sigma_sine_fee_volatility/png/<pid>_volatility_1_price_steps10000.png`
- PnL panel: `abm_results/scenarios/sigma_sine_fee_volatility/png/<pid>_volatility_6_pnl_steps10000.png`
- Fee panel: `abm_results/scenarios/sigma_sine_fee_volatility/png/<pid>_volatility_7_fee_steps10000.png`

### `sigma_sine_fee_toxicity` (`fee_mode: toxicity`)

- Price panel: `abm_results/scenarios/sigma_sine_fee_toxicity/png/<pid>_toxicity_1_price_steps10000.png`
- PnL panel: `abm_results/scenarios/sigma_sine_fee_toxicity/png/<pid>_toxicity_6_pnl_steps10000.png`
- Fee panel: `abm_results/scenarios/sigma_sine_fee_toxicity/png/<pid>_toxicity_7_fee_steps10000.png`

---

## Interpretation

- **Static 5 bps fee (`sigma_sine_fee_static`)**
  - The passive LPs earn about **19k** in fees but suffer roughly **43k** in LVR, leaving **hedged PnL ≈ −24k**. Unhedged PnL is even more negative (≈ −140k) because it also includes market risk.
  - In this noisy, high-vol environment with efficient arb and best-ex smart routing, a fixed 5 bps fee is far below the continuous-time LVR bound, so LPs lose substantially on liquidity provision even before accounting for price exposure.

- **Volatility-linked fee (`sigma_sine_fee_volatility`)**
  - The controller ramps the fee up and down with observed volatility: average fee rises to **~37 bps**, occasionally spiking above **100 bps**.
  - Passive LP fees increase to about **49k**, but realized LVR also grows to about **55k**, so **hedged PnL remains slightly negative (≈ −7k)**. Unhedged PnL is close to break-even (≈ −1k) in this single run.
  - Intuitively, charging more during volatile periods helps narrow the gap between fees and adverse selection, but with this configuration volatility-based fees do **not quite** reach `fees − LVR ≥ 0` for passive LPs.

- **Toxicity-linked fee (`sigma_sine_fee_toxicity`)**
  - Here the controller responds directly to basis/twisting between CEX and DEX prices. The fee becomes extremely aggressive: average fee is **~176 bps**, with frequent excursions toward the cap of **500 bps**.
  - Passive LPs earn about **47k** in fees against **~45k** of LVR, so **hedged PnL turns positive (≈ +2.3k)**. Unhedged PnL is strongly positive (≈ +94.6k) in this run.
  - This shows that, in this environment, it *is* possible to push `fees − LVR` above zero for passive LPs, but only by tolerating very high fee levels that are likely to be problematic for DEX users in practice.

---

## Grid Search over Fee Sensitivities (Noisy-Sine Regime)

To complement the single-seed runs above, a parameter sweep was run using the base scenario config at `abm_results/scenarios/test.yml` (configured to match the noisy-sine volatility profile). For reproducible sweeps with the current code, see `run_parameter_grid_2d_violin_parallel.py` and the `run_parameter_surface_*` helpers. The summary CSV and plots for this note live under:

- CSV: `abm_results/grid_search/grid_summary.csv`

<p align="center">
  <img src="../abm_results/grid_search/plots/sigma_noisy_sine_0.00015_amp0.0001_per10000_noise5e-5.png" alt="Grid search over fee sensitivities in the noisy-sine regime" width="900"/>
</p>

Looking at **passive LP hedged PnL** (`series_key = "lp_pnl_passive"`) in the grid:

- **Static fees** (`fee_mode = static`, constant 5 bps):
  - Mean final hedged PnL across seeds is around **−19k** in this regime, confirming the single-run static scenario: static low fees systematically undercharge for LVR.
- **Volatility-linked fees** (`fee_mode = volatility`, varying `k_sigma`):
  - Even at the best `k_sigma` in this sweep, mean hedged PnL for passive LPs stays **negative (≈ −2k)**, with substantial dispersion across seeds. Volatility-only signals help but do not fully offset adverse selection.
- **Toxicity-linked fees** (`fee_mode = toxicity`, varying `k_basis`):
  - For moderate `k_basis` settings, passive LP hedged PnL remains negative; for more aggressive toxicity sensitivity (e.g. `k_basis ≈ 0.089`) the grid reports **slightly positive mean hedged PnL (≈ +140)**, but with large cross-seed variability (std ≈ 1.5k).

Overall, the grid search supports the same qualitative picture as the single-seed scenarios:

- At realistic volatility, **static and purely volatility-linked schedules leave passive LPs with negative hedged PnL**.
- **Toxicity-aware schedules can, in principle, get `fees − LVR` near or above zero**, but only at high fee levels and with significant run-to-run variability in outcomes.

---

## Takeaways and Next Steps

- In a realistic, noisy high-volatility regime with efficient arbitrage and best-ex smart routing, **static and volatility-linked fees leave passive LPs with negative hedged PnL** in this configuration.
- A strongly toxicity-sensitive fee schedule can **flip hedged PnL positive**, but at the cost of **very high average and peak fees**.
- These are **single-seed, single-scenario** results; a full study should:
  - average across many seeds per scenario,
  - track explicit DEX efficiency metrics (volume, smart-router CEX routing share, realized spreads),
  - and explore softer toxicity settings to find regions where hedged PnL is near zero with more realistic fee levels. 
