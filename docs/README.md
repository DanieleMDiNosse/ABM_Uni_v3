---
title: Model Overview
nav_order: 2
---

# ABM Uni v3 Simulation (ongoing)

<p align="center">
  <img src="../abm_results/cex_dex_price.png" alt="Simulation example" width="500"/>
</p>

Agent-Based Market (ABM) simulator for a Uniswap v3 style pool. The project focuses on **microstructure effects** such as block mempools, asynchronous LP management, dynamic fee schedules, MEV and realistic arbitrage/trader interactions.

The implementation lives in `scripts/run.py` and is configured via YAML files. Example scenario configs in this repo live under `abm_results/scenarios/` (see for example `abm_results/scenarios/test.yml`). Per-scenario results, plots, and verbose logs are written under `abm_results/scenarios/<scenario_name>/`.

---

## High-Level Features
- **Full Uniswap v3 math**: concentrated-liquidity pool with tick crossing math, tick-aware liquidity net, span-by-span fee accounting.
- **Rich agent roster**: smart router, noise trader, arbitrageur, strategic LP cohorts (passive + active narrow), and a JIT MEV-style LP (“Jiter”).
- **Block-aware mempool**: the simulator runs in mempool execution mode (`block_time` micro-steps per block; the current implementation requires `block_time > 1`). It freezes the validated snapshot, runs `block_time` micro-steps that diffuse the CEX and probabilistically enqueue smart/noise intents, then enqueues a single arb intent plus LP intents (burn/recenter/mint) and replays the shuffled mempool (arb first) against the live pool.
- **Validated price snapshots**: at the end of every block the simulator freezes both the DEX state (tick/S) and a CEX mark. Swap intents (smart/noise) and the arbitrage target are formed off this “last validated” snapshot (`agent_S_ref`, `agent_tick_ref`, `cex_ref_for_agents`), while the CEX path still diffuses during micro-steps for the rebalancing benchmark, fee signals, and some LP diagnostics. All intents are executed together at the block boundary via the mempool replay.
- **Dynamic fee controller** with five modes (see **[Fee Schedules](fee_schedules.md)** for formulas):
  - `static` fixes the fee at `f0`.
  - `volatility_cex` sets the fee based on an EWMA realized-volatility estimate from the **CEX** price series.
  - `volatility_dex` sets the fee based on an EWMA realized-volatility estimate from the **DEX** price series.
  - `toxicity` sets the fee based on the fee-adjusted log basis (in ticks).
  - `lvr_fee_ewma` applies a feedback update based on an EWMA of the per-step (LVR - fees) gap normalized by DEX notional.
  Fee moves are clipped by `fee_step_bps_min/max` and gated by `fee_cooldown`.
- **Liquidity bootstrapping**: simulations always start from an evolved/sharded binomial hill that allocates `initial_total_L` across synthetic *seed* LPs (`is_seed=True`) that provide background liquidity and can optionally be plotted; these seed LPs are excluded from the strategic LP cohorts and PnL statistics.
- **LP width rule**: narrow LPs size their ranges off an EWMA of `|log-return|` of the CEX mid (`ref.m`) with half-life `basis_half_life`, plus a configurable binomial noise term (`binom_n`, `binom_p`), then clamp to `[w_min_ticks, w_max_ticks]`.
- **Comprehensive telemetry**: per-agent PnL series split by smart router vs. noise trader, liquidity history, fee path, target bands, LP wallet/wealth (hedged vs. unhedged), micro-time traces (per block), and verbose logs under `<results_root>/logs/` (e.g. `abm_results/scenarios/<scenario_name>/logs/` when running via `python -m scripts.run --config ...`).

For the *detailed* and math-accurate description of each agent’s decision rules and execution logic, see:
- **[Agent Behaviour Details](agents_spec.md)**

---

### Reference Market (CEX)
- Implemented as `ReferenceMarket` in `core/utils.py`.
- Modeled as a GBM over the CEX mid-price with drift `cex_mu` and volatility `σ_t`, plus a permanent impact term of the form
  `impact = kappa * sign(Δa) * |Δa|^{1+xi}` applied to the CEX price in token1 units per token0.
- Volatility can be:
  - **static** (`cex_sigma_mode: static`, using `cex_sigma`);
  - **regime-switching** (`cex_sigma_mode: regime`, two-state Markov chain over `σ_L`/`σ_H` with transition probabilities `p_LL`/`p_HH`);
  - a **noisy sine wave** (`cex_sigma_mode: noisy_sine`) following `σ_t = max(cex_sigma_floor, σ_center + A·sin(2πt/period) + ε_t)` with `ε_t ~ N(0, cex_sigma_sine_noise)`;
  - or **Heston-like stochastic volatility** (`cex_sigma_mode: heston`), where the variance `v_t = σ_t^2` follows a mean-reverting square-root process with parameters `cex_heston_kappa`, `cex_heston_theta`, `cex_heston_sigma_v`, correlation `cex_heston_rho`, and optional initial variance `cex_heston_v0` (falling back to `cex_sigma^2` when omitted).
  The center for the noisy-sine mode defaults to `cex_sigma` (or the midpoint of `cex_sigma_low`/`cex_sigma_high` when provided); in scenarios using `noisy_sine` the amplitude is typically specified explicitly via `cex_sigma_sine_amp`. The active path is returned as `cex_sigma_series` and `cex_regime_series`.
- In non-Heston modes the diffusion step uses `m ← m · exp(cex_mu - 0.5 · σ_t^2 + σ_t · z)` with `z ~ N(0,1)`, so `σ_t` is interpreted directly as the per-microstep volatility (no squaring). In Heston mode, the variance `v_t` and price `m_t` are updated jointly with correlated Gaussian shocks while keeping `σ_t = sqrt(v_t)` in the returned series.
- The CEX diffuses during intra-block micro-steps (`ref.diffuse_only()`). Permanent impact is applied **immediately** every time an action “touches the CEX” (smart-router CEX legs, arb hedge legs, LP mint/burn conversions, JIT burn conversion) via `ref.apply_impact_only(Δa)`.

#### Heston Volatility Mode (details)
- **Continuous-time model** (conceptual): in Heston mode the CEX price `m_t` and variance `v_t` are thought of as solving
    $$
    \mathrm{d}\log m_t = (\mu - \tfrac{1}{2} v_t)\,\mathrm{d}t + \sqrt{v_t}\,\mathrm{d}W_t^{(1)} \\
    \mathrm{d}v_t = \kappa(\theta - v_t)\,\mathrm{d}t + \sigma_v \sqrt{v_t}\,\mathrm{d}W_t^{(2)}
    $$
    
  with correlation 
  $$
  \mathrm{corr}(W^{(1)}, W^{(2)}) = \rho
  $$
  
- **Discrete-time implementation** (per micro-step, `Δt = 1` second):
  - The simulator stores the variance as `_heston_v = v_t` and uses `σ_t = sqrt(v_t)` in plots and outputs (`cex_sigma_series`).
  - Each diffusion step (`ReferenceMarket._diffuse_heston`) draws `z1, z2 ~ N(0, 1)` i.i.d. and performs:
    - `v_t = max(_heston_v, 0)` (guard against numerical drift),
    - `dv = kappa * (theta - v_t) * dt + sigma_v * sqrt(max(v_t, 0)) * sqrt(dt) * z1`,
    - `v_next = max(0, v_t + dv)` (full truncation to keep variance non-negative),
    - `_heston_v = v_next` and `sigma = sqrt(max(v_next, 0))` (this `sigma` is what ends up in `cex_sigma_series`),
    - `z_price = rho * z1 + sqrt(1 - rho^2) * z2` (correlated shock for the price, with `rho` clipped to `[-1, 1]`),
    - `log_m_next = log(m_t) + (mu - 0.5 * v_t) * dt + sqrt(max(v_t, 0)) * sqrt(dt) * z_price`,
    - `m_t` is updated to `exp(log_m_next)` and floored at `1e-12` for numerical stability.
- **Initialization and parameter validation**:
  - Heston mode is enabled by `cex_sigma_mode: heston` in the YAML config; `simulate(...)` fails fast if any of the required parameters are missing:
    - `cex_heston_kappa`, `cex_heston_theta`, `cex_heston_sigma_v`, `cex_heston_rho`.
  - Constraints enforced at startup:
    - `cex_heston_kappa > 0`,
    - `cex_heston_theta ≥ 0`,
    - `cex_heston_sigma_v ≥ 0`,
    - `cex_heston_rho ∈ [-1, 1]`.
  - The initial variance is chosen as:
    - `v0 = cex_heston_v0` if provided (must be strictly positive), or
    - `v0 = cex_sigma^2` if `cex_heston_v0` is omitted (requires `cex_sigma > 0`).
    In both cases `sigma_for_ref = sqrt(v0)` is used as the initial per-step volatility in logs.
- **Configuration knobs** (YAML, under `simulate:`):
  - `cex_sigma_mode: heston` — activates the Heston engine.
  - `cex_sigma` — per-step volatility used as `sqrt(v0)` when `cex_heston_v0` is omitted.
  - `cex_heston_kappa` — mean reversion speed of the variance process (higher values pull `v_t` back to `theta` more aggressively).
  - `cex_heston_theta` — long-run variance level.
  - `cex_heston_sigma_v` — volatility of the variance process (controls how “noisy” `v_t` is).
  - `cex_heston_rho` — correlation between price shocks and variance shocks; negative values generate the usual “leverage effect”.
  - `cex_heston_v0` — optional explicit initial variance; when omitted the code uses `cex_sigma^2`.
- **Outputs and plotting**:
  - `cex_sigma_series` continues to contain the *per-step* volatility used in the GBM/Heston update; in Heston mode this is `sqrt(v_t)` at each step.
  - The existing PnL figure (`6_pnl`) uses a volatility subplot whenever the sigma path is dynamic. Heston mode enables this subplot (`sigma_panel = True`) so you can inspect `cex_sigma_series` under the agent PnL panel.
  - The `cex_regime_series` remains available for consistency; in Heston mode it is a simple label (`"H"`) and not used for logic.

## Simulation Flow
1. **Initialization**:
   - Parse CLI (`python -m scripts.run --config path/to/config.yml`).
   - Load scenario from YAML (see next section).
   - Seed RNGs, build empty pool, generate LP roster, bootstrap liquidity.
2. **Per step (block)**:
   - Copy the validated snapshot into agent-facing variables (`agent_S_ref`, `agent_tick_ref`, `cex_ref_for_agents`).
   - Update/adapt reference CEX (diffusion + impact) and evolve EWMA signals.
   - Run micro-steps that diffuse the CEX and enqueue smart/noise intents against the snapshot, then insert an arb intent (using the same snapshot), add LP intents, and replay the mempool (arb first) against the live pool.
   - Apply fees, update LP positions, and settle agent PnL at the post-impact CEX price.
   - Update the dynamic fee controller, log state, and finally capture the new validated snapshot (live DEX + CEX) for the next iteration.
3. **Post-processing**:
   - Generate the default plot suite (prices, LP stats, PnLs, fee path).
   - Compute DEX log-return autocorrelation (saved under `abm_results/png` and `abm_results/html`).
   - Optionally render liquidity GIFs.

---

## Configuration

Scenario YAML files contain a top-level `fee_mode` label plus a `simulate` mapping with every parameter accepted by `simulate(...)`. The loader fails fast on missing/extra keys. A minimal template:

```yaml
fee_mode: static            # scenario label + default fee mode
simulate:
  block_time: 5             # micro-steps per block (mempool execution; must be > 1)
  T: 750                    # number of blocks
  seed: 7
  cex_mu: 0.0
  cex_sigma_mode: static    # "static", "regime", "noisy_sine", or "heston"
  cex_sigma: 0.0015         # used when mode=static
  cex_sigma_low: 0.0001     # regime-switching: σ_L
  cex_sigma_high: 0.002     # regime-switching: σ_H (> σ_L)
  cex_sigma_p_LL: 0.98      # P(Z_{t+1}=L | Z_t=L)
  cex_sigma_p_HH: 0.95      # P(Z_{t+1}=H | Z_t=H)
  cex_sigma_regime_init: L  # starting regime ("L" or "H")
  cex_sigma_sine_period: 10000   # noisy_sine: steps per full cycle
  cex_sigma_sine_amp: 0.0005     # noisy_sine: amplitude (must be set explicitly)
  cex_sigma_sine_noise: 0.0      # noisy_sine: white-noise std added to σ_t (must be set explicitly)
  cex_sigma_sine_phase: 0.0      # noisy_sine: phase offset (radians)
  cex_sigma_floor: 0.0           # noisy_sine: lower bound for σ_t
  # Heston: stochastic variance over σ_t^2 (only used when cex_sigma_mode: heston)
  # cex_heston_kappa: 1.0        # mean reversion speed of variance
  # cex_heston_theta: 1.0e-6     # long-run variance level
  # cex_heston_sigma_v: 0.1      # volatility of variance
  # cex_heston_rho: -0.5         # correlation between price and variance shocks
  # cex_heston_v0: 2.25e-8       # optional initial variance; defaults to cex_sigma^2 when omitted
  # Trader arrivals (Poisson intensities).
  # Prefer per-second knobs (micro-step = 1 second): expected per block scales with block_time.
  smart_trades_per_second: 1.6    # per second => ~8 per block when block_time=5
  noise_trades_per_second: 1.2    # per second => ~6 per block when block_time=5
  # smart_trades_per_block: 8.0   # legacy per-block knob (ignored if *_per_second is set)
  # noise_trades_per_block: 6.0   # legacy per-block knob (ignored if *_per_second is set)

  # LP event targets (Poisson means). Same convention: prefer per-second knobs.
  narrow_mints_per_second: 40.0   # per second => ~200 per block when block_time=5
  # narrow_mints_per_block: 200.0 # legacy per-block knob (ignored if *_per_second is set)
  passive_lp_share: 0.2
  passive_mints_per_second: 12.0  # per second => ~60 per block when block_time=5
  passive_burns_per_second: 2.0   # per second => ~10 per block when block_time=5
  # passive_mints_per_block: 60.0   # legacy per-block knob (ignored if *_per_second is set)
  # passive_burns_per_block: 10.0   # legacy per-block knob (ignored if *_per_second is set)
  passive_width_pct: 5.0
  passive_width_ticks: 500
  N_LP: 500                # number of strategic LP agents (excluding the seed binomial-hill LPs)
  tau: 20
  tau_seconds: 100          # optional: per-second LP review clock (mean waiting time in seconds)
  w_min_ticks: 10
  w_max_ticks: 1_774_540
  basis_half_life: 20
  slope_s: 0.15
  binom_n: 10
  binom_p: 0.5
  trader_mean: 1.0
  trader_sigma: 0.6
  theta_T: 0.95
  slippage_tolerance: 0.01
  mint_mu: 0.05
  mint_sigma: 0.01
  theta_TP: 0.1
  theta_SL: 0.25
  initial_binom_N: 450
  initial_total_L: 500000.0
  k_out_min: 3
  k_out_max: 8
  visualize: true
  skip_step: 300
  f0: 0.003
  f_min: 0.0005
  f_max: 0.05
  fee_half_life: 10
  k_sigma: 50.0
  k_basis: 0.0001
  fee_step_bps_min: 0.001
  fee_step_bps_max: 20.0
  fee_cooldown: 0
```

Any argument of `simulate(...)` can be overridden in the YAML. Keep `fee_mode` in sync with the controller you intend to test.

For a key-by-key description of the telemetry returned by `simulate(...)` (prices, PnLs, fees, LP wealth, activity, etc.), see **[Simulation Outputs](simulation_outputs.md)**.

---

## Running a Scenario
```bash
python -m scripts.run --config abm_results/scenarios/test.yml
```

Outputs:
- `abm_results/scenarios/<scenario_name>/logs/<pid>_verbose_steps_<fee_mode>_<n>.txt`: human-readable log per step and mempool replay summaries (includes micro-time traces; omitted when `light_mode=True`).
- `abm_results/scenarios/<scenario_name>/png/` & `abm_results/scenarios/<scenario_name>/html/`: figures summarizing prices, liquidity, agent PnLs (smart vs. noise vs. arb, hedged vs. unhedged LPs), fee path, and width signals. PNG export relies on Kaleido (needs Chrome); HTML files are always written.
- Optional liquidity GIFs can be generated via `utils.make_liquidity_gif(...)` using the recorded `liq_history` and `tick_history` series.
- JSON-like dict returned by `simulate` with all recorded series (see tail of `scripts/run.py` for exact keys, including wallet vs. wealth, fee signals, and the active CEX volatility/regime path). Micro-step traces are written to the verbose log but are not currently returned in the output dict.

---

## Batch Runners & Analysis Helpers
- `scripts/run_multiple.py`: run a single scenario config over many seeds (parallel) and plot mean ± std bands for agent PnL series under `abm_results/scenarios/<scenario>/multi_runs/{html,png}/`.
- `scripts/run_parameter_surface_nd_pnl_fee_dashboard.py`: ND parameter sweeps (cache-only) writing CSV + metadata under `abm_results/grid_search/dashboard_nd/data/`; worker runs use isolated temp output folders and the cache fingerprint includes an effective config-content hash.
- `scripts/build_parameter_surface_nd_pnl_fee_dashboard.py`: build the standalone HTML dashboard from the cached CSV under `abm_results/grid_search/dashboard_nd/html/` (reads cache metadata when available, while staying compatible with older caches).
- Grid enumeration/seeding details: `docs/nd_grid_sampling_methods.qmd`.
- `scripts/run_experiment_design.py`: run experiment designs (grid/LHS/Sobol/Saltelli/adaptive refine/BayesOpt) defined in an experiment YAML under `abm_results/experiments/`, caching point summaries under `abm_results/experiments_runs/<tag>/data/`.
- `scripts/build_experiment_design_dashboard.py`: build a standalone HTML dashboard for sampled designs from `points_<tag>.csv` (scatter + filtering + optional binned heatmap).
- `scripts/analyze_experiment_design.py`: screening (permutation importance), Saltelli Sobol indices (when applicable), and top-point summaries from an experiment cache.
- `scripts/sigma_calibration.py`: derive realistic per-second `cex_sigma` from Binance 1s ETH/USDC data (CSV/Parquet/pickle) and optionally persist the computed series.
- `scripts/visualize_distributions.py`: generate and save figures for the stochastic components used by the simulator (Heston price/volatility paths, binomial-hill initial liquidity, binomial width noise, log-normal trader and LP mint-size distributions, and geometric LP review clocks), useful for validating input distributions and for documentation figures.

Reproduction recipe (experiment designs):
```bash
conda activate main

# 1) Preview + run (examples live under abm_results/experiments/)
python -m scripts.run_experiment_design --experiment abm_results/experiments/example_lhs_screening.yml --dry-run
python -m scripts.run_experiment_design --experiment abm_results/experiments/example_lhs_screening.yml

# 2) Build the sampled-design dashboard (use the printed cache/meta paths)
python -m scripts.build_experiment_design_dashboard --cache abm_results/experiments_runs/<tag>/data/points_<tag>.csv --meta abm_results/experiments_runs/<tag>/data/meta_<tag>.json

# 3) Analyze (screening; Sobol indices only if the design is sobol_saltelli)
python -m scripts.analyze_experiment_design --cache abm_results/experiments_runs/<tag>/data/points_<tag>.csv --meta abm_results/experiments_runs/<tag>/data/meta_<tag>.json --metric fee_mean
```

For further questions or ideas, open an issue or start a discussion in this repository. Happy simulating!
