# ABM Uni v3 Simulation (ongoing)

<p align="center">
  <img src="abm_results/cex_dex_price.png" alt="Simulation example" width="500"/>
</p>

Agent-based market (ABM) simulator for a Uniswap v3 style pool that extends the Angeris et al. model (“An analysis of Uniswap markets”). The project focuses on **microstructure effects** such as block mempools, asynchronous LP management, dynamic fee schedules, and realistic arbitrage/trader interactions.

The implementation lives in `run.py` and is configured via YAML files consumed by `utils.load_simulation_parameters` (top-level `fee_mode` + a complete `simulate` section). Scenario examples live under `scenarios/` (high/low-vol templates) and `tests/test.yml`. Results, plots, and verbose logs are written to `abm_results/`.

---

## High-Level Features
- **Full Uniswap v3 math**: concentrated-liquidity pool with tick-aware liquidity net, span-by-span fee accounting, and range re-centering logic.
- **Rich agent roster**:
  - **Smart router**: opportunistic trader enforcing best-execution vs. a reference CEX.
  - **Noise trader**: flow provider without valuation discipline, used to stress spreads/liquidity.
  - **Arbitrageur**: clears price discrepancies between the DEX and the CEX reference band; in block mode the arb executes **before** any mempool order (pre-trade CEX vs. DEX snapshot).
  - **LPs**: passive baselines and active narrow LPs. Each LP carries a budget, cooldown, and rebalancing benchmark to compute Loss-versus-Rebalancing (LVR).
- **Block-aware mempool**:
  - `block_time == 1`: deterministic schedule `LP bucket A → smart+noise → LP bucket B → arb → LP bucket C`.
  - `block_time > 1`: freeze the validated snapshot, then run `block_time` micro-steps that diffuse the CEX and probabilistically enqueue smart/noise intents; at the block boundary enqueue a single arb intent plus LP intents (burn/recenter/mint) and replay the shuffled mempool (arb first) against the live pool.
- **Validated price snapshots**: at the end of every block the simulator freezes both the DEX state (tick/S) and the CEX mark. During the following block LPs, noise traders, the arbitrageur, and the smart router all reference this shared “last validated” snapshot (`agent_S_ref`, `agent_tick_ref`, `cex_ref_for_agents`) when forming orders; in block mode the CEX path still diffuses in the background for rebalancing and diagnostics, but mempool orders are priced off the frozen snapshot and executed together at the block boundary.
- **Dynamic fee controller** with four modes:
  - `static` fixes the fee at `f0`.
  - `volatility` adds a multiple of EWMA(|log-return|).
  - `toxicity` adds a multiple of the fee-adjusted log basis (in ticks).
  - `gas` maps a GAS-style (score-driven) volatility state into the fee using `gas_alpha`, `gas_beta`, `gas_omega`, and `k_gas_sigma`.
  Fee moves are clipped by `fee_step_bps_min/max` and gated by `fee_cooldown`.
- **Liquidity bootstrapping**: simulations always start from an evolved/sharded binomial hill that allocates `initial_total_L` across synthetic LPs and can optionally be plotted.
- **LP width rule**: narrow LPs size their ranges off an EWMA of the fee-adjusted basis plus a configurable binomial noise term (`binom_n`, `binom_p`), then clamp to `[w_min_ticks, w_max_ticks]`.
- **Comprehensive telemetry**: per-agent PnL series split by smart router vs. noise trader, liquidity history, fee path, target bands, LP wallet/wealth (hedged vs. unhedged), micro-time traces (block mode), and verbose logs under `abm_results/logs/`.

---

## Agent Behaviour Details

### Reference Market (CEX)
- Implemented as `ReferenceMarket` in `utils.py`.
- Modeled as a GBM over the CEX mid-price with drift `cex_mu` and volatility `σ_t`, plus a permanent impact term of the form
  `impact = kappa * sign(Δa) * |Δa|^{1+xi}` applied to the CEX price in token1 units per token0.
- Volatility can be **static** (`cex_sigma_mode: static`, using `cex_sigma`), **regime-switching** (`cex_sigma_mode: regime`, two-state Markov chain over `σ_L`/`σ_H` with transition probabilities `p_LL`/`p_HH`), or a **noisy sine wave** (`cex_sigma_mode: noisy_sine`) following `σ_t = max(cex_sigma_floor, σ_center + A·sin(2πt/period) + ε_t)` with `ε_t ~ N(0, cex_sigma_sine_noise)`. The center defaults to `cex_sigma` (or the midpoint of `cex_sigma_low`/`cex_sigma_high` when provided); in scenarios using `noisy_sine` the amplitude is typically specified explicitly via `cex_sigma_sine_amp`. The active path is returned as `cex_sigma_series` and `cex_regime_series`.
- The diffusion step uses `m ← m · exp(cex_mu - 0.5 · σ_t^2 + σ_t · z)` with `z ~ N(0,1)`, so `σ_t` is interpreted directly as the per-microstep volatility (no squaring).
- In non-block mode (`block_time == 1`), each simulation step calls `ref.step(Δa_cex)`, which first applies impact from the net arbitrage flow and then diffuses via GBM.
- In block mode (`block_time > 1`), the CEX only diffuses during intra-block micro-steps (`ref.diffuse_only()`), while impact from the arbitrage is applied once at the end via `ref.apply_impact_only(Δa_cex)`. This decouples diffusion from impact and matches the code in `run.py`.

### Arbitrageur
- Encoded in `arbitrage_to_target` and the `arb` branch of `execute_mempool_orders` in `run.py`.
- At each step, uses the **validated** snapshot of the CEX price from the end of the previous block (`cex_ref_for_agents`) to define a no-arb band `[m·r, m/r]`, where `r = 1 - fee`.
- In non-block mode, the arbitrageur trades directly against the live pool using the current `ref.m`.
- In block mode, an `arb` intent is inserted into the mempool at the start of the block, and it executes first when the mempool is replayed. The arb’s trade size and direction are determined by bringing the DEX price back into the no-arb band, with per-span fees allocated via the same fee callback used for traders.
- PnL is measured in token1 by tracking token flows vs. the CEX price after impact is applied for that block.

### Smart Router
- Implemented via `execute_trader("smart", ...)` and smart-router branches in `execute_mempool_orders`.
- Per potential trade:
  - Draws a token1 notional from a log-normal distribution (`trader_mean`, `trader_sigma`).
  - Randomly chooses direction `X_to_Y` or `Y_to_X` with equal probability, converting the notionals into dx/dy so the expected trade *value* is symmetric across directions.
  - Applies a best-execution check vs. the CEX price referenced by agents (`theta_T`): if the quoted DEX execution is worse than the CEX benchmark by more than this threshold, the trade is skipped.
  - Enforces a maximum relative slippage bound (`slippage_tolerance`) at execution time; trades violating slippage are also skipped.
- Trade *arrival rates* are configured via `smart_trades_per_block`, interpreted as the **expected number of smart-router intents per block**. Internally this is converted to a per-step/per-micro-step Bernoulli probability `p_smart ≈ smart_trades_per_block / block_time` (clipped to `[0,1]`) that is sampled each micro-step in block mode and each step in non-block mode.
- In non-block mode, smart-router trades execute immediately in the step schedule. In block mode, the smart router simply enqueues intents into the mempool during micro-steps; those intents are executed later in random order when the mempool is replayed.

### Noise Trader
- Implemented via `execute_trader("noise", ...)` and noise branches in `execute_mempool_orders`.
- Shares the same log-normal size process as the smart router, but **does not** enforce best execution vs. CEX: it only enforces slippage constraints.
- Trade arrival is controlled by `noise_trades_per_block`, interpreted as the **expected number of noise intents per block** and converted to a per-step/per-micro-step Bernoulli probability `p_noise ≈ noise_trades_per_block / block_time` (clipped to `[0,1]`) applied at the same cadence as the smart router.
- Provides “uninformed” flow that stresses spreads and liquidity. In block mode these trades are also enqueued into the mempool during micro-steps and executed during the mempool replay.

### Liquidity Providers
- Defined in `agents.py` (`LPAgent`, `Position`, and `RebalancerState`) with management logic in `run.py`.
- **Types**:
  - *Passive baselines* (`is_passive=True` for a fraction `passive_lp_share` of LPs): wide fixed-width ranges, probabilistic mint/burn rules driven by the block-level targets `passive_mints_per_block`, `passive_burns_per_block`, and width `passive_width_ticks`.
  - *Active narrow LPs* (`is_active_narrow=True` and `is_passive=False`): concentrate liquidity near the current mid, recenter after they have been out of range for a random number of steps between `k_out_min` and `k_out_max`, and follow an EWMA-driven width signal with binomial noise.
- **Decision process** (per LP):
  - Each LP carries an internal review clock with inter-review times drawn from a geometric distribution with mean `tau`. In any block an LP is either *not due* (clock has not fired yet) or *due* (clock hits zero); only due LPs are allowed to act.
  - LP activity knobs are specified as **target counts per block** across the population: `narrow_mints_per_block`, `passive_mints_per_block`, and `passive_burns_per_block`. Given `block_time = B`, the passive share, and `N_LP`, the simulator converts these into per-LP Bernoulli probabilities, e.g. `p_narrow_mint ≈ narrow_mints_per_block / (B · N_narrow)`, `p_passive_mint ≈ passive_mints_per_block / (B · N_passive)`, and `p_passive_burn ≈ passive_burns_per_block / (B · N_passive)` (clipped to `[0,1]`), and flips these coins only for LPs whose review clock is due and that are not in cooldown. Realized mint/burn counts therefore fluctuate around the targets and also scale with how many LPs happen to be due in that block (≈ `1/τ` fraction on average) and with cooldowns.
  - After a burn, an LP enters a cooldown for several steps during which it cannot mint again.
  - Narrow LPs track how many consecutive steps their position has been out-of-range (`out_steps`). Once this reaches `k_out_threshold`, they enqueue a recenter intent targeting a symmetric band around the agent’s reference price.
  - For **active narrow LPs only** (`is_passive=False`), TP/SL logic evaluates each position’s PnL in token1 terms using `Position.PnL_y(agent_S_ref, validated_cex)` and burns positions whose PnL exceeds `theta_TP` or drops below `-theta_SL` times the initial HODL value. Passive LPs do **not** use TP/SL; they exit positions only via the probabilistic burn rule.
- **Scheduling and execution**:
  - In non-block mode, LP burns and narrow-LP recenter actions execute directly during their bucket(s) in the per-step schedule A/B/C. New mints are currently handled only in the block-mode mempool path, so for fully budget- and wallet-consistent LP dynamics you should prefer `block_time > 1`.
  - In block mode, all LP actions (burn, recenter, mint) are added to the mempool as intents (`lp_burn`, `lp_recenter`, `lp_mint`) and executed alongside trader orders when the mempool is replayed.
- **Budgets & bootstrap**:
  - Each LP carries a liquidity budget `L_budget` and tracked live deployment `L_live`; new mints are clipped by both a per-step cap (fraction of `L_budget`) and remaining budget.
  - `bootstrap_initial_binomial_hill_sharded` distributes `initial_total_L` across a set of seed LPs so early burns are staggered and the book has a smooth “hill” shape.
- **Wealth tracking**:
  - `RebalancerState` maintains a self-financing benchmark that delta-hedges the LP’s token0 exposure using the CEX price; LVR is computed as the difference between LP wealth (wallet plus open position mark-to-market) and this benchmark.
  - Simulation outputs track total and cohort-level PnLs (active vs. passive), rebalancer PnL, wallet balances, and mark-to-market wealth.

---

## Simulation Flow
1. **Initialization**:
   - Parse CLI (`python run.py --config path/to/config.yml`).
   - Load scenario from YAML (see next section).
   - Seed RNGs, build empty pool, generate LP roster, bootstrap liquidity.
2. **Per step (block)**:
   - Copy the validated snapshot into agent-facing variables (`agent_S_ref`, `agent_tick_ref`, `cex_ref_for_agents`).
   - Update/adapt reference CEX (diffusion + impact) and evolve EWMA signals.
   - Randomize actor order depending on `block_time`. In block mode: run micro-steps that diffuse the CEX and enqueue smart/noise + LP intents against the snapshot, then insert an arb intent (using the same snapshot) and replay the mempool (arb first) against the live pool.
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
  block_time: 5             # 1 => synchronous mode; >1 => mempool mode
  T: 750                    # number of blocks
  seed: 7
  cex_mu: 0.0
  cex_sigma_mode: static    # "static", "regime", or "noisy_sine"
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
  smart_trades_per_block: 8.0    # expected smart-router intents per block
  noise_trades_per_block: 6.0    # expected noise intents per block
  narrow_mints_per_block: 200.0  # expected narrow LP mints per block (total across narrow LPs)
  passive_lp_share: 0.2
  passive_mints_per_block: 60.0  # expected passive LP mints per block (total across passive LPs)
  passive_burns_per_block: 10.0  # expected passive LP burns per block (total across passive LPs)
  passive_width_ticks: 500
  N_LP: 500
  tau: 20
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

---

## Running a Scenario
```bash
python run.py --config scenarios/high_vol_static.yml
```

Outputs:
- `abm_results/logs/verbose_steps_<fee_mode>_<n>.txt`: human-readable log per step and mempool replay summaries (includes micro-time traces when `block_time>1`).
- `abm_results/png/` & `abm_results/html/`: figures summarizing prices, liquidity, agent PnLs (smart vs. noise vs. arb, hedged vs. unhedged LPs), fee path, and width signals. PNG export relies on Kaleido (needs Chrome); HTML files are always written.
- Optional `abm_results/liquidity_evolution_<fee_mode>.gif` if `make_liquidity_gif` is enabled.
- JSON-like dict returned by `simulate` with all recorded series (see tail of `run.py` for exact keys, including wallet vs. wealth, fee signals, micro-step history, and the active CEX volatility/regime path).

---

## Batch Runners & Analysis Helpers
- `run_fee_sweep.py --config fee_sweep_config.yml`: sweep `k_sigma`/`k_basis` (or static baseline) with parallel simulations; writes a CSV plus matplotlib summary of active/passive LP wealth.
- `run_scenarios_mean_std.py --scenarios-dir scenarios/ --runs 5`: run every YAML scenario multiple times and emit mean ± std PnL charts for each agent class.
- `run_parameter_grid_mean_std.py`: grid-search `cex_sigma` × fee sensitivity for all fee modes (static, volatility, toxicity, gas) and plot mean ± std PnL per fee mode.
- `sigma_calibration.py`: derive realistic per-second `cex_sigma` from Binance 1s ETH/USDC data (CSV/Parquet/pickle) and optionally persist the computed series.

For further questions or ideas, open an issue or start a discussion in this repository. Happy simulating!
