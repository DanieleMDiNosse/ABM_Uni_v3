# ABM Uni v3 Simulation (ongoing)

<p align="center">
  <img src="abm_results/cex_dex_price.png" alt="Simulation example" width="500"/>
</p>

Agent-based market (ABM) simulator for a Uniswap v3 style pool that extends the Angeris et al. model (“An analysis of Uniswap markets”). The project focuses on **microstructure effects** such as block mempools, asynchronous LP management, dynamic fee schedules, and realistic arbitrage/trader interactions.

The implementation lives in `run.py` and can be configured through YAML scenario files (for example `abm_mempool_config.yml`). Results, plots, and verbose logs are written to `abm_results/`.

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
  - `block_time > 1`: (i) freeze the validated snapshot, (ii) run the arbitrage at that price, (iii) diffuse the CEX for each micro-step, (iv) enqueue LP/trader intents, and (v) execute the shuffled mempool to mimic intra-block ordering. When desired the block length itself can be re-drawn from a bounded Zipf(α) distribution.
- **Validated price snapshots**: at the end of every block the simulator freezes both the DEX state (tick/S) and the CEX mark. During the following block LPs, noise traders, and the arbitrageur make decisions against this shared “last validated” snapshot, while the smart router still references the frozen DEX state but measures best execution against the live diffused CEX mark. This keeps order placement aligned with what would be visible on-chain while letting the router react to the most recent CEX price.
- **Dynamic fee controller** with three modes:
  - `static` fixes the fee at `f0`.
  - `volatility` adds a multiple of EWMA(|log-return|).
  - `toxicity` adds a multiple of the fee-adjusted log basis (in ticks).
  Fee moves are clipped by `fee_step_bps_min/max` and gated by `fee_cooldown`.
- **Liquidity bootstrapping**: simulations always start from an evolved/sharded binomial hill that allocates `initial_total_L` across synthetic LPs and can optionally be plotted.
- **Comprehensive telemetry**: per-agent PnL series, liquidity history, fee path, target bands, LP wallet/wealth, and block-level log files.

---

## Agent Behaviour Details

### Reference Market (CEX)
- Implemented as `ReferenceMarket` in `utils.py`.
- Modeled as a GBM over the CEX mid-price with drift `cex_mu` and volatility `cex_sigma`, plus a permanent impact term of the form
  `impact = kappa * sign(Δa) * |Δa|^{1+xi}` applied to the CEX price in token1 units per token0.
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
- In non-block mode, smart-router trades execute immediately in the step schedule. In block mode, the smart router simply enqueues intents into the mempool during micro-steps; those intents are executed later in random order when the mempool is replayed.

### Noise Trader
- Implemented via `execute_trader("noise", ...)` and noise branches in `execute_mempool_orders`.
- Shares the same log-normal size process as the smart router, but **does not** enforce best execution vs. CEX: it only enforces slippage constraints.
- Provides “uninformed” flow that stresses spreads and liquidity. In block mode these trades are also enqueued into the mempool during micro-steps and executed during the mempool replay.

### Liquidity Providers
- Defined in `agents.py` (`LPAgent`, `Position`, and `RebalancerState`) with management logic in `run.py`.
- **Types**:
  - *Passive baselines* (`is_passive=True` for a fraction `passive_lp_share` of LPs): wide fixed-width ranges, probabilistic mint/burn rules (`passive_mint_prob`, `passive_burn_prob`, `passive_width_ticks`).
  - *Active narrow LPs* (`is_active_narrow=True` and `is_passive=False`): concentrate liquidity near the current mid, recenter after they have been out of range for a random number of steps between `k_out_min` and `k_out_max`, and follow an EWMA-driven width signal with binomial noise.
- **Decision process** (per LP):
  - Review times are drawn from a geometric distribution with mean `tau`; only LPs whose review clocks fire in a given step are allowed to act.
  - After a burn, an LP enters a cooldown for several steps during which it cannot mint again.
  - Narrow LPs track how many consecutive steps their position has been out-of-range (`out_steps`). Once this reaches `k_out_threshold`, they enqueue a recenter intent targeting a symmetric band around the agent’s reference price.
  - TP/SL logic evaluates each position’s PnL in token1 terms using `Position.PnL_y(agent_S_ref, validated_cex)` and burns positions whose PnL exceeds `theta_TP` or drops below `-theta_SL` times the initial HODL value.
- **Scheduling and execution**:
  - In non-block mode, LPs act directly during their bucket(s) in the per-step schedule A/B/C.
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
   - Randomize actor order depending on `block_time`. In block mode: arb at the snapshot price, diffuse micro-steps, enqueue intents (all referencing the snapshot), then replay the mempool against the live pool.
   - Apply fees, update LP positions, and settle agent PnL at the post-impact CEX price.
   - Update the dynamic fee controller, log state, and finally capture the new validated snapshot (live DEX + CEX) for the next iteration.
3. **Post-processing**:
   - Generate the default plot suite (prices, LP stats, PnLs, fee path).
   - Compute DEX log-return autocorrelation (saved under `abm_results/png` and `abm_results/html`).
   - Optionally render liquidity GIFs.

---

## Configuration

Scenario YAML files follow the schema:

```yaml
fee_mode: static            # scenario label + default fee mode
simulate:
  block_time: 5             # 1 => synchronous mode; >1 => mempool mode
  T: 750                    # number of blocks
  seed: 7
  cex_mu: 0.0
  cex_sigma: 0.0015
  p_trade: 0.7
  noise_floor: 0.5
  p_lp_narrow: 0.95
  passive_lp_share: 0.2
  passive_mint_prob: 0.3
  passive_burn_prob: 0.05
  passive_width_ticks: 500
  N_LP: 500
  tau: 20
  w_min_ticks: 10
  w_max_ticks: 1_774_540
  basis_half_life: 20
  slope_s: 0.15
  binom_n: 10
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
  k_out: 5
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
python run.py --config scenarios/abm_mempool_config.yml
```

Outputs:
- `abm_results/verbose_steps_<scenario>.txt`: human-readable log per step and mempool replay summaries.
- `abm_results/png/` & `abm_results/html/`: figures summarizing prices, liquidity, agent PnLs, and fee path.
- Optional `abm_results/liquidity_evolution_<fee_mode>.gif` if `make_liquidity_gif` is enabled.
- JSON-like dict returned by `simulate` (see tail of `run.py` for exact keys).
- Plots are rendered with Plotly; PNG export relies on Kaleido (which in turn needs Chrome available). HTML files are always written.

---

For further questions or ideas, open an issue or start a discussion in this repository. Happy simulating!
