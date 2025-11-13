# ABM Uni v3 Simulation

Agent-based market (ABM) simulator for a Uniswap v3 style pool that extends the Angeris et al. model (“An analysis of Uniswap markets”). The project focuses on **microstructure effects** such as block mempools, asynchronous LP management, dynamic fee schedules, and realistic arbitrage/trader interactions.

The implementation lives in `run.py` and can be configured through YAML scenario files (for example `abm_mempool_config.yml`). Results, plots, and verbose logs are written to `abm_results/`.

---

## High-Level Features
- **Full Uniswap v3 math**: concentrated-liquidity pool with tick-aware liquidity net, span-by-span fee accounting, and range re-centering logic.
- **Rich agent roster**:
  - **Smart router**: opportunistic trader enforcing best-execution vs. a reference CEX.
  - **Noise trader**: flow provider without valuation discipline, used to stress spreads/liquidity.
  - **Arbitrageur**: clears price discrepancies between the DEX and the CEX reference band; in block mode the arb executes **before** any mempool order (pre-trade CEX vs. DEX snapshot).
  - **LPs**: passive baselines, active narrow, active wide. Each LP carries a budget, cooldown, and rebalancing benchmark to compute Loss-versus-Rebalancing (LVR).
- **Block-aware mempool**:
  - `block_size == 1`: deterministic schedule `LP bucket A → smart+noise → LP bucket B → arb → LP bucket C`.
  - `block_size > 1`: micro diffusion on the CEX, intents are queued, arbitrage executes **first**, then LP + trader intents are shuffled together and executed to mimic intra-block random ordering.
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
- Modeled as a GBM with drift `cex_mu`, volatility `cex_sigma`, and impact function `kappa * sign(Δa) * |Δa|^{1+xi}`.
- Diffuses every micro time-step. At the end of each block the arbitrageur observes the latest CEX price and trades the DEX back into the `[m * r, m / r]` band.

### Arbitrageur
- Runs `arbitrage_to_target(arb_ref_m)` to bring the pool price inside the tolerance band.
- Arbitrage fires **before** mempool execution using the pre-mempool DEX price and the current CEX snapshot. This mirrors a MEV searcher inserting the backrun at the front of the block.
- PnL is measured in token1 using the same settlement convention as other agents (CEX price after the block’s net impact).

### Smart Router
- Samples swaps from log-normal size distributions (`trader_mean`, `trader_sigma`).
- Enforces:
  - Best execution vs. CEX (`theta_T` threshold). Smart Router trades on DEX only if the DEX quote is no less that `theta_T` times CEX quote.
  - Slippage tolerance (`slippage_tolerance`).
- Intents are routed into the mempool and executed if liquidity is available. Token flows are tracked and PnL is settled **after** the CEX incorporates arbitrage impact, mimicking settlement on block confirmation.

### Noise Trader
- Shares the same size distribution as the smart router but **skips** the best-execution check (only slippage is enforced). Useful for stress testing liquidity by supplying “uninformed” flow.

### Liquidity Providers
- **Classes**:
  - *Passive baselines* (`passive_lp_share`): wide default ranges, probabilistic mint/burn rules (`passive_mint_prob`, `passive_burn_prob`, `passive_width_ticks`).
  - *Active narrow LPS*: concentrate liquidity near the mid, recenter after `k_out` steps out of range, follow an EWMA-driven width rule with binomial noise.
  <!-- - *Active wide LPS*: optional additional cohort (`active_wide_lp_enabled`), slower to react but still “active”. -->
- **Scheduler**:
  - Each LP has a geometric review clock (`tau`) and a cooldown after burning.
  - All due LPs inject their intents into the mempool (burns/recenters/mints) and are shuffled with trader flow.
  <!-- - In single-step mode due LPs are split across three buckets to interleave with other actors deterministically. -->
- **Budgets & bootstrap**:
  - Liquidity budgets derived from `initial_total_L`.
  - Early in the sim, `bootstrap_initial_binomial_hill_sharded` seeds liquidity and ensures every LP is funded.
- **Wealth tracking**:
  - Rebalancer benchmark tracks token0 exposure and PnL vs. a continuous delta-hedging strategy (`lp.rebalancer`).
  - Metrics include realized wallet value, mark-to-market wealth, and hedged Fee-LVR by cohort.

---

## Simulation Flow
1. **Initialization**:
   - Parse CLI (`python run.py --config path/to/config.yml`).
   - Load scenario from YAML (see next section).
   - Seed RNGs, build empty pool, generate LP roster, bootstrap liquidity.
2. **Per step**:
   - Update reference CEX (diffusion and, after swaps, impact).
   - Evolve EWMA signals for LP widths and fee controller.
   - Randomize actor order depending on `block_size`.
   - Execute arbitrage.
   - Run mempool or direct swaps, apply fees, update LP positions.
   - Settle smart-router, noise and arbitrageur PnL at the post-impact CEX price.
   - Update fee schedule for the next step, record telemetry, and log to `abm_results/verbose_steps_*`.
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
  block_size: 5             # 1 => synchronous mode; >1 => mempool mode
  T: 750                    # number of blocks
  seed: 7
  cex_mu: 0.0
  cex_sigma: 0.0015
  p_trade: 0.7
  noise_floor: 0.5
  p_lp_narrow: 0.95
  p_lp_wide: 0.7
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
python run.py --config abm_mempool_config.yml
```

Outputs:
- `abm_results/verbose_steps_<scenario>.txt`: human-readable log per step and mempool replay summaries.
- `abm_results/png/` & `abm_results/html/`: figures summarizing prices, liquidity, agent PnLs, and fee path.
- Optional `abm_results/liquidity_evolution_<fee_mode>.gif` if `make_liquidity_gif` is enabled.
- JSON-like dict returned by `simulate` (see tail of `run.py` for exact keys).

---

For further questions or ideas, open an issue or start a discussion in this repository. Happy simulating!

