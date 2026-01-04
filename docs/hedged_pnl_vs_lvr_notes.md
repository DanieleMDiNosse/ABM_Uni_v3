---
title: Hedged PnL vs LVR
nav_order: 5
---

# Hedged PnL vs LVR in the ABM Uni v3 Simulator

This note explains why, under the **current model and parameter choices**, it is difficult to obtain **positive hedged PnL** for passive LPs (i.e. `fees > LVR`), and sketches realistic strategies that could move the system toward `fees − LVR > 0`.

It is meant to complement:

- `LP_PnL.md` (PnL definitions),
- `LVR_explanation.md` (rebalancing benchmark implementation),
- `fee_schedules.md` (dynamic fee controller),
- `README.md` (agent and microstructure overview).

---

## 1. Recap: What Hedged PnL Measures

From `LP_PnL.md` and `LVR_explanation.md`:

- Let $V_t^{LP}$ be LP wealth (wallet + mark‑to‑market of open positions) at CEX price $m_t$
- Let $V_t^{reb}$ be the value of the **rebalancing benchmark** that:
  - holds the same token0 exposure path as the LP,
  - but always trades at the **CEX mid** instead of the DEX price.
- Let $F_t$ be cumulative fees (token0/1 fees valued at $m_t$)
- Let $\text{LVR}_t$ be **loss‑versus‑rebalancing**, i.e. the cumulative adverse selection cost of trading at stale/mispriced DEX quotes instead of at the CEX.

The theory (and the simulator) enforce

$$
V_t^{LP} = V_t^{reb} + F_t - \text{LVR}_t,
$$

so that

$$
\text{PnL}^{\text{hedged}}_t
= V_t^{LP} - V_t^{reb}
= F_t - \text{LVR}_t.
$$

Thus:

- **Unhedged PnL** = 
  $$
  V_t^{LP} - V_0^{LP}
  $$
   (full economic outcome, includes market beta).
- **Hedged PnL** = 
  $$
  F_t - \text{LVR}_t
  $$
   (liquidity‑provision economics only).

You can never improve hedged PnL by taking more price risk; only the balance between **fees collected** and **adverse selection costs** matters.

---

## 2. Current Microstructure: Why Adverse Selection Dominates

The simulator implements a fairly harsh environment for passive LPs:

### 2.1 Reference Market and Volatility

- The CEX price is a GBM (`ReferenceMarket` in `utils.py`) with per‑second volatility `cex_sigma` (or regime switching between `cex_sigma_low`/`high`).
- For realistic values (e.g. `cex_sigma_low=1e-4`, `cex_sigma_high=2e-3`), the **continuous‑time LVR formulas** in Milionis–Moallemi–Roughgarden imply a strictly positive instantaneous LVR rate even *before* modeling any microstructure frictions.
- Intuitively: whenever price diffuses, an LP in a convex AMM *must* trade “against” the drift of informed prices on average, which is exactly what LVR captures.

### 2.2 Arbitrageur: Nearly Frictionless and Always Present

- The arbitrageur (`arbitrage_to_target` plus arb branches in `run.py`) sees a **validated snapshot** of the CEX price and trades the pool back into a **no‑arb band**:
  - lower bound: 
$$
    m_t (1 - f_t)
    $$
  - upper bound: 
    $$
    m_t / (1 - f_t)
    $$
  where $f_t$ is the current taker fee.
- Without extra costs, this arb is:
  - perfectly myopic and well‑capitalized,
  - permitted to trade every block,
  - executed *before* other mempool flow in block mode.
- Even with the optional `flash_loan_fee`, the arb only skips when **total profit < liquidity‑taker fee + flash fee**. Whenever the mispricing is big enough, it executes and realizes gains that, in the frictionless theory, are exactly LVR for the LPs.

Effectively, you have **continuous monitoring in discrete time**: any significant mispricing is rapidly harvested.

### 2.3 Smart Router: Best Execution Against CEX

- The smart router agent (`execute_trader` in `run.py`) enforces:
  - best execution vs the CEX mid (`theta_T` threshold),
  - a slippage window vs a baseline quote (to avoid very bad DEX fills).
- This means:
  - When the DEX is marginally *worse* than the CEX, smart flow **routes to the CEX** instead of the DEX; the CEX leg contributes to reference-price impact and smart-router PnL, but does not interact with AMM liquidity.
  - When the DEX is marginally *better*, smart flow will happily trade on the DEX and *harvest* favorable quotes.

So the *informed* or “non‑noise” trader flow hitting the AMM is **systematically tilted** toward states where the DEX is attractive for traders and unattractive for LPs.

### 2.4 Noise Trader: Adds Volume but Not “Free” Fees

- The noise trader submits random orders with no valuation discipline, governed by `noise_trades_per_block`, lognormal size parameters, and the same slippage checks.
- Noise trades:
  - do pay fees and thus boost $F_t$
  - but they also move price away from the CEX and create mispricings, which the arbitrageur then clears.
- Because LVR is path‑dependent, *every* cycle “noise trade → mispricing → arb” contributes positive LVR, and only a fraction of the noise trades generate fees large enough to counter it.

### 2.5 LPs: Always Exposed, Without Information Edge

- Passive LPs:
  - hold finite-width ranges (e.g. 2–10% bands in your configs),
  - are present across most of the time horizon (seed binomial hill from background LPs with `is_seed=True` + ongoing mints by the strategic passive cohort),
  - enter and exit positions according to **simple probabilistic rules** tied to review clocks (`tau`): when a passive LP’s review clock fires, it may mint a new wide range with probability derived from `passive_mints_per_block`, and may randomly burn one of its existing ranges with probability derived from `passive_burns_per_block`. Passive LPs never recenter based on out‑of‑range thresholds and do **not** use TP/SL (`theta_TP`, `theta_SL`); those thresholds apply only to active narrow LPs.
- They **do not control**:
  - when informed flow arrives,
  - where the CEX price is diffusing,
  - or when arb jumps in.

Structurally, they are exactly on the wrong side of:

1. Noise moves that create mispricings,
2. Arbitrage that removes those mispricings,
3. Smart flow that selectively trades when the DEX is favorable.

### 2.6 Dynamic Fee Controller: Mostly Reactive

- `fee_mode: volatility` uses an EWMA of absolute log‑returns to push fees up when $|\log m_t - \log m_{t-1}|$ is large.
- `fee_mode: toxicity` uses an EWMA of **fee‑adjusted basis** (DEX–CEX log gap) in ticks.
- Both modes are:
  - **lagged** (EWMA with half‑life over multiple blocks),
  - **step‑capped** (`fee_step_bps_max`),
  - and optionally throttled (`fee_cooldown`).

In practice, this means:

- LVR often starts increasing as volatility or basis widens **before** the controller has time to raise fees.
- Once fees rise enough to protect LPs, the smart router routes away a lot of competitive flow; arb frequency may fall, but so does fee income.

The controller is “fair” in the sense of responding to observed toxicity, but it rarely gets ahead of LVR in a way that would make $F_t - \text{LVR}_t$ positive on average.

---

## 3. Why Hedged PnL Is Hard to Make Positive

Putting Sections 1–2 together:

1. **LVR is structurally ≥ 0** whenever there is any mispricing between DEX and CEX. In continuous time with GBM prices and frictionless arb, Milionis et al. show LVR per unit time is strictly positive for any nonzero volatility.
2. In your simulation:
   - Arbitrage is nearly frictionless and “fast block” (blocks are short, and arb sees every block).
   - Smart flow is best‑ex against the CEX.
   - Noise flow both generates and then indirectly amplifies LVR via arb.
3. Fees are:
   - charged on all executed trades,
   - but partially self‑defeating: once they become large enough, they deter flow and reduce volume.

Under these conditions, the model is very close to the **idealized LVR upper bound**: the LP pays almost the full continuous‑time adverse selection cost, and there is relatively little “dumb” flow or structural friction to subsidize them.

As a result, for passive LPs you empirically observe:

- $F_t$ increasing over time but
- $\text{LVR}_t$ increasing at least as fast,
- so that $F_t - \text{LVR}_t \leq 0$ on most long runs.

This is not a coding bug; it is the expected outcome given:

- efficient arbitrage,
- best‑execution smart flow,
- and realistic volatility.

In other words, you have built exactly the world that the LVR theory warns **is bad for LPs**, and the simulator is faithfully reporting that.

---

## 4. What Would Be Needed to Achieve `fees > LVR`?

If you want positive hedged PnL for passive LPs in this framework, you need to change the **economics**, not the accounting. Broadly, you need at least one of:

1. More **inelastic/uninformed flow** that pays fees without generating much LVR.
2. Less **efficient arbitrage** so that realized LVR is strictly below its frictionless bound.
3. A more **aggressive or anticipatory fee schedule** that charges substantially more during the most toxic states.

Below are concrete levers within your simulator.

### 4.1 Increase Uninformed or Execution‑Lazy Flow

Goal: create volume that pays fees but does not systematically trade at mispriced quotes.

Possible changes:

- Increase `noise_trades_per_block` relative to `smart_trades_per_block` so a larger fraction of trades are noise.
- Allow a fraction of smart flow to be “execution‑lazy”:
  - use a looser `theta_T` for some trader cohort,
  - or disable best‑ex checks for a subset of flow,
  so they still route to the DEX even when the CEX is slightly better.
- Introduce “sticky venue preference”:
  - some traders always use the DEX up to a moderate slippage bound,
  - even if the CEX is marginally better.

These changes are meant to mimic real‑world behaviour where some order flow is *not* perfectly price‑sensitive.

### 4.2 Soften the Arbitrageur

Goal: reduce realized LVR by making arbitrage less than perfectly efficient.

You have already added `flash_loan_fee` (per‑notional cost); additional options:

- Add a **fixed gas cost** or minimum profit threshold per arb:
  - skip arb trades whose expected profit is below that threshold.
  - this eliminates many small LVR events but leaves most fees unchanged.
- Limit arb capital or per‑block notional:
  - arb only partially clears large mispricings within a block,
  - leaving some residual “cheap” states that can be exploited by other flow or future LP repositioning.
- Randomize arb arrival:
  - in some blocks, the arb simply doesn’t show up,
  - or arrives after traders rather than before.

All of these bring realized LVR **below** its frictionless theoretical value, giving fees a chance to outrun it.

### 4.3 More Toxicity‑Aware and Aggressive Fees

Goal: ensure that when mispricing is large (high LVR states), fees are *very* high.

Options within the current controller:

- Prefer `fee_mode: toxicity` over pure `volatility` in high‑vol regimes:
  - toxicity mode uses an EWMA of excess log‑basis (beyond the fee band),
  - which is more closely tied to states where LVR is large.
- Shorten `fee_half_life` so the controller responds within a few blocks, not tens of blocks.
- Increase:
  - baseline `f0`,
  - lower bound `f_min`,
  - and upper bound `f_max` (e.g. allow realistic 5–100 bps tiers).
- Reduce or remove `fee_cooldown` unless you observe unhealthy oscillations.

The tradeoff: very high fees will push smart flow away and may reduce inelastic flow too; you probably need to tune these jointly with 4.1.

### 4.4 LP Design and Risk Profile

Goal: make “passive LPs” look more like fee‑earning HODLers and less like aggressive market makers.

Ideas:

- Use **wider ranges** for passive LPs (hundreds–thousands of ticks) so they behave closer to HODL with fee income.
- Reduce passive churn:
  - lower `passive_burns_per_block`,
  - lengthen `tau` (review interval).
- For **active narrow LPs** (which do use TP/SL), adjust `theta_TP`/`theta_SL` to avoid frequent rebalance events that realize LVR; for purely passive baselines you typically keep these thresholds high and rely on the probabilistic burn rule instead.
- Consider separating:
  - very wide “baseline” LPs (fee‑earning, low LVR per unit),
  - from narrower, more active LPs whose hedged PnL you do *not* expect to be positive but that improve pool quality.

### 4.5 Market Environment and Volatility Structure

Goal: operate in regimes where theoretical LVR is smaller relative to fee income.

You can:

- Lower overall `cex_sigma` or spend more time in the low‑σ regime in `cex_sigma_mode: regime`.
- Introduce **mean reversion** in the CEX process:
  - more back‑and‑forth price action around a central value,
  - which can, for some flow mixes, lead to higher fee income without proportionally raising LVR.

These are more speculative and model‑dependent, but they are levers you already have in the GBM and regime‑switching setup.

---

## 5. Practical Path Forward

To systematically explore “can I make hedged PnL positive?”:

1. Define a small set of **target scenarios** (e.g., low‑vol / high‑noise / frictional arb).
2. For each scenario, use `run_parameter_grid_2d_violin_parallel.py` (2D sweeps) or `run_parameter_surface_nd_pnl_fee_dashboard.py` (multi-parameter sweeps) to sweep over, then render the HTML with `build_parameter_surface_nd_pnl_fee_dashboard.py`:
   - `noise_trades_per_block`, `smart_trades_per_block`, `theta_T`,
   - `f0`, `f_min`, `f_max`, `k_sigma`, `k_basis`, `fee_half_life`,
   - `flash_loan_fee` or additional arb costs.
3. Track, for passive LPs:
   - `lp_fee_value_passive` (fees),
   - `lp_lvr_passive` (LVR),
   - `lp_pnl_passive` (hedged PnL = fees − LVR).
4. Identify regions where the **mean** of `lp_pnl_passive` over many seeds is:
   - close to zero (LPs roughly break‑even on liquidity provision),
   - or slightly positive (LPs net beneficiaries of the flow mix).

If, even after softening arbitrage and injecting more inelastic flow, hedged PnL stays strongly negative, that is valuable evidence supporting the pessimistic LVR story: in a world of mostly informed flow and efficient arb, structural LP losses are hard to avoid, regardless of fee schedule.

This simulator is well‑suited to explore exactly that frontier. The challenge is not a missing term in the accounting, but finding realistic mechanisms and parameter regimes where `fees > LVR` can hold without assuming obviously unrealistic flow behaviour. 
