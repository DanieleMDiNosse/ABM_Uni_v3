---
title: Loss-Versus-Rebalancing (LVR)
nav_order: 4
---

# Loss-Versus-Rebalancing (LVR)

## 1. What is LVR?

**Loss-Versus-Rebalancing (LVR)** is a metric introduced by Milionis, Moallemi, and Roughgarden (2023) to quantify the *structural* cost borne by liquidity providers in an Automated Market Maker (AMM).

The key idea is that AMMs are *passive*: they post a price that becomes stale whenever the
external market moves. Arbitrageurs then “pick off” the pool by trading until the AMM price
realigns with the external reference price. LVR measures the cumulative value extracted by
these arbitrage trades (equivalently: the opportunity cost of providing liquidity versus a
frictionless benchmark that trades at the external price).

### Intuition: the rebalancing benchmark

To isolate liquidity-provision economics from market risk, LVR compares the LP’s outcome to a
**rebalancing benchmark** with two defining properties:

- **Inventory matching:** at every instant $t$, the benchmark holds the same inventory of token0
  as the LP/AMM, denoted $x_t$.
- **Frictionless execution:** whenever $x_t$ changes (because trades move the pool along its curve),
  the benchmark executes that same inventory change at the *external* reference price $m_t$ (CEX mid),
  rather than at the stale AMM price $P^{DEX}_t$.

Because the benchmark executes the same sequence of inventory changes but at better prices
(sell at $m_t > P^{DEX}_t$ when price rises, buy at $m_t < P^{DEX}_t$ when price falls), its value
is higher. That “missing value” is LVR.

In much of the LVR literature (and in this repo’s accounting), it is convenient to keep **fees**
as an explicit separate term. Let:

- $V_t^{LP}$ be total LP wealth (wallet + mark-to-market of open positions) valued at $m_t$,
- $F_t$ be cumulative fees earned (in both tokens) valued at $m_t$,
- $V_t^{reb}$ be the value of the rebalancing benchmark.

Then the defining decomposition is:

$$
\boxed{
V_t^{LP} = V_t^{reb} + F_t - \mathrm{LVR}_t.
}
$$

Equivalently,

$$
\mathrm{LVR}_t = V_t^{reb} + F_t - V_t^{LP}.
$$

Note: if you define $V_t^{LP}$ *excluding* fees, then the simpler identity
$\mathrm{LVR}_t = V_t^{reb} - V_t^{LP}$ is recovered.

---

## 2. Continuous-time LVR formulas (theory / intuition)

Milionis et al. show that when the external reference price $m_t$ follows a diffusion with
instantaneous volatility $\sigma_t$, LVR accumulates according to the volatility and the AMM’s
**marginal liquidity**.

Over a time interval $[0,T]$:

$$
\mathrm{LVR}_{[0,T]}
= \int_0^T \frac{\sigma_t^2 m_t^2}{2}\, \left|x^{*\prime}(m_t)\right|\, dt,
$$

where:
- $x^*(p)$ is the AMM inventory of token0 as a function of price $p$ (the “demand curve”),
- $\left|x^{*\prime}(p)\right| = \left|\frac{dx}{dp}\right|$ is the *marginal liquidity*, i.e. how
  aggressively the AMM trades against price changes.

### Specialization to Uniswap v3 (within an active range)

In Uniswap v3, within a tick range where liquidity $L_t$ is active and with square-root price
$s=\sqrt{p}$, token0 inventory takes the form:

$$
x(s) = L_t\left(\frac{1}{s} - \frac{1}{s_b}\right),
$$

where $s_b$ is the upper square-root price bound of the active range.

Using $p=s^2$ and the chain rule:

$$
\left|\frac{dx}{dp}\right|
= \left|\frac{dx}{ds}\frac{ds}{dp}\right|
= \left|\left(-\frac{L_t}{s^2}\right)\left(\frac{1}{2\sqrt{p}}\right)\right|
= \frac{L_t}{2p\sqrt{p}}.
$$

Plugging into the general formula (with $p=m_t$) yields the *instantaneous* Uniswap v3 LVR rate:

$$
\boxed{
\ell_t^{v3}
= \frac{\sigma_t^2 L_t \sqrt{m_t}}{4}.
}
$$

Key intuition:
- LVR scales **linearly** in active liquidity $L_t$,
- **quadratically** in volatility $\sigma_t$,
- and increases with the price level through $\sqrt{m_t}$.

Crucially, because $L_t$ is *concentrated* liquidity, a position only incurs LVR when the
reference price lies inside its active tick range (when its active liquidity is $>0$).

---

## 3. How LVR is computed in this ABM (discrete-time benchmark)

The Agent-Based Model (ABM) computes LVR using the **Rebalancing Benchmark** approach. This is the most accurate way to measure LVR in a simulation because it captures the exact path-dependent losses from discrete trades and price jumps, rather than relying on a theoretical approximation (like $\sigma^2$).

### Rebalancing benchmark in discrete time

Fix an LP and index blocks by $t=0,1,2,\dots$. Let $m_t$ be the CEX mid at the **end of block**
$t$. Let:

- $x_t$ be the LP’s *total exposure in token0* at time $t$ (summed over all its AMM positions),
- $y_t$ be its cash holdings in token1.

The LP’s mark-to-market wealth is:

$$
V_t^{LP} = x_t m_t + y_t.
$$

We construct a benchmark portfolio $(x_t^{reb}, y_t^{reb})$ such that:
- it always holds the same token0 exposure as the LP: $x_t^{reb} = x_t$,
- it trades only at the CEX mid $m_t$,
- it is self-financing.

Initialize at $t=0$ with $V_0^{reb} = V_0^{LP}$.

Then update it in two ways:

1. **Price moves (passive holding).** When $m_{t-1} \to m_t$ while holding $x_{t-1}^{reb}$:
   $$
   V_t^{reb} = V_{t-1}^{reb} + x_{t-1}^{reb}(m_t - m_{t-1}).
   $$
2. **Rebalancing trades (active adjustments).** Whenever the LP exposure changes within block $t$
   from $x_{t-}^{LP}$ to $x_{t+}^{LP}$ due to AMM interactions (mint/burn, or swaps moving price
   within/outside ranges), the benchmark trades at $m_t$ to match:
   $$
   x_{t+}^{reb} = x_{t+}^{LP},
   \qquad
   y_{t+}^{reb} = y_{t-}^{reb} - (x_{t+}^{reb} - x_{t-}^{reb})m_t.
   $$

By construction, the benchmark value is always:

$$
V_t^{reb} = x_t^{reb} m_t + y_t^{reb}.
$$

### Mapping to the implementation (`scripts/run.py`)

The computation is handled by `RebalancerState` in `core/agents.py` and updated inside `scripts/run.py`.

The benchmark’s *price-move accrual* is implemented by `_accrue_price_move`:

```python
# scripts/run.py: _accrue_price_move
delta = M_new - rb.last_M
rb.cumulative_R += rb.x_prev * delta
rb.last_M = M_new
```

The benchmark’s *rebalancing trade* is implemented by `_rebalance_lp_to_target`:

```python
# scripts/run.py: _rebalance_lp_to_target
dx = x_target - rb.x_prev
rb.cash_y -= dx * M_now
rb.x_prev = x_target
```

The benchmark value used for PnL bookkeeping is:

$$
V_t^{reb} = V_0^{reb} + R_t
\quad\text{with}\quad
R_t := \sum_{k\le t} x_{k-1}^{reb}(m_k - m_{k-1}),
$$

which corresponds to:
```python
rebal_value_now = rb.initial_rebal_value_y + rb.cumulative_R
```

At end-of-block, the simulator computes:
- LP wealth (mark-to-market) $V_t^{LP}$ via `lp_wealth_y(lp, pool.S, ref.m)`,
- fee value $F_t$ via `lp_total_fee_earned_value_y(lp, ref.m)`,
- benchmark value $V_t^{reb}$ via `rb.initial_rebal_value_y + rb.cumulative_R`.

It then records the *hedged PnL*:

$$
\mathrm{PnL}^{hedged}_t
:=
V_t^{LP} - V_t^{reb}
= F_t - \mathrm{LVR}_t,
$$

so LVR is recovered as the identity:

$$
\boxed{
\mathrm{LVR}_t = F_t - \mathrm{PnL}^{hedged}_t.
}
$$

---

## 4. Outputs and diagnostics

The main per-block series are (each split into `total` / `active` / `passive` cohorts):

- `lp_fee_value_*_series`: $F_t$ (cumulative fees, valued at $m_t$).
- `lp_rebal_value_*_series`: $V_t^{reb}$ (benchmark value).
- `lp_rebal_*_series`: $R_t = V_t^{reb} - V_0^{reb}$ (benchmark PnL path).
- `lp_pnl_*`: hedged PnL $V_t^{LP} - V_t^{reb} = F_t - \mathrm{LVR}_t$.
- `lp_lvr_*_series`: LVR, computed as `lp_fee_value_*_series - lp_pnl_*`.
- `lp_unhedged_*`: unhedged PnL $V_t^{LP} - V_0^{LP}$.

These are the quantities typically plotted in the dashboards and consumed by batch runners such as
`scripts/run_multiple.py`, `scripts/run_parameter_surface_nd_pnl_fee_dashboard.py`, and
`scripts/build_parameter_surface_nd_pnl_fee_dashboard.py`.
