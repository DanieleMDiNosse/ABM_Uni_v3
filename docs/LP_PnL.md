---
title: LP PnL
nav_order: 5
---

# PnL Measurement for Uniswap v3 Liquidity Providers

This note summarizes how to define and interpret **unhedged** and **hedged** PnL for Uniswap v3 LPs, and how these objects relate to **fees**, **LVR** and **impermanent loss**.

The goal is to use a PnL framework that:
- is **correct for Uniswap v3** (fees in both tokens, concentrated liquidity),
- is suitable for an **ABM with external CEX prices**,  
- separates **market risk** (beta) from **liquidity-provision economics** (fees vs adverse selection / LVR).

---

## Implementation conventions (this repo)

The definitions below are generic, but `scripts/run.py` uses the following concrete conventions:

- Tokens: `X = token0` (“ETH-like”), `Y = token1` (“USDC-like”).
- Numéraire: all values are reported in token1 units (`Y`).
- External (CEX) price: `m_t = ref.m` is the CEX mid in `Y per X`.
- DEX price state: `S_t = pool.S` (sqrt-price), so `P^{DEX}_t = pool.price = S_t^2` is also `Y per X`.
- LP wealth mark-to-market: `V^{LP}_t` is computed by `lp_wealth_y(lp, S_t, m_t)` (wallet + open positions, incl. uncollected fees).

---

## 1. Uniswap v3 LP position: notation and mark-to-market

Consider a v3 pool with two tokens, which we call **X** and **Y**.  
Let all values be expressed in a chosen **numéraire** (e.g. USDC).

At time $t$:

- External (CEX) mid price (in units of `Y per X`):
  $$
  m_t
  $$
- LP’s **principal (liquidity) balances** in the pool:
  $$
  x_t \text{ units of X}, \qquad y_t \text{ units of Y}.
  $$
- LP’s **uncollected fee balances** on those open positions:
  $$
  f^X_t \text{ units of X}, \qquad f^Y_t \text{ units of Y}.
  $$
- LP’s **wallet balances** held outside the pool:
  $$
  w^X_t \text{ units of X}, \qquad w^Y_t \text{ units of Y}.
  $$

We define:

1. **Mark-to-market value of the principal (liquidity) only**
   $$
   V^{\text{liq}}_t := x_t m_t + y_t.
   $$

2. **Mark-to-market value of uncollected fees**
   $$
   F^{\text{uncol}}_t := f^X_t m_t + f^Y_t.
   $$

3. **Open-position value (principal + uncollected fees)**
   $$
   V^{\text{pos}}_t := V^{\text{liq}}_t + F^{\text{uncol}}_t.
   $$

4. **Wallet value**
   $$
   W_t := w^X_t m_t + w^Y_t.
   $$

5. **Total LP wealth (wallet + open positions)**
   $$
   V^{\text{LP}}_t := W_t + V^{\text{pos}}_t.
   $$

At the initial time $t=0$:

$$
V^{\text{liq}}_0 = x_0 m_0 + y_0, 
\quad
F^{\text{uncol}}_0 = f^X_0 m_0 + f^Y_0,
\quad
V^{\text{LP}}_0 = W_0 + V^{\text{liq}}_0 + F^{\text{uncol}}_0.
$$

In many v3 setups, fees start at zero:
$$
f^X_0 = f^Y_0 = 0, \quad F^{\text{uncol}}_0 = 0,
\quad \Rightarrow \quad V^{\text{LP}}_0 = W_0 + V^{\text{liq}}_0.
$$

---

## 2. Unhedged PnL: total economic outcome

The **unhedged PnL** is simply the change in total mark-to-market value of the LP position:

$$
\boxed{
\text{PnL}^{\text{LP}}_t := V^{\text{LP}}_t - V^{\text{LP}}_0.
}
$$

Replacing with the definitions:

$$
\text{PnL}^{\text{LP}}_t 
= V^{\text{LP}}_t - V^{\text{LP}}_0.
$$

If you prefer to think in terms of “wallet”, “mark-to-market of liquidity”, and “uncollected fees”, you can rewrite:

- **Mark-to-market of liquidity (principal)**:
  $$
  \text{MtM}^{\text{liq}}_t := V^{\text{liq}}_t = x_t m_t + y_t.
  $$

Then:

$$
\boxed{
\text{PnL}^{\text{LP}}_t 
= W_t + \text{MtM}^{\text{liq}}_t + F^{\text{uncol}}_t - V^{\text{LP}}_0.
}
$$

This matches the simulator’s accounting: LP wealth is “wallet + open positions” marked at the CEX mid.

### Interpretation


$\text{PnL}^{\text{LP}}_t$ includes:

- pure **price moves** (beta exposure to X and Y),
- the effect of **being in an AMM**:
  - trading at possibly stale prices versus CEX,
  - plus **fees** earned from all swaps.

This is the **full economic outcome** for the LP.  
In the ABM, this is the first quantity you should always compute and store.

---

## 3. Rebalancing benchmark and Loss-versus-Rebalancing (LVR)

To understand how much of the LP’s PnL comes from **liquidity provision** vs **market risk**, we compare the LP to a **rebalancing strategy** that:

- holds the **same target exposure path** as the LP (same “delta” as the pool),
- but always trades at **fair external prices** (CEX mid), not at the AMM price.

### 3.1. Rebalancing strategy

In `scripts/run.py`, the benchmark is implemented as a **self-financing delta-hedge** at the CEX price $m_t$.

Define the LP’s *token0 exposure* (in token units) at the current pool state:
$$
x^{\text{target}}_t
:=
\texttt{lp\_token0\_exposure}(lp, S_t)
$$
which (by design) includes:
- token0 principal inside open positions,
- uncollected token0 fees,
- plus any token0 held in the LP wallet (typically zero for strategic LPs in this repo).

The benchmark holds:
- $x^{\text{prev}}_t$ units of token0, and
- a cash account in token1,
and is initialized to match the LP’s initial wealth:

$$
V^{\text{reb}}_0 = V^{\text{LP}}_0.
$$

Whenever the CEX price moves $m \to m'$, the benchmark accrues (in token1):
$$
\Delta R = x^{\text{prev}} (m' - m),
\qquad
R_t = \sum \Delta R.
$$
In code this accrual happens on **every** diffusion/impact update via `_broadcast_price_move(M_new)`.

Whenever the LP’s exposure changes (mint/burn, or a swap moves the pool price across the LP’s range), the benchmark *rebalances* by setting:
$$
x^{\text{prev}}_t \leftarrow x^{\text{target}}_t
$$
at the current CEX price $m_t$ (adjusting the cash account accordingly, with no immediate PnL).

With this self-financing construction:

$$
\text{PnL}^{\text{reb}}_t = V^{\text{reb}}_t - V^{\text{reb}}_0 = R_t,
\qquad
V^{\text{reb}}_t = V^{\text{reb}}_0 + R_t.
$$

This benchmark carries (by construction) the same *token0 price exposure* as the LP, while abstracting away AMM execution effects.

### 3.2. LVR: loss from adverse selection

Define **Loss-Versus-Rebalancing (LVR)**, denoted $\text{LVR}_t$, as the cumulative loss the LP suffers from being forced to trade at mispriced AMM quotes instead of at fair CEX prices, along the entire path up to time $t$.

Also define the **cumulative fees earned** (realized + uncollected), valued in token1 at the current CEX mid:
$$
F^{\text{cum}}_t := f^{X,\text{cum}}_t \, m_t + f^{Y,\text{cum}}_t.
$$
In the implementation, $f^{X,\text{cum}}_t, f^{Y,\text{cum}}_t$ correspond to the LP’s cumulative fee counters (`fees0_earned`, `fees1_earned`).

The key result (in the continuous-time / frictionless framework) is:

$$
\boxed{
V^{\text{LP}}_t
= V^{\text{reb}}_t + F^{\text{cum}}_t - \text{LVR}_t.
}
$$

Rearranging:

$$
V^{\text{LP}}_t - V^{\text{reb}}_t = F^{\text{cum}}_t - \text{LVR}_t.
$$

This decomposition says:

- Start from what you’d have with a purely **rebalancing trader** $(V^{\text{reb}}_t)$,
- Then add **fees** (a positive contribution for LPs),
- Then subtract **LVR** (a negative contribution capturing adverse selection / execution loss versus CEX).

In an idealized setting with only arbitrage flow and no external frictions, LVR is closely related to value captured by arbitrageurs. In this ABM, LVR is computed from LP wealth vs the benchmark and should be interpreted as a **residual adverse-selection term**, not as “arb PnL” one-for-one.

---

## 4. Hedged PnL: isolating liquidity-provision economics

To remove market risk and focus on the **economics of liquidity provision**, define the **hedged PnL** as:

$$
\text{PnL}^{\text{hedged}}_t
:= \text{PnL}^{\text{LP}}_t - \text{PnL}^{\text{reb}}_t.
$$

Using the initial condition $V^{\text{reb}}_0 = V^{\text{LP}}_0$,

$$
\text{PnL}^{\text{hedged}}_t
= (V^{\text{LP}}_t - V^{\text{LP}}_0) - (V^{\text{reb}}_t - V^{\text{reb}}_0)
= V^{\text{LP}}_t - V^{\text{reb}}_t.
$$

Now plug in the LVR decomposition:

$$
V^{\text{LP}}_t = V^{\text{reb}}_t + F^{\text{cum}}_t - \text{LVR}_t
\quad \Rightarrow \quad
V^{\text{LP}}_t - V^{\text{reb}}_t = F^{\text{cum}}_t - \text{LVR}_t.
$$

Therefore:

$$
\boxed{
\text{PnL}^{\text{hedged}}_t 
= V^{\text{LP}}_t - V^{\text{reb}}_t
= F^{\text{cum}}_t - \text{LVR}_t.
}
$$

### Why does mark-to-market disappear here?

The **mark-to-market (principal) component** appears in both $V^{\text{LP}}_t$ and $V^{\text{reb}}_t$.  
By construction, the benchmark is continuously rebalanced to match the LP’s **token0 exposure path** $x^{\text{target}}_t$ (including uncollected token0 fees), and all valuation uses the same external price $m_t$.

- The **pure price risk** (beta) is **shared** by both strategies.
- When we take the difference $V^{\text{LP}}_t - V^{\text{reb}}_t$,
  the **beta effect cancels**.
- What remains is the **net liquidity-provision PnL**:
  $$
  \text{PnL}^{\text{hedged}}_t = \underbrace{F^{\text{cum}}_t}_{\text{fees}} - \underbrace{\text{LVR}_t}_{\text{adverse selection}}.
  $$

So:
- **Unhedged PnL** answers:  
  *“How much money did the LP make or lose in total?”*
- **Hedged PnL** answers:  
  *“Given the same market exposure (delta path), how much did the LP earn from providing liquidity, net of adverse selection?”*

This is exactly what you want for studying **MEV, adverse selection, and fee design**.

---

## 5. Impermanent Loss (IL) vs LVR

For completeness, recall the standard **impermanent loss** definition.

Let the LP’s initial **principal** token amounts be:

$$
x^{\text{init}} := x_0,
\quad
y^{\text{init}} := y_0.
$$

Define a **HODL strategy** that just keeps these tokens in a wallet, never trading:

$$
V^{\text{HODL}}_t := x^{\text{init}} m_t + y^{\text{init}}.
$$

Then **impermanent loss on principal** at time $t$ is:

$$
\boxed{
\text{IL}_t := V^{\text{liq}}_t - V^{\text{HODL}}_t.
}
$$

Interpretation:

- $\text{IL}_t$ measures how much better or worse the LP is compared to **just HODLing** the initial tokens.
- It mixes:
  - **market-direction risk** (price movements of X, Y),
  - and **AMM-specific effects** (fees, slippage, adverse selection).

### Why you should use LVR instead of IL for adverse selection

- IL answers: *“Was I better off than just HODLing?”*  
  That’s a useful practitioner quantity, but it does **not cleanly separate market risk from adverse selection.**
- LVR + hedged PnL answer: *“Given the same exposure path, what did liquidity provision earn me net of adverse selection?”*  
  This is exactly what matters for your ABM focus on **negative externalities of adverse selection and MEV**.

You generally do **not** want to subtract both IL and LVR anywhere in PnL calculations – they are two different comparisons to two different benchmarks (HODL vs rebalancing).


The pair **(unhedged PnL, hedged PnL)** gives you:

- the **full economic outcome** for the LPs, and
- a clean measure of how your mechanisms (dynamic fees, MEV protection, etc.) **tame adverse selection** and **shift the balance between fees and LVR**.

---

## 6. How the Simulator Reports These Quantities

`scripts/run.py` keeps LP wealth self-consistent by debiting wallets on mint, crediting full value (principal + fees) on burn, and delta-hedging via `RebalancerState`. The returned series map directly to the objects above:

- `lp_wallet_series`, `lp_wallet_active_series`, `lp_wallet_passive_series`: realized token1 wallet after mints/burns.
- `lp_wealth_series` (+ active/passive splits): wallet + mark-to-market of open positions, i.e., $V^{\text{LP}}_t$.
- `lp_fee_value_*_series`: cumulative fees *earned* (realized + uncollected), marked to the CEX price.
- `lp_unhedged_*`: unhedged PnL $V^{\text{LP}}_t - V^{\text{LP}}_0$.
- `lp_rebal_value_*_series` and `lp_rebal_*_series`: rebalancing benchmark value $V^{\text{reb}}_t$ and its PnL path.
- `lp_pnl_*` (hedged) = $F^{\text{cum}}_t - \text{LVR}_t$; `lp_lvr_*` = $\text{LVR}_t$.

Notes (matching current `scripts/run.py`):
- `lp_pnl_*` is computed as $\sum_i (V^{LP,i}_t - V^{reb,i}_t)$ at end-of-block, where `V^{reb,i}_t = initial_rebal_value_y + cumulative_R`.
- `lp_lvr_*` is computed as the identity `lp_fee_value_*_series - lp_pnl_*` (so the decomposition holds by construction), and it should not be expected to match `arb_pnl_*` exactly (flash fees, CEX impact, and settlement conventions differ).
- In `light_mode: true`, most recorder series (including LP PnL/LVR) are disabled and returned as empty lists.
- As of the current implementation, `lp_wallet_*_series` and `lp_wealth_*_series` are declared/returned but not populated inside the main loop (they will be empty). You can reconstruct cohort total wealth as `lp_pnl_total[t] + lp_rebal_value_total_series[t]` (and analogously for active/passive).

All series are split into total/active/passive cohorts (seed LPs are excluded; Jiter has its own `jiter_*` series) and are consumed by batch runners such as `scripts/run_multiple.py`, `scripts/run_parameter_surface_nd_pnl_fee_dashboard.py`, and `scripts/build_parameter_surface_nd_pnl_fee_dashboard.py`.
