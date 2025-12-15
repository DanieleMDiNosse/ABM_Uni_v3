---
title: LP PnL
nav_order: 3
---

# PnL Measurement for Uniswap v3 Liquidity Providers

This note summarizes how to define and interpret **unhedged** and **hedged** PnL for Uniswap v3 LPs, and how these objects relate to **fees**, **LVR** and **impermanent loss**.

The goal is to use a PnL framework that:
- is **correct for Uniswap v3** (fees in both tokens, concentrated liquidity),
- is suitable for an **ABM with external CEX prices**,  
- separates **market risk** (beta) from **liquidity-provision economics** (fees vs adverse selection / LVR).

---

## 1. Uniswap v3 LP position: notation and mark-to-market

Consider a v3 pool with two tokens, which we call **X** and **Y**.  
Let all values be expressed in a chosen **numéraire** (e.g. USDC).

At time $t$:

- External (CEX) prices:
  \[
  P^X_t, \quad P^Y_t
  \]
- LP’s **principal (liquidity) balances** in the pool:
  \[
  x_t \text{ units of X}, \qquad y_t \text{ units of Y}.
  \]
- LP’s **fee balances** (uncollected or notionally in the “fee bucket”):
  \[
  f^X_t \text{ units of X}, \qquad f^Y_t \text{ units of Y}.
  \]

We define:

1. **Mark-to-market value of the principal (liquidity) only**
   \[
   V^{\text{liq}}_t := x_t P^X_t + y_t P^Y_t.
   \]

2. **Mark-to-market value of fees**
   \[
   F_t := f^X_t P^X_t + f^Y_t P^Y_t.
   \]

3. **Total LP position value (principal + fees)**
   \[
   V^{\text{LP}}_t := V^{\text{liq}}_t + F_t.
   \]

At the initial time $t=0$:

\[
V^{\text{liq}}_0 = x_0 P^X_0 + y_0 P^Y_0, 
\quad
F_0 = f^X_0 P^X_0 + f^Y_0 P^Y_0,
\quad
V^{\text{LP}}_0 = V^{\text{liq}}_0 + F_0.
\]

In many v3 setups, fees start at zero:
\[
f^X_0 = f^Y_0 = 0, \quad F_0 = 0,
\quad \Rightarrow \quad V^{\text{LP}}_0 = V^{\text{liq}}_0.
\]

---

## 2. Unhedged PnL: total economic outcome

The **unhedged PnL** is simply the change in total mark-to-market value of the LP position:

\[
\boxed{
\text{PnL}^{\text{LP}}_t := V^{\text{LP}}_t - V^{\text{LP}}_0.
}
\]

Replacing with the definitions:

\[
\text{PnL}^{\text{LP}}_t 
= (V^{\text{liq}}_t + F_t) - V^{\text{LP}}_0.
\]

If you prefer to think in terms of “mark-to-market of liquidity” and “fees” separately, you can rewrite:

- **Mark-to-market of liquidity (principal) at time $t$**:
  \[
  \text{MtM}^{\text{liq}}_t := V^{\text{liq}}_t = x_t P^X_t + y_t P^Y_t.
  \]

Then:

\[
\boxed{
\text{PnL}^{\text{LP}}_t 
= \text{MtM}^{\text{liq}}_t + F_t - V^{\text{LP}}_0.
}
\]

This is exactly the informal description:

> **PnL = mark-to-market (liquidity) + fees − initial value**

provided that “mark-to-market” refers to the value of the **principal** only.

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

Let $(x_t, y_t)$ be the LP’s pool balances at each time $t$.  
Define a **rebalancing portfolio** that continuously adjusts its holdings to match this path, but trades at $P^X_t, P^Y_t$:

- Value of the rebalancing strategy: $V^{\text{reb}}_t$

We choose the initial condition so that:

\[
V^{\text{reb}}_0 = V^{\text{LP}}_0.
\]

Then its PnL is:

\[
\text{PnL}^{\text{reb}}_t := V^{\text{reb}}_t - V^{\text{reb}}_0.
\]

This strategy carries exactly the **same market exposure** as the LP, but without adverse selection or fee income from AMM trades.

### 3.2. LVR: loss from adverse selection

Define **Loss-Versus-Rebalancing (LVR)**, denoted $\text{LVR}_t$, as the cumulative loss the LP suffers from being forced to trade at mispriced AMM quotes instead of at fair CEX prices, along the entire path up to time $t$.

The key result (in the continuous-time / frictionless framework) is:

\[
\boxed{
V^{\text{LP}}_t
= V^{\text{reb}}_t + F_t - \text{LVR}_t.
}
\]

Rearranging:

\[
V^{\text{LP}}_t - V^{\text{reb}}_t = F_t - \text{LVR}_t.
\]

This decomposition says:

- Start from what you’d have with a purely **rebalancing trader** $(V^{\text{reb}}_t)$,
- Then add **fees** (a positive contribution for LPs),
- Then subtract **LVR** (a negative contribution, equal to arbitrageur profits coming from adverse selection).

LVR is closely related to arbitrage profits between CEX and AMM: in many settings, **sum of arbitrage profits = LVR** (with opposite sign for LPs).

---

## 4. Hedged PnL: isolating liquidity-provision economics

To remove market risk and focus on the **economics of liquidity provision**, define the **hedged PnL** as:

\[
\text{PnL}^{\text{hedged}}_t
:= \text{PnL}^{\text{LP}}_t - \text{PnL}^{\text{reb}}_t.
\]

Using the initial condition $V^{\text{reb}}_0 = V^{\text{LP}}_0$,

\[
\text{PnL}^{\text{hedged}}_t
= (V^{\text{LP}}_t - V^{\text{LP}}_0) - (V^{\text{reb}}_t - V^{\text{reb}}_0)
= V^{\text{LP}}_t - V^{\text{reb}}_t.
\]

Now plug in the LVR decomposition:

\[
V^{\text{LP}}_t = V^{\text{reb}}_t + F_t - \text{LVR}_t
\quad \Rightarrow \quad
V^{\text{LP}}_t - V^{\text{reb}}_t = F_t - \text{LVR}_t.
\]

Therefore:

\[
\boxed{
\text{PnL}^{\text{hedged}}_t 
= V^{\text{LP}}_t - V^{\text{reb}}_t
= F_t - \text{LVR}_t.
}
\]

### Why does mark-to-market disappear here?

The **mark-to-market (principal) component** appears in both 
$V^{\text{LP}}_t$ and $V^{\text{reb}}_t$.  
By construction, the rebalancing strategy holds the **same path of token balances** $(x_t, y_t)$ as the LP (up to fees), and trades at the same external prices for those holdings.

- The **pure price risk** (beta) is **shared** by both strategies.
- When we take the difference $V^{\text{LP}}_t - V^{\text{reb}}_t$,
  the **beta effect cancels**.
- What remains is the **net liquidity-provision PnL**:
  \[
  \text{PnL}^{\text{hedged}}_t = \underbrace{F_t}_{\text{fees}} - \underbrace{\text{LVR}_t}_{\text{adverse selection}}.
  \]

So:
- **Unhedged PnL** answers:  
  *“How much money did the LP make or lose in total?”*
- **Hedged PnL** answers:  
  *“Given the same market exposure (delta path), how much did the LP earn from providing liquidity, net of adverse selection?”*

This is exactly what you want for studying **MEV, adverse selection, and fee design**.

---

## 5. Impermanent Loss (IL) vs LVR

For completeness, recall the standard **impermanent loss** definition.

Let the LP’s initial token amounts (principal + any initial fees) be:

\[
x^{\text{init}} := x_0 + f^X_0,
\quad
y^{\text{init}} := y_0 + f^Y_0.
\]

Define a **HODL strategy** that just keeps these tokens in a wallet, never trading:

\[
V^{\text{HODL}}_t := x^{\text{init}} P^X_t + y^{\text{init}} P^Y_t.
\]

Then IL at time $t$ is:

\[
\boxed{
\text{IL}_t := V^{\text{LP}}_t - V^{\text{HODL}}_t.
}
\]

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

`run.py` keeps LP wealth self-consistent by debiting wallets on mint, crediting full value (principal + fees) on burn, and delta-hedging via `RebalancerState`. The returned series map directly to the objects above:

- `lp_wallet_series`, `lp_wallet_active_series`, `lp_wallet_passive_series`: realized token1 wallet after mints/burns.
- `lp_wealth_series` (+ active/passive splits): wallet + mark-to-market of open positions, i.e., $V^{\text{LP}}_t$.
- `lp_fee_value_*_series`: fees marked to the CEX price.
- `lp_unhedged_*`: unhedged PnL $V^{\text{LP}}_t - V^{\text{LP}}_0$.
- `lp_rebal_value_*_series` and `lp_rebal_*_series`: rebalancing benchmark value $V^{\text{reb}}_t$ and its PnL path.
- `lp_pnl_*` (hedged) = $F_t - \text{LVR}_t$; `lp_lvr_*` = $\text{LVR}_t$.

All series are split into total/active/passive cohorts for easier cohort-level analysis and are consumed by the batch runners (`run_scenarios_mean_std.py`, `run_parameter_grid_mean_std_parallel.py`, etc.).
