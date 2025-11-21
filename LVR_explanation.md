# Loss-Versus-Rebalancing (LVR)

## 1. What is LVR?

**Loss-Versus-Rebalancing (LVR)** is a metric introduced by Milionis, Moallemi, and Roughgarden (2023) to quantify the cost of providing liquidity in an Automated Market Maker (AMM). It represents the "adverse selection" cost incurred by Liquidity Providers (LPs) when arbitrageurs trade against the pool to align its price with an external market (CEX).

### Mathematical Definition

In a continuous-time setting, the instantaneous LVR is defined as:

$$
LVR_t = \frac{1}{2} \sigma_t^2 P_t
$$

Where:
*   $\sigma_t$ is the instantaneous volatility of the asset price.
*   $P_t$ is the price of the asset.

The cumulative LVR over a period $[0, T]$ is the integral of this instantaneous cost:

$$
LVR_{[0, T]} = \int_0^T \frac{1}{2} \sigma_t^2 P_t \, dt
$$

### Intuition: The Rebalancing Benchmark

LVR can be understood by comparing the LP's portfolio to a **Rebalancing Benchmark**.

*   **LP Portfolio ($V_t^{LP}$):** The value of the assets held in the AMM pool. This portfolio's composition changes automatically as the price moves (selling as price rises, buying as price falls).
*   **Rebalancing Benchmark ($V_t^{Rebal}$):** A hypothetical self-financing portfolio that replicates the LP's exposure (delta) but trades at the external market price (CEX). It holds the same amount of the risky asset ($x_t$) as the LP but executes trades at the "fair" CEX price rather than the "stale" AMM price.

The difference in value between these two portfolios (excluding fees collected by the LP) is exactly the LVR:

$$
V_T^{Rebal} - V_T^{LP} = LVR_{[0, T]} - \text{Fees}_{[0, T]}
$$

Or, rearranging to isolate LVR:

$$
LVR_{[0, T]} = (V_T^{Rebal} - V_T^{LP}) + \text{Fees}_{[0, T]}
$$

In the absence of fees, the LP always loses relative to the benchmark because the LP sells to arbitrageurs at a price lower than the CEX price (when price rises) or buys at a price higher than the CEX price (when price falls).

## 2. How LVR is Computed in the ABM

The Agent-Based Model (ABM) computes LVR using the **Rebalancing Benchmark** approach. This is the most accurate way to measure LVR in a simulation because it captures the exact path-dependent losses from discrete trades and price jumps, rather than relying on a theoretical approximation (like $\sigma^2$).

### Implementation Details

The computation is handled by the `RebalancerState` class in `agents.py` and updated in `run.py`.

#### A. Tracking the Benchmark Value

The benchmark portfolio consists of:
1.  **Risky Asset ($x_{prev}$):** The amount of token0 held by the LP.
2.  **Cash ($cash\_y$):** The amount of token1 (numéraire) held.

The benchmark value is updated in two ways:

1.  **Price Moves (Passive Holding):**
    When the CEX price ($M$) moves from $M_{old}$ to $M_{new}$, the value of the risky asset holding changes. The simulation tracks the cumulative PnL from these moves in `cumulative_R`:
    ```python
    # run.py: _accrue_price_move
    delta = M_new - rb.last_M
    rb.cumulative_R += rb.x_prev * delta
    ```
    This approximates the integral $\int x_t dM_t$.

2.  **Rebalancing Trades (Active Trading):**
    When the LP's exposure ($x$) changes (due to minting/burning or price moves within the pool), the benchmark "rebalances" to match the new target exposure ($x_{target}$). It buys or sells the difference $(x_{target} - x_{prev})$ at the current CEX price ($M_{now}$).
    ```python
    # run.py: _rebalance_lp_to_target
    dx = x_target - rb.x_prev
    rb.cash_y -= dx * M_now  # Buy/sell at fair market price
    rb.x_prev = x_target
    ```
    This ensures the benchmark always matches the LP's delta.

#### B. Calculating LVR (Fee-LVR)

At the end of each step, the simulation calculates the **Hedged PnL**, which is the difference between the LP's actual wealth change and the benchmark's return.

```python
# run.py (simplified)
delta_rebal = rb.cumulative_R - rb.last_cumulative_R
delta_wealth = wealth_now - rb.last_wealth_y

# hedged_step = Actual LP Return - Benchmark Return
hedged_step = delta_wealth - delta_rebal

rb.hedged_pnl_cum += hedged_step
```

Since $V^{LP} = V^{Rebal} - LVR + Fees$, we have:
$$
\text{hedged\_pnl\_cum} = V^{LP} - V^{Rebal} = \text{Fees} - LVR
$$

The simulation logs this metric as **"Fee-LVR"**. To isolate LVR, one would simply subtract the collected fees:
$$
LVR = \text{Fees} - \text{hedged\_pnl\_cum}
$$
(Note: Since `hedged_pnl_cum` is typically negative, $LVR$ is a positive cost).

#### C. Wealth Conservation in LP Operations

To ensure the $V_T^{LP}$ term (LP Wealth) is tracked correctly across rebalancing events, the simulation enforces strict wealth conservation during `burn` and `mint` operations:

1.  **Burning:** When a position is burned, its **full value** (Principal + Uncollected Fees) is credited to the LP's wallet.
    ```python
    # run.py: burn_any
    realized_value = pos.position_value_y_now(pool.S, ref.m) + pos.fees_value_y(ref.m)
    lp.wallet_y += realized_value
    ```
    This converts the position's mark-to-market value into cash without any loss or gain (other than the PnL already accrued).

2.  **Minting:** When a new position is minted, the cost of the liquidity (in token1 terms) is debited from the LP's wallet.
    ```python
    # run.py: lp_mint
    cost_y = amt0 * ref.m + amt1
    lp.wallet_y -= cost_y
    ```
    This converts cash into a position of equal initial value.

By ensuring that `Wealth = Wallet + Open_Positions` is invariant during rebalancing, the simulation guarantees that changes in `hedged_pnl_cum` reflect only genuine economic performance (Fees - LVR) and not accounting artifacts.

<!-- ## 3. Verification of Correctness

The implementation in the ABM is **correct** and robust.

1.  **Path Dependence:** By tracking the benchmark explicitly, the model correctly accounts for LVR generated by discrete price jumps (e.g., CEX impact) and specific trade sequences, which analytical formulas might miss in a discrete-time setting.
2.  **Dynamic Exposure:** The model correctly updates the benchmark's exposure ($x_{prev}$) whenever the LP's position changes (via `_rebalance_lp_to_target`), ensuring the benchmark remains a valid delta-hedge.
3.  **Causality Handling:** Even though the simulation has a known issue with CEX impact causality (impact applied after trades), the LVR calculation correctly captures the loss *given* that sequence of events. If the LP holds a position while the CEX price jumps (due to impact), the benchmark correctly accrues the gain on that position, while the LP (which sold to the trader before the jump) misses out. The difference is correctly recorded as LVR.

Therefore, the `Fee-LVR` metric produced by the simulation is a reliable measure of the LP's profitability relative to a hedged baseline. -->
