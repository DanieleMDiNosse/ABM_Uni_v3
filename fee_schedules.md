# Fee Schedules

This document explains the fee schedules implemented in the simulation, including the mathematical formulas and their implementation details.

The fee logic is primarily implemented in `run.py` within the `simulate` function.

## Fee Modes

There are three supported fee modes:

1.  **Static** (`static`)
2.  **Volatility-based** (`volatility`)
3.  **Toxicity-based** (`toxicity`)

### 1. Static Fee

In the static mode, the fee remains constant throughout the simulation.

$$ f_t = f_0 $$

*   $f_0$: Baseline fee (parameter `f0`).

### 2. Volatility-based Fee

This mode adjusts the fee based on the realized volatility of the CEX price. The idea is to increase fees during periods of high volatility to compensate LPs for increased LVR (Loss Versus Rebalancing).

**Formula:**

$$ f_{raw} = f_0 + k_\sigma \cdot \hat{\sigma}^2_t $$

**Components:**

*   **Log-return observation:**
    $$ vol\_obs_t = |\ln(m_t) - \ln(m_{t-1})| $$
    where $m_t$ is the CEX price at step $t$.

*   **EWMA of Squared Volatility:**
    $$ \hat{\sigma}^2_t = \text{EWMA}(vol\_obs_t^2) $$
    The Exponentially Weighted Moving Average (EWMA) is updated at each step using a half-life parameter (`fee_half_life`).

*   **Parameters:**
    *   $f_0$: Baseline fee.
    *   $k_\sigma$: Scaling factor for volatility (parameter `k_sigma`).

### 3. Toxicity-based Fee

This mode adjusts the fee based on the "toxic" flow, measured by the arbitrage opportunity size (basis) that exceeds the current fee band. This effectively measures how far the DEX price is lagging behind the CEX price.

**Formula:**

$$ f_{raw} = f_0 + k_{basis} \cdot \text{basis\_ticks}_t $$

**Components:**

*   **Fee Band (Log Space):**
    $$ \text{fee\_band\_ln} = -\ln(1 - f_{current}) $$
    This represents the price impact of the current fee.

*   **Log Price Gap:**
    $$ \text{log\_gap}_t = |\ln(P_{DEX, t}) - \ln(P_{CEX, t})| $$

*   **Excess Basis (Observation):**
    $$ B_{obs, t} = \max(0, \text{log\_gap}_t - \text{fee\_band\_ln}) $$
    This measures how much the price gap exceeds the fee band.

*   **EWMA of Basis:**
    $$ B_{hat, t} = \text{EWMA}(B_{obs, t}) $$
    Smoothed using `fee_half_life`.

*   **Basis in Ticks:**
    $$ \text{basis\_ticks}_t = \frac{B_{hat, t}}{\text{TICK\_LN}} $$
    Converts the log-basis into an equivalent number of ticks.

*   **Parameters:**
    *   $f_0$: Baseline fee.
    *   $k_{basis}$: Scaling factor for basis ticks (parameter `k_basis`).

## Fee Update Mechanism (Controller)

Regardless of the mode, the calculated raw fee $f_{raw}$ goes through a controller that applies clamping, smoothing, and hysteresis before updating the actual pool fee.

1.  **Clamping:**
    The target fee is clamped within a configured range:
    $$ f_{tgt} = \text{clamp}(f_{raw}, f_{min}, f_{max}) $$

2.  **Step Size Limits & Hysteresis:**
    The fee is only updated if the change is significant enough, and the change per step is limited.

    *   **Minimum Change Threshold:**
        If $|f_{tgt} - f_{current}| < \text{fee\_step\_bps\_min}$, no update occurs.

    *   **Maximum Step Size:**
        The change is capped at $\text{fee\_step\_bps\_max}$.
        $$ \Delta f = \text{sign}(f_{tgt} - f_{current}) \cdot \min(|f_{tgt} - f_{current}|, \text{fee\_step\_bps\_max}) $$

    *   **New Fee:**
        $$ f_{new} = f_{current} + \Delta f $$

3.  **Cooldown:**
    After a fee update, a cooldown period (`fee_cooldown` steps) is enforced during which the fee cannot be changed again.

## Implementation Reference

*   **File:** `run.py`
*   **Function:** `simulate`
*   **Relevant Section:** "Dynamic fee controller" block inside the main simulation loop.
