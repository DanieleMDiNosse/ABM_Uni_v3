---
title: Fee Schedules
nav_order: 6
---

# Fee Schedules

This document explains the fee schedules implemented in the simulation, including the mathematical formulas and their implementation details.

The fee logic is primarily implemented in `run.py` within the `simulate` function.

## Exponentially Weighted Moving Average (EWMA)

Several fee controllers smooth noisy per-step signals using an Exponentially Weighted Moving Average (EWMA). Given a per-step observation series $x_t$ (e.g., absolute log-returns or a basis signal), the EWMA state $v_t$ is updated recursively as:

$$ v_t = \lambda v_{t-1} + (1-\lambda)x_t $$

This is equivalent to the common EMA form $v_t = (1-\alpha)v_{t-1} + \alpha x_t$ with $\alpha = 1-\lambda$.

**What controls how closely it follows the underlying signal?** The smoothing parameter $\alpha$ (or, equivalently, $\lambda$ / the half-life) controls responsiveness:

* Larger $\alpha$ (smaller half-life) puts more weight on the newest observation, so $v_t$ tracks $x_t$ more closely but is noisier.
* Smaller $\alpha$ (larger half-life) puts more weight on past values, so $v_t$ is smoother but lags changes in $x_t$.

In this codebase, the EWMA is parameterized by a *half-life in steps* (e.g. `fee_half_life`). The decay is chosen so that the influence of past information halves every `half_life_steps`:

$$ \lambda = \exp\left(-\frac{\ln 2}{\text{half\_life\_steps}}\right) \quad\Rightarrow\quad \alpha = 1-\lambda $$

## Fee Modes

There are four supported fee modes:

1.  **Static** (`static`)
2.  **Volatility-based** (`volatility`)
3.  **Toxicity-based** (`toxicity`)
4.  **LVR-gap EWMA-based** (`lvr_fee_ewma`)

### 1. Static Fee

In the static mode, the fee remains constant throughout the simulation.

$$ f_t = f_0 $$

*   $f_0$: Baseline fee (parameter `f0`).

### 2. Volatility-based Fee

This mode adjusts the fee based on the realized volatility of the CEX price. The idea is to increase fees during periods of high volatility to compensate LPs for increased LVR (Loss Versus Rebalancing).

**Formula:**

$$ f_{raw} = f_0 + k_\sigma \cdot \hat{\sigma}_t \cdot \sqrt{\text{block\_time}} $$

**Components:**

*   **Log-return observation:**
    $$ vol\_obs_t = |\ln(m_t) - \ln(m_{t-1})| $$
    where $m_t$ is the CEX price at step $t$.

*   **EWMA of absolute log-returns:**
    $$ \hat{\sigma}_t = \text{EWMA}(vol\_obs_t) $$
    The Exponentially Weighted Moving Average (EWMA) is updated at each step using a half-life parameter (`fee_half_life`). No squaring is applied; the controller works directly with the smoothed absolute log-return.

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

*   **Excess Basis (Observation, uses the *current* pool fee):**
    $$ B_{obs, t} = \max(0, \text{log\_gap}_t - \text{fee\_band\_ln}) $$
    This measures how much the price gap exceeds the fee band; gaps inside the fee band contribute 0.

*   **EWMA of Basis:**
    $$ B_{hat, t} = \text{EWMA}(B_{obs, t}) $$
    Smoothed using `fee_half_life`.

*   **Basis in Ticks:**
    $$ \text{basis\_ticks}_t = \frac{B_{hat, t}}{\text{TICK\_LN}} $$
    Converts the log-basis into an equivalent number of ticks.

*   **Parameters:**
    *   $f_0$: Baseline fee.
    *   $k_{basis}$: Scaling factor for basis ticks (parameter `k_basis`).

### 4. LVR-gap EWMA-based Fee

This mode adjusts the fee using the EWMA of the per-step gap between LVR and fees, normalized by the executed DEX notional. The controller raises fees when LVR exceeds fees and lowers them when fees exceed LVR.

**Formula:**

$$ g_{obs, t} = \frac{\Delta \text{LVR}_t - \Delta \text{Fees}_t}{\text{Notional}_t} $$

$$ g_{\hat{t}} = \text{EWMA}(g_{obs, t}) $$

$$ f_{raw} = f_{current} + k_{lvr} \cdot g_{\hat{t}} $$

**Components:**

*   **Per-step increments (pool-wide totals):**
    $$ \Delta \text{LVR}_t = \text{LVR}_t - \text{LVR}_{t-1} $$
    $$ \Delta \text{Fees}_t = \text{Fees}_t - \text{Fees}_{t-1} $$
    where $\text{LVR}_t$ and $\text{Fees}_t$ are the cumulative pool-wide totals over all non-seed LPs (active + passive).

*   **DEX notional (token1 units):**
    $$ \text{Notional}_t = \sum_{i \in \text{DEX swaps}} |\text{input}_i| $$
    This uses the absolute input notional of executed DEX swaps in token1 units (including arbitrage). If $\text{Notional}_t = 0$, the controller skips the update for that step.

*   **Parameters:**
    *   $k_{lvr}$: Feedback gain (parameter `k_lvr`), applied around the current fee.

## Fee Update Mechanism (Controller)

Regardless of the mode, the calculated raw fee $f_{raw}$ goes through a controller that applies clamping, smoothing, and hysteresis before updating the actual pool fee.

1.  **Clamping:**
    The target fee is clamped within a configured range:
    $$ f_{tgt} = \text{clamp}(f_{raw}, f_{min}, f_{max}) $$

2.  **Step Size Limits & Hysteresis:**
    The fee is only updated if the change is significant enough, and the change per step is limited.

    *   **Minimum Change Threshold:**
        If $|f_{tgt} - f_{current}| < \text{fee\_step\_bps\_min} / 10{,}000$, no update occurs.

    *   **Maximum Step Size:**
        The change is capped at $\text{fee\_step\_bps\_max} / 10{,}000$.
        $$ \Delta f = \text{sign}(f_{tgt} - f_{current}) \cdot \min(|f_{tgt} - f_{current}|, \text{fee\_step\_bps\_max} / 10{,}000) $$

    *   **New Fee:**
        $$ f_{new} = f_{current} + \Delta f $$
        If scheduled, `f_new` is staged in `fee_next` and only committed once any cooldown has elapsed.

3.  **Cooldown:**
    After staging a fee update, a cooldown period (`fee_cooldown` steps) is enforced during which additional changes are ignored.

## Implementation Reference

*   **File:** `run.py`
*   **Function:** `simulate`
*   **Relevant Section:** "Dynamic fee controller" block inside the main simulation loop.
*   **Diagnostics:** the simulation returns `fee_series`, `fee_sigma_series`, `fee_basis_ticks_series`, `fee_signal_series`, and `fee_imb_series` for plotting/debugging.
