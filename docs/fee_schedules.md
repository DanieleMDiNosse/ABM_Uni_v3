---
title: Fee Schedules
nav_order: 6
---

# Fee Schedules

This document explains the fee schedules implemented in the simulation, including the mathematical formulas and their implementation details.

The fee logic is primarily implemented in `run.py` within the `simulate` function.

## Exponentially Weighted Moving Average (EWMA)

Several fee controllers smooth noisy per-step signals using an Exponentially Weighted Moving Average (EWMA). Given a per-step observation series $x_t$ (e.g., squared log-returns, a basis signal, or an LVR-gap signal), the EWMA state $v_t$ is updated recursively as:

$$ v_t = \lambda v_{t-1} + (1-\lambda)x_t $$

This is equivalent to the common EMA form $v_t = (1-\alpha)v_{t-1} + \alpha x_t$ with $\alpha = 1-\lambda$.

**What controls how closely it follows the underlying signal?** The smoothing parameter $\alpha$ (or, equivalently, $\lambda$ / the half-life) controls responsiveness:

* Larger $\alpha$ (smaller half-life) puts more weight on the newest observation, so $v_t$ tracks $x_t$ more closely but is noisier.
* Smaller $\alpha$ (larger half-life) puts more weight on past values, so $v_t$ is smoother but lags changes in $x_t$.

In this codebase, the EWMA is parameterized by a *half-life in steps* (e.g. `fee_half_life`). The decay is chosen so that the influence of past information halves every `half_life_steps`:

$$ \lambda = \exp\left(-\frac{\ln 2}{\tau_{\text{hl}}}\right) \quad\Rightarrow\quad \alpha = 1-\lambda $$
where $\tau_{\text{hl}}$ is the half-life in steps.

## Fee Modes

There are five supported fee modes:

1.  **Static** (`static`)
2.  **Volatility-based (CEX)** (`volatility_cex`)
3.  **Volatility-based (DEX)** (`volatility_dex`)
4.  **Toxicity-based** (`toxicity`)
5.  **LVR-gap EWMA-based** (`lvr_fee_ewma`)

### 1. Static Fee

In the static mode, the fee remains constant throughout the simulation.

$$ f_t = f_0 $$

*   $f_0$: Initial fee level (parameter `f0`). In `static` mode this is the fixed fee; in the dynamic modes it is only the starting value at $t=0$.

### 2. Volatility-based Fee (CEX)

This mode adjusts the fee based on the realized volatility of the **CEX** price (`ref.m`). The idea is to increase fees during periods of high volatility to compensate LPs for increased LVR (Loss Versus Rebalancing).

**Formula:**

$$ f_{raw} = k_\sigma \cdot \hat{\sigma}_t $$

Since this controller is *pure-signal* (no additive baseline), when $\hat{\sigma}_t$ is small the applied fee will typically be clamped to $f_{min}$ by the update mechanism below.

**Components:**

*   **Log-return observation:**
    $$ r_t = \ln(m_t) - \ln(m_{t-1}) $$
    $$ v_t = r_t^2 $$
    where $m_t$ is the CEX price at step $t$.

*   **EWMA volatility estimate:**
    $$ \hat{\sigma^2}_t = \text{EWMA}(v_t) $$
    $$ \hat{\sigma}_t = \sqrt{\hat{\sigma^2}_t} $$
    The Exponentially Weighted Moving Average (EWMA) is updated at each step using a half-life parameter (`fee_half_life`). In the code, `fee_sigma_series` stores $\hat{\sigma^2}_t$ (see plotting label "EWMA(σ^2)").

*   **Parameters:**
    *   $k_\sigma$: Scaling factor for volatility (parameter `k_sigma`).

### 3. Volatility-based Fee (DEX)

Same controller as above, but with the volatility observation computed from the **DEX** price (`pool.price`):

$$ r^{\text{DEX}}_t = \ln(P_{DEX,t}) - \ln(P_{DEX,t-1}) \quad,\quad v^{\text{DEX}}_t = (r^{\text{DEX}}_t)^2 $$

This mode is selected with `fee_mode = "volatility_dex"` and uses the same $f_{raw} = k_\sigma \cdot \hat{\sigma}_t$ mapping, with $\hat{\sigma}_t$ estimated from the DEX price series.

### 4. Toxicity-based Fee

This mode adjusts the fee based on the "toxic" flow, measured by the arbitrage opportunity size (basis) that exceeds the current fee band. This effectively measures how far the DEX price is lagging behind the CEX price.

**Formula:**

$$ f_{raw} = k_{\text{basis}} \cdot \beta^{\text{ticks}}_t $$

Since this controller is *pure-signal* (no additive baseline), when $\beta^{\text{ticks}}_t$ is small the applied fee will typically be clamped to $f_{min}$ by the update mechanism below.

**Components:**

*   **Fee Band (Log Space):**
    $$ \ell_f = -\ln(1 - f_{current}) $$
    This represents the price impact of the current fee.

*   **Log Price Gap:**
    $$ \gamma_t = |\ln(P_{DEX, t}) - \ln(P_{CEX, t})| $$

*   **Excess Basis (Observation, uses the *current* pool fee):**
    $$ B_{obs, t} = \max(0, \gamma_t - \ell_f) $$
    This measures how much the price gap exceeds the fee band; gaps inside the fee band contribute 0.

*   **EWMA of Basis:**
    $$ B_{hat, t} = \text{EWMA}(B_{obs, t}) $$
    Smoothed using `fee_half_life`.

*   **Basis in Ticks:**
    $$ \beta^{\text{ticks}}_t = \frac{\hat{B}_t}{\ln(1.0001)} $$
    Converts the log-basis into an equivalent number of ticks.

*   **Parameters:**
    *   $k_{basis}$: Scaling factor for basis ticks (parameter `k_basis`).

### 5. LVR-gap EWMA-based Fee

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
    This uses the absolute input notional of executed DEX swaps in token1 units (including arbitrage). Concretely, inputs in token0 are converted to token1 notional using the *pre-execution DEX price* for that swap. If $\text{Notional}_t = 0$, the controller skips the update for that step and leaves the fee unchanged.

*   **Parameters:**
    *   $k_{lvr}$: Feedback gain (parameter `k_lvr`), applied around the current fee.

## Fee Update Mechanism (Controller)

Regardless of the mode, the calculated raw fee $f_{raw}$ goes through a controller that applies clamping, smoothing, and hysteresis before updating the actual pool fee.

The implementation is staged (commit→reveal): the controller computes signals from the end-of-step state, produces a pending fee `fee_next`, and the pool fee `pool.f` is only updated once the cooldown timer has elapsed.

1.  **Step Size Limits & Clamping:**
    Let $f_{current}$ be the current pool fee at the end of the step.

    *   **Minimum Change Threshold:**
        If $|f_{raw} - f_{current}| < \delta_{\min} / 10{,}000$, no pending update is staged (where $\delta_{\min} = \texttt{fee\_step\_bps\_min}$).

    *   **Maximum Step Size:**
        The change is capped at $\delta_{\max} / 10{,}000$ (where $\delta_{\max} = \texttt{fee\_step\_bps\_max}$).
        $$ \Delta f = \text{sign}(f_{raw} - f_{current}) \cdot \min(|f_{raw} - f_{current}|, \delta_{\max} / 10{,}000) $$

    *   **Candidate New Fee (clamped):**
        $$ f_{new} = \text{clamp}(f_{current} + \Delta f, f_{min}, f_{max}) $$

2.  **Cooldown / Commit Timing:**
    After staging a pending update (`fee_next = f_new`), a cooldown counter (`fee_cooldown_left`) counts down each step. Once the counter reaches 0, the current pool fee is set to `fee_next` and `fee_next` is cleared. This ensures the pool fee changes at most once every `fee_cooldown` steps. (Implementation detail: while the cooldown counts down, `fee_next` can be overwritten; the value that gets applied is the latest staged fee when the counter reaches 0.)

## Implementation Reference

*   **File:** `run.py`
*   **Function:** `simulate`
*   **Relevant Section:** "Dynamic fee controller" block inside the main simulation loop.
*   **Diagnostics:** the simulation returns `fee_series`, `fee_sigma_series`, `fee_basis_ticks_series`, `fee_signal_series`, and `fee_imb_series` for plotting/debugging.
