# Fee Schedules

This document explains the fee schedules implemented in the simulation, including the mathematical formulas and their implementation details.

The fee logic is primarily implemented in `run.py` within the `simulate` function.

## Fee Modes

There are four supported fee modes:

1.  **Static** (`static`)
2.  **Volatility-based** (`volatility`)
3.  **Volatility-oracle-based** (`volatility_oracle`)
4.  **Toxicity-based** (`toxicity`)

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

### 3. Volatility-Oracle-Based Fee

This mode adjusts the fee using the *instantaneous* CEX volatility signal from the reference market, without any additional smoothing. Instead of estimating volatility from realized log-returns, the controller directly consumes the per-step volatility path \( \sigma_t \) used by the CEX process (e.g. regime, noisy-sine, or Heston).

**Per-step formula (conceptual):**

$$ f_{raw} = f_0 + k_\sigma \cdot \sigma_t \cdot \sqrt{\text{block\_time}} $$

where:

*   **Oracle volatility:**
    $$ \sigma_t = \text{ReferenceMarket.sigma at step } t $$
    which is exactly the volatility used when diffusing the CEX price in
    `utils.ReferenceMarket` (static / regime / noisy-sine / Heston).
*   **Parameters:**
    *   $f_0$: Baseline fee.
    *   $k_\sigma$: Scaling factor for volatility (parameter `k_sigma`),
        re-used from the standard volatility mode.

**Timing behaviour:**

*   **Non-block mode (`block_time == 1`):**  
    - At the end of step \(t\), after the CEX update, the controller reads
      the current \(\sigma_t\), computes \(f_{raw}\) as above, and stages a
      new fee via the same clamped / step-limited / cooldown mechanism used
      in the standard volatility mode.  
    - The staged fee becomes active on step \(t+1\), so there is a
      one-step lag between \(\sigma_t\) and the fee actually seen by trades.

*   **Block mode (`block_time > 1`):**  
    - In addition to the end-of-step diagnostics, the simulator lets the
      fee react *within* each block at micro-step granularity. At each
      micro-step \(k\) inside block \(t\), **before** enqueuing smart/noise
      intents, it:
        1. reads the current oracle volatility \(\sigma_{t,k} = \text{ReferenceMarket.sigma}\);
        2. computes a micro-step raw fee
           \[
             f_{raw}^{(micro)} = f_0 + k_\sigma \cdot \sigma_{t,k} \cdot \sqrt{\text{block\_time}};
           \]
        3. clamps and step-limits this value using the same
           `f_min` / `f_max` / `fee_step_bps_min` / `fee_step_bps_max`
           logic; if the implied change is large enough, it updates
           `pool.f` **immediately** (no `fee_next`, no cooldown gating).  
    - As a result, trades within the same block can experience different
      fees as \(\sigma_{t,k}\) evolves. The block-level controller at the
      end of the step records the volatility signal and fee path for
      plotting but does not stage an additional fee move in this mode.

### 4. Toxicity-based Fee

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
