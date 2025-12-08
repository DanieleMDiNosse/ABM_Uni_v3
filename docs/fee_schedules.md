# Fee Schedules

This document explains the fee schedules implemented in the simulation, including the mathematical formulas and their implementation details.

The fee logic is primarily implemented in `run.py` within the `simulate` function.

## Fee Modes

There are four supported fee modes:

1.  **Static** (`static`)
2.  **Volatility-based** (`volatility`)
3.  **Toxicity-based** (`toxicity`)
4.  **GAS-based** (`gas`)

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

### 4. GAS-based Fee

This mode replaces the EWMA with a **score-driven (GAS) volatility state** and charges based on both the level of that state and the “surprise” in the most recent return.

**State and score (Gaussian log-variance):**

*   Observation: $r_t = \log m_t - \log m_{t-1}$.
*   State: $f_t = \log \sigma_t^2$.
*   Score: $s_t = \tfrac{1}{2}\left(\frac{r_t^2}{\sigma_t^2} - 1\right)$ with $\sigma_t^2 = e^{f_t}$.
*   Update: $f_{t+1} = \omega + \beta f_t + \alpha s_t$.
*   Derived level: $\hat{\sigma}_t = \exp(\tfrac{1}{2} f_t)$.

**Fee mapping:**

$$ f_{raw} = f_0 + k_{\text{gas\_sigma}} \cdot \hat{\sigma}_t . $$
<!-- $$ f_{raw} = f_0 + k_{\text{gas\_sigma}} \cdot \hat{\sigma}_t + k_{\text{gas\_score}} \cdot \max(0, s_t). $$ -->

**Parameters:**

*   `gas_alpha` ($\alpha$): score weight.
*   `gas_beta` ($\beta$): persistence of the log-variance state.
*   `gas_omega` ($\omega$): drift term.
*   `k_gas_sigma`: fee sensitivity to the GAS volatility level $\hat{\sigma}_t$.
<!-- *   `k_gas_score`: fee sensitivity to positive surprises $s_t$. -->

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
