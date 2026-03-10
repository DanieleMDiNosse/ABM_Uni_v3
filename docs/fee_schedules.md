---
title: Fee Schedules
nav_order: 6
---

# Fee Schedules

This page documents the fee controllers implemented in `scripts/run.py`. It describes the controller signals, the five supported fee modes, and the shared commit/reveal update logic.

## Shared EWMA State

Several controllers smooth a noisy per-step observation series $x_t$ with an exponentially weighted moving average:

$$ v_t = \lambda v_{t-1} + (1-\lambda)x_t $$

The implementation parameterizes the decay by `fee_half_life`:

$$ \lambda = \exp\left(-\frac{\ln 2}{\tau_{\text{hl}}}\right) \quad\Rightarrow\quad \alpha = 1-\lambda $$
where $\tau_{\text{hl}}$ is the half-life in simulation steps.

The controller uses three observation families:

- **Volatility observation**
  - CEX: $r_t^{\text{cex}} = \log m_t - \log m_{t-1}$
  - DEX: $r_t^{\text{dex}} = \log P_t^{\text{dex}} - \log P_{t-1}^{\text{dex}}$
  - EWMA input: $(r_t)^2$
- **Fee-adjusted basis observation**
  - fee band in log space: $\ell_f = -\log(1-f_t)$
  - raw gap: $\gamma_t = |\log P_t^{\text{dex}} - \log m_t|$
  - excess basis: $B_t = \max(0, \gamma_t - \ell_f)$
- **LVR gap observation**
  - $\Delta \mathrm{gap}_t = (\Delta \mathrm{LVR}_t - \Delta \mathrm{Fees}_t) / \mathrm{DEXNotional}_t$
  - this update is skipped when `dex_notional_y_this <= 0`

## Fee Modes

### 1. `static`

Static mode keeps the pool fee fixed at its initial value `f0`:

$$ f_t = f_0 $$

No controller signal is used and no updates are staged.

### 2. `volatility_cex`

This mode uses the CEX return series:

$$
\sigma^{2,\text{ewma}}_t = \mathrm{EWMA}\!\left(\left(\log m_t - \log m_{t-1}\right)^2\right)
$$

and maps it to a raw fee target via:

$$
f^{\text{raw}}_t = k_\sigma \sqrt{\sigma^{2,\text{ewma}}_t}
$$

Notes:

- `fee_sigma_series` stores the EWMA state $\sigma^{2,\text{ewma}}_t$, not its square root.
- `fee_signal_series` stores the controller signal actually used for this mode, which is again $\sigma^{2,\text{ewma}}_t`.
- Because the controller is pure-signal rather than baseline-plus-spread, low volatility regimes usually drive the staged target toward `f_min`.

### 3. `volatility_dex`

Same controller as above, but the observation is computed from the DEX price series:

$$
\sigma^{2,\text{ewma}}_t = \mathrm{EWMA}\!\left(\left(\log P_t^{\text{dex}} - \log P_{t-1}^{\text{dex}}\right)^2\right)
$$

and the raw target is still:

$$
f^{\text{raw}}_t = k_\sigma \sqrt{\sigma^{2,\text{ewma}}_t}
$$

### 4. `toxicity`

This mode reacts to the fee-adjusted cross-venue log gap. First compute:

$$
B_t = \max(0, |\log P_t^{\text{dex}} - \log m_t| - \ell_f), \qquad \ell_f = -\log(1-f_t)
$$

Then smooth and convert to tick units:

$$
\hat B_t = \mathrm{EWMA}(B_t), \qquad \beta_t^{\text{ticks}} = \hat B_t / \log(1.0001)
$$

The raw target is:

$$
f_t^{\text{raw}} = k_{\text{basis}} \, \beta_t^{\text{ticks}}
$$

Diagnostics:

- `fee_basis_ticks_series` stores $\beta_t^{\text{ticks}}$.
- `fee_signal_series` also stores $\beta_t^{\text{ticks}}$ in this mode.

### 5. `lvr_fee_ewma`

This is the only feedback controller centered on the current fee level. It compares the latest increase in LP LVR to the latest increase in LP fee value:

$$
\Delta \mathrm{gap}_t
= \frac{\Delta \mathrm{LVR}_t - \Delta \mathrm{Fees}_t}{\mathrm{DEXNotional}_t}
$$

with

- $\Delta \mathrm{LVR}_t = \mathrm{LVR}_t - \mathrm{LVR}_{t-1}$
- $\Delta \mathrm{Fees}_t = \mathrm{Fees}_t - \mathrm{Fees}_{t-1}$
- `DEXNotional_t = dex_notional_y_this`

The smoothed gap signal is:

$$
g_t = \mathrm{EWMA}(\Delta \mathrm{gap}_t)
$$

and the raw target is:

$$
f_t^{\text{raw}} = f_t + k_{\text{lvr}} g_t
$$

If there is no DEX notional in the current step, the update is skipped and the fee is left unchanged for that step.

## Shared Update Mechanism

Regardless of fee mode, the raw target passes through the same staged controller.

### 1. Start-of-block commit

At the start of each block:

- if `fee_cooldown_left > 0`, decrement it by one;
- if `fee_next` is staged and `fee_cooldown_left <= 0`, commit it to `pool.f`.

This is why the controller is effectively commit/reveal: signals are computed from end-of-step data, but the new fee applies from the next block onward.

### 2. End-of-step staging

Given the current fee `pool.f`, define:

- `min_step = fee_step_bps_min / 1e4`
- `max_step = fee_step_bps_max / 1e4`
- `\Delta f = f_t^{\text{raw}} - pool.f`

If $|\Delta f| < \mathrm{min\_step}$, nothing is staged.

Otherwise, use the clipped step:

$$
\Delta f^{\text{clip}}
= \mathrm{sign}(\Delta f)\min(|\Delta f|, \mathrm{max\_step})
$$

and then clamp the staged fee to the allowed range:

$$
f_{t+1}^{\text{staged}} = \mathrm{clip}(pool.f + \Delta f^{\text{clip}}, f_{\min}, f_{\max})
$$

If the staged value differs from the current fee, it is written to `fee_next`. If no cooldown is already active, the controller starts one with `fee_cooldown_left = fee_cooldown`.

Important implementation detail: while the cooldown counts down, the pending value can be overwritten by a newer staged fee. The committed fee is therefore the latest `fee_next` available when the cooldown expires.

## Diagnostics Returned By `simulate(...)`

- `fee_series`: the currently applied fee path (`pool.f`) before the newly staged value is committed next block.
- `fee_sigma_series`: EWMA of squared volatility observations.
- `fee_basis_ticks_series`: EWMA fee-adjusted basis in tick units.
- `fee_signal_series`: controller-specific plotted signal.
- `fee_imb_series`: end-of-step reserve imbalance in the active band; useful for diagnostics, but not a fee-controller input.

## Implementation Reference

- File: `scripts/run.py`
- Function: `simulate(...)`
- Relevant block: `# ================== Dynamic fee controller ==================`
