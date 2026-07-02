---
title: Fee Schedules
nav_order: 6
---

# Fee Schedules

This page documents the fee controllers implemented in `scripts/run.py`. It describes the controller signals, the supported fee modes, the optional EWMA smoothing switch, and the shared commit/reveal update logic.

## Shared EWMA State

Several controllers can smooth a noisy per-step observation series $x_t$ with an exponentially weighted moving average:

$$ v_t = \lambda v_{t-1} + (1-\lambda)x_t $$

The implementation parameterizes the decay by `fee_half_life`:

$$ \lambda = \exp\left(-\frac{\ln 2}{\tau_{\text{hl}}}\right) \quad\Rightarrow\quad \alpha = 1-\lambda $$
where $\tau_{\text{hl}}$ is the half-life in simulation steps.

The YAML/simulation parameter `fee_use_ewma` controls whether the controller observation is smoothed before it is mapped to a fee target:

- `fee_use_ewma: true` uses the EWMA state $v_t$.
- `fee_use_ewma: false` uses the raw observation $x_t$ directly.
- If omitted or set to `null`, legacy per-mode defaults are preserved: `volatility_cex`, `volatility_dex`, `toxicity`, and `lvr_fee_ewma` use EWMA smoothing; `linear_asymmetric` uses the raw block-open signed gap.

The bounded fee-step rule below is still applied after the target is computed, so disabling EWMA removes signal smoothing but not the per-block step-size cap.

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

and maps either the EWMA-smoothed signal or the raw squared return, depending on `fee_use_ewma`, to a raw fee target via:

$$
f^{\text{raw}}_t = k_\sigma \sqrt{\sigma^{2,\text{ewma}}_t}
$$

Notes:

- `fee_sigma_series` stores the controller signal actually used: the EWMA state $\sigma^{2,\text{ewma}}_t$ when smoothing is enabled, otherwise the raw squared return observation.
- `fee_signal_series` stores the same controller signal for this mode.
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

- `fee_basis_ticks_series` stores $\beta_t^{\text{ticks}}$ computed from the smoothed excess basis when `fee_use_ewma: true`, or from the raw current excess basis when `fee_use_ewma: false`.
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

When `fee_use_ewma: true`, the smoothed gap signal is:

$$
g_t = \mathrm{EWMA}(\Delta \mathrm{gap}_t)
$$

and the raw target is:

$$
f_t^{\text{raw}} = f_t + k_{\text{lvr}} g_t
$$

If there is no DEX notional in the current step, the update is skipped and the fee is left unchanged for that step.

When `fee_use_ewma: false`, the target uses the raw valid $\Delta \mathrm{gap}_t$ observation instead of $g_t$.

### 6. `linear_asymmetric`

This mode implements the linear asymmetric dynamic-fee approximation motivated by Baggiani, Herdegen, and Sánchez-Betancourt (arXiv:2506.02869v1). The pool keeps separate input fees for the two swap directions:

- `fee_x_to_y`: fee charged when a trader sells token0 (`X`) into the pool and receives token1 (`Y`), pushing the token1-per-token0 DEX price downward.
- `fee_y_to_x`: fee charged when a trader sells token1 (`Y`) into the pool and receives token0 (`X`), pushing the DEX price upward.

The signed block-open signal is the log DEX/oracle price gap:

$$
z_t = \log P_t^{\text{dex}} - \log m_t .
$$

The raw linear targets are:

$$
f^{x\to y,\text{raw}}_t = f_0 + k_{\text{asym}} z_t,
\qquad
f^{y\to x,\text{raw}}_t = f_0 - k_{\text{asym}} z_t,
$$

where `asymmetric_fee_slope` is $k_{\text{asym}}$.

Sign convention:

- If $P_t^{\text{dex}} > m_t$, token0 is expensive on the DEX. The adverse-selection direction is `X_to_Y`, so `fee_x_to_y` rises and `fee_y_to_x` falls.
- If $P_t^{\text{dex}} < m_t$, token0 is cheap on the DEX. The adverse-selection direction is `Y_to_X`, so `fee_y_to_x` rises and `fee_x_to_y` falls.
- If $P_t^{\text{dex}} = m_t$, both directional targets equal `f0`.

When `fee_use_ewma: true`, the signed gap $z_t$ is replaced by its EWMA-smoothed value before the two directional targets are computed. When `fee_use_ewma: false`, the raw block-open $z_t$ is used directly. Both directional targets are clipped to `[f_min, f_max]` and moved with the same bounded step rule used by the other causal block-open controllers. The legacy `fee_series` output records the midpoint of the two directional fees for backward-compatible aggregations. The actual applied directional fee paths are returned as `fee_x_to_y_series` and `fee_y_to_x_series`, and `fee_signal_series` records the controller signal after the optional smoothing switch.

## Shared Update Mechanism

Regardless of fee mode, the raw target passes through the same bounded step controller.

### 1. Block-open update for volatility/toxicity/asymmetric schedules

At the start of each block, volatility, toxicity, and linear-asymmetric schedules compute their signal from pre-block information and apply the resulting bounded fee step before LP, trader, JIT, or arbitrage execution in that block.

### 2. Delayed staging for realized-LVR feedback

The optional LVR-feedback artifact is different because its signal uses realized outcomes from the current block. Its bounded target is staged in `fee_next` and committed at the start of the next block.

### 3. Bounded step rule

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

For volatility/toxicity schedules, the clipped and clamped value is applied immediately to `pool.f` at block open. For `linear_asymmetric`, the same rule is applied independently to the `X_to_Y` and `Y_to_X` directional fee paths. For realized-LVR feedback, the clipped and clamped value is written to `fee_next` and committed on the following block.

## Diagnostics Returned By `simulate(...)`

- `fee_series`: the currently applied fee path (`pool.f`) before the newly staged value is committed next block. In `linear_asymmetric`, this is the midpoint of the two directional fees for backward-compatible summaries.
- `fee_x_to_y_series`: directional input fee applied to `X_to_Y` swaps; equals `fee_series` for symmetric modes.
- `fee_y_to_x_series`: directional input fee applied to `Y_to_X` swaps; equals `fee_series` for symmetric modes.
- `fee_use_ewma`: boolean resolved smoothing choice after applying legacy per-mode defaults.
- `fee_sigma_series`: volatility-controller signal after the optional smoothing switch.
- `fee_basis_ticks_series`: fee-adjusted basis signal in tick units after the optional smoothing switch.
- `fee_signal_series`: controller-specific plotted signal.
- `fee_imb_series`: end-of-step reserve imbalance in the active band; useful for diagnostics, but not a fee-controller input.

## Implementation Reference

- File: `scripts/run.py`
- Function: `simulate(...)`
- Relevant block: `# ================== Dynamic fee controller ==================`
