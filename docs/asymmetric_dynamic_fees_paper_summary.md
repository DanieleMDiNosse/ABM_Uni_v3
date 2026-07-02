---
title: Asymmetric Dynamic Fees Paper Summary
nav_order: 7
---

# Asymmetric Dynamic Fees Paper Summary

Source: Baggiani, Herdegen, and Sánchez-Betancourt, “Optimal Dynamic Fees in Automated Market Makers,” arXiv:2506.02869v1.

This note summarizes the paper’s model and the asymmetric fee schedules it proposes. It is a reading note for future implementation in this repository, not evidence that the schedules are already implemented.

## Scientific claim

An AMM that can set direction-specific dynamic fees should charge different fees for buys and sells. The optimal policy has two regimes:

1. raise the fee on the trade direction that is attractive to arbitrageurs, thereby reducing adverse-selection flow; and
2. lower, possibly subsidize, the opposite direction to attract noise traders and increase useful trading volume.

The paper’s simulations find that these asymmetric dynamic fees collect more fee revenue than constant fees, and that simple linear approximations are close to the computed optimal policies for the reported parameter sets.

## Model object

The venue is a constant-product AMM with two assets:

- `X`: riskless asset / numéraire;
- `Y`: risky asset;
- `S_t`: external CEX/oracle midprice of `Y` in units of `X`;
- `Z(y)`: AMM instantaneous exchange rate when the pool holds quantity `y` of risky asset `Y`.

Trades move the risky-asset inventory on a discrete grid:

- a sell into the AMM increases AMM inventory from `y` to `y + Δ⁺(y)`;
- a buy from the AMM decreases AMM inventory from `y` to `y - Δ⁻(y)`.

The paper uses two direction-specific proportional fee functions:

- `p(t, y)`: fee for selling `Y` into the AMM;
- `m(t, y)`: fee for buying `Y` from the AMM.

The fee-adjusted exchange rates faced by liquidity takers are:

```text
sell Y into AMM:  Z_p⁺(y) = (1 - p(y)) Z⁺(y)
buy Y from AMM:  Z_m⁻(y) = (1 + m(y)) Z⁻(y)
```

Fees are collected in units of `X` outside the pool. In the paper’s mathematical problem, `p` and `m` are not constrained to `[0, 1]`; negative fees are allowed and mean the venue pays liquidity takers to trade in that direction. For this repository, any implementation should explicitly decide whether negative fees are allowed or whether the schedule is clipped to existing `f_min`/`f_max` bounds.

## Liquidity-taker arrival model

Buy and sell arrivals are controlled point processes. Their intensities depend exponentially on the profitability of trading against the AMM relative to the external price. In simplified form:

```text
λ_buy  increases when buying Y from the AMM is cheap relative to S_t
λ_sell increases when selling Y to the AMM is expensive relative to S_t
```

The paper uses an Avellaneda-Stoikov-style sensitivity parameter `k > 0`. Larger `k` means liquidity-taker arrivals are more sensitive to price differences; smaller `k` means the AMM needs stronger fee incentives to shape flow.

The venue maximizes expected cumulative fee revenue, optionally minus a penalty for AMM/oracle price misalignment:

```text
E[ cumulative fees - ∫ P(Y_t, S_t) dt ]
```

A common penalty studied in the paper is:

```text
P(Y_t, S_t) = φ ( Z(Y_t) - S_t )²
```

where `φ ≥ 0` controls how much the venue values keeping the AMM quote aligned with the external price.

## Fee schedule 1: exact HJB optimizer

The general dynamic-programming solution gives optimal direction-specific fees in terms of the reduced value function `g(t, y, s)`:

```text
p*(t, y) = [g(t, y, s) - g(t, y + Δ⁺(y), s)] / [Z⁺(y) Δ⁺(y)]
           + 1 / [k Z⁺(y) Δ⁺(y)]

m*(t, y) = [g(t, y, s) - g(t, y - Δ⁻(y), s)] / [Z⁻(y) Δ⁻(y)]
           + 1 / [k Z⁻(y) Δ⁻(y)]
```

Interpretation:

- the first term is a marginal continuation-value term: how much future venue value changes if inventory moves one grid step;
- the second term is a positive markup term controlled by demand sensitivity `k` and trade size/notional.

This is the economically clean object, but it requires solving the HJB/value-function problem and is not the simplest implementation target for an ABM experiment.

## Fee schedule 2: constant-oracle approximation

Section 4 assumes a constant external price `S_t = S_0` (`σ = 0`). With the transform `exp(k g(t, y)) = w(t, y)`, the nonlinear HJB becomes a linear ODE system. The resulting optimal fees are:

```text
p*(t, y_i) = [1 + log( w(t, y_i) / w(t, y_{i+1}) )] / [k Z⁺(y_i) Δ⁺(y_i)]

m*(t, y_i) = [1 + log( w(t, y_i) / w(t, y_{i-1}) )] / [k Z⁻(y_i) Δ⁻(y_i)]
```

where `w(t, y_i)` is obtained from a matrix exponential over the inventory grid.

Main qualitative pattern when `φ = 0`:

- if the AMM price is high relative to the external price (`Z(y) > S_t`), selling `Y` into the AMM is attractive to arbitrageurs; the AMM raises the sell fee `p` and lowers the buy fee `m`;
- if the AMM price is low relative to the external price (`Z(y) < S_t`), buying `Y` from the AMM is attractive to arbitrageurs; the AMM raises the buy fee `m` and lowers the sell fee `p`.

Around the reference inventory `y_0`, the paper shows that a linear approximation to `p*` and `m*` closely matches the optimal curves for its baseline parameters.

Effect of `k`:

- smaller `k` means order flow is less sensitive to mispricing;
- optimal fees increase roughly like `1/k` in the reported limiting analysis.

Effect of `φ`:

- for `φ = 0`, the policy mainly separates arbitrage-penalizing and noise-attracting regimes;
- for larger `φ`, the policy increasingly tries to keep the AMM quote aligned with the external price, changing both the slope and level of the asymmetric fees.

## Fee schedule 3: second-order stochastic-oracle approximation

Section 5 allows a stochastic external price:

```text
S_t = S_0 + σ W_t
```

The paper uses a second-order approximation with constant trade sizes `δ⁺` and `δ⁻`. The reduced value function is approximated by a quadratic form:

```text
g(t, y, s) = y² A(t) + y B(t, s) + C(t, s)
B(t, s) = s b₁(t) + b₀(t)
```

The resulting fees are linear in inventory `y` and external price `s`:

```text
p*(t, y) = -[(2y + δ⁺) A(t) + B(t, s)] / Z⁺(y)
           + 1 / [k Z⁺(y) δ⁺]

m*(t, y) = -[(-2y + δ⁻) A(t) - B(t, s)] / Z⁻(y)
           + 1 / [k Z⁻(y) δ⁻]
```

The paper emphasizes that these fees:

- are direction-specific (`p*` for sells, `m*` for buys);
- depend linearly on inventory and external price under the approximation;
- do not depend directly on `σ` in the final fee formula, although `σ` enters the value-function approximation problem;
- preserve the same two-regime economic intuition: penalize the arbitrage direction and subsidize or cheapen the opposite direction.

## Practical schedule suggested by the paper

For implementation in this repository, the paper’s most relevant proposal is not necessarily the full HJB solver. The implementable takeaway is a bounded asymmetric linear fee rule driven by current pool/oracle mispricing and inventory state.

A minimal ABM-compatible schedule should expose two applied fees per block/step:

```text
fee_sell_t = fee charged when trader sells token into the pool
fee_buy_t  = fee charged when trader buys token from the pool
```

A simple linearized version consistent with the paper’s figures is:

```text
mispricing_t = log(P_dex_t) - log(P_oracle_t)

fee_sell_raw_t = base_fee + slope * mispricing_t
fee_buy_raw_t  = base_fee - slope * mispricing_t
```

with clipping and optional step-size limits:

```text
fee_sell_t = clip_with_step_limit(fee_sell_raw_t, f_min, f_max)
fee_buy_t  = clip_with_step_limit(fee_buy_raw_t,  f_min, f_max)
```

Sign convention for the rule above:

- `mispricing_t > 0` means the AMM price is high relative to the oracle. Selling into the AMM is the adverse-selection/arbitrage direction, so `fee_sell_t` rises and `fee_buy_t` falls.
- `mispricing_t < 0` means the AMM price is low relative to the oracle. Buying from the AMM is the adverse-selection/arbitrage direction, so `fee_buy_t` rises and `fee_sell_t` falls.

This is the closest direct analogue of the paper’s “linear dynamic fees” result. The exact implementation must map `sell`/`buy` to the repository’s token0/token1 swap directions and numéraire conventions before changing simulator behavior.

## Implementation questions for ABM_Uni_v3

Before coding, the following choices should be explicit in YAML and tests:

1. Direction convention: define whether `fee_sell` means selling token0, selling token1, or selling the risky asset `Y` into the pool.
2. Oracle convention: use the existing CEX/reference price and state whether it is token1 per token0 or token0 per token1.
3. Mispricing signal: choose log-price gap, tick gap, inventory gap, or a combination.
4. Timing: compute the asymmetric fees from block-open information and apply them to same-block flow, preserving causal timing.
5. Bounds: decide whether negative fees are allowed; default ABM implementation should probably clamp to non-negative `f_min` unless the experiment explicitly tests rebates.
6. Step limits: decide whether sell and buy fees share the existing `fee_step_bps_min`/`fee_step_bps_max` logic or have separate limits.
7. Diagnostics: record both fee paths separately, plus the signed mispricing signal used to set them.

## Minimal verification checks for implementation

A later code change should include at least these checks:

- when `P_dex = P_oracle`, the two directional fees are equal to the baseline fee, up to clipping;
- when `P_dex > P_oracle`, the sell/adverse-direction fee is greater than or equal to the buy/opposite-direction fee;
- when `P_dex < P_oracle`, the buy/adverse-direction fee is greater than or equal to the sell/opposite-direction fee;
- fee bounds and per-step update limits are respected independently for both directions;
- the fee signal is causal: it uses only block-open or previously committed state, not realized same-block outcomes.

## Known limitations of this reading note

- The paper’s model has fixed AMM depth and no LP add/remove behavior over the horizon; this differs from the full ABM in this repository.
- The paper maximizes venue fee revenue, while this repository often evaluates LP PnL, LVR, arbitrage, JIT behavior, and welfare-style diagnostics.
- The paper’s fees are paid outside the pool and can be negative; Uniswap-style implementation constraints may require non-negative, protocol-valid fees.
- The paper uses a discrete inventory grid and HJB approximations; a Uniswap v3 pool has concentrated liquidity, ticks, and potentially multiple LP ranges.
