---
title: Agent Behaviour Details
nav_order: 3
---

# Agent Behaviour Details

This page describes the *implemented* agent logic in `run.py` (snapshot timing, mempool ordering, slippage/best-ex checks, and how CEX impact is applied).

Key global semantics used throughout:
- **Validated snapshot for agent decisions** (at block start): `agent_S_ref`, `agent_tick_ref`, `cex_ref_for_agents` are copied from the *end of the previous block* and held fixed for intent formation.
- **Mempool execution**: intents are collected during micro-steps; at the block boundary the mempool is replayed with **arb first**, then everything else shuffled.
- **CEX impact timing**: permanent impact is applied **immediately** whenever the CEX is touched (arb hedges, smart-router CEX routing, LP mint/burn conversions, Jiter burn conversions). There is no “apply net impact at end-of-block” aggregation in the current engine.

---

## Arbitrageur
- Encoded by `preview_arbitrage_to_target(...)` + `arbitrage_to_target(...)` and the `typ == "arb"` branch inside `execute_mempool_orders`.
- **Information / timing**:
  - The arb target uses the **validated** CEX snapshot `arb_ref_m = cex_ref_for_agents` (end of previous block), not the intra-block diffused `ref.m`.
  - The arb intent is inserted once per block and executed **first** in the mempool replay.
- **No-arb band** (as implemented):
  - Let taker fee `f_t = pool.f`, `r_t = 1 - f_t`, and `flash_loan_mult = 1 + flash_loan_fee`.
  - The arb band is constructed from the snapshot `m_ref = cex_ref_for_agents` as:
    $$
      P^{\min} = \frac{m_{\text{ref}}\,r_t}{1+\phi},\qquad
      P^{\max} = \frac{m_{\text{ref}}(1+\phi)}{r_t}.
    $$
- **Profitability filter (pre-trade)**:
  - The arb **previews** the full path on a cloned pool and computes an *expected* net profit that includes flash funding.
  - The arb is skipped if the preview direction is `None` or if `expected_profit <= 0`.
- **Execution + hedge semantics**:
  - If the DEX is cheap (“up”): buy token0 on DEX (input token1), then **sell** the token0 output on the CEX; CEX impact is applied immediately.
  - If the DEX is expensive (“down”): sell token0 on DEX (input token0), then **buy** token0 on the CEX to hedge/repay; CEX impact is applied immediately.
  - In both directions the DEX leg is executed via `swap_exact_to_target(...)` with span-by-span fee allocation; LP rebalancers touched by the swap are batch-updated right after the arb swap.
- **PnL measurement**:
  - DEX flow PnL is settled at the **end-of-step** CEX price `settlement_m = ref.m` (after all immediate impacts), while the arb *decision* and band are still formed off `cex_ref_for_agents`.

---

## Smart Router
- Implemented via `maybe_enqueue_smart_router_intent(...)` (submission during micro-steps) and the `'agent': 'smart'` branches in `execute_mempool_orders` (execution at block boundary).
- **Arrival process**:
  - During each micro-step: `N ~ Poisson(smart_trades_per_block / block_time)`.
- **Trade sizing and direction**:
  - Draw token1 notional `Y_not ~ exp(N(trader_mean, trader_sigma^2))`.
  - Random direction `side ∈ {X_to_Y, Y_to_X}`.
- **Best-execution routing (implemented)**:
  - At submission, the router compares a DEX quote against a CEX benchmark using the *current* CEX price `m_now = ref.m`.
  - If the DEX quote is not competitive, the trade is executed **on the CEX immediately**:
    - **X_to_Y**: sell token0 on CEX (negative `Δa`), receive `dy = dx * m_now`.
    - **Y_to_X**: buy token0 on CEX (positive `Δa`), receive `dx = dy / m_now`.
    - CEX impact is applied immediately; this route is recorded as “executed” with **realized PnL = 0** (fair exchange at the current CEX price), i.e. it bypasses later DEX settlement revaluation.
  - Otherwise, the router enqueues a DEX swap intent into the mempool.
- **Execution-time slippage gate (implemented)**:
  - At mempool replay, the engine re-quotes the live pool and **skips** if:
    `final_quote <= min_output`.
  - `min_output` is computed at submission from a *baseline* that uses the **validated DEX snapshot price** and the **current fee**:
    - Baseline for `X_to_Y`: `dx * pool.r * (agent_S_ref^2)`.
    - Baseline for `Y_to_X`: `(dy * pool.r) / (agent_S_ref^2)`.
    - `min_output = baseline * (1 - slippage_tolerance)`.

---

## Noise Trader
- Implemented via `maybe_enqueue_noise_trader_intent(...)` and the `'agent': 'noise'` branches in `execute_mempool_orders`.
- Same size process and arrival process as smart router, but **no best-execution check**: it always enqueues a DEX swap intent (subject to the same execution-time slippage gate based on the validated DEX snapshot baseline described above).
- Trades execute in shuffled mempool order (arb remains first), and DEX flow PnL is settled at end-of-step `ref.m`.

---

## Liquidity Providers (strategic passive + active narrow)
- LP objects: `LPAgent`, `Position`; management logic in `run.py` via mempool intents `lp_mint`, `lp_burn`, `lp_recenter`.
- **Scheduling / eligibility**:
  - Each non-JIT LP has a review clock (`next_review`) drawn from a geometric distribution with mean `tau`. Only “due” LPs can act.
  - After burns, LPs enter a cooldown (random 3–8 blocks) during which they cannot mint.
  - Block-level Poisson targets schedule actions among eligible LPs:
    - `narrow_mints_per_block`, `passive_mints_per_block`, `passive_burns_per_block`.
- **Active narrow TP/SL burns (implemented)**:
  - For active narrow LPs only (`is_passive=False`), each open position is evaluated with `pos.PnL_y(agent_S_ref, ref.m)` and burned if it breaches:
    - TP: `pnl >= theta_TP * pos.hodl0_value_y`
    - SL: `pnl <= -theta_SL * pos.hodl0_value_y`
- **Recenter rule (implemented)**:
  - Each position tracks consecutive out-of-range blocks (`out_steps`), based on the **validated** tick snapshot `agent_tick_ref`.
  - If `out_steps >= k_out_threshold` (per-LP random integer in `[k_out_min, k_out_max]`), the LP enqueues:
    1) burn of the old position, then
    2) mint of a new range centered around `agent_S_ref` using the **current** `w_ticks` (same width signal as new narrow mints).
- **Cash-budgeted minting (implemented)**:
  - Strategic LPs hold **token1 cash only** (`wallet_y`); `wallet_x` is kept at 0 by construction.
  - A utilization factor is drawn as:
    - `z ~ LogNormal(mint_mu, mint_sigma)`
    - `eta = z / (1 + z)`  (note: this differs from a `min(1, z)` spec)
  - For a candidate range and reference snapshot `S_ref = agent_S_ref`, with CEX price at execution `m_exec = ref.m`:
    - Compute deposit coefficients `(a0, a1)` using `minted_amounts_at_S(1.0, sa, sb, S_ref)`.
    - `L_max = wallet_y / (a0 * m_exec + a1)`, `L_new = eta * L_max`.
  - **Execution-time composition**: token amounts are computed using the *live* pool price `S_exec = pool.S` at execution.
  - **Immediate CEX conversion + impact**:
    - On mint: LP effectively buys the required token0 on CEX (positive `Δa`) at `m_exec`; impact is applied immediately.
    - On burn: LP converts any withdrawn token0 into token1 on CEX (negative `Δa`) at `m_exec`; impact is applied immediately.
- **Passive LP ranges**:
  - Passive LPs can be parameterized either by `passive_width_pct` (preferred) or `passive_width_ticks` fallback.
  - When `passive_width_pct` is set, the code builds a symmetric ±% band in **price**, snaps to tick spacing, and ensures `upper > lower`.

---

## Narrow-LP width signal (implemented)
This is the signal used for **new narrow mints** and **recenters**.

- The engine updates an EWMA of **absolute** CEX log-returns once per block:
  $$
  v_t = \left|\log m_t - \log m_{t-1}\right|,\qquad D_t = \text{EWMA}(v_t;\; \text{half-life}=\text{basis_half_life})
  $$
- Convert to “ticks” units using `TICK_LN = log(1.0001)`:
  $$
  \text{vol\_ticks}_t = D_t / \log(1.0001)
  $$
- Add mean-zero binomial noise (in tick units, snapped to spacing):
  $$
  K_t \sim \text{Binomial}(\text{binom\_n}, \text{binom\_p}),\quad
  \text{noise\_ticks}_t = (K_t - n p)\cdot \text{tick\_spacing}
  $$
- Width before snapping:
  $$
  w^{\text{raw}}_t = w_{\min} + \text{slope\_s}\cdot \text{vol\_ticks}_t + \text{noise\_ticks}_t
  $$
- Then the code snaps to the tick grid and clamps to `[w_min_ticks, w_max_ticks]` in *band units* (multiples of `tick_spacing`).

---

## Jiter (JIT LP searcher, implemented)
- Enabled when `p_jit > 0`, `N_jit > 0`, `liquidity_perc_jit > 0`.
- **Arrival**: each block, joins with Bernoulli probability `p_jit`.
- **Target selection (implemented)**:
  - Observes the current mempool (excluding arb special-casing) and targets the **single largest swap intent** by input size, normalized to token1 using `cex_ref_for_agents`:
    - if intent is `dx` (token0 in), compare `dx * cex_ref_for_agents`
    - if intent is `dy` (token1 in), compare `dy`
  - Wraps the target as: `jit_mint` → target swap → `jit_burn`.
- **Single-tick placement with boundary handling**:
  - Mints a one-tick position `[lower, lower + tick_spacing)` near the current active tick, with a direction-aware adjustment if `pool.S` is numerically on the band boundary (to avoid immediate-cross + numerical blowups).
- **Liquidity sizing (implemented)**:
  - Let `L_existing_band` be active liquidity at the chosen tick band.
  - Share target uses:
    $$
      L_{\text{share}} = \frac{q}{1-q}\,L_{\text{existing}},\quad q=\text{liquidity\_perc\_jit}
    $$
  - Also computes a minimum `L_needed` to keep the targeted swap inside the tick when possible, then sets:
    $$
      L_{\text{target}} = \max\left(L_{\text{share}},\,\max(0,\,L_{\text{needed}} - L_{\text{existing}})\right)
    $$
  - Caps the mint to avoid astronomically large liquidity (a numeric stability guard).
- **Flash funding + profitability filter (implemented)**:
  - Models a flash cost on the **token1 value** of principal at the mint-time CEX price `m_exec = ref.m`:
    - `flash_fee_y = (amt0 * m_exec + amt1) * flash_loan_fee`
  - Computes an *expected fee capture* for the targeted swap:
    - total fee value is `amount_in * pool.f` (if fees in token1) or `amount_in * pool.f * m_exec` (if fees in token0 valued in token1),
    - Jiter’s fee share is `L_target / (L_existing + L_target)`.
  - Skips the JIT if `expected_fee_capture_y - flash_fee_y <= 0`.
- **Burn + conversion (implemented)**:
  - After the targeted swap, burns the one-tick position, converts any token0 to token1 at the current `ref.m`, and applies immediate CEX impact from that conversion.