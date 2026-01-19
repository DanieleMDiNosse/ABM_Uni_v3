---
title: Agent Behaviour Details
nav_order: 3
---

# Agent Behaviour Details

This page documents the *implemented* agent logic in `run.py`.
It is intentionally “code-first”: when in doubt, trust the implementation over any paper/spec.

The most important “global semantics” to understand before reading any individual agent:
- **Discrete blocks with micro-steps**: one simulation step `t` is one “block”. Inside each block there are `block_time = B` *micro-steps* where only the CEX price diffuses and trader intents arrive.
- **Validated snapshots for intent formation** (block start): `agent_S_ref`, `agent_tick_ref`, `cex_ref_for_agents` are copied from the **end of the previous block** and held fixed as the “validated state” for many decisions (slippage baselines, LP recenter/out-of-range counting, arb target price).
- **Mempool execution at the block boundary**: intents are collected during micro-steps, but the DEX (pool) state does not change until the mempool is replayed at the end of the block.
- **Execution ordering**: mempool replay executes **arbitrage first**, then shuffles all other intents uniformly at random; JIT liquidity, when enabled, is inserted as `jit_mint → target_swap → jit_burn` around one targeted swap.
- **CEX impact is immediate**: every time an action “touches the CEX” (smart-router CEX routing, arb hedge leg, LP mint/burn conversion, JIT burn conversion), permanent impact is applied immediately via `ref.apply_impact_only(Δa)` and the rebalancing benchmark is updated via `_broadcast_price_move(ref.m)`.
---

## Notation (as used in code)
- Tokens:
  - `X` / token0: “ETH-like”
  - `Y` / token1: “USDC-like”
- Prices:
  - `m = ref.m`: CEX mid price in `Y per X`
  - `S = pool.S`: DEX sqrt-price, so `P = pool.price = S²` is also `Y per X`
- Fees:
  - `f = pool.f`: taker fee
  - `r = pool.r = 1 - f`: fee multiplier used in quotes/swaps (fee-on-input)
- Time:
  - `t`: block/step index, `t = 0, 1, …, T-1`
  - `k`: micro-step index inside a block, `k = 0, 1, …, B-1`
- “Validated snapshot” variables (copied at block start from end of previous block):
  - `agent_S_ref`: validated DEX sqrt-price
  - `agent_tick_ref`: validated active tick
  - `cex_ref_for_agents`: validated CEX price

---

## Block timeline (what happens when)

At a high level, each block `t` follows this structure:

1) **Freeze validated snapshot** (decision reference):
   - `agent_S_ref ← validated_S`
   - `agent_tick_ref ← validated_tick`
   - `cex_ref_for_agents ← validated_cex`

2) **(Optional) apply a scheduled fee update** (dynamic fee controller commit→reveal):
   - If `fee_next` is staged and cooldown allows it, the pool fee `pool.f` is updated at the *start* of the block.

3) **Update bookkeeping benchmark** (LVR rebalancer) at block start:
   - `_rebalance_all(ref.m, pool.S)` aligns the benchmark exposures to current positions before any within-block moves.

4) **Compute the active-LP width signal** `w_ticks` once per block:
   - Uses an EWMA of `|log m_t - log m_{t-1}|` plus binomial noise; see “Narrow-LP width signal”.

5) **Micro-step loop (mempool intake; DEX frozen)**
   - Repeat `B = block_time` times:
     - Sample smart-router arrivals: `n_smart ~ Poisson(smart_trades_per_block / B)` and enqueue smart intents or execute on CEX immediately.
     - Sample noise-trader arrivals: `n_noise ~ Poisson(noise_trades_per_block / B)` and enqueue noise intents.
     - Diffuse the CEX price: `ref.diffuse_only()`, then `_broadcast_price_move(ref.m)`.
   - During this loop, **DEX price does not move**; only CEX moves.

6) **Append block arb intent** (executes first later):
   - `mempool_orders.append({"type": "arb", "arb_ref_m": cex_ref_for_agents})`

7) **Schedule LP intents into the same mempool**
   - Determine which LPs are “due” this block (geometric review clock with mean `tau`, plus cooldown handling).
   - Enqueue burns, recenters, and new mints (Poisson targets) as `lp_burn`, `lp_recenter`, `lp_mint` orders.

8) **Plan JIT target (optional)**
   - If Jiter is enabled and triggers this block (`Bernoulli(p_jit)`), select the **single largest swap intent** by input size (normalized to token1 using `cex_ref_for_agents`) and mark it as a JIT target.

9) **Replay the mempool (DEX execution)**
   - Execute all mempool orders with:
     - **arb first**, then
     - **all non-arb orders shuffled**, with
     - `jit_mint → target_swap → jit_burn` inserted around the targeted swap (if any).
   - LP mint/burn conversions and arb hedge legs apply immediate CEX impact during this phase.

10) **Settle PnL and update “validated” snapshot**
   - Traders’ DEX flows are settled at the end-of-block CEX price `ref.m`.
   - The arb accumulator is settled at `arb_ref_m = cex_ref_for_agents` (validated snapshot used for the arb target).
   - Then:
     - `validated_S ← pool.S`
     - `validated_tick ← pool.tick`
     - `validated_cex ← ref.m`

---

## Mempool order types (actual dict payloads)

The mempool is a Python list of dictionaries `mempool_orders`. These are the order types the engine understands:

- **Swap intent** (smart/noise):
  - `{"type": "swap", "agent": "smart"|"noise", "side": "X_to_Y"|"Y_to_X", "amount": dx_or_dy, "unit": "dx"|"dy", "m_submit": m_now, "min_output": min_out, "jit_target": optional_id}`
- **Arb intent**:
  - `{"type": "arb", "arb_ref_m": cex_ref_for_agents}`
- **LP mint**:
  - `{"type": "lp_mint", "lp_id": lp.id, "lower": tick, "upper": tick, "eta": eta}`
- **LP burn**:
  - `{"type": "lp_burn", "lp_id": lp.id, "lower": tick, "upper": tick, "L": pos.L}`
- **LP recenter** (burn+mint in one atomic mempool action):
  - `{"type": "lp_recenter", "lp_id": lp.id, "old_lower": tick, "old_upper": tick, "old_L": pos.L, "new_lower": tick, "new_upper": tick, "eta": eta}`
- **JIT wrappers** (injected at execution time, not part of original intake):
  - `{"type": "jit_mint", "target_id": some_id}`
  - `{"type": "jit_burn", "target_id": some_id}`

---

## Arbitrageur
- Encoded by `preview_arbitrage_to_target(...)`, `arbitrage_to_target(...)`, and the `type == "arb"` branch inside `execute_mempool_orders()`.
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
  - A tiny tolerance is used around the band (`1e-9`) to avoid noisy flip-flopping.
- **Profitability filter (pre-trade)**:
  - The arb **previews** the full path on a cloned pool and computes an *expected* net profit that includes flash funding.
  - The arb is skipped if the preview direction is `None` or if `expected_profit <= 0`.
  - Preview uses the same fee-on-input model (`pool.r`) and tick-crossing logic as the live arb, but does **not** allocate fees to LP positions (it is a dry-run).
- **Profit computation (as coded)**:
  - Let `prev_in` be the pre-fee DEX input required to hit the band edge in the preview.
  - For `"up"` (DEX cheap, input is `dy`):
    - `expected_flash_fee = (flash_loan_mult - 1) * prev_in`
    - `expected_profit = prev_x_out * arb_ref_m - flash_loan_mult * prev_in`
  - For `"down"` (DEX expensive, input is `dx`):
    - `notional_y = prev_in * arb_ref_m`
    - `expected_flash_fee = (flash_loan_mult - 1) * notional_y`
    - `expected_profit = prev_y_out - flash_loan_mult * notional_y`
- **Execution + hedge semantics**:
  - If the DEX is cheap (“up”): buy token0 on DEX (input token1), then **sell** the token0 output on the CEX; CEX impact is applied immediately.
  - If the DEX is expensive (“down”): sell token0 on DEX (input token0), then **buy** token0 on the CEX to hedge/repay; CEX impact is applied immediately.
  - In both directions the DEX leg is executed via `swap_exact_to_target(...)` with span-by-span fee allocation:
    - `"up"` allocates fees in token1 per span
    - `"down"` allocates fees in token0 per span
  - LP rebalancers touched by the arb swap are batch-updated via `_flush_pending_rebalance()` right after the arb swap finishes.
- **PnL measurement**:
  - The arb accumulator records only the DEX leg flows (with flash funding folded into the input via `flash_loan_mult`).
  - **Settlement uses `arb_ref_m` (the validated snapshot), not the end-of-block `ref.m`**:
    - `arb_acc.settle(arb_ref_m)`
  - The hedge leg does still move the simulated CEX price immediately via impact (`ref.apply_impact_only(...)`), affecting *everyone else* and future steps; it is just not re-marked into the arb PnL number.

---

## Traders (Smart Router + Noise Trader)

Both trader types share the same size distribution and micro-step arrival process:
- **Arrival per micro-step**:
  - `n_smart ~ Poisson(smart_trades_per_block / block_time)`
  - `n_noise ~ Poisson(noise_trades_per_block / block_time)`
- **Size** (token1 notional):
  - `Y_notional ~ exp(N(trader_mean, trader_sigma²))`
- **Direction**:
  - Randomly chosen each intent: `side ∈ {X_to_Y, Y_to_X}`

### Smart Router
- Implemented via `maybe_enqueue_smart_router_intent(m_now)` (submission during micro-steps) and the `'agent': 'smart'` branches in `execute_mempool_orders()` (execution at block boundary).
- **Arrival process**:
  - During each micro-step: `N ~ Poisson(smart_trades_per_block / block_time)`.
- **Trade sizing and direction**:
  - Draw token1 notional `Y_not ~ exp(N(trader_mean, trader_sigma^2))`.
  - Random direction `side ∈ {X_to_Y, Y_to_X}`.
- **Best-execution routing (implemented)**:
  - At submission, the router compares a DEX quote against a CEX benchmark using the *current* CEX price `m_now = ref.m` (this is the current micro-step price, not the validated snapshot).
  - If the DEX quote is not competitive, the trade is executed **on the CEX immediately**:
    - **X_to_Y**: sell token0 on CEX (negative `Δa`), receive `dy = dx * m_now`.
    - **Y_to_X**: buy token0 on CEX (positive `Δa`), receive `dx = dy / m_now`.
    - CEX impact is applied immediately; this route is recorded as “executed” with **realized PnL = 0** (fair exchange at the current CEX price), i.e. it bypasses later DEX settlement revaluation.
  - Otherwise, the router enqueues a DEX swap intent into the mempool.
- **Competitiveness test (exact comparisons in code)**:
  - For `X_to_Y`:
    - Convert the drawn token1 notional to a token0 size using the current CEX: `dx = Y_notional / m_now`.
    - Compute a live DEX quote: `dy_out_dex = pool.quote_x_to_y(dx)`.
    - Compare against the CEX value in token1: `dy_out_cex = dx * m_now`.
    - Route to CEX if: `dy_out_dex < theta_T * dy_out_cex`.
  - For `Y_to_X`:
    - Use `dy = Y_notional`.
    - Compute a live DEX quote: `dx_out_dex = pool.quote_y_to_x(dy)`.
    - Compare against the CEX output in token0: `dx_out_cex = dy / m_now`.
    - Route to CEX if: `dx_out_dex < theta_T * dx_out_cex`.
- **Execution-time slippage gate (implemented)**:
  - At mempool replay, the engine re-quotes the live pool and **skips** if `final_quote <= min_output` (note the inclusive `<=`).
  - `min_output` is computed at submission from a *baseline* that uses:
    - the **validated DEX snapshot price** `agent_S_ref²`, and
    - the **current pool fee** via `pool.r`.
    - Baseline for `X_to_Y`: `dx * pool.r * (agent_S_ref^2)`.
    - Baseline for `Y_to_X`: `(dy * pool.r) / (agent_S_ref^2)`.
    - `min_output = baseline * (1 - slippage_tolerance)`.
- **DEX execution details**:
  - Swaps execute via `pool.swap_x_to_y(dx)` or `pool.swap_y_to_x(dy)` with fee allocation callback `allocate_fees`.
  - After each swap, `_flush_pending_rebalance()` updates the LVR benchmark for LPs whose positions were touched.
  - If the swap ends up being a numerical/no-liquidity no-op (`used_amount <= EPS_LIQ`), the pool tick/S is rolled back to the pre-swap state.
- **PnL settlement**:
  - DEX swap flows are revalued at the end-of-block CEX price: `sr_acc.settle(settlement_m)` with `settlement_m = ref.m`.
  - CEX-routed trades contribute `0` realized PnL (by construction) and are excluded from later settlement revaluation.

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
  - After burns, LPs enter a cooldown (random 3–8 blocks) during which they are not selected as “due” (and therefore will not schedule new actions). Cooldown is enforced block-to-block; it does not retroactively cancel already-enqueued intents in the current mempool.
  - Block-level Poisson targets schedule actions among eligible LPs:
    - `narrow_mints_per_block`, `passive_mints_per_block`, `passive_burns_per_block`.
- **LP types in the code**:
  - **Active narrow** (`is_active_narrow=True`, `is_passive=False`): can TP/SL burn and recenter; uses the dynamic `w_ticks` width signal for new mints and recenters.
  - **Passive** (`is_passive=True`): does not TP/SL burn or recenter; can be randomly burned (`passive_burns_per_block`) and can mint with a passive width rule.
  - **Seed/background LPs** (`is_seed=True`): created by `bootstrap_initial_binomial_hill_sharded(...)` to provide initial liquidity; excluded from cohort PnL/wealth aggregates but otherwise behave like passive LPs in scheduling.
- **Initialization (implemented)**:
  - **Active vs passive composition**:
    - The simulation assigns *exactly* `round((1 - passive_lp_share) * N_LP)` LPs as “active narrow” by shuffling LP indices once (seeded RNG) and taking the first `target_active`.
    - The remaining LPs are marked passive.
  - **Initial liquidity (“binomial hill”)**:
    - The pool is bootstrapped by `bootstrap_initial_binomial_hill_sharded(pool, ref, LPs, N=initial_binom_N, L_total=initial_total_L, num_seed_lps=20, seed_lp_id_base=10_000, seed_is_passive=True)`.
    - This creates 20 seed LPs (IDs `10_000…10_019`) and distributes one-tick positions across them so their future burns are staggered.
  - **Initial cash inventories (cash-budgeted LPs)**:
    - Strategic LPs (non-seed, non-JIT) start with token1 cash only:
      - Compute total seed principal at `t=0`: `seed_total_x`, `seed_total_y`.
      - Value it at the initial CEX price `m0`: `initial_seed_value_y = seed_total_x * m0 + seed_total_y`.
      - Set each strategic LP’s wallet to `wallet_y = initial_seed_value_y / N_LP`, `wallet_x = 0`.
    - Seed LPs start with their liquidity already deployed as positions and begin with `wallet_y = 0`.
  - **Note**: `LPAgent.mintProb` is set to `0.0` and is not used for targeting; mint/burn events are scheduled explicitly via Poisson targets in the mempool scheduler.
- **Active narrow TP/SL burns (implemented)**:
  - For active narrow LPs only (`is_passive=False`), each open position is evaluated with `pos.PnL_y(agent_S_ref, ref.m)` and burned if it breaches:
    - TP: `pnl >= theta_TP * pos.hodl0_value_y`
    - SL: `pnl <= -theta_SL * pos.hodl0_value_y`
- **Passive random burns (implemented)**:
  - For passive LPs, burns are not TP/SL-triggered. Instead:
    - Collect all positions belonging to **due** passive LPs.
    - Draw `n_burn_intents ~ Poisson(passive_burns_per_block)` and sample that many positions uniformly without replacement.
    - Enqueue each sampled position as an `lp_burn` intent.
- **Recenter rule (implemented)**:
  - Each position tracks consecutive out-of-range blocks (`out_steps`), based on the **validated** tick snapshot `agent_tick_ref`.
  - If `out_steps >= k_out_threshold` (per-LP random integer in `[k_out_min, k_out_max]`), the LP enqueues:
    1) burn of the old position, then
    2) mint of a new range centered around `agent_S_ref` using the **current** `w_ticks` (same width signal as new narrow mints).
- **Cash-budgeted minting (implemented)**:
  - Strategic LPs hold **token1 cash only** (`wallet_y`); `wallet_x` is kept at 0 by construction.
  - A utilization factor is drawn as:
    - `z ~ LogNormal(mint_mu, mint_sigma)`
    - `eta = min(1, z)`  (with guards for non-finite draws)
  - For a candidate range and reference snapshot `S_ref = agent_S_ref`, with CEX price at execution `m_exec = ref.m`:
    - Compute deposit coefficients `(a0, a1)` using `minted_amounts_at_S(1.0, sa, sb, S_ref)`.
    - `L_max = wallet_y / (a0 * m_exec + a1)`, `L_new = eta * L_max`.
  - **Mint composition (as implemented)**:
    - The token amounts used for wallet debits and for the stored `Position.amt*_init` are computed at the **validated snapshot** `S_ref = agent_S_ref` via `minted_amounts_at_S(L_new, sa, sb, S_ref)` (not at the live execution-time `pool.S`).
    - This is a bookkeeping convention: the pool only tracks liquidity, while per-position amounts are tracked in `Position` objects.
  - **Immediate CEX conversion + impact**:
    - On mint: LP effectively buys the required token0 on CEX (positive `Δa`) at `m_exec`; impact is applied immediately.
    - On burn: LP converts any withdrawn token0 into token1 on CEX (negative `Δa`) at `m_exec`; impact is applied immediately.
  - **Cashflow identities (how wallets are updated)**:
    - On mint, after choosing `(amt0, amt1)` and execution-time CEX price `m_exec`:
      - `cost_y = amt0 * m_exec + amt1`
      - `wallet_y ← wallet_y - cost_y`, `wallet_x ← 0`
    - On burn, at execution-time `(S_exec = pool.S, m_exec = ref.m)`:
      - Compute current principal: `(amt0_now, amt1_now) = pos.current_amounts(S_exec)`
      - Include uncollected fees: `amt0_total = amt0_now + fees0`, `amt1_total = amt1_now + fees1`
      - Convert to token1: `wallet_y ← wallet_y + amt1_total + amt0_total * m_exec`, `wallet_x ← 0`
      - Set `lp.cooldown ← randint(3, 8)` (implemented as `np.random.randint(3, 9)`).
- **Mint range construction (tick-based, active and passive fallback)**:
  - For a desired total width in ticks `width_ticks`:
    - `n_bands = max(1, round(width_ticks / tick_spacing))`
    - `upper = lower + n_bands * tick_spacing`
  - The “centered around `S_now`” logic solves for `lower` such that the arithmetic mean of sqrt-price boundaries equals `S_now`:
    - Let `g = pool.g` and `base_s = pool.base_s` so that `S(tick) = base_s * g^tick`.
    - Define `denom = 1 + g^(n_bands * tick_spacing)`.
    - Compute a real-valued `lower_real = log((2*S_now/base_s)/denom, g)` and snap to the tick grid.
- **Passive LP ranges**:
  - Passive LPs can be parameterized either by `passive_width_pct` (preferred) or `passive_width_ticks` fallback.
  - When `passive_width_pct` is set, the code builds a symmetric ±% band in **price**, snaps to tick spacing, and ensures `upper > lower`.
  - Specifically, with `half = passive_width_pct / 200`:
    - `P_low = P_now * (1 - half)`, `P_high = P_now * (1 + half)`
    - `S_low = S_now * sqrt(1 - half)`, `S_high = S_now * sqrt(1 + half)`
    - ticks are computed from `log(S/base_s, g)` and then snapped so that `lower` is floored to the tick grid and `upper` is ceiled to the next tick grid boundary.

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
  - **Implementation detail**:
    - The targeted swap dict is tagged in-place with `o["jit_target"] = id(o)`.
    - During mempool replay, the execution engine injects wrapper orders around any swap carrying that tag.
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
  - If the JIT is skipped at this stage, the engine removes the `jit_target` bookkeeping entry so the later `jit_burn` becomes a no-op.
- **Flash-funded accounting model (implemented)**:
  - Jiter is treated as flash-funded, so it is allowed to run negative inventories temporarily:
    - At mint, the model subtracts the minted principal (and the flash fee) from the Jiter wallet.
    - The flash fee is also accumulated in `flash_fees_paid_y`.
  - Unlike regular LP mints, Jiter does not apply a CEX conversion impact at mint time (it is “borrowed”, not bought).
- **Burn + conversion (implemented)**:
  - After the targeted swap, burns the one-tick position, converts any token0 to token1 at the current `ref.m`, and applies immediate CEX impact from that conversion.
  - The JIT “success” activity marker is recorded only if the targeted swap actually executed (i.e., it was not skipped by the execution-time slippage gate).

---

## Rebalancing benchmark (LVR bookkeeping) — how LP “hedged PnL” is computed

This model computes loss-versus-rebalancing (LVR) by comparing each LP’s wealth to a *delta-hedged benchmark* tracked in `LPAgent.rebalancer` (`RebalancerState` in `agents.py`).

Key mechanics:
- **Benchmark token0 exposure**:
  - `x_target = lp_token0_exposure(lp, S_now)` includes token0 in wallet plus token0 principal plus uncollected token0 fees.
- **Benchmark accrual on every CEX move**:
  - Whenever the CEX price changes (diffusion or impact), `_broadcast_price_move(M_new)` accrues:
    - `ΔR = x_prev * (M_new - last_M)`
    - `cumulative_R += ΔR`
    - `last_M ← M_new`
- **Rebalancing to the new exposure**:
  - `_rebalance_lp_to_target(lp, M_now, S_now)` updates the benchmark’s `x_prev` to match `x_target` (and adjusts benchmark cash) whenever:
    - positions change (mint/burn), or
    - the pool price moves and a position was “touched” by a swap (batched through `_flush_pending_rebalance()`), or
    - the engine calls `_rebalance_all(...)` at block boundaries.
- **Reported hedged PnL**:
  - At end of block, LP wealth is marked to market in token1:
    - `V_lp = lp_wealth_y(lp, pool.S, ref.m)`
  - The benchmark value is:
    - `V_reb = initial_rebal_value_y + cumulative_R`
  - Hedged PnL is reported as:
    - `PnL_hedged = V_lp - V_reb`
  - (The model also records fees earned and can compute `LVR = fees_value - PnL_hedged` at the cohort level.)

---

## CEX impact model (ReferenceMarket)

Whenever an agent “touches the CEX”, `run.py` applies **permanent additive impact** immediately:
- The signed CEX trade size is `Δa` in **token0 units**:
  - `Δa > 0`: buy token0 (pushes `m` up)
  - `Δa < 0`: sell token0 (pushes `m` down)
- Impact function (from `ReferenceMarket.apply_impact_only` in `utils.py`):
  - `impact = kappa * sign(Δa) * |Δa|^(1 + xi)`
  - `m ← max(1e-12, m + impact)`
- In `run.py`, `kappa` is set when constructing the reference market (currently `kappa = 1e-3`).

Diffusion is separate and happens via `ref.diffuse_only()` during the micro-step loop:
- In non-Heston modes, this is a GBM-style multiplicative update `m ← m * exp( … )` using `mu` and the current `sigma`.
- In Heston mode, variance is evolved first and then used for the price update.

---

## DEX fee allocation (how LPs earn fees)

Swaps charge fees on the **input** token (fee-on-input). During swap execution, the pool calls the callback `allocate_fees(...)` once per tick-span segment:
- Callback signature: `allocate_fees(token, fee_amt, tick_snapshot, L_snapshot)`
  - `token ∈ {"x","y"}` indicates which token the fee is paid in (matches swap direction).
  - `tick_snapshot` is the tick band being traversed for that segment.
  - `L_snapshot` is the active liquidity in that band before the segment is applied.
- Positions are indexed by tick coverage in `positions_by_tick[tick]`.
  - Fees are distributed pro-rata within that tick band: each position gets `fee_amt * (pos.L / total_L_in_band)`.
  - Token0 fees accumulate in `pos.fees0`; token1 fees accumulate in `pos.fees1`.
  - LP-level cumulative fee counters `lp.fees0_earned` / `lp.fees1_earned` are also incremented.
- Even if `fee_amt == 0`, the callback still marks LP owners as “touched” so the LVR benchmark can be rebalanced consistently after swaps.
