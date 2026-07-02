---
title: LP Width Mint Signals
nav_order: 10
---

# LP width mint signals (beyond CEX volatility)

## Goal / scope
Explore alternative **signals** an *active (narrow) LP* can use to choose the **range width** when minting a Uniswap v3-style position, beyond the current “CEX realized-vol proxy” rule used in this repo.

This note focuses on *width* (risk/exposure) rather than *center* (where to place liquidity). In practice these are coupled, but separating them keeps hypotheses falsifiable.

---

## Definitions / units (v3-style)

- Tokens: `X = token0`, `Y = token1`.
- Numéraire: values in `Y`.
- Prices:
  - External “efficient” mid (oracle/CEX): $m_t$ in `Y per X`.
  - DEX sqrt-price: $S_t$; DEX price $P^{DEX}_t = S_t^2$ in `Y per X`.
- Tick: $\mathrm{tick}(P) = \lfloor \log_{1.0001}(P) \rfloor$.
- Tick spacing: `tick_spacing` (grid constraint).
- Width (this note): total width in **ticks**, `w_ticks`, snapped to `tick_spacing`.
  - Half-width in ticks is $w/2$.
  - In code/docs you may also see “bands” = multiples of `tick_spacing`.

**Working assumption:** the LP has a fixed capital budget and uses width as its main “gamma/risk knob” (narrower → more concentrated liquidity near spot).

---

## Baseline (current repo)

Current active-LP width rule (see `scripts/run.py` around the “Active LP width rule” block):

- Volatility signal: EWMA of absolute CEX log-return
  $$
  \hat v_t \approx \mathrm{EWMA}(|\Delta \log m_t|)
  $$
  converted to ticks by dividing by `TICK_LN = log(1.0001)`.
- Add a mean-zero **binomial noise** term (in ticks).
- Map to width and clamp:
  $$
  w_t = \mathrm{clip}(w_{\min} + s\,\hat v_t^{\text{(ticks)}} + \epsilon_t,\; w_{\min},\; w_{\max})
  $$
  then snap to the tick grid.

This note proposes **other** candidates for the signal that feeds into the width mapping.

---

## Ranking criteria: “applicability in real AMMs”

I rank ideas primarily by how deployable they are for a real Uniswap v3 LP:

1. **Data availability**: onchain-only > onchain + public oracle > requires offchain feeds > requires low-latency private infra.
2. **Manipulation resistance**: can a small actor spoof the signal cheaply?
3. **Directness**: does the signal connect to the LP’s economic objective (fees vs adverse selection/LVR vs reposition cost)?
4. **Parameter stability**: likely to work across assets/venues without constant retuning.

Applicability is not the same as “best in simulation”; it’s closer to “could a real LP plausibly run this without heroic infra”.

---

## Ranked ideas (summary)

| Rank | Signal family | What it tries to measure | Data required | Applicability |
|---:|---|---|---|---|
| 1 | **Net-edge (fees vs adverse selection / LVR proxy)** | Is LP being paid enough for risk right now? | onchain fee growth + oracle price | High |
| 2 | **Rebalance-cost / latency aware width** | Optimal tradeoff: in-range time vs churn cost | oracle/DEX vol + gas/latency | High |
| 3 | **Liquidity-competition / fee-density** | Where is fee revenue per unit liquidity highest? | onchain liquidity + swap path | High–Med |
| 4 | **Swap markout toxicity** | Are takers “informed” vs future mid? | swaps + oracle mid (markout) | Med |
| 5 | **VPIN-style volume imbalance (AMM-adapted)** | Order-flow imbalance → informed flow probability | swaps (direction+size) | Med |
| 6 | **Cross-venue basis stress** | Arbitrage pressure / adverse selection risk | DEX price + oracle/CEX | Med |
| 7 | **Jump / tail-risk indicators** | “gap risk” that kills narrow ranges | oracle/DEX price series | Med–Low |
| 8 | **MEV / mempool indicators** | Sandwich/JIT intensity (execution toxicity) | mempool + builder data | Low |

Notes:
- Several items still use a volatility estimate somewhere; the difference is **what else** they condition on (fees, toxicity, gas, competition).
- VPIN/toxicity ideas are included because they connect to *adverse selection* rather than “diffusive volatility”.

---

## 1) Net-edge width: fees vs (LVR / adverse selection) proxy

### Idea
Treat width as the LP’s “risk exposure control”. Narrow when the environment offers good compensation (fees high relative to adverse selection), and widen (or stop minting) when compensation is poor.

This is closest to the economic objective:
$$
\text{Expected net} \approx \text{Fees} - \text{Adverse selection / LVR} - \text{Rebalance cost}.
$$

### Concrete signals (examples)
Pick a window $[t-W, t]$ and compute per-unit-liquidity rates:

- **Fee rate**: realized fees earned per unit liquidity (or per unit capital) from onchain fee growth inside the LP’s active range.
- **LVR proxy** (oracle-based): use an “efficient price” $m_t$ and measure how much price movement occurred while the LP was providing liquidity near spot.
  - In practice you can use a markout-style proxy (see section 4) or an oracle-based decomposition (see repo docs `docs/LP_PnL.md` and `docs/LVR_explanation.md`).

Define a net-edge score, e.g.
$$
z_t = \frac{\widehat{\text{Fees}}_t - \widehat{\text{LVR}}_t}{\max(\widehat{\text{Fees}}_t, \epsilon)}.
$$

### Mapping to width
Simple monotone mapping:
- widen when $z_t$ decreases (fees not covering LVR),
- narrow when $z_t$ increases.

Example:
$$
w_t = \mathrm{clip}\Big(w_0 \cdot \exp(-k\,z_t),\; w_{\min},\; w_{\max}\Big).
$$

### Why it may work (and when it won’t)
- Pro: directly answers “is LP being paid for risk?” rather than guessing via volatility alone.
- Con: needs an oracle “efficient price” and can be noisy over short windows; also partially circular (LP’s own width affects realized fee rate).

**Applicability:** high (if the LP already runs an oracle feed, which most serious active LPs do).

**Reference (theory context):**
- Milionis, Moallemi, Roughgarden, Zhang, *Automated Market Making and Loss-Versus-Rebalancing*, arXiv:2208.06046v5 (May 27, 2024). (PDF in this repo: `literature/LVR.pdf`)
- Milionis, Moallemi, Roughgarden, *Automated Market Making and Arbitrage Profits in the Presence of Fees*, arXiv:2305.14604v2 (Jul 23, 2025). (PDF: `literature/LVR-fees.pdf`)

---

## 2) Width from rebalance-cost + expected time-in-range (latency/gas-aware)

### Idea
Width controls how often you go out-of-range and must rebalance. If rebalancing is expensive (gas high, latency high, risk of missing moves), you want **wider** ranges even if volatility is unchanged.

### Concrete signal
Pick:
- a volatility estimate (DEX realized vol, oracle vol, or the existing CEX vol proxy),
- a target rebalance horizon $H$ (in seconds/blocks) derived from gas budget + ops constraints.

Then set width so that the probability of staying in range over $H$ is at least some $p$:
$$
\Pr\big(|\Delta \log m| \le w/2\big) \ge p.
$$

For a diffusive model $\Delta \log m \sim \mathcal N(0, \sigma^2 H)$, this becomes:
$$
\frac{w}{2} \approx z_p\,\sigma\sqrt{H}
\quad\Rightarrow\quad
w \propto \sigma\sqrt{H}.
$$

### Mapping to width
This is already a “width formula”; the key difference vs the current repo is that $H$ (rebalance cadence) is explicit and can depend on:
- gas price / L2 vs L1,
- bot uptime,
- max daily tx budget,
- (optionally) measured JIT/MEV risk (rebalance transactions being targeted).

### Pros/cons
- Pro: very practical; forces a falsifiable tradeoff (more time in range vs more fees per capital).
- Con: still needs a volatility estimate and assumes some model for exit times.

**Applicability:** high (most real active LP strategies already reason in these terms).

---

## 3) Liquidity competition / fee-density around spot (onchain microstructure)

### Idea
In v3, “how narrow should I be?” depends on how much **other** liquidity is sitting near spot and where volume tends to print. If the near-spot region is overcrowded, ultra-narrow positions can have poor marginal fee capture even if volatility is low.

### Onchain-only proxies
- **Local depth**: total active liquidity (and its slope across ticks) near current tick.
- **Fee density**: observed fee growth per tick region, estimated from swap path + tick-crossing frequencies.
- **Volume-at-tick** proxy: histogram of swaps by tick (or by price interval) over a recent window.

### Mapping to width
Choose width to hit a target “share of volume” or “expected fee per unit capital”:
- narrower when fee-density is sharply peaked at spot and competition is low,
- wider when fee density is flatter or the spot region is crowded.

A simple discrete optimizer:
1. Define candidate widths (bands): $w \in \{w_1,\dots,w_K\}$.
2. For each $w$, estimate expected fee capture:
   $$
   \widehat{\text{Fees}}(w) \approx \sum_{\text{ticks in range}} \frac{\text{expected volume at tick} \times f}{\text{total liquidity at tick} + L_{\text{you}}(w)}
   $$
   (heuristic; useful as a ranking even if not perfectly calibrated).
3. Pick the width that maximizes $\widehat{\text{Fees}}(w) - \lambda,\widehat{\text{Risk}}(w) - \text{rebalance cost}(w)$.

### Pros/cons
- Pro: onchain-only and directly linked to v3’s “crowding” economics.
- Con: computationally heavier; volume-at-price estimates are noisy and can be regime-dependent.

**Applicability:** High–Med (very plausible, but the “expected volume at tick” estimation is work).

---

## 4) Swap markout toxicity (adverse selection measured directly)

### Idea
Compute “markouts” of DEX swaps relative to a future efficient price. If takers consistently get better-than-future prices, flow is toxic for LPs → widen (or increase fees; but here we focus on width).

### Concrete signal (per swap $i$)
Let $p_i$ be the swap’s execution price and $m_{t_i+\Delta}$ an oracle mid after some delay $\Delta$ (e.g., 1 block, 30s, 5m). Define a signed markout:
$$
\text{markout}_i
= s_i \cdot \frac{m_{t_i+\Delta} - p_i}{p_i}
$$
where $s_i = +1$ if the swap is “buy X” (price-up direction) and $s_i=-1$ if “sell X”.

Aggregate over a rolling window with notional weights to get $\widehat{\text{tox}}_t$.

### Mapping to width
- widen when toxicity increases (positive markouts),
- narrow when toxicity is low/negative.

### Pros/cons
- Pro: interpretable “ground truth” microstructure metric for informed flow.
- Con: requires an oracle mid and a delay choice $\Delta$; susceptible to measurement noise in low-volume pools.

**Applicability:** medium (used in TradFi market making; less common for onchain LPs but feasible).

---

## 5) VPIN-style order flow imbalance (AMM-adapted)

### Idea
VPIN (Volume-synchronized Probability of Informed Trading) estimates informed trading probability from persistent buy/sell imbalance in equal-volume buckets.

Even if the original VPIN is a CEX construct, a *variant* can be computed from AMM swap events:
- swaps have a clear direction,
- sizes are observable,
- you can build volume buckets in `Y` notional.

### Concrete signal (sketch)
1. Choose a bucket volume $V$ in `Y` units (e.g., $V = 10^5$ USDC).
2. Stream swaps and fill buckets until total notional reaches $V$.
3. In each bucket $b$, compute buy and sell volume, $V_b^+$ and $V_b^-$, and imbalance:
   $$
   I_b = \frac{|V_b^+ - V_b^-|}{V}.
   $$
4. VPIN-like signal:
   $$
   \text{VPIN}_t = \frac{1}{B} \sum_{b=t-B+1}^{t} I_b.
   $$

### ABM-specific considerations (this repo)

In this simulator, “DEX order flow” is a **mixture of three distinct sources**, and VPIN behaves very differently depending on whether you include each component.

#### 5.1 Flow decomposition: who generates DEX swaps?

At the block level (`scripts/run.py`):

- **Noise trader (`agent: noise`)**:
  - Always submits DEX swaps (no best-ex check).
  - Direction is random (`X_to_Y` vs `Y_to_X`).
  - Notional is sampled from the **same** log-normal as smart router:
    $$
    \text{notional}_Y \sim \exp(\mathcal N(\texttt{trader mean}, \texttt{trader sigma}^2)).
    $$
- **Smart router (`agent: smart`)**:
  - Direction is random, but venue is **state-dependent**:
    it executes on the DEX only when the DEX quote is competitive vs the current CEX mid `m_now`;
    otherwise it routes to the CEX (and the trade never appears in DEX swap flow).
  - This best-ex selection is implemented as:
    - For `X_to_Y`: execute on DEX only if `quote_x_to_y(dx) >= theta_T * dx * m_now`.
    - For `Y_to_X`: execute on DEX only if `quote_y_to_x(dy) >= theta_T * dy / m_now`.
  - **Implication:** when the DEX is *overpriced* vs `m_now`, the smart router will (mostly) execute `X_to_Y` on DEX and route `Y_to_X` to CEX; when the DEX is *underpriced*, the opposite happens. So *even with random trade direction sampling*, smart-router **DEX** flow becomes one-sided when there is cross-venue mispricing.
- **Arbitrageur (`type: arb`)**:
  - There is **at most one** arbitrage intent per block, and it executes **first** in the mempool.
  - The arb swap is directionally deterministic: it buys on the DEX when the DEX is cheap and sells when the DEX is expensive (relative to the no-arb band around the last validated CEX price).
  - It is skipped if in-band or previewed to be unprofitable (liquidity fee + flash-loan fee).

#### 5.2 Why naive VPIN can be “mostly arb + best-ex selection”

If you compute VPIN on **all executed DEX swaps**, high VPIN episodes can be driven by:

1. **Arbitrage dominance:** a single, large, one-sided arb trade can contribute a large fraction of notional in a bucket, mechanically pushing imbalance toward 1 even if noise is symmetric.
2. **Smart-router venue selection:** when the DEX is mispriced vs the current CEX mid, smart router executes on DEX primarily on the “favorable” side and routes the other side to CEX. This creates DEX-side imbalance even though the smart router is not “directional” in how it samples trade side.
3. **Noise dilution:** increasing the preferred `noise_trades_per_second` arrival rate (or legacy `noise_trades_per_block`) adds symmetric volume that tends to *lower* VPIN (in expectation), but the effect depends strongly on the bucket size vs the trade-size distribution (next point).
4. **Slippage filtering + block ordering:** submitted swaps can be skipped at execution if the realized quote violates `slippage_tolerance`. Since the arbitrage intent executes first and the remaining swaps are randomly shuffled within the block, this can create additional one-sided selection in what actually executes on the DEX.

So in this ABM, VPIN is better interpreted as a proxy for **cross-venue mispricing pressure** (arb + best-ex filtering) than as “probability of privately informed trading” in the original TradFi sense.

#### 5.3 Bucket design matters more with log-normal order sizes

Because both noise and smart notional are log-normal, trade sizes are heavy-tailed. Two practical implications:

- If $V$ is too small relative to typical trade size, many buckets will be dominated by **one trade**, and $I_b$ will be near 1 whenever that trade is one-sided (which is always).
- A robust VPIN implementation should allow **splitting a single trade across multiple volume buckets** (standard in volume-synchronized constructions); otherwise VPIN becomes overly spiky and overly sensitive to rare large draws.

#### 5.4 Measurement choice: executed vs submitted swaps

In this simulator, swaps are **submitted** during intra-block micro-steps but **executed** together at the block boundary. For VPIN, this matters:

- VPIN should be computed on **executed** swaps (post slippage checks), not on submitted intents, otherwise you will overcount flow that never touches the pool.
- For `X_to_Y` swaps the natural `Y`-notional is `dx_in × price` (use either the pre-swap DEX price or an oracle `m` consistently); for `Y_to_X` swaps it is simply `dy_in`.
- Because non-arb swaps are shuffled within a block, VPIN computed on the *execution order* has an additional randomness component. A lower-variance alternative is to compute a **block-level imbalance** first,
  $$
  I_t^{\\text{block}} = \\frac{|V_t^+ - V_t^-|}{V_t^+ + V_t^-},
  $$
  and then smooth it (EWMA or rolling mean). This sacrifices the “equal-volume bucket” property but matches the simulator’s time discretization.

### Mapping to width
- widen with higher VPIN (more imbalance → higher adverse selection),
- narrow when VPIN is low (more symmetric flow).

### Pros/cons
- Pro: onchain-only; simple; connects to toxicity intuition.
- Con: AMM order flow is heavily shaped by arbitrage; VPIN can spike during trending markets even if it’s “mechanical” rather than privately informed.

### Practical VPIN variants to test in this ABM

To make VPIN informative (rather than a near-duplicate of “arb pressure”), compute **actor-conditioned** and **arb-adjusted** variants:

- `VPIN_total`: all executed DEX swaps (noise + smart + arb). Useful as a “basis-stress” indicator.
- `VPIN_ex_arb`: exclude `type: arb` swaps. This targets toxicity in **taker flow** that LPs face aside from explicit arb blocks.
- `VPIN_smart` vs `VPIN_noise`: sanity check that noise-only VPIN is near 0 (up to sampling error) while smart-only VPIN rises when the DEX is mispriced (due to best-ex selection).
- `arb_share` (recommended companion metric):
  $$
  \text{arb\_share}_t = \frac{\text{arb notional}_Y}{\text{total DEX notional}_Y}
  $$
  over a rolling window. If VPIN spikes coincide with high `arb_share`, you’re mostly measuring “arb dominance”.

These decompositions are particularly natural in this repo because swaps already carry `agent` labels (`smart` / `noise`) and a distinct `type: arb`.

**Applicability:** medium (feasible, but interpretation in AMMs needs care).

**Canonical reference (original VPIN):**
- Easley, López de Prado, O’Hara, *Flow Toxicity and Liquidity in a High-Frequency World*, Review of Financial Studies (2012). (Not included in this repo.)

---

## 6) Cross-venue basis stress (arb pressure as a width signal)

### Idea
When DEX price deviates from the oracle/CEX price, arbitrage flows are likely imminent; these flows are typically *toxic* for LPs (they capture LVR).

Define basis in ticks:
$$
b_t = \frac{\log(P_t^{DEX}) - \log(m_t)}{\log(1.0001)}.
$$

### Signal and mapping
- Use EWMA of $|b_t|$ or of basis volatility.
- widen when $|b_t|$ is large or highly volatile (expect more arb and faster tick movement).

### Pros/cons
- Pro: directly measures the “mispricing” that motivates arbitrage.
- Con: depends on oracle quality; basis can be transient and/or manipulated in low-liquidity pools.

**Applicability:** medium.

---

## 7) Jump / tail-risk indicators (gap risk management)

### Idea
Narrow ranges are disproportionately harmed by jumps/gaps (fast exits and missed fee capture), even if average volatility is stable. Detect jumpy regimes and widen aggressively.

### Concrete signals (examples)
- Realized kurtosis / extreme returns in a window.
- Jump detection via bipower variation (separating continuous vs jump variation).
- “max draw” in log-price over a short horizon.

### Pros/cons
- Pro: targets the failure mode of very narrow LPing.
- Con: noisy, lagging; harder to calibrate robustly across assets.

**Applicability:** medium–low (useful as an overlay rather than the main driver).

---

## 8) MEV / mempool indicators (sandwich + JIT intensity)

### Idea
When MEV intensity is high (sandwiches, JIT liquidity), being ultra-narrow at the top of the book can be structurally disadvantaged. Use MEV indicators to widen or reduce exposure.

### Signals (examples)
- Fraction of swaps that are sandwiched (requires backrun classification).
- JIT share: minted liquidity that appears right before swaps and burns right after.
- Block-level reorg/latency risk.

### Pros/cons
- Pro: relevant for real LP profitability.
- Con: requires heavy infra / chain-specific MEV analytics; signals are noisy and can be gamed.

**Applicability:** low for most LPs; more plausible for professional market makers.

---

## Suggested research protocol (how to falsify in this repo)

Even if the immediate output of this note is “ideas”, it helps to phrase each as a testable claim.

### Minimal evaluation metrics
For each width signal policy:
- **Net LP PnL** (hedged and unhedged) in `Y` (see `docs/LP_PnL.md`).
- **Fee revenue** vs **LVR proxy** decomposition (if oracle-based).
- **Time in range** and **recenter frequency** (turnover).
- **Tail outcomes**: worst drawdown, worst single-step loss.

### Reproduction baseline
Run the existing baseline scenario:

1. Activate env: `conda activate main`
2. Run: `python -m scripts.run --config configs/scenarios/section4_microstructure_model0_static.yml`

Outputs land under `abm_results/scenarios/<scenario_name>/` (a new run subfolder), including LP PnL plots and the standard price/fee diagnostics. The narrow-LP width path is currently used for plotting and logging, but `simulate()` does not persist a raw `w_ticks_series` in its returned output dict.

### Next step (implementation sketch, not done here)
Add a config switch like `active_lp_width_mode: {volatility_cex, net_edge, vpin, markout, ...}` and implement each signal as a separate function to keep comparisons honest.

---

## Known limitations / open questions

- **Width vs center coupling:** some toxicity signals should arguably change *center* (skew) rather than only width.
- **Arbitrage confounding:** many “toxic-flow” metrics on AMMs are dominated by arbitrage that tracks oracle moves; separating “informed” vs “mechanical” flow may require careful conditioning on oracle updates.
- **Self-referentiality:** realized fees/toxicity depend on your own width; policies should be evaluated in counterfactuals or using exogenous pool-level signals where possible.
