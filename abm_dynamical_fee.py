# abm.py

"""
ABM Step 1 â€” Uniswap v3 price dynamics with arbitrage, noise traders, and adaptive LPs
======================================================================================

Purpose
-------
This script implements the *first* step of the ABM for Uniswap v3: a single-pool DEX
with concentrated liquidity, fee-on-input swaps, a reference CEX price, noise traders,
a fast (riskâ€“neutral) arbitrageur, and a simpleâ€”yet data-drivenâ€”LP heuristic for
minting/burning ranges. Blocks, the mempool, and strategic MEV are **not** modeled
here (they arrive in later steps). The goal is to reproduce the co-evolution P_t (DEX)
and m_t (CEX) while making every action and liquidity transition explicit and auditable.

High-level design choices
-------------------------
â€¢ **Pool state** is Uniswap v3â€“style with **tick spacing**:
  - S = sqrt(price) so that DEX price is P = SÂ² (Y per X).
  - The tick grid is geometric in S with ratio g (>1). Tick i has lower boundary
    s_i = base_s Â· g^i.
  - The *active band* is **[tick, tick+tick_spacing)**; with 5 bps pools we use
    tick_spacing = 10. s_upper(i) = s_lower(i) Â· g^{tick_spacing}.
  - Active liquidity L_active is the cumulative sum over a *liquidity_net* map:
    liquidity_net[k] = Î”L when crossing **upward** through boundary k.

â€¢ **Fee model**: A constant pool fee f is charged **on input**; r â‰¡ 1â€“f is the
  retained fraction used to move S. Fees are accrued to LPs pro-rata by **L**
  in the active band at the instant of fill.

â€¢ **Swaps only move price when liquidity exists.**
  If L_active == 0 (a â€œdesertâ€), the price stands still. When a swap is attempted,
  the engine **bridges to the next initialized boundary in the swap direction** and
  then starts consuming there (no amount is consumed while traversing empty space).
  This matches Uniswap v3 behavior: price changes only where liquidity exists.

â€¢ **Arbitrage target**: either the no-arb fee band [m_tÂ·r, m_t/r] (default) or
  parity m_t. The arbitrageur walks S band-by-band **through non-empty ranges**
  until the target is hit or a desert is encountered.

â€¢ **Reference market (CEX)**: m_t follows geometric Brownian motion (Î¼, Ïƒ) with a
  **signed** linear impact from the DEX arbitrage notional (token1 units).
  Positive signed notional (Yâ†’X on the DEX) nudges m *up*; negative nudges m *down*.

â€¢ **LP heuristic (width rule)**:
  Width in ticks is driven by the **EWMA of fee-adjusted dislocation** and a
  **mean-zero binomial noise term** to keep heterogeneous behavior across LPs.

Event order within a step t
---------------------------
(1) **Record pre-state:** store P_{t-1}, m_{t-1}, the **pre-step fee band**
    [m_{t-1}Â·r, m_{t-1}/r], and L_active (start-of-step).

(2) **Actor phase (randomized per step):**
    The set {LPs adjust, Noise trader (optional), Arbitrageur} is **shuffled**
    uniformly at random each step and executed in that order. This allows causality
    to change across steps (e.g., arb may move price before LP thresholds are checked).

(3) **CEX update (m_t):**
    GBM step plus a signed impact proportional to arbitrage notional *realized this step*.

(4) **Record post-state.**

Mechanics and formulas (v3 within a spacing band)
-------------------------------------------------
[omitted here for brevity â€” identical to previous version]
"""

from __future__ import annotations

# ===== Standard library =====
import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Callable

from bisect import bisect_right, bisect_left

# ===== Third-party =====
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm


# =============================================================================
# Plot styling (global)
# =============================================================================
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 12

plt.rcParams.update({
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": LABEL_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
})
plt.rcParams["axes.grid"] = True


# =============================================================================
# Global utilities & tolerances
# =============================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]."""
    return max(lo, min(hi, x))

def lcm(a: int, b: int) -> int:
    """Least common multiple (for aligning width to w_min and grid)."""
    return abs(a * b) // math.gcd(a, b) if a and b else 0

def next_numbered_path(base: Path, extension: str = ".txt") -> Path:
    """
    Return the first path of the form `{stem}_{n}{extension}` that does not exist yet.
    Ensures the parent directory exists before returning the candidate.
    """
    base = Path(base)
    directory = base.parent if base.parent != Path("") else Path(".")
    directory.mkdir(parents=True, exist_ok=True)
    if base.suffix:
        stem = base.stem
        ext = base.suffix
    else:
        stem = base.name
        ext = extension
    idx = 0
    while True:
        candidate = directory / f"{stem}_{idx}{ext}"
        if not candidate.exists():
            return candidate
        idx += 1

# Numerical epsilons (tuned to the scale of this toy model)
EPS_LIQ = 1e-18        # active liquidity ~ zero for swaps
EPS_PRICE_CHANGE = 1e-10
EPS_BOUNDARY = 1e-12
EPS_LIQ2 = 1e-7       # active liquidity ~ zero for LP

# log(1 tick) for Uniswap v3 tick ratio 1.0001 on price P (since ticks are on sqrt-price, P ticks use ln(1.0001))
TICK_LN = math.log(1.0001)


# =============================================================================
# Simple EWMA helper (for the fee-adjusted absolute basis)
# =============================================================================

class EWMA:
    """
    Exponentially Weighted Moving Average with half-life parameterization.

    Let Î = exp(-ln 2 / half_life_steps). On update with observation x_t:
        v_t = Î v_{t-1} + (1 - Î) x_t

    We use this to smooth the *fee-adjusted absolute basis* B_t that drives the LP
    width rule (see simulate(): Eq. (10)(12) in the PDF for the model-side notation).
    """
    def __init__(self, half_life_steps: int, init: float = 0.0):
        # decay Î» so that value halves every 'half_life_steps'
        self.lambda_ = math.exp(-math.log(2.0) / max(1, half_life_steps))
        self.v = init

    def update(self, x: float) -> float:
        self.v = self.lambda_ * self.v + (1.0 - self.lambda_) * x
        return self.v


# =============================================================================
# Pool (Uniswap v3â€“style, spacing-aware)
# =============================================================================

@dataclass
class V3Pool:
    """
    Minimal Uniswap v3 pool state for a single asset pair, **spacing-aware**.

    Grid & price:
      â€¢ S := sqrt(P) where P is token1 per token0. Grid ratio g (>1) is geometric in S.
      â€¢ Tick i has lower boundary s_i = base_s Â· g^i and upper s_{i+Î”} with Î” = tick_spacing.
      â€¢ The active **band** is [tick, tick+tick_spacing). Active liquidity L_active is the
        prefix-sum of `liquidity_net` up to `tick`.

    Liquidity book:
      â€¢ `liquidity_net[k]` stores the Î”L applied when crossing **upward** through boundary k.
      â€¢ `recompute_active_L()` rebuilds L_active robustly from liquidity_net.

    Fees & swaps:
      â€¢ Fee f is **on input**; r = 1 - f is the retained fraction used to move S.
        Fees are accrued **per band segment**: if a swap consumes input Δq_eff in a band,
        the fee charged there is Δq_eff·(1/r - 1) and is allocated to LPs pro-rata by
        the band's active L at the *start* of that segment.
      â€¢ Swaps consume liquidity only where it exists. If L_active == 0 (â€œdesertâ€), price
        does not move. The engine outside the raw swap functions can **bridge** to the next
        initialized band and then swap. See `swap_exact_to_target(...)` and `ensure_liquidity(...)`.

    Invariants:
      â€¢ Price only changes via swaps; L_active only changes via (re)mint/burn.
      â€¢ We clamp tiny drifts to zero with EPS_* guards to avoid numerical ghosts.
    """
    g: float
    base_s: float
    tick: int
    S: float
    f: float
    liquidity_net: Dict[int, float] = field(default_factory=dict)
    L_active: float = 0.0
    tick_spacing: int = 5  # 5 bps pool default

    # ----- derived properties -----
    @property
    def r(self) -> float:
        return 1.0 - self.f

    @property
    def price(self) -> float:
        return self.S * self.S

    # ----- spacing helpers -----
    def _snap(self, i: int) -> int:
        s = self.tick_spacing
        return (i // s) * s

    # ----- tick/price helpers for a **spacing band** -----
    def s_lower(self, i: Optional[int] = None) -> float:
        if i is None:
            i = self.tick
        return self.base_s * (self.g ** i)

    def s_upper(self, i: Optional[int] = None) -> float:
        return self.s_lower(i) * (self.g ** self.tick_spacing)

    # ----- core state maintenance -----
    def __post_init__(self) -> None:
        self.tick = self._snap(self.tick)
        self.recompute_active_L()

    def recompute_active_L(self) -> None:
        # numerically robust accumulation of active liquidity up to current tick
        L = math.fsum(dL for t, dL in self.liquidity_net.items() if t <= self.tick)
        # clamp tiny drift to zero
        if abs(L) < EPS_LIQ2:
            L = 0.0
        self.L_active = L


    def add_liquidity_range(self, lower_tick: int, upper_tick: int, L: float) -> None:
        lower_tick = self._snap(lower_tick)
        upper_tick = self._snap(upper_tick)
        assert lower_tick < upper_tick, "Range must be non-empty after snap"

        self.liquidity_net[lower_tick] = self.liquidity_net.get(lower_tick, 0.0) + L
        self.liquidity_net[upper_tick] = self.liquidity_net.get(upper_tick, 0.0) - L

        if lower_tick <= self.tick or upper_tick <= self.tick:
            self.recompute_active_L()

        if abs(self.L_active) < EPS_LIQ2:
            self.L_active = 0.0

    def _cross_up_once(self):
        self.tick += self.tick_spacing
        self.recompute_active_L()

    def _cross_down_once(self):
        self.tick -= self.tick_spacing
        self.recompute_active_L()


    # ----- exact v3 swaps for the noise trader (spacing aware) -----
    def swap_x_to_y(self, dx_in: float, fee_cb: Optional[Callable[[str, float, int, float], None]] = None) -> Tuple[float, float, float]:
        if dx_in <= 0 or self.L_active <= 0:
            return 0.0, 0.0, 0.0

        dx_eff = dx_in * self.r
        dx_used = 0.0
        dy_out = 0.0

        while dx_eff > EPS_LIQ and self.L_active > 0:
            S_lo = self.s_lower()
            if self.S <= S_lo + EPS_BOUNDARY:
                self.S = S_lo
                self._cross_down_once()
                continue

            dx_to = self.L_active * (1 / S_lo - 1 / self.S)

            if dx_eff < dx_to - EPS_BOUNDARY:
                # Snapshot tick/L for this segment before applying it
                _tick_snap = self.tick; _L_snap = self.L_active
                S_new = 1 / (1 / self.S + dx_eff / self.L_active)
                dy = self.L_active * (S_new - self.S)
                self.S = S_new
                dx_used += dx_eff
                dy_out += -dy
                # Per-segment fee (token X) on *input* with fee-on-input model
                _fee_seg = (dx_eff / self.r) - dx_eff
                if fee_cb and _fee_seg > 0.0 and _L_snap > 0.0:
                    fee_cb("x", _fee_seg, _tick_snap, _L_snap)
                dx_eff = 0.0
            else:
                # Snapshot tick/L for this segment before crossing
                _tick_snap = self.tick; _L_snap = self.L_active
                dy = self.L_active * (S_lo - self.S)
                self.S = S_lo
                dx_eff -= dx_to
                dx_used += dx_to
                dy_out += -dy
                # Per-segment fee (token X)
                _fee_seg = (dx_to / self.r) - dx_to
                if fee_cb and _fee_seg > 0.0 and _L_snap > 0.0:
                    fee_cb("x", _fee_seg, _tick_snap, _L_snap)
                self._cross_down_once()

        dx_pre = dx_used / self.r if self.r > 0 else dx_used
        fee_x = dx_pre - dx_used
        return dx_pre, dy_out, fee_x

    def swap_y_to_x(self, dy_in: float, fee_cb: Optional[Callable[[str, float, int, float], None]] = None) -> Tuple[float, float, float]:
        if dy_in <= 0 or self.L_active <= 0:
            return 0.0, 0.0, 0.0

        dy_eff = dy_in * self.r
        dy_used = 0.0
        dx_out = 0.0

        while dy_eff > EPS_LIQ and self.L_active > 0:
            S_hi = self.s_upper()
            if self.S >= S_hi - EPS_BOUNDARY:
                self.S = S_hi
                self._cross_up_once()
                continue

            dy_to = self.L_active * (S_hi - self.S)

            if dy_eff < dy_to - EPS_BOUNDARY:
                # Snapshot tick/L for this segment before applying it
                _tick_snap = self.tick; _L_snap = self.L_active
                S_new = self.S + dy_eff / self.L_active
                dx = self.L_active * (1 / self.S - 1 / S_new)
                self.S = S_new
                dy_used += dy_eff
                dx_out += dx
                # Per-segment fee (token Y) on *input*
                _fee_seg = (dy_eff / self.r) - dy_eff
                if fee_cb and _fee_seg > 0.0 and _L_snap > 0.0:
                    fee_cb("y", _fee_seg, _tick_snap, _L_snap)
                dy_eff = 0.0
            else:
                # Snapshot tick/L for this segment before crossing
                _tick_snap = self.tick; _L_snap = self.L_active
                dx = self.L_active * (1 / self.S - 1 / S_hi)
                self.S = S_hi
                dy_eff -= dy_to
                dx_out += dx
                # Per-segment fee (token Y)
                _fee_seg = (dy_to / self.r) - dy_to
                if fee_cb and _fee_seg > 0.0 and _L_snap > 0.0:
                    fee_cb("y", _fee_seg, _tick_snap, _L_snap)
                self._cross_up_once()

        dy_pre = dy_used / self.r if self.r > 0 else dy_used
        fee_y = dy_pre - dy_used
        return dy_pre, dx_out, fee_y


# =============================================================================
# Sparse boundary index (for fast next boundary lookups)
# =============================================================================

class BoundaryIndex:
    """
    Sparse index over boundaries with non-zero `liquidity_net` entries.

    Purpose:
      â€¢ O(log B) lookup for the next initialized boundary upward/downward from a tick.
      â€¢ Used by non-mutating quotes and by the "desert-bridging" logic so we can find
        the next band that actually has liquidity without touching pool state.

    Contract:
      â€¢ Call `mark_dirty()` whenever `liquidity_net` changes; reads auto-refresh lazily.
    """
    def __init__(self, liquidity_net: Dict[int, float]):
        self.liq = liquidity_net
        self.keys = sorted([k for k, v in liquidity_net.items() if abs(v) > EPS_LIQ])
        self.dirty = False

    def mark_dirty(self) -> None:
        self.dirty = True

    def _ensure(self) -> None:
        if self.dirty:
            self.keys = sorted([k for k, v in self.liq.items() if abs(v) > EPS_LIQ])
            self.dirty = False

    def next_up(self, tick: int) -> Optional[int]:
        self._ensure()
        i = bisect_right(self.keys, tick)
        return self.keys[i] if i < len(self.keys) else None

    def prev_down(self, tick: int) -> Optional[int]:
        self._ensure()
        i = bisect_left(self.keys, tick) - 1
        return self.keys[i] if i >= 0 else None


# =============================================================================
# LP positions & agents
# =============================================================================

@dataclass
class Position:
    """
    A concentrated liquidity position [lower, upper) with liquidity L.

    Amounts at price S (classic v3 algebra):
      Let s_a = lower boundary sqrt-price, s_b = upper boundary sqrt-price.
      If S <= s_a:  (token0, token1) = (L(1/s_a - 1/s_b), 0)
      If S >= s_b:  (0, L(s_b - s_a))
      Else:         (L(1/S - 1/s_b), L(S - s_a))

    Valuation and PnL (token1 numÃ©raire):
      â€¢ `hodl0_value_y = amt0_init * m + amt1_init` at mint time.
      â€¢ Mark-to-market IL_y(S, m) = position_value_y_now(S, m) - hodl0_value_y  (â‰¤ 0 typically).
      â€¢ Fees accrue separately as (fees0 * m + fees1).  PnL_y = IL_y + fees_value_y.
    """
    owner: int
    lower: int
    upper: int
    L: float
    sa: float
    sb: float
    amt0_init: float
    amt1_init: float
    fees0: float = 0.0
    fees1: float = 0.0
    hodl0_value_y: float = 0.0

    def in_range(self, tick: int) -> bool:
        return self.lower <= tick < self.upper

    def current_amounts(self, S: float) -> Tuple[float, float]:
        if S <= self.sa:
            return (self.L * (1 / self.sa - 1 / self.sb), 0.0)
        if S >= self.sb:
            return (0.0, self.L * (self.sb - self.sa))
        return (self.L * (1 / S - 1 / self.sb), self.L * (S - self.sa))

    def hodl_value_y_now(self, m: float) -> float:
        return self.amt0_init * m + self.amt1_init

    def position_value_y_now(self, S: float, m: float) -> float:
        a0, a1 = self.current_amounts(S)
        return a0 * m + a1

    def fees_value_y(self, m: float) -> float:
        return self.fees0 * m + self.fees1

    def IL_y(self, S: float, m: float) -> float:
        return self.position_value_y_now(S, m) - self.hodl_value_y_now(m)

    def PnL_y(self, S: float, m: float) -> float:
        return self.IL_y(S, m) + self.fees_value_y(m)


@dataclass
class LPAgent:
    id: int
    positions: List[Position] = field(default_factory=list)
    mintProb: float = 0.5
    is_active_narrow: bool = False
    # --- async scheduler state ---
    review_rate: float = 0.2      # p for geometric clock; weâ€™ll set = 1/tau at runtime
    next_review: int = 0          # steps until this LP may act again
    cooldown: int = 0             # after a burn, sit out this many steps
    can_act: bool = False         # per-segment mask (set by the micro-scheduler)
    L_budget: float = 0.0
    L_live: float = 0.0
    wallet_y: float = 0.0



# =============================================================================
# Reference market (CEX)
# =============================================================================

@dataclass
class ReferenceMarket:
    m: float            # CEX price of token A in token B (B per A)
    mu: float           # drift (per step) of log-returns
    sigma: float        # vol (per step) of log-returns
    kappa: float        # impact scale (price units per A^(1+xi))
    xi: float = 0.0     # impact exponent (xi = 0 => linear in |Î”a|)

    def step(self, delta_a_cex_signed: float) -> float:
        """
        Apply permanent, additive impact from the CEX trade in token A units,
        then diffuse via GBM. Returns the impact applied (for debugging).
        """
        # 1) permanent impact (additive), Eq. (7)
        impact = self.kappa * math.copysign(abs(delta_a_cex_signed)**(1.0 + self.xi),
                                            delta_a_cex_signed)
        self.m = max(1e-12, self.m + impact)

        # 2) diffusion, Eq. (8)
        z = np.random.normal()
        self.m *= math.exp(self.mu - 0.5 * self.sigma**2 + self.sigma * z)
        self.m = max(1e-12, self.m)
        return impact

# =============================================================================
# Builders
# =============================================================================

def build_demo_pool() -> Tuple[V3Pool, float]:
    f = 0.003
    g = np.sqrt(1.0001)
    m0 = 2000.0
    S0 = math.sqrt(m0)
    base_s = S0 / math.sqrt(g)
    pool = V3Pool(g=g, base_s=base_s, tick=0, S=S0, f=f, liquidity_net={}, tick_spacing=10)
    def add(width, L): pool.add_liquidity_range(-width, +width, L)
    add(10, 5_000.0); add(50, 10_000.0); add(150, 20_000.0); add(500, 30_000.0)
    return pool, m0


def build_empty_pool() -> Tuple[V3Pool, float]:
    f = 0.003
    g = np.sqrt(1.0001)
    m0 = 2000.0
    S0 = math.sqrt(m0)
    base_s = S0 / math.sqrt(g)
    pool = V3Pool(g=g, base_s=base_s, tick=0, S=S0, f=f, liquidity_net={}, tick_spacing=10)
    return pool, m0


def minted_amounts_at_S(L: float, sa: float, sb: float, S: float) -> Tuple[float, float]:
    """
    Given liquidity L and range [sa, sb) in sqrt-price, return minted (token0, token1)
    at current S. Mirrors `Position.current_amounts` but without instantiating a Position.

    See Position docstring for the closed forms used here.
    """
    if S <= sa:
        return L * (1 / sa - 1 / sb), 0.0
    elif S >= sb:
        return 0.0, L * (sb - sa)
    else:
        return L * (1 / S - 1 / sb), L * (S - sa)


def bootstrap_initial_binomial_hill(
    pool: V3Pool,
    ref: ReferenceMarket,
    LPs: List[LPAgent],
    N: int = 400,
    L_total: float = 70_000.0,
    seed_lp_id: int = 10_000,
    seed_mint_prob: float = 0.0,
    min_L_per_tick: float = 1e-9,
    plot: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> LPAgent:
    seed = LPAgent(id=seed_lp_id, mintProb=seed_mint_prob)
    center_tick = pool._snap(pool.tick)
    S_entry = pool.S

    ticks: List[int] = []
    L_vals: List[float] = []

    denom = float(2 ** N)
    for k in range(N + 1):
        w = math.comb(N, k) / denom
        L_i = w * L_total
        if L_i < min_L_per_tick:
            continue

        rel = k - (N // 2)
        lower = center_tick + rel * pool.tick_spacing
        upper = lower + pool.tick_spacing
        sa = pool.s_lower(lower)
        sb = pool.s_upper(upper)

        amt0, amt1 = minted_amounts_at_S(L_i, sa, sb, S_entry)

        pos = Position(
            owner=seed.id, lower=lower, upper=upper, L=L_i, sa=sa, sb=sb,
            amt0_init=amt0, amt1_init=amt1, hodl0_value_y=amt0 * ref.m + amt1
        )
        pool.add_liquidity_range(lower, upper, L_i)
        seed.positions.append(pos)

        ticks.append(lower)
        L_vals.append(L_i)

    pool.recompute_active_L()
    LPs.append(seed)

    if plot and len(ticks) > 0:
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            created_fig = True
        ax.bar(ticks, L_vals, width=pool.tick_spacing, align="edge")
        ax.set_xlabel("Tick", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Liquidity per band (L)", fontsize=LABEL_FONT_SIZE)
        ax.set_title(title or f"Initial binomial hill (N={N}, total L={L_total:,.0f})", fontsize=TITLE_FONT_SIZE)
        ax.grid(True, axis="y", alpha=0.25)
        if created_fig:
            plt.tight_layout()

    return seed

def bootstrap_initial_binomial_hill_sharded(
    pool: V3Pool,
    ref: ReferenceMarket,
    LPs: List[LPAgent],
    N: int = 400,
    L_total: float = 70_000.0,
    num_seed_lps: int = 20,
    seed_lp_id_base: int = 10_000,
    seed_mint_prob: float = 0.0,
    min_L_per_tick: float = 1e-9,
    tau: int = 20,
    plot: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> List[LPAgent]:
    """
    Split the binomial hill across `num_seed_lps` seed LPs so burns are staggered.
    Each seed LP has its own review clock; all have mintProb=0 and is_active_narrow=False.
    """
    assert num_seed_lps >= 1
    center_tick = pool._snap(pool.tick)
    S_entry = pool.S

    # prepare seed LPs
    seed_LPs: List[LPAgent] = []
    for j in range(num_seed_lps):
        sid = seed_lp_id_base + j
        lp = LPAgent(id=sid, mintProb=seed_mint_prob, is_active_narrow=False)
        # async timing so they act at different steps
        lp.review_rate = 1.0 / max(1, tau)
        lp.next_review = int(np.random.geometric(lp.review_rate))
        lp.cooldown = 0
        lp.can_act = False
        seed_LPs.append(lp)

    # binomial weights once
    ticks: List[int] = []
    L_vals: List[float] = []
    denom = float(2 ** N)
    tick_specs: List[Tuple[int, float]] = []  # (lower_tick, L_i)

    for k in range(N + 1):
        w = math.comb(N, k) / denom
        L_i = w * L_total
        if L_i < min_L_per_tick:
            continue
        rel = k - (N // 2)
        lower = center_tick + rel * pool.tick_spacing
        tick_specs.append((lower, L_i))
        ticks.append(lower)
        L_vals.append(L_i)

    # round-robin assign ticks to seed LPs
    for idx, (lower, L_i) in enumerate(tick_specs):
        lp = seed_LPs[idx % num_seed_lps]
        upper = lower + pool.tick_spacing
        sa = pool.s_lower(lower)
        sb = pool.s_upper(upper)
        amt0, amt1 = minted_amounts_at_S(L_i, sa, sb, S_entry)

        pos = Position(
            owner=lp.id, lower=lower, upper=upper, L=L_i, sa=sa, sb=sb,
            amt0_init=amt0, amt1_init=amt1, hodl0_value_y=amt0 * ref.m + amt1,
        )
        pool.add_liquidity_range(lower, upper, L_i)
        lp.positions.append(pos)

    pool.recompute_active_L()

    # append all seeds to LPs list
    for lp in seed_LPs:
        LPs.append(lp)

    # optional plot (same look as the single-LP version)
    if plot and len(ticks) > 0:
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            created_fig = True
        ax.bar(ticks, L_vals, width=pool.tick_spacing, align="edge")
        ax.set_xlabel("Tick", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Liquidity per band (L)", fontsize=LABEL_FONT_SIZE)
        ax.set_title(title or f"Initial binomial hill (N={N}, total L={L_total:,.0f}, seeds={num_seed_lps})", fontsize=TITLE_FONT_SIZE)
        ax.grid(True, axis="y", alpha=0.25)
        if created_fig:
            plt.tight_layout()

    return seed_LPs


def add_static_binomial_hill(
    pool: V3Pool,
    N: int = 400,
    L_total: float = 70_000.0,
    min_L_per_tick: float = 1e-9,
    plot: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> None:
    center_tick = pool._snap(pool.tick)
    ticks: List[int] = []
    L_vals: List[float] = []
    denom = float(2 ** N)
    for k in range(N + 1):
        w = math.comb(N, k) / denom
        L_i = w * L_total
        if L_i < min_L_per_tick:
            continue
        rel = k - (N // 2)
        lower = center_tick + rel * pool.tick_spacing
        upper = lower + pool.tick_spacing
        pool.add_liquidity_range(lower, upper, L_i)
        ticks.append(lower); L_vals.append(L_i)

    pool.recompute_active_L()

    if plot and len(ticks) > 0:
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            created_fig = True
        ax.bar(ticks, L_vals, width=pool.tick_spacing, align="edge")
        ax.set_xlabel("Tick", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Liquidity per band (L)", fontsize=LABEL_FONT_SIZE)
        ax.set_title(title or f"Initial binomial hill (N={N}, total L={L_total:,.0f})", fontsize=TITLE_FONT_SIZE)
        ax.grid(True, axis="y", alpha=0.25)
        if created_fig:
            plt.tight_layout()


def bootstrap_initial_liquidity_as_lp(
    pool: V3Pool,
    ref: ReferenceMarket,
    LPs: List[LPAgent],
    hill: List[Tuple[int, float]],
    seed_lp_id: int = 999,
    seed_mint_prob: float = 0.0,
) -> LPAgent:
    seed = LPAgent(id=seed_lp_id, mintProb=seed_mint_prob)
    for width, L in hill:
        lower, upper = pool._snap(pool.tick - width), pool._snap(pool.tick + width)
        if upper <= lower:
            upper = lower + pool.tick_spacing
        sa, sb = pool.s_lower(lower), pool.s_upper(upper)
        S_entry = pool.S
        amt0, amt1 = minted_amounts_at_S(L, sa, sb, S_entry)

        pos = Position(
            owner=seed.id, lower=lower, upper=upper, L=L, sa=sa, sb=sb,
            amt0_init=amt0, amt1_init=amt1, hodl0_value_y=amt0 * ref.m + amt1
        )
        assert abs(pos.PnL_y(pool.S, ref.m)) <= 1e-9 * max(1.0, pos.hodl0_value_y), \
            "Non-zero PnL at bootstrap mint"

        pool.add_liquidity_range(lower, upper, L)
        seed.positions.append(pos)

    pool.recompute_active_L()
    LPs.append(seed)
    return seed


# =============================================================================
# Simulation
# =============================================================================

def simulate(
    T: int = 120,
    seed: int = 7,
    use_fee_band: bool = True,
    cex_mu: float = 0.0,
    cex_sigma: float = 0.02,
    p_trade: float = 0.6,
    noise_floor: float = 0.0,
    p_lp_narrow: float = 0.7,
    p_lp_wide: float = 0.15,
    N_LP: int = 12,
    tau: int = 20,
    # --- LP width via EWMA(B_t) + binomial noise ---
    w_min_ticks: int = 25,
    w_max_ticks: int = 5000,
    basis_half_life: int = 60,   # steps
    slope_s: float = 1.0,        # ticks per (basis-in-ticks)
    # --- binomial noise parameters (already present; now used) ---
    binom_n: int = 8,
    binom_p: float = 0.5,
    # --- lognormal noise parameters (new; not yet used) ---
    trader_mean: float = -2.0,
    trader_sigma: float = 1.0,
    theta_T: float = 1.0,
    # --- other params ---
    mint_mu: float = 0.01,
    mint_sigma: float = 0.02,
    theta_TP: float = 0.003,
    theta_SL: float = 0.01,
    evolve_initial_hill: bool = True,
    initial_binom_N: int = 400,
    initial_total_L: float = 70_000.0,
    plot_initial_hill: bool = False,
    k_out: int = 2,
    visualize: bool = True,
    skip_step: int = 0,
    # --- Dynamic fee controller (new) ---
    fee_mode: str = "volatility",      # "static" | "volatility" | "toxicity" | "imbalance" | "hybrid"
    f0: float = 0.003,             # baseline fee (e.g., 30 bps)
    f_min: float = 0.0005,         # 5 bps
    f_max: float = 0.02,           # 200 bps safety cap
    fee_half_life: int = 20,       # EWMA half-life (steps) for signals
    k_sigma: float = 0.02,         # adds ~k_sigma * EWMA(|logret|) to fee
    k_basis: float = 2e-5,         # fee per tick of dislocation (basis in ticks)
    # k_imb: float = 0.002,          # fee += k_imb * |imbalance|, imbalance in [0,1]
    fee_step_bps_min: float = 0.5, # do not change fee unless ≥ 0.5 bps move
    fee_step_bps_max: float = 5.0, # max step per update (bps)
    fee_cooldown: int = 1,         # blocks between fee changes (hysteresis)
):
    """
    Run a Step-1 ABM with a Uniswap v3–style pool.

    - noise_floor (float in [0,1]): with this probability, the noise trader executes on the DEX even if the DEX quote fails the relative-value check.
Run a Step-1 ABM of a Uniswap v3â€“style pool with noise traders, a band-targeting
    arbitrageur, and adaptive LPs. **Actor order is randomized each step.**
    """
    initial_params = dict(locals())
    np.random.seed(seed)
    random.seed(seed)

    # --- Build pool + reference market + LP agents ----------------------------
    if evolve_initial_hill:
        pool, m0 = build_empty_pool()
        ref = ReferenceMarket(m=m0, mu=cex_mu, sigma=cex_sigma, kappa=1e-3)

        LPs: List[LPAgent] = []
        for i in range(N_LP):
            is_narrow = random.random() < 0.7
            mintProb = p_lp_narrow if is_narrow else p_lp_wide
            LPs.append(LPAgent(id=i, mintProb=mintProb, is_active_narrow=is_narrow))
            lp = LPs[-1]
            lp.review_rate = 1.0 / max(1, tau)
            lp.next_review = int(np.random.geometric(lp.review_rate))
            lp.cooldown = 0
            lp.can_act = False

        # Distribute initial_total_L across LPs (each gets ~equal share)
        L_SCALE = initial_total_L / max(1, N_LP)
        for lp in LPs:
            lp.L_budget = 2.0 * L_SCALE   # each LP can deploy up to ~2Ã— their fair share
            lp.L_live = 0.0               # tracked across mints/burns

        bootstrap_initial_binomial_hill_sharded(
            pool, ref, LPs,
            N=initial_binom_N,
            L_total=initial_total_L,
            num_seed_lps=20,
            seed_lp_id_base=10_000,
            seed_mint_prob=0.0,
            tau=tau,
            plot=plot_initial_hill
        )

        # ensure budgets exist for every LP, including the just-appended seed
        for lp in LPs:
            if lp.L_budget <= 0.0:
                lp.L_budget = 2.0 * L_SCALE
            if lp.L_live < 0.0:
                lp.L_live = 0.0

    else:
        pool, m0 = build_empty_pool()
        ref = ReferenceMarket(m=m0, mu=0.0, sigma=0.02, kappa=1e-3)
        LPs = []
        for i in range(N_LP):
            is_narrow = random.random() < 0.7
            mintProb = 0.55 if is_narrow else 0.15
            LPs.append(LPAgent(id=i, mintProb=mintProb, is_active_narrow=is_narrow))
            lp = LPs[-1]                      # the one we just appended
            lp.review_rate = 1.0 / max(1, tau)
            lp.next_review = int(np.random.geometric(lp.review_rate))
            lp.cooldown = 0
            lp.can_act = False

        add_static_binomial_hill(pool, N=initial_binom_N, L_total=initial_total_L, plot=plot_initial_hill)

    bidx = BoundaryIndex(pool.liquidity_net)

    # ------------------ Recorders ------------------
    P_series, M_series = [], []
    X_active_end, Y_active_end = [], []
    band_lo_pre, band_hi_pre = [], []
    band_lo_post, band_hi_post = [], []
    L_end, L_pre_step = [], []
    L_pre_trader, L_pre_arb_eff = [], []
    trader_y_series, arb_y_series = [], []
    trader_steps, trader_dirs = [], []
    arb_steps, arb_dirs = [], []
    mint_steps, mint_sizes, burn_steps, burn_sizes = [], [], [], []
    mint_widths = []
    liq_history: List[Dict[int, float]] = []
    tick_history: List[int] = []
    delta_a_cex_series = []
    # --- PnL recorders ---
    trader_pnl_steps = []       # realized per-step PnL (token1)
    arb_pnl_steps = []          # realized per-step PnL (token1)
    lp_pnl_total_series = []    # mark-to-market total LP PnL across all LPs (token1)
    lp_pnl_active_series = []   # mark-to-market PnL for active (narrow) LPs
    lp_pnl_passive_series = []  # mark-to-market PnL for passive (wide) LPs
    trader_exec_count = []
    arb_exec_count = []

    # --- Split PnL/flow recorders for Smart Router vs Noise Trader ---
    sr_pnl_steps = []
    noise_pnl_steps = []
    sr_exec_count = []
    noise_exec_count = []
    sr_y_series = []
    noise_y_series = []

    # Determine verbose log file path for this run
    verbose_log_path = next_numbered_path(Path(f"abm_results/verbose_steps_{fee_mode}"))
    verbose_log_path_str = str(verbose_log_path)

    with open(verbose_log_path_str, "a") as f:
        f.write("# Simulation parameters\n")
        for key in sorted(initial_params):
            f.write(f"{key} = {initial_params[key]}\n")
        f.write("\n")


    # --- LP wealth recorders (new) ---
    lp_wallet_series = []      # realized wallet (token1)
    lp_wealth_series = []      # wallet + open PnL (token1)
    lp_wallet_active_series = []
    lp_wallet_passive_series = []
    lp_wealth_active_series = []
    lp_wealth_passive_series = []
    # --- Dynamic fee signal recorders (new) ---
    fee_sigma_series = []          # EWMA abs log-return (σ̂)
    fee_basis_ticks_series = []    # EWMA fee-adjusted basis, in ticks
    fee_imb_series = []            # EWMA |imbalance| in [0,1]
    fee_signal_series = []         # controller signal actually used (per fee_mode)
    # --- EWMA(B_t) state for LP width rule ---
    ewma_B = EWMA(half_life_steps=basis_half_life)

    # --- Dynamic fee controller state (new) ---
    pool.f = float(f0)  # controller baseline overrides builder default
    fee_next: Optional[float] = None
    fee_cooldown_left: int = 0
    fee_series: List[float] = []

    # EWMA signals for controllers
    ewma_sigma_fee = EWMA(half_life_steps=fee_half_life, init=0.0)  # |log m_t - log m_{t-1}|
    ewma_basis_fee = EWMA(half_life_steps=fee_half_life, init=0.0)  # fee-adjusted log gap
    ewma_imb_fee   = EWMA(half_life_steps=fee_half_life, init=0.0)  # value-imbalance
    prev_m_for_vol = ref.m

    # ------------------ Helpers ------------------
    def allocate_fees(token: str, fee_amt: float, tick_snapshot: int, L_snapshot: float) -> None:
        if fee_amt <= 0 or L_snapshot <= 0:
            return
        for lp in LPs:
            for pos in lp.positions:
                if pos.in_range(tick_snapshot):
                    share = pos.L / L_snapshot
                    if token == "x":
                        pos.fees0 += share * fee_amt
                    else:
                        pos.fees1 += share * fee_amt

    def tick_from_S(S: float) -> int:
        raw = int(math.floor(math.log(max(S, 1e-18) / pool.base_s, pool.g)))
        return pool._snap(raw)

    def ensure_liquidity(direction: str) -> Tuple[bool, int, float, float]:
        """
        Non-mutating: given 'direction' ('up' or 'down'), return a candidate band
        we could bridge into *without* touching pool state.

        Returns: (ok, new_tick, new_S, L_at_band)
        """
        if pool.L_active > EPS_LIQ:
            return True, pool.tick, pool.S, pool.L_active

        curr_tick = tick_from_S(pool.S)

        if direction == "up":
            nb = bidx.next_up(curr_tick)
            if nb is None:
                return False, pool.tick, pool.S, 0.0
            new_tick = pool._snap(nb)
            new_S = pool.s_lower(new_tick)
        else:
            pb = bidx.prev_down(curr_tick)
            if pb is None:
                return False, pool.tick, pool.S, 0.0
            new_tick = pool._snap(pb - pool.tick_spacing)
            new_S = pool.s_upper(new_tick)

        # compute L at that band without mutating pool
        L = math.fsum(dL for k, dL in pool.liquidity_net.items() if k <= new_tick)
        if abs(L) < EPS_LIQ:
            return False, pool.tick, pool.S, 0.0
        return True, new_tick, new_S, L


    def mint_lp(lp: LPAgent, width_ticks: int) -> None:
        # Use an INTEGER number of spacing bands around the active band.
        # Min total width = tick_spacing; all widths multiples of tick_spacing.
        n_bands = max(1, int(round(width_ticks / pool.tick_spacing)))
        

        X = abs(np.random.normal(mint_mu, mint_sigma))
        want = X * L_SCALE
        cap_step = 0.25 * lp.L_budget
        cap_left = max(0.0, lp.L_budget - lp.L_live)
        L_new = max(0.0, min(want, cap_step, cap_left))
        if L_new <= 0: 
            return

        # Center around current sqrt price S (approximately), not the snapped active band.
        S_now = pool.S
        s = pool.tick_spacing
        nb = n_bands
        denom = (1.0 + (pool.g ** (nb * s + s)))
        if denom <= 0.0:
            denom = 1.0
        lower_real = math.log((2.0 * S_now / pool.base_s) / denom, pool.g)
        lower = pool._snap(int(round(lower_real)))
        upper = lower + nb * s
        if upper <= lower:
            upper = lower + pool.tick_spacing

        sa, sb = pool.s_lower(lower), pool.s_upper(upper)
        amt0, amt1 = minted_amounts_at_S(L_new, sa, sb, pool.S)

        pos = Position(
            owner=lp.id, lower=lower, upper=upper, L=L_new, sa=sa, sb=sb,
            amt0_init=amt0, amt1_init=amt1, hodl0_value_y=amt0 * ref.m + amt1,
        )

        assert abs(pos.PnL_y(pool.S, ref.m)) <= 1e-9 * max(1.0, pos.hodl0_value_y), "Non-zero PnL at mint"

        pool.add_liquidity_range(lower, upper, L_new)
        pool.recompute_active_L()

        if abs(pool.L_active) < EPS_LIQ2:
            pool.L_active = 0.0

        bidx.mark_dirty()
        lp.positions.append(pos)

        mint_steps.append(t)
        mint_sizes.append(L_new)
        mint_widths.append(upper - lower)

        with open(verbose_log_path_str, "a") as f:
            f.write(f"[t={t:03d}] LP{lp.id} MINT L={L_new:.4f} [{lower},{upper}) | L_active={pool.L_active:.4f}\n")
        lp.L_live = getattr(lp, "L_live", 0.0) + L_new

    def burn_any(lp: LPAgent, idx: int) -> None:
        pos = lp.positions.pop(idx)
        # Realize PnL into LP wallet at burn time (fees + IL vs floating HODL)
        realized_y = pos.PnL_y(pool.S, ref.m)
        lp.wallet_y = getattr(lp, 'wallet_y', 0.0) + float(realized_y)
        pool.add_liquidity_range(pos.lower, pos.upper, -pos.L)
        pool.recompute_active_L()

        if abs(pool.L_active) < EPS_LIQ2:
            pool.L_active = 0.0
        elif pool.L_active < 0.0:
            # still materially negative => logic error rather than rounding
            raise AssertionError(f"L_active underflow after burn: {pool.L_active}")

        bidx.mark_dirty()
        burn_steps.append(t)
        burn_sizes.append(pos.L)

        with open(verbose_log_path_str, "a") as f:
            f.write(f"[t={t:03d}] LP{lp.id} BURN L={pos.L:.4f} [{pos.lower},{pos.upper}) | L_active={pool.L_active:.4f}\n")

        lp.cooldown = np.random.randint(3, 9)  # 3–8 steps of “hands off”
        lp.L_live = max(0.0, getattr(lp, "L_live", 0.0) - pos.L)


    def reserves_in_active_tick() -> Tuple[float, float]:
        if pool.L_active <= EPS_LIQ:
            return 0.0, 0.0
        sa, sb = pool.s_lower(), pool.s_upper()
        S_eff = min(max(pool.S, sa), sb)
        x = pool.L_active * max(0.0, 1.0 / S_eff - 1.0 / sb)
        y = pool.L_active * max(0.0, S_eff - sa)
        return x, y

    # ----- Arbitrage internals (unchanged) -----
    def fast_span_up(to_S: float, target_S: float) -> Tuple[float, float, float]:
        S0, L, r = pool.S, pool.L_active, pool.r
        S1 = min(to_S, target_S)
        if S1 <= S0 or L <= 0:
            return 0.0, 0.0, 0.0
        dy_eff = L * (S1 - S0)
        dy_pre = dy_eff / r
        dx_out = L * (1 / S0 - 1 / S1)
        fee_y = dy_pre - dy_eff
        pool.S = S1
        pool.tick = tick_from_S(pool.S)
        return dy_pre, dx_out, fee_y

    def fast_span_down(to_S: float, target_S: float) -> Tuple[float, float, float]:
        S0, L, r = pool.S, pool.L_active, pool.r
        S1 = max(to_S, target_S)
        if S1 >= S0 or L <= 0:
            return 0.0, 0.0, 0.0
        dx_eff = L * (1 / S1 - 1 / S0)
        dx_pre = dx_eff / r
        dy_out = L * (S0 - S1)
        fee_x = dx_pre - dx_eff
        pool.S = S1
        pool.tick = tick_from_S(pool.S)
        return dx_pre, dy_out, fee_x

    def swap_exact_to_target(target_price: float, direction: str, fee_cb: Optional[Callable[[str, float, int, float], None]] = None) -> Tuple[float, float, float, float, float]:
        target_S = math.sqrt(max(1e-18, target_price))

        # --- Desert bridge (peek â†’ optionally apply) ---
        bridged = False
        prev_tick, prev_S = pool.tick, pool.S
        if pool.L_active <= EPS_LIQ:
            ok, new_tick, new_S, _ = ensure_liquidity(direction)
            if not ok:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            pool.tick, pool.S = new_tick, new_S
            pool.recompute_active_L()
            bridged = True

        total_in = total_out = fee_x = fee_y = 0.0
        L_first = 0.0

        if direction == "up":
            while pool.L_active > 0 and pool.S < target_S - EPS_BOUNDARY:
                S_hi = pool.s_upper()
                L_before = pool.L_active
                _tick_before = pool.tick
                dy, dx, f = fast_span_up(S_hi, target_S)
                if dy > 0 and L_first == 0.0:
                    L_first = L_before
                # Per-span fee allocation (token Y)
                if fee_cb and f > 0.0 and L_before > 0.0:
                    fee_cb("y", f, _tick_before, L_before)
                fee_y += f; total_in += dy; total_out += dx
                if pool.S >= target_S - EPS_BOUNDARY:
                    break
                pool._cross_up_once()
                if pool.L_active <= 0:
                    break
            if total_in <= EPS_LIQ and bridged:
                pool.tick, pool.S = prev_tick, prev_S
                pool.recompute_active_L()
            return total_in, total_out, fee_x, fee_y, L_first

        else:  # "down"
            while pool.L_active > 0 and pool.S > target_S + EPS_BOUNDARY:
                S_lo = pool.s_lower()
                L_before = pool.L_active
                _tick_before = pool.tick
                dx, dy, f = fast_span_down(S_lo, target_S)
                if dx > 0 and L_first == 0.0:
                    L_first = L_before
                # Per-span fee allocation (token X)
                if fee_cb and f > 0.0 and L_before > 0.0:
                    fee_cb("x", f, _tick_before, L_before)
                fee_x += f; total_in += dx; total_out += dy
                if pool.S <= target_S + EPS_BOUNDARY:
                    break
                pool._cross_down_once()
                if pool.L_active <= 0:
                    break
            if total_in <= EPS_LIQ and bridged:
                pool.tick, pool.S = prev_tick, prev_S
                pool.recompute_active_L()
            return total_in, total_out, fee_x, fee_y, L_first


    def arbitrage_to_target() -> Tuple[float, float, float, Optional[str], float, float, float]:
        """
        Returns:
        in_used        = total input amount into the DEX (dy for 'up', dx for 'down')
        x_out_from_dex = token A out from the DEX (dx_out for 'up'; 0.0 for 'down')
        y_out_from_dex = token B out from the DEX (0.0 for 'up'; dy_out for 'down')
        direction      = 'up' or 'down' or None
        fee_x, fee_y
        L_first
        """
        P = pool.price
        r = pool.r
        if use_fee_band:
            lo, hi = ref.m * r, ref.m / r
            if P < lo * (1 - 1e-9):
                # up: returns (dy_in, dx_out, fee_x=0, fee_y, L_first)
                dy_in, dx_out, fx, fy, Lff = swap_exact_to_target(lo, "up", fee_cb=allocate_fees)
                return dy_in, dx_out, 0.0, ("up" if dy_in > 0 else None), fx, fy, Lff
            if P > hi * (1 + 1e-9):
                # down: returns (dx_in, dy_out, fee_x, fee_y=0, L_first)
                dx_in, dy_out, fx, fy, Lff = swap_exact_to_target(hi, "down", fee_cb=allocate_fees)
                return dx_in, 0.0, dy_out, ("down" if dx_in > 0 else None), fx, fy, Lff
            return 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0
        else:
            target = ref.m
            if abs(P - target) / max(target, 1e-12) < 1e-9:
                return 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0
            direction = "up" if P < target else "down"
            ain, aout, fx, fy, Lff = swap_exact_to_target(target, direction)
            # aout is dx_out only for 'up'; for 'down' we set aout=0.0
            return (ain, aout if direction == "up" else 0.0, 0.0 if direction == "up" else aout, direction, fx, fy, Lff)


    # ------------------ Main loop ------------------
    for t in range(T):
        # --- Apply any committed fee update (commit→reveal) ---
        if fee_cooldown_left > 0:
            fee_cooldown_left -= 1
        if fee_next is not None and fee_cooldown_left <= 0:
            pool.f = clamp(fee_next, f_min, f_max)
            fee_next = None
        r = pool.r

        # Pre-step band window
        band_lo_pre.append(ref.m * r)
        band_hi_pre.append(ref.m / r)

        # Record start-of-step active L and price
        L_pre_step.append(pool.L_active)
        P_before = pool.price

        # ---------------------------------------------------------------------
        # LP width rule: EWMA of fee-adjusted absolute basis B_t + binomial noise
        # ---------------------------------------------------------------------
        # B_t = max(0, |ln P - ln m| - ln(1/(1-f)))
        fee_band_ln = -math.log1p(-pool.f)  # ln(1/(1-f))
        log_gap = abs(math.log(max(pool.price, 1e-18)) - math.log(max(ref.m, 1e-18)))
        B_t = max(0.0, log_gap - fee_band_ln)
        D_t = ewma_B.update(B_t)  # smoothed actionable dislocation

        # Deterministic width component from EWMA basis (in ticks)
        basis_in_ticks = D_t / TICK_LN

        # --- Mean-zero binomial noise term (in ticks) ---
        # draw K ~ Bin(n, p), center by n p, and scale by tick_spacing to live on the grid
        noise_ticks = 0.0
        if binom_n > 0 and 0.0 < binom_p < 1.0:
            K = np.random.binomial(binom_n, binom_p)
            noise_ticks = (K - binom_n * binom_p) * pool.tick_spacing  # non-negative noise per spec

        # Map to width in ticks: w = clip(w_min + slope * basis_in_ticks + noise_ticks, w_min, w_max)
        w_unclipped = w_min_ticks + slope_s * basis_in_ticks + noise_ticks
        step_width_ticks = pool.tick_spacing  # total width snaps to tick_spacing
        w_ticks = int(round(w_unclipped / step_width_ticks)) * step_width_ticks
        # Enforce minimum based on w_min_ticks (rounded up to spacing multiple), not just one band
        _min_bands = max(1, (w_min_ticks + step_width_ticks - 1) // step_width_ticks)
        _max_bands = max(1, w_max_ticks // step_width_ticks)
        w_ticks = max(_min_bands * step_width_ticks, min(w_ticks, _max_bands * step_width_ticks))
        # w_ticks = int(round(max(w_min_ticks, min(w_unclipped, w_max_ticks))))
        # ---------------------------------------------------------------------

        # --- Per-step accumulators (so we can randomize actor order) ---
        trader_y_this = 0.0
        arb_y_this = 0.0
        trader_pnl_this = 0.0
        arb_pnl_this = 0.0
        _trader_execs = 0
        _arb_execs = 0
        # Split per-actor accumulators
        sr_y_this = 0.0
        noise_y_this = 0.0
        sr_pnl_this = 0.0
        noise_pnl_this = 0.0
        _sr_execs = 0
        _noise_execs = 0
        delta_a_cex_this = 0.0
        L_pre_trader_this = np.nan
        L_pre_arb_eff_this = np.nan
        dir_arb_this: Optional[str] = None

        # --- Actor routines (closures) ---
        def act_LPs():
            # ----- burns (TP/SL) -----
            for lp in LPs:
                if hasattr(lp, "can_act") and not lp.can_act:
                    continue
                to_burn = []
                for i, pos in enumerate(lp.positions):
                    pnl = pos.PnL_y(pool.S, ref.m)
                    if pnl >= theta_TP * pos.hodl0_value_y or pnl <= -theta_SL * pos.hodl0_value_y:
                        to_burn.append(i)
                for i in reversed(to_burn):
                    burn_any(lp, i)   # sets lp.cooldown

            # ----- re-center (narrow LPs only) -----
            for lp in LPs:
                if hasattr(lp, "can_act") and not lp.can_act:
                    continue
                to_recenters: List[int] = []
                for i, pos in enumerate(lp.positions):
                    in_rng = pos.in_range(pool.tick)
                    out_steps = getattr(pos, "out_steps", 0)
                    out_steps = 0 if in_rng else out_steps + 1
                    setattr(pos, "out_steps", out_steps)
                    if lp.is_active_narrow and out_steps >= k_out:
                        to_recenters.append(i)

                for i in reversed(to_recenters):
                    pos = lp.positions[i]
                    width = pos.upper - pos.lower
                    L_same = pos.L
                    burn_any(lp, i)
                    # Center around current sqrt price S (approximately), not the snapped active band.
                    n_bands = max(1, int(round(width / pool.tick_spacing)))
                    S_now = pool.S
                    s = pool.tick_spacing
                    nb = n_bands
                    denom = (1.0 + (pool.g ** (nb * s + s)))
                    if denom <= 0.0:
                        denom = 1.0
                    lower_real = math.log((2.0 * S_now / pool.base_s) / denom, pool.g)
                    lower = pool._snap(int(round(lower_real)))
                    upper = lower + nb * s
                    sa, sb = pool.s_lower(lower), pool.s_upper(upper)
                    amt0, amt1 = minted_amounts_at_S(L_same, sa, sb, pool.S)
                    newpos = Position(
                        owner=lp.id, lower=lower, upper=upper, L=L_same, sa=sa, sb=sb,
                        amt0_init=amt0, amt1_init=amt1, hodl0_value_y=amt0 * ref.m + amt1,
                    )
                    pool.add_liquidity_range(lower, upper, L_same)
                    bidx.mark_dirty()
                    lp.positions.append(newpos)
                    mint_steps.append(t); mint_sizes.append(L_same); mint_widths.append(upper - lower)
                    with open(verbose_log_path_str, "a") as f:
                        f.write(f"[t={t:03d}] LP{lp.id} RECENTER L={L_same:.4f} [{lower},{upper}) | L_active={pool.L_active:.4f}\n")

            # ----- probabilistic mints (blocked during cooldown) -----
            for lp in LPs:
                if hasattr(lp, "can_act") and not lp.can_act:
                    continue
                if getattr(lp, "cooldown", 0) > 0:
                    continue
                if random.random() < lp.mintProb:
                    mint_lp(lp, w_ticks)

            pool.recompute_active_L()
            if -1e-9 < pool.L_active < 0.0:
                pool.L_active = 0.0

        
            # ----- Non-mutating Uni v3 quotes (spacing-aware, can bridge deserts) -----
        def _active_L_at_tick_local(tick_i: int) -> float:
            L = 0.0
            for k, dL in pool.liquidity_net.items():
                if k <= tick_i:
                    L += dL
            return L

        def quote_x_to_y(dx_in: float) -> float:
            if dx_in <= 0:
                return 0.0
            r = pool.r
            dx_eff = dx_in * r

            tick_loc = tick_from_S(pool.S)
            S_loc = pool.S
            L_loc = _active_L_at_tick_local(tick_loc)

            # bridge through a desert (downwards for X->Y)
            if L_loc <= EPS_LIQ:
                pb = bidx.prev_down(tick_loc)
                if pb is None:
                    return 0.0
                tick_loc = pool._snap(pb - pool.tick_spacing)
                S_loc = pool.s_upper(tick_loc)
                L_loc = _active_L_at_tick_local(tick_loc)
                if L_loc <= EPS_LIQ:
                    return 0.0

            dy_out = 0.0
            while dx_eff > EPS_LIQ and L_loc > EPS_LIQ:
                S_lo = pool.s_lower(tick_loc)
                if S_loc <= S_lo + EPS_BOUNDARY:
                    S_loc = S_lo
                    tick_loc -= pool.tick_spacing
                    L_loc = _active_L_at_tick_local(tick_loc)
                    continue
                dx_to = L_loc * (1.0 / S_lo - 1.0 / S_loc)
                if dx_eff < dx_to - EPS_BOUNDARY:
                    S_new = 1.0 / (1.0 / S_loc + dx_eff / L_loc)
                    dy = L_loc * (S_new - S_loc)     # negative
                    dy_out += -dy
                    dx_eff = 0.0
                else:
                    dy = L_loc * (S_lo - S_loc)      # negative
                    dy_out += -dy
                    dx_eff -= dx_to
                    S_loc = S_lo
                    tick_loc -= pool.tick_spacing
                    L_loc = _active_L_at_tick_local(tick_loc)
            return max(0.0, dy_out)

        def quote_y_to_x(dy_in: float) -> float:
            if dy_in <= 0:
                return 0.0
            r = pool.r
            dy_eff = dy_in * r

            tick_loc = tick_from_S(pool.S)
            S_loc = pool.S
            L_loc = _active_L_at_tick_local(tick_loc)

            # bridge through a desert (upwards for Y->X)
            if L_loc <= EPS_LIQ:
                nb = bidx.next_up(tick_loc)
                if nb is None:
                    return 0.0
                tick_loc = pool._snap(nb)
                S_loc = pool.s_lower(tick_loc)
                L_loc = _active_L_at_tick_local(tick_loc)
                if L_loc <= EPS_LIQ:
                    return 0.0

            dx_out = 0.0
            while dy_eff > EPS_LIQ and L_loc > EPS_LIQ:
                S_hi = pool.s_upper(tick_loc)
                if S_loc >= S_hi - EPS_BOUNDARY:
                    S_loc = S_hi
                    tick_loc += pool.tick_spacing
                    L_loc = _active_L_at_tick_local(tick_loc)
                    continue
                dy_to = L_loc * (S_hi - S_loc)
                if dy_eff < dy_to - EPS_BOUNDARY:
                    S_new = S_loc + dy_eff / L_loc
                    dx = L_loc * (1.0 / S_loc - 1.0 / S_new)
                    dx_out += dx
                    dy_eff = 0.0
                else:
                    dx = L_loc * (1.0 / S_loc - 1.0 / S_hi)
                    dx_out += dx
                    dy_eff -= dy_to
                    S_loc = S_hi
                    tick_loc += pool.tick_spacing
                    L_loc = _active_L_at_tick_local(tick_loc)
            return max(0.0, dx_out)

        def act_smart_router():
            nonlocal sr_y_this, L_pre_trader_this, sr_pnl_this, _sr_execs, delta_a_cex_this, trader_pnl_this, trader_y_this

            # Arrival as before for 'smart' flow
            if random.random() >= p_trade:
                return

            side = random.choice(["X_to_Y", "Y_to_X"])
            L_pre_trader_this = pool.L_active
            P_pre = pool.price
            m_now = ref.m

            if side == "X_to_Y":
                dx = np.exp(np.random.normal(loc=trader_mean, scale=trader_sigma))
                if dx <= 0:
                    return
                # Decision rule: execute only if DEX output >= theta_T * CEX output
                dy_dex_quote = quote_x_to_y(dx)
                dy_cex = dx * m_now
                if dy_dex_quote < theta_T * dy_cex:
                    return

                prev_tick, prev_S = pool.tick, pool.S
                bridged = False
                if pool.L_active <= EPS_LIQ:
                    ok, new_tick, new_S, _ = ensure_liquidity("down")
                    if not ok:
                        return
                    pool.tick, pool.S = new_tick, new_S
                    pool.recompute_active_L()
                    bridged = True

                used_dx_pre, dy_out_real, fee_x = pool.swap_x_to_y(dx, fee_cb=allocate_fees)
                if used_dx_pre <= EPS_LIQ:
                    if bridged:
                        pool.tick, pool.S = prev_tick, prev_S
                        pool.recompute_active_L()
                    return

                trader_steps.append(t); trader_dirs.append("down")
                # Update SR and aggregate trader metrics
                sr_y_this += -P_pre * used_dx_pre
                trader_y_this += -P_pre * used_dx_pre
                pnl = (dy_out_real - used_dx_pre * m_now)
                sr_pnl_this += pnl
                trader_pnl_this += pnl
                _sr_execs += int(used_dx_pre > 0)

            else:
                dy = np.exp(np.random.normal(loc=trader_mean, scale=trader_sigma))
                if dy <= 0:
                    return

                dx_dex_quote = quote_y_to_x(dy)
                dx_cex = dy / max(m_now, 1e-18)
                if dx_dex_quote < theta_T * dx_cex:
                    return

                prev_tick, prev_S = pool.tick, pool.S
                bridged = False
                if pool.L_active <= EPS_LIQ:
                    ok, new_tick, new_S, _ = ensure_liquidity("up")
                    if not ok:
                        return
                    pool.tick, pool.S = new_tick, new_S
                    pool.recompute_active_L()
                    bridged = True

                used_dy_pre, dx_out_real, fee_y = pool.swap_y_to_x(dy, fee_cb=allocate_fees)
                if used_dy_pre <= EPS_LIQ:
                    if bridged:
                        pool.tick, pool.S = prev_tick, prev_S
                        pool.recompute_active_L()
                    return

                trader_steps.append(t); trader_dirs.append("up")
                sr_y_this += +used_dy_pre
                trader_y_this += +used_dy_pre
                pnl = (dx_out_real * m_now - used_dy_pre)
                sr_pnl_this += pnl
                trader_pnl_this += pnl
                _sr_execs += int(used_dy_pre > 0)

        def act_noise_trader():
            np.random.seed(t * 1000 + 999)  # separate stream from smart router
            nonlocal noise_y_this, L_pre_trader_this, noise_pnl_this, _noise_execs, trader_pnl_this, trader_y_this
            # Baseline probability of exogenous noise trade (arrives only on DEX)
            p_noise = noise_floor
            if random.random() >= p_noise:
                return

            side = random.choice(["X_to_Y", "Y_to_X"])
            L_pre_trader_this = pool.L_active
            P_pre = pool.price
            m_now = ref.m

            if side == "X_to_Y":
                dx = np.exp(np.random.normal(loc=trader_mean, scale=trader_sigma))
                if dx <= 0:
                    return

                prev_tick, prev_S = pool.tick, pool.S
                bridged = False
                if pool.L_active <= EPS_LIQ:
                    ok, new_tick, new_S, _ = ensure_liquidity("down")
                    if not ok:
                        return
                    pool.tick, pool.S = new_tick, new_S
                    pool.recompute_active_L()
                    bridged = True

                used_dx_pre, dy_out_real, fee_x = pool.swap_x_to_y(dx, fee_cb=allocate_fees)
                if used_dx_pre <= EPS_LIQ:
                    if bridged:
                        pool.tick, pool.S = prev_tick, prev_S
                        pool.recompute_active_L()
                    return

                trader_steps.append(t); trader_dirs.append("down")
                noise_y_this += -P_pre * used_dx_pre
                trader_y_this += -P_pre * used_dx_pre
                pnl = (dy_out_real - used_dx_pre * m_now)
                noise_pnl_this += pnl
                trader_pnl_this += pnl
                _noise_execs += int(used_dx_pre > 0)

            else:
                dy = np.exp(np.random.normal(loc=trader_mean, scale=trader_sigma))
                if dy <= 0:
                    return

                prev_tick, prev_S = pool.tick, pool.S
                bridged = False
                if pool.L_active <= EPS_LIQ:
                    ok, new_tick, new_S, _ = ensure_liquidity("up")
                    if not ok:
                        return
                    pool.tick, pool.S = new_tick, new_S
                    pool.recompute_active_L()
                    bridged = True

                used_dy_pre, dx_out_real, fee_y = pool.swap_y_to_x(dy, fee_cb=allocate_fees)
                if used_dy_pre <= EPS_LIQ:
                    if bridged:
                        pool.tick, pool.S = prev_tick, prev_S
                        pool.recompute_active_L()
                    return

                trader_steps.append(t); trader_dirs.append("up")
                noise_y_this += +used_dy_pre
                trader_y_this += +used_dy_pre
                pnl = (dx_out_real * m_now - used_dy_pre)
                noise_pnl_this += pnl
                trader_pnl_this += pnl
                _noise_execs += int(used_dy_pre > 0)

        def act_arbitrageur():
            nonlocal arb_y_this, L_pre_arb_eff_this, dir_arb_this, delta_a_cex_this, arb_pnl_this, _arb_execs

            in_used, x_out_from_dex, y_out_from_dex, dir_arb, fee_x_arb, fee_y_arb, L_first = arbitrage_to_target()
            delta_a_cex_this = 0.0
            if in_used > 0 and dir_arb is not None:
                L_pre_arb_eff_this = L_first
                dir_arb_this = dir_arb
                arb_steps.append(t); arb_dirs.append(dir_arb)

                if dir_arb == "up":
                    # DEX cheap: buy A on DEX (A out), sell A on CEX @ m_now
                    delta_a_cex_this = -x_out_from_dex
                    arb_y_this = +in_used
                    arb_pnl_this += (x_out_from_dex * ref.m - in_used)
                    _arb_execs += int(in_used > 0)
                    # Fees already allocated per span via fee_cb
                else:
                    # DEX expensive: sell A on DEX (A in), buy A on CEX @ m_now
                    delta_a_cex_this = +in_used
                    arb_y_this = -pool.price * in_used
                    arb_pnl_this += (y_out_from_dex - in_used * ref.m)
                    _arb_execs += int(in_used > 0)
                    # Fees already allocated per span via fee_cb

        # --- async LP micro-scheduler: A â†’ trader â†’ B â†’ arb â†’ C ---

        # figure out which LPs are due to act this step
        due = []
        for i, lp in enumerate(LPs):
            if lp.cooldown > 0:
                lp.cooldown -= 1
                # let the review clock keep ticking while cooling down
                lp.next_review = max(1, lp.next_review - 1)
                continue
            lp.next_review -= 1
            if lp.next_review <= 0:
                due.append(i)
                lp.next_review = int(np.random.geometric(lp.review_rate))

        # split due LPs into 3 buckets so we can interleave them
        random.shuffle(due)
        n = len(due)
        k1 = np.random.binomial(n, 1/3) if n > 0 else 0
        k2 = np.random.binomial(n - k1, 1/2) if (n - k1) > 0 else 0
        bucketA = due[:k1]
        bucketB = due[k1:k1+k2]
        bucketC = due[k1+k2:]

        def _enable(indices):
            s = set(indices)
            for j, lp in enumerate(LPs):
                lp.can_act = (j in s)

        # run the schedule
        _enable(bucketA)
        act_LPs()

        _enable(bucketB)
        act_smart_router()
        act_noise_trader()
        act_LPs()

        _enable(bucketC)
        act_arbitrageur()
        act_LPs()

        # disable everyone for next step
        _enable([])

        # ---- CEX update  ----
        ref.step(delta_a_cex_this)

        # ================== Dynamic fee controller  ==================
        # Signals based on END-OF-STEP state; new fee applies NEXT step.

        # 1) Volatility of CEX (abs log-return)
        try:
            vol_obs = abs(math.log(max(ref.m, 1e-18)) - math.log(max(prev_m_for_vol, 1e-18)))
        except ValueError:
            vol_obs = 0.0
        prev_m_for_vol = ref.m
        sigma_hat = ewma_sigma_fee.update(vol_obs**2)  # The LVR is proportional to volatility squared ($\sigma^2$), as shown by Milionis, et al. (2023)

        # 2) Toxicity / basis (fee-adjusted log gap)
        fee_band_ln = -math.log1p(-pool.f)  # ln(1/(1-f))
        log_gap = abs(math.log(max(pool.price, 1e-18)) - math.log(max(ref.m, 1e-18)))
        B_obs = max(0.0, log_gap - fee_band_ln)
        B_hat = ewma_basis_fee.update(B_obs)
        basis_ticks = B_hat / TICK_LN   # convert log-gap to "ticks"

        # Record raw signals for diagnostics/plotting
        fee_sigma_series.append(sigma_hat)
        fee_basis_ticks_series.append(basis_ticks)

        # Select controller
        f_raw = pool.f
        if fee_mode == "volatility":
            f_raw = f0 + k_sigma * sigma_hat
        elif fee_mode == "toxicity":
            f_raw = f0 + k_basis * basis_ticks
        else:
            f_raw = pool.f  # "static": no change


        # Controller signal used for plotting (depends on fee_mode)
        if fee_mode == "volatility":
            ctrl_sig = sigma_hat
        elif fee_mode == "toxicity":
            ctrl_sig = basis_ticks
        else:
            ctrl_sig = 0.0
        fee_signal_series.append(ctrl_sig)
        # Clip and apply hysteresis (min/max step in bps, cooldown)
        f_tgt = clamp(f_raw, f_min, f_max)
        min_step = fee_step_bps_min / 1e4
        max_step = fee_step_bps_max / 1e4
        delta_f = f_tgt - pool.f
        if abs(delta_f) >= min_step:
            step = math.copysign(min(abs(delta_f), max_step), delta_f)
            f_new = clamp(pool.f + step, f_min, f_max)
            if abs(f_new - pool.f) >= 1e-12:
                fee_next = f_new
                fee_cooldown_left = max(0, int(fee_cooldown))

        # record current fee (before next-step commit)
        fee_series.append(pool.f)
        # ==================================================================

        # ---- Record end-of-step + invariants ----
        P_after = pool.price
        P_series.append(P_after)
        M_series.append(ref.m)
        delta_a_cex_series.append(delta_a_cex_this)

        x_e, y_e = reserves_in_active_tick()
        X_active_end.append(x_e)
        Y_active_end.append(y_e)

        band_lo_post.append(ref.m * r)
        band_hi_post.append(ref.m / r)
        L_end.append(pool.L_active)
        # ---- PnL bookkeeping ----
        trader_pnl_steps.append(trader_pnl_this)
        arb_pnl_steps.append(arb_pnl_this)
        trader_exec_count.append(_trader_execs)
        arb_exec_count.append(_arb_execs)
        lp_total = 0.0
        lp_total_active = 0.0
        lp_total_passive = 0.0
        lp_wallet_total = 0.0
        lp_wallet_active = 0.0
        lp_wallet_passive = 0.0
        for lp in LPs:
            wallet_y = getattr(lp, 'wallet_y', 0.0)
            lp_wallet_total += wallet_y
            lp_open_pnl = 0.0
            for pos in lp.positions:
                lp_open_pnl += pos.PnL_y(pool.S, ref.m)
            lp_total += lp_open_pnl
            if lp.is_active_narrow:
                lp_total_active += lp_open_pnl
                lp_wallet_active += wallet_y
            else:
                lp_total_passive += lp_open_pnl
                lp_wallet_passive += wallet_y
        lp_pnl_total_series.append(lp_total)
        lp_pnl_active_series.append(lp_total_active)
        lp_pnl_passive_series.append(lp_total_passive)

        # Wealth accounting (wallet + open PnL)
        lp_wallet_series.append(lp_wallet_total)
        lp_wallet_active_series.append(lp_wallet_active)
        lp_wallet_passive_series.append(lp_wallet_passive)
        lp_wealth_series.append(lp_wallet_total + lp_total)
        lp_wealth_active_series.append(lp_wallet_active + lp_total_active)
        lp_wealth_passive_series.append(lp_wallet_passive + lp_total_passive)
        # store per-step trader/arb details (now that order is randomized)
        trader_y_series.append(trader_y_this)
        arb_y_series.append(arb_y_this)
        L_pre_trader.append(L_pre_trader_this)

        sr_y_series.append(sr_y_this)
        noise_y_series.append(noise_y_this)
        sr_pnl_steps.append(sr_pnl_this)
        noise_pnl_steps.append(noise_pnl_this)
        sr_exec_count.append(_sr_execs)
        noise_exec_count.append(_noise_execs)
        L_pre_arb_eff.append(L_pre_arb_eff_this)

        price_moved = abs(P_after - P_before) > EPS_PRICE_CHANGE
        had_fill = (abs(trader_y_this) > 0) or (abs(arb_y_this) > 0)
        had_L_event = (t in mint_steps) or (t in burn_steps)
        if price_moved and not (had_fill or had_L_event):
            raise RuntimeError(
                f"DEX price changed at t={t} without swap or LP Î”L. L_active={pool.L_active:.4f}. Change {abs(P_after - P_before)}"
            )

        # Save verbose step info to a txt file in abm_results
        log_line = (
        f"[t={t:03d}] DEX={pool.price:.4f} | CEX={ref.m:.4f} | "
        f"traderY={trader_y_this:.2f} | arb_dir={dir_arb_this} arbY={arb_y_this:.2f} | "
        f"L={pool.L_active:.4f} | w_ticks={w_ticks}"
        )
        with open(verbose_log_path_str, "a") as f:
            f.write(log_line + "\n")

        liq_history.append(dict(pool.liquidity_net))
        tick_history.append(pool.tick)

    # =============================================================================
    # Plotting
    # =============================================================================
    P_series = np.array(P_series)
    M_series = np.array(M_series)
    X_active_end = np.array(X_active_end)
    Y_active_end = np.array(Y_active_end)
    band_lo_pre = np.array(band_lo_pre)
    band_hi_pre = np.array(band_hi_pre)
    band_lo_post = np.array(band_lo_post)
    band_hi_post = np.array(band_hi_post)
    L_end = np.array(L_end)
    L_pre_step = np.array(L_pre_step)
    L_pre_trader = np.array(L_pre_trader)
    L_pre_arb_eff = np.array(L_pre_arb_eff)
    steps = np.arange(len(P_series))
    trader_pnl_steps = np.array(trader_pnl_steps)
    arb_pnl_steps = np.array(arb_pnl_steps)
    trader_pnl_cum = np.cumsum(trader_pnl_steps)
    arb_pnl_cum = np.cumsum(arb_pnl_steps)
    sr_pnl_steps = np.array(sr_pnl_steps)
    noise_pnl_steps = np.array(noise_pnl_steps)
    sr_pnl_cum = np.cumsum(sr_pnl_steps)
    noise_pnl_cum = np.cumsum(noise_pnl_steps)
    lp_pnl_total_series = np.array(lp_pnl_total_series)
    lp_pnl_active_series = np.array(lp_pnl_active_series)
    lp_pnl_passive_series = np.array(lp_pnl_passive_series)
    lp_wallet_series = np.array(lp_wallet_series)
    lp_wallet_active_series = np.array(lp_wallet_active_series)
    lp_wallet_passive_series = np.array(lp_wallet_passive_series)
    lp_wealth_series = np.array(lp_wealth_series)
    lp_wealth_active_series = np.array(lp_wealth_active_series)
    lp_wealth_passive_series = np.array(lp_wealth_passive_series)
    fee_sigma_series = np.array(fee_sigma_series)
    fee_basis_ticks_series = np.array(fee_basis_ticks_series)
    fee_imb_series = np.array(fee_imb_series)
    fee_signal_series = np.array(fee_signal_series)

    # --- Visualization skip window ---
    s0 = max(0, int(skip_step))
    steps_v = steps[s0:]
    P_series_v = P_series[s0:]
    M_series_v = M_series[s0:]
    X_active_end_v = X_active_end[s0:]
    Y_active_end_v = Y_active_end[s0:]
    band_lo_pre_v = band_lo_pre[s0:]
    band_hi_pre_v = band_hi_pre[s0:]
    band_lo_post_v = band_lo_post[s0:]
    band_hi_post_v = band_hi_post[s0:]
    L_end_v = L_end[s0:]
    L_pre_step_v = L_pre_step[s0:]
    L_pre_trader_v = L_pre_trader[s0:]
    L_pre_arb_eff_v = L_pre_arb_eff[s0:]
    trader_pnl_steps_v = trader_pnl_steps[s0:]
    arb_pnl_steps_v = arb_pnl_steps[s0:]
    trader_pnl_cum_v = trader_pnl_cum[s0:]
    arb_pnl_cum_v = arb_pnl_cum[s0:]
    sr_pnl_cum_v = sr_pnl_cum[s0:]
    noise_pnl_cum_v = noise_pnl_cum[s0:]
    lp_wealth_series_v = lp_wealth_series[s0:]
    lp_wealth_active_series_v = lp_wealth_active_series[s0:]
    lp_wealth_passive_series_v = lp_wealth_passive_series[s0:]
    fee_series_v = fee_series[s0:]
    fee_sigma_series_v = fee_sigma_series[s0:]
    fee_basis_ticks_series_v = fee_basis_ticks_series[s0:]
    fee_imb_series_v = fee_imb_series[s0:]
    fee_signal_series_v = fee_signal_series[s0:]
    trader_y_v = np.array(trader_y_series)[s0:]
    arb_y_v = np.array(arb_y_series)[s0:]
    
    if visualize:
        # Î”L per step (aggregate)
        mint_step_sum = np.zeros_like(P_series)
        for s, L in zip(mint_steps, mint_sizes):
            if 0 <= s < len(mint_step_sum):
                mint_step_sum[s] += L
        burn_step_sum = np.zeros_like(P_series)
        for s, L in zip(burn_steps, burn_sizes):
            if 0 <= s < len(burn_step_sum):
                burn_step_sum[s] += L


        mint_step_sum_v = mint_step_sum[s0:]
        burn_step_sum_v = burn_step_sum[s0:]
        
        # ===== Separate figures instead of a single multi-subplot figure =====
        from pathlib import Path as _Path
        _out_dir = _Path("abm_results")
        _out_dir.mkdir(parents=True, exist_ok=True)
        _prefix = f"abm_fee_{fee_mode}_{cex_sigma}"

        # Common helper to save + tidy
        def _save_fig(fig, name):
            fig.tight_layout()
            fig.savefig(_out_dir / f"{_prefix}_{name}.png", dpi=150)
        
        # ----- 1) Price panel -----
        fig1, ax = plt.subplots(figsize=(15, 4.5))
        ax.fill_between(steps_v, band_lo_post_v, band_hi_post_v, color="lightgray", alpha=0.35, label="No-arb fee band")
        ax.plot(steps_v, band_lo_post_v, color="#888", linestyle=":", linewidth=1.2)
        ax.plot(steps_v, band_hi_post_v, color="#888", linestyle=":", linewidth=1.2)
        ax.plot(steps_v, P_series_v, label="DEX price P_t", linewidth=2)
        ax.plot(steps_v, M_series_v, "--", label="CEX price m_t", linewidth=1.6)

        # --- Vertical-only offsets to avoid marker overlap ---
        s0 = max(0, int(skip_step))
        off_y = (50 / 1e4) * P_series   # baseline: ~50 bps of price

        # Trader markers (stars)
        # t_up = [s for s, d in zip(trader_steps, trader_dirs) if d == "up" and s >= s0]
        # t_dn = [s for s, d in zip(trader_steps, trader_dirs) if d == "down" and s >= s0]
        # if t_up:
        #     arr = np.array(t_up, dtype=int)
        #     ax.scatter(arr, P_series[arr] - 1.00*off_y[arr], marker="*", color="#008B8B", s=50, label="Trader Y→X (↑)")
        # if t_dn:
        #     arr = np.array(t_dn, dtype=int)
        #     ax.scatter(arr, P_series[arr] - 1.60*off_y[arr], marker="*", color="#20B2AA", s=50, label="Trader X→Y (↓)")

        # Arbitrage markers (triangles)
        up = [s for s, d in zip(arb_steps, arb_dirs) if d == "up" and s >= s0]
        dn = [s for s, d in zip(arb_steps, arb_dirs) if d == "down" and s >= s0]
        if len(arb_steps) > 0:
            arb_abs = np.array([abs(arb_y_series[s]) for s in arb_steps])
            max_abs = float(max(1e-12, np.max(arb_abs)))
            _scale = lambda a: 30 + 120 * (a / max_abs)
        else:
            _scale = lambda a: 30
        if up:
            arr = np.array(up, dtype=int)
            ax.scatter(arr, P_series[arr] + 1.00*off_y[arr], marker="^", color="green",
                       s=[_scale(abs(arb_y_series[s])) for s in up], label="Arb (↑ to target)")
        if dn:
            arr = np.array(dn, dtype=int)
            ax.scatter(arr, P_series[arr] + 1.60*off_y[arr], marker="v", color="red",
                       s=[_scale(abs(arb_y_series[s])) for s in dn], label="Arb (↓ to target)")

        # LP markers (circles/crosses), stacked higher than arb
        # if len(mint_steps) + len(burn_steps) > 0:
        #     maxL = 1e-12
        #     if mint_sizes: maxL = max(maxL, max(mint_sizes))
        #     if burn_sizes: maxL = max(maxL, max(burn_sizes))
        #     scaleL = lambda L: 30 + 120 * (L / maxL)
        #     if mint_steps:
        #         mint_steps_v = [s for s in mint_steps if s >= s0]
        #         if mint_steps_v:
        #             arr = np.array(mint_steps_v, dtype=int)
        #             ax.scatter(arr, P_series[arr] + 2.40*off_y[arr],
        #                        marker="o", facecolors="none", edgecolors="#6a0dad",
        #                        s=[scaleL(mint_sizes[mint_steps.index(s)]) for s in mint_steps_v],
        #                        label="LP mint/center")
        #     if burn_steps:
        #         burn_steps_v = [s for s in burn_steps if s >= s0]
        #         if burn_steps_v:
        #             arr = np.array(burn_steps_v, dtype=int)
        #             ax.scatter(arr, P_series[arr] + 3.20*off_y[arr],
        #                        marker="x", color="#ff8c00",
        #                        s=[scaleL(burn_sizes[burn_steps.index(s)]) for s in burn_steps_v],
        #                        label="LP burn")

        ax.set_xlim(steps_v[0]-0.5, steps_v[-1]+0.5)
        ax.set_ylabel("Price (token1 per token0)", fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("Step", fontsize=LABEL_FONT_SIZE)
        ax.set_title("CEX vs DEX Price", fontsize=TITLE_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=LEGEND_FONT_SIZE)
        ax.margins(y=0.14)
        _save_fig(fig1, "1_price")
        # plt.close(fig1)

        # ----- 2) Notionals -----
        fig2, ax = plt.subplots(figsize=(15, 3.6))
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.3)
        ax.bar(steps_v, trader_y_v, width=0.8, alpha=0.5, label="Trader notional (token1, signed)")
        ax.plot(steps_v, arb_y_v, label="Arbitrage notional (token1, signed)", lw=1.8)
        ax.set_ylabel("Notional (token1, signed)", fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left")
        _save_fig(fig2, "2_notional")
        # plt.close(fig2)

        # ----- 3) Liquidity traces -----
        fig3, ax = plt.subplots(figsize=(15, 3.6))
        ax.plot(steps_v, L_end_v, lw=1.8, label="Active L (end of step)")
        ax.plot(steps_v, L_pre_step_v, lw=1.0, ls="--", label="Active L (start of step)")
        ax.plot(steps_v, L_pre_trader_v, lw=1.0, ls=":", label="Active L (before trader)")
        ax.plot(steps_v, L_pre_arb_eff_v, lw=1.2, ls="-.", label="Active L (before arb, effective)")
        ax.set_ylabel("Active L", fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left")
        for s_idx, L in zip(steps_v, L_end_v):
            if L <= 1e-9:
                ax.axvspan(s_idx - 0.5, s_idx + 0.5, color="red", alpha=0.05, lw=0)
        _save_fig(fig3, "3_activeL")
        # plt.close(fig3)

        # ----- 4) L per step -----
        fig4, ax = plt.subplots(figsize=(15, 3.6))
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.3)
        ax.bar(steps_v, mint_step_sum_v, width=0.8, alpha=0.6, label="LP mint/center L (>0)", color="#6a0dad")
        ax.bar(steps_v, -burn_step_sum_v, width=0.8, alpha=0.6, label="LP burn L (<0)", color="#ff8c00")
        ax.set_ylabel("L per step", fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left")
        _save_fig(fig4, "4_L_per_step")
        # plt.close(fig4)

        # ----- 5) Active-band reserves -----
        fig5, ax = plt.subplots(figsize=(15, 3.6))
        ax.plot(steps_v, X_active_end_v * P_series_v, lw=1.8, label="token0 value in active band (≈ token1 units)")
        ax.plot(steps_v, Y_active_end_v, lw=1.8, label="token1 in active band (Y)")
        ax.set_xlabel("Step", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Active-band reserves", fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left")
        for s_idx, L in zip(steps_v, L_end_v):
            if L <= 1e-9:
                ax.axvspan(s_idx - 0.5, s_idx + 0.5, color="red", alpha=0.05, lw=0)
        _save_fig(fig5, "5_active_reserves")
        # plt.close(fig5)

        # ----- 6) PnL panel -----
        fig6, ax = plt.subplots(figsize=(15, 3.6))
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.3)
        ax.plot(steps_v, sr_pnl_cum_v, lw=1.8, label="Smart Router cumulative PnL (token1)")
        # ax.plot(steps_v, noise_pnl_cum_v, lw=1.8, label="Noise Trader cumulative PnL (token1)")
        ax.plot(steps_v, arb_pnl_cum_v, lw=1.8, label="Arbitrageur cumulative PnL (token1)")
        ax.plot(steps_v, lp_wealth_series_v, lw=1.8, label="LPs Total PnL (wallet+open, token1)")
        ax.plot(steps_v, lp_wealth_active_series_v, lw=1.5, linestyle="--",
                label="Active LPs PnL (wallet+open)")
        ax.plot(steps_v, lp_wealth_passive_series_v, lw=1.5, linestyle=":",
                label="Passive LPs PnL (wallet+open)")
        ax.set_ylabel("PnL", fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("Step", fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left")
        _save_fig(fig6, "6_pnl")
        # plt.close(fig6)

        # ----- 7) Fee panel + twin signal -----
        fig7, ax = plt.subplots(figsize=(15, 3.6))
        ax2b = ax.twinx()
        ax.plot(steps_v, fee_series_v, lw=1.8, label="Fee")
        ax.set_ylabel("Fee", fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("Step", fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        if fee_mode == "volatility":
            ax2b.plot(steps_v, fee_sigma_series_v, lw=1.2, linestyle="--", label="σ̂ (abs log-return)")
            ax2b.set_ylabel("σ̂", fontsize=LABEL_FONT_SIZE)
        elif fee_mode == "toxicity":
            ax2b.plot(steps_v, fee_basis_ticks_series_v, lw=1.2, linestyle="--", label="Basis (ticks)")
            ax2b.set_ylabel("Basis (ticks)", fontsize=LABEL_FONT_SIZE)
        else:
            ax2b.plot(steps_v, fee_signal_series_v, lw=1.2, linestyle="--", label="Controller signal")
            ax2b.set_ylabel("Signal", fontsize=LABEL_FONT_SIZE)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2b.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=LEGEND_FONT_SIZE, loc="upper left")
        _save_fig(fig7, "7_fee")
        # plt.show()
        # plt.close(fig7)

    return {
        "DEX_price": P_series,
        "CEX_price": M_series,
        "band_lo_pre": band_lo_pre,
        "band_hi_pre": band_hi_pre,
        "band_lo_post": band_lo_post,
        "band_hi_post": band_hi_post,
        "L_active_end": L_end,
        "L_pre_step": L_pre_step,
        "L_pre_trader": L_pre_trader,
        "L_pre_arb_eff": L_pre_arb_eff,
        "trader_notional_y": trader_y_series,
        "arb_notional_y": arb_y_series,
        "trader_steps": trader_steps,
        "trader_dirs": trader_dirs,
        "arb_steps": arb_steps,
        "arb_dirs": arb_dirs,
        "mint_steps": mint_steps,
        "mint_sizes": mint_sizes,
        "mint_widths": mint_widths,
        "burn_steps": burn_steps,
        "burn_sizes": burn_sizes,
        "liq_history": liq_history,
        "tick_history": tick_history,
        "x_active_reserves": X_active_end.tolist(),
        "y_active_reserves": Y_active_end.tolist(),
        "grid_base_s": pool.base_s,
        "grid_g": pool.g,
        "trader_pnl_steps": trader_pnl_steps.tolist(),
        "arb_pnl_steps": arb_pnl_steps.tolist(),
        "trader_pnl_cum": trader_pnl_cum.tolist(),
        "arb_pnl_cum": arb_pnl_cum.tolist(),
        "smart_router_pnl_steps": sr_pnl_steps.tolist(),
        "noise_trader_pnl_steps": noise_pnl_steps.tolist(),
        "smart_router_pnl_cum": sr_pnl_cum.tolist(),
        "noise_trader_pnl_cum": noise_pnl_cum.tolist(),
        "smart_router_notional_y": sr_y_series,
        "noise_trader_notional_y": noise_y_series,
        "smart_router_exec_count": sr_exec_count,
        "noise_trader_exec_count": noise_exec_count,
        "lp_pnl_total": lp_pnl_total_series.tolist(),
        "lp_pnl_active": lp_pnl_active_series.tolist(),
        "lp_pnl_passive": lp_pnl_passive_series.tolist(),
        "trader_exec_count": trader_exec_count,
        "fee_series": fee_series,
        "fee_mode": fee_mode,
        "f_min": f_min,
        "f_max": f_max,


        "fee_sigma_series": fee_sigma_series.tolist(),
        "fee_basis_ticks_series": fee_basis_ticks_series.tolist(),
        "fee_imb_series": fee_imb_series.tolist(),
        "fee_signal_series": fee_signal_series.tolist(),
        "lp_wallet_series": lp_wallet_series.tolist(),
        "lp_wealth_series": lp_wealth_series.tolist(),
        "lp_wallet_active_series": lp_wallet_active_series.tolist(),
        "lp_wallet_passive_series": lp_wallet_passive_series.tolist(),
        "lp_wealth_active_series": lp_wealth_active_series.tolist(),
        "lp_wealth_passive_series": lp_wealth_passive_series.tolist(),
        "arb_exec_count": arb_exec_count,
    }


# =============================================================================
# Visualize liquidity per tick over time (GIF)
# =============================================================================

def make_liquidity_gif(
    liq_history: List[Dict[int, float]],
    tick_history: List[int],
    base_s: float,             # <â€” NEW
    g: float,                  # <â€” NEW
    out_path: str = "abm_results/liquidity_evolution.gif",
    fps: int = 10,
    dpi: int = 120,
    pad_frac: float = 0.05,
    downsample_every: int = 1,
    center_line: bool = True,  # draw the active **band center** (P_lower * g)
):
    """
    Animate liquidity per 1-tick bin, with the x-axis in **price** (P = S^2).

    Bars:
      left edge  = P_lower(i) = (base_s * g**i)^2
      width      = Î”P(i) = P_lower(i) * (g**2 - 1)
      height     = active liquidity in that 1-tick bin

    Vertical line:
      at the active bandâ€™s **center** price (default) or at the lower edge.
    """
    assert len(liq_history) == len(tick_history), "Mismatched histories."

    if downsample_every > 1:
        liq_history = liq_history[::downsample_every]
        tick_history = tick_history[::downsample_every]

    # ----- collect the universe of tick boundaries we ever touch -----
    all_boundaries = set()
    for snap in liq_history:
        all_boundaries.update(k for k, v in snap.items() if abs(v) > EPS_LIQ)
    if not all_boundaries:
        all_boundaries = {0}
    tmin = min(all_boundaries) - 5
    tmax = max(all_boundaries) + 5

    boundaries = np.arange(tmin, tmax + 1, dtype=int)   # tick boundaries (1-tick step)
    tick_axis = boundaries[:-1]                          # left edge tick of each 1-tick bin

    # ----- build L frames (unchanged) -----
    L_frames = []
    ymax = 1e-12
    for snap in tqdm(liq_history, desc="Building L frames"):
        delta = np.zeros_like(boundaries, dtype=float)
        for k, dL in snap.items():
            if tmin <= k <= tmax:
                delta[k - tmin] += dL
        L_per_tick = np.cumsum(delta)[:-1]              # active L in each 1-tick bin
        L_frames.append(L_per_tick)
        ymax = max(ymax, float(np.max(L_per_tick)))

    # ----- convert ticks -> price axis & widths -----
    # P_lower(i) = (base_s * g**i)^2
    g2 = g * g
    P_lower = (base_s * (g ** tick_axis)) ** 2
    dP = P_lower * (g2 - 1.0)                            # width of each 1-tick bin in price
    x_left = float(P_lower[0])
    x_right = float(P_lower[-1] + dP[-1])

    # where to draw the vertical line for the active band
    def active_line_price(tick_i: int) -> float:
        P_lo = float((base_s * (g ** tick_i)) ** 2)
        return P_lo * (g if center_line else 1.0)        # center = geometric mean => Ã—g

    # ----- plot/animate -----
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(P_lower, L_frames[0], width=dP, align="edge", color="#4C78A8")
    vline_x = active_line_price(tick_history[0])
    tick_line = ax.axvline(vline_x, color="crimson", lw=2, alpha=0.9,
                           label=("Active band (center)" if center_line else "Active band (lower edge)"))

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(0.0, ymax * (1.0 + pad_frac))
    ax.set_xlabel("Price (token1 per token0)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Active liquidity per 1-tick bin", fontsize=LABEL_FONT_SIZE)
    ax.set_title("Liquidity vs Price â€” evolution", fontsize=TITLE_FONT_SIZE)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)

    txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.8))

    def update(frame_idx: int):
        L = L_frames[frame_idx]
        for rect, h in zip(bars, L):
            rect.set_height(float(h))
        tick_line.set_xdata([active_line_price(tick_history[frame_idx]),
                             active_line_price(tick_history[frame_idx])])
        txt.set_text(f"step = {frame_idx * downsample_every}")
        return (*bars, tick_line, txt)

    anim = animation.FuncAnimation(fig, update, frames=len(L_frames), blit=False)
    writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=dpi)
    # plt.close(fig)
    print(f"[GIF] wrote {out_path}")



# =============================================================================
# Scenario presets
# =============================================================================
# Common parameters
T = 1000
skip_step = 300
seed = 7
use_fee_band = True 
cex_mu = 0.00
p_trade = 0.7
noise_floor = 0.5
p_lp_narrow = 0.95
p_lp_wide = 0.7
N_LP = 500
tau = 20
w_min_ticks = 10
w_max_ticks = 1_774_540
basis_half_life = 20
slope_s = 0.15
binom_n = 10
binom_p = 0.5
trader_mean = 1
trader_sigma = 0.6
theta_T = 0.95
mint_mu = 0.05
mint_sigma = 0.01
theta_TP = 0.10
theta_SL = 0.25
evolve_initial_hill = True
initial_binom_N = 450
initial_total_L = 500_000
k_out = 5
visualize = True
f0 = 0.003
f_min = 0.0005
f_max = 0.05
fee_half_life = 10
k_sigma = 50
k_basis = 0.0001
fee_step_bps_min = 0.001
fee_step_bps_max = 20.0
fee_cooldown = 0


# ---- Helper to merge common parameters with per-scenario overrides ----
def WITH(**overrides):
    base = dict(
        T=T, seed=seed, use_fee_band=use_fee_band,
        cex_mu=cex_mu,
        p_trade=p_trade, noise_floor=noise_floor, p_lp_narrow=p_lp_narrow, p_lp_wide=p_lp_wide,
        N_LP=N_LP, tau=tau,
        w_min_ticks=w_min_ticks, w_max_ticks=w_max_ticks, basis_half_life=basis_half_life, slope_s=slope_s,
        binom_n=binom_n, binom_p=binom_p,
        trader_mean=trader_mean, trader_sigma=trader_sigma, theta_T=theta_T,
        mint_mu=mint_mu, mint_sigma=mint_sigma,
        theta_TP=theta_TP, theta_SL=theta_SL, evolve_initial_hill=evolve_initial_hill,
        initial_binom_N=initial_binom_N, initial_total_L=initial_total_L,
        k_out=k_out, visualize=visualize,
        f0=f0, f_min=f_min, f_max=f_max, fee_half_life=fee_half_life,
        k_sigma=k_sigma, k_basis=k_basis,
        fee_step_bps_min=fee_step_bps_min, fee_step_bps_max=fee_step_bps_max, fee_cooldown=fee_cooldown,
    )
    base.update(overrides)
    # Ensure explicit defaults for items commonly varied per scenario
    if 'cex_sigma' not in base:
        base['cex_sigma'] = 0.012
    if 'fee_mode' not in base:
        base['fee_mode'] = 'static'
    return base

SCENARIOS = {
    # Baseline static: band-targeting arb, smart-routed trader (toxic), static fee.
    "static_baseline": WITH(
        cex_sigma=0.015, fee_mode="static"
    ),

    "volatility_baseline": WITH(
        cex_sigma=0.015, fee_mode="volatility"
    ),

    "toxicity_baseline": WITH(
        cex_sigma=0.015, fee_mode="toxicity"
    ),
}


# =============================================================================
# Entrypoint
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ABM with configurable fee modes.")
    parser.add_argument(
        "--fee-mode",
        choices=list(SCENARIOS.keys()),
        default="static_baseline",
        help="Scenario/fee mode preset to run."
    )
    parser.add_argument(
        "--fee-param",
        type=float,
        help="Override fee controller parameter (k_sigma for volatility, k_basis for toxicity)."
    )
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Force visualization panels on after the simulation."
    )
    viz_group.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Skip generating visualization panels after the simulation completes."
    )
    parser.set_defaults(visualize=None)
    args = parser.parse_args()

    scenario = args.fee_mode
    params = SCENARIOS[scenario].copy()

    if args.fee_param is not None:
        if scenario == "volatility_baseline":
            params["k_sigma"] = args.fee_param
        elif scenario == "toxicity_baseline":
            params["k_basis"] = args.fee_param
        else:
            parser.error("--fee-param is not applicable when fee-mode is static_baseline.")

    if args.visualize is not None:
        params["visualize"] = args.visualize

    print(f"[scenario] {scenario}")
    if args.fee_param is not None:
        param_name = "k_sigma" if scenario == "volatility_baseline" else "k_basis"
        print(f"[override] {param_name} = {args.fee_param}")
    if args.visualize is not None:
        print(f"[override] visualize = {args.visualize}")

    out = simulate(**params)

    # Compute DEX log returns
    dex_prices = np.array(out["DEX_price"][skip_step:])
    dex_returns = np.diff(np.log(dex_prices))

    # Compute autocorrelation up to lag 15
    max_lag = 15
    autocorr = [np.corrcoef(dex_returns[:-lag], dex_returns[lag:])[0, 1] for lag in range(1, max_lag + 1)]

    # Visualize autocorrelation
    plt.figure(figsize=(7, 4))
    plt.bar(range(max_lag), autocorr, color="#20B2AA", edgecolor="k", alpha=0.7)
    plt.xlabel("Lag", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Autocorrelation", fontsize=LABEL_FONT_SIZE)
    plt.title("Autocorrelation of DEX log returns (up to lag 15)", fontsize=TITLE_FONT_SIZE)
    plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(out['mint_widths'], bins=30, color="#4C78A8", edgecolor="k", alpha=0.7)
    plt.xlabel("Mint width (ticks)", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Frequency", fontsize=LABEL_FONT_SIZE)
    plt.title(f"Mint widths — scenario: {scenario}", fontsize=TITLE_FONT_SIZE)
    plt.tight_layout()
    # make_liquidity_gif(
    #     liq_history=out["liq_history"],
    #     tick_history=out["tick_history"],
    #     base_s=out["grid_base_s"],
    #     g=out["grid_g"],
    #     out_path="abm_results/liquidity_evolution.gif",
    #     fps=10,
    #     dpi=100,
    #     downsample_every=1,
    # )
    plt.show()
