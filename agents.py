"""
Agent definitions and behavior functions for LP agents.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    EPS_LIQ2, EPS_LIQ, EPS_BOUNDARY, TITLE_FONT_SIZE, LABEL_FONT_SIZE,
    minted_amounts_at_S, ReferenceMarket
)
from uniswapv3_pool import V3Pool, BoundaryIndex


# =============================================================================
# LP positions & agents
# =============================================================================


@dataclass
class RebalancerState:
    """
    Tracks the per-LP rebalancing benchmark used for loss-versus-rebalancing (LVR).
    All quantities are valued in token1 (“y”) unless otherwise indicated.
    """
    x_prev: float = 0.0          # token0 units held after last rebalance
    cash_y: float = 0.0          # token1 cash balance of the benchmark
    cumulative_R: float = 0.0    # cumulative rebalancing PnL (token1)
    last_M: float = 0.0          # last CEX price used for accrual (token1 per token0)
    last_wealth_y: float = 0.0   # LP wealth snapshot (wallet + mark-to-market) at last update
    last_cumulative_R: float = 0.0  # snapshot of cumulative_R at last wealth observation
    hedged_pnl_cum: float = 0.0  # cumulative hedged PnL = wealth - rebal benchmark
    initialized: bool = False

    def reset(self) -> None:
        self.x_prev = 0.0
        self.cash_y = 0.0
        self.cumulative_R = 0.0
        self.last_M = 0.0
        self.last_wealth_y = 0.0
        self.last_cumulative_R = 0.0
        self.hedged_pnl_cum = 0.0
        self.initialized = False


@dataclass
class Position:
    """
    A concentrated liquidity position [lower, upper) with liquidity L.

    Amounts at price S (classic v3 algebra):
      Let s_a = lower boundary sqrt-price, s_b = upper boundary sqrt-price.
      If S <= s_a:  (token0, token1) = (L(1/s_a - 1/s_b), 0)
      If S >= s_b:  (0, L(s_b - s_a))
      Else:         (L(1/S - 1/s_b), L(S - s_a))

    Valuation and PnL (token1 numéraire):
      • `hodl0_value_y = amt0_init * m + amt1_init` at mint time.
      • Mark-to-market IL_y(S, m) = position_value_y_now(S, m) - hodl0_value_y  (≤ 0 typically).
      • Fees accrue separately as (fees0 * m + fees1).  PnL_y = IL_y + fees_value_y.
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
    is_passive: bool = False
    # --- async scheduler state ---
    review_rate: float = 0.2      # p for geometric clock; we'll set = 1/tau at runtime
    next_review: int = 0          # steps until this LP may act again
    cooldown: int = 0             # after a burn, sit out this many steps
    can_act: bool = False         # per-segment mask (set by the micro-scheduler)
    L_budget: float = 0.0
    L_live: float = 0.0
    wallet_y: float = 0.0
    rebalancer: RebalancerState = field(default_factory=RebalancerState)


def lp_token0_exposure(lp: LPAgent, S: float) -> float:
    """
    Aggregate token0 exposure for an LP at sqrt-price S, including uncollected token0 fees.
    """
    total = 0.0
    for pos in lp.positions:
        amt0, _ = pos.current_amounts(S)
        total += amt0 + pos.fees0
    return total


def lp_mark_to_market_y(lp: LPAgent, S: float, m: float) -> float:
    """
    Mark-to-market value (token1) of all open positions including uncollected fees.
    """
    total = 0.0
    for pos in lp.positions:
        total += pos.position_value_y_now(S, m) + pos.fees_value_y(m)
    return total


def lp_wealth_y(lp: LPAgent, S: float, m: float) -> float:
    """
    Total LP wealth (token1) = wallet holdings + mark-to-market open value.
    """
    wallet = getattr(lp, "wallet_y", 0.0)
    return wallet + lp_mark_to_market_y(lp, S, m)


# =============================================================================
# Bootstrap functions for initial liquidity
# =============================================================================

# def bootstrap_initial_binomial_hill(
#     pool: V3Pool,
#     ref: ReferenceMarket,
#     LPs: List[LPAgent],
#     N: int = 400,
#     L_total: float = 70_000.0,
#     seed_lp_id: int = 10_000,
#     seed_mint_prob: float = 0.0,
#     min_L_per_tick: float = 1e-9,
#     plot: bool = False,
#     ax: Optional[plt.Axes] = None,
#     title: Optional[str] = None,
# ) -> LPAgent:
#     """Bootstrap initial liquidity as a binomial hill with a single seed LP."""
#     seed = LPAgent(id=seed_lp_id, mintProb=seed_mint_prob)
#     center_tick = pool._snap(pool.tick)
#     S_entry = pool.S

#     ticks: List[int] = []
#     L_vals: List[float] = []

#     denom = float(2 ** N)
#     for k in range(N + 1):
#         w = math.comb(N, k) / denom
#         L_i = w * L_total
#         if L_i < min_L_per_tick:
#             continue

#         rel = k - (N // 2)
#         lower = center_tick + rel * pool.tick_spacing
#         upper = lower + pool.tick_spacing
#         sa = pool.s_lower(lower)
#         sb = pool.s_upper(upper)

#         amt0, amt1 = minted_amounts_at_S(L_i, sa, sb, S_entry)

#         pos = Position(
#             owner=seed.id, lower=lower, upper=upper, L=L_i, sa=sa, sb=sb,
#             amt0_init=amt0, amt1_init=amt1, hodl0_value_y=amt0 * ref.m + amt1
#         )
#         pool.add_liquidity_range(lower, upper, L_i)
#         seed.positions.append(pos)

#         ticks.append(lower)
#         L_vals.append(L_i)

#     pool.recompute_active_L()
#     LPs.append(seed)

#     if plot and len(ticks) > 0:
#         created_fig = False
#         if ax is None:
#             fig, ax = plt.subplots(figsize=(10, 3.5))
#             created_fig = True
#         ax.bar(ticks, L_vals, width=pool.tick_spacing, align="edge")
#         ax.set_xlabel("Tick", fontsize=LABEL_FONT_SIZE)
#         ax.set_ylabel("Liquidity per band (L)", fontsize=LABEL_FONT_SIZE)
#         ax.set_title(title or f"Initial binomial hill (N={N}, total L={L_total:,.0f})", fontsize=TITLE_FONT_SIZE)
#         ax.grid(True, axis="y", alpha=0.25)
#         if created_fig:
#             plt.tight_layout()

#     return seed




def bootstrap_initial_liquidity_as_lp(
    pool: V3Pool,
    ref: ReferenceMarket,
    LPs: List[LPAgent],
    hill: List[Tuple[int, float]],
    seed_lp_id: int = 999,
    seed_mint_prob: float = 0.0,
) -> LPAgent:
    """Bootstrap initial liquidity from a pre-defined hill specification."""
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

