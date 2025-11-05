from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from collections import deque
from bisect import bisect_right, bisect_left
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ========= Utilities & tolerances =========

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

EPS_LIQ = 1e-18
EPS_LIQ2 = 1e-7       # robust clamp for small numerical drift in L_active
EPS_PRICE_CHANGE = 1e-10
EPS_BOUNDARY = 1e-12


# ========= Pool (Uniswap v3–style, spacing-aware) =========

@dataclass
class V3Pool:
    g: float
    base_s: float
    tick: int
    S: float
    f: float
    liquidity_net: Dict[int, float] = field(default_factory=dict)
    L_active: float = 0.0
    tick_spacing: int = 10

    @property
    def r(self) -> float:
        return 1.0 - self.f

    @property
    def price(self) -> float:
        return self.S * self.S

    def _snap(self, i: int) -> int:
        s = self.tick_spacing
        return (i // s) * s

    def s_lower(self, i: Optional[int] = None) -> float:
        if i is None:
            i = self.tick
        return self.base_s * (self.g ** i)

    def s_upper(self, i: Optional[int] = None) -> float:
        return self.s_lower(i) * (self.g ** self.tick_spacing)

    def __post_init__(self) -> None:
        self.tick = self._snap(self.tick)
        self.recompute_active_L()

    def recompute_active_L(self) -> None:
        # robust fsum + clamp tiny negatives to 0
        L = math.fsum(dL for t, dL in self.liquidity_net.items() if t <= self.tick)
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
        if -1e-9 < self.L_active < 0.0:
            self.L_active = 0.0

    def _cross_up_once(self):
        self.tick += self.tick_spacing
        self.recompute_active_L()

    def _cross_down_once(self):
        self.tick -= self.tick_spacing
        self.recompute_active_L()

    # exact swaps within spacing bands
    def swap_x_to_y(self, dx_in: float) -> Tuple[float, float, float]:
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
                S_new = 1 / (1 / self.S + dx_eff / self.L_active)
                dy = self.L_active * (S_new - self.S)
                self.S = S_new
                dx_used += dx_eff
                dy_out += -dy
                dx_eff = 0.0
            else:
                dy = self.L_active * (S_lo - self.S)
                self.S = S_lo
                dx_eff -= dx_to
                dx_used += dx_to
                dy_out += -dy
                self._cross_down_once()
        dx_pre = dx_used / self.r if self.r > 0 else dx_used
        fee_x = dx_pre - dx_used
        return dx_pre, dy_out, fee_x

    def swap_y_to_x(self, dy_in: float) -> Tuple[float, float, float]:
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
                S_new = self.S + dy_eff / self.L_active
                dx = self.L_active * (1 / self.S - 1 / S_new)
                self.S = S_new
                dy_used += dy_eff
                dx_out += dx
                dy_eff = 0.0
            else:
                dx = self.L_active * (1 / self.S - 1 / S_hi)
                self.S = S_hi
                dy_eff -= dy_to
                dy_used += dy_to
                dx_out += dx
                self._cross_up_once()
        dy_pre = dy_used / self.r if self.r > 0 else dy_used
        fee_y = dy_pre - dy_used
        return dy_pre, dx_out, fee_y


# ========= Boundary index =========

class BoundaryIndex:
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


# ========= LP positions & agents =========

@dataclass
class Position:
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
        return self.position_value_y_now(S, m) - self.hodl0_value_y
    def PnL_y(self, S: float, m: float) -> float:
        return self.IL_y(S, m) + self.fees_value_y(m)

@dataclass
class LPAgent:
    id: int
    positions: List[Position] = field(default_factory=list)
    mintProb: float = 0.5
    is_active_narrow: bool = False
    # async/scheduling & risk controls
    review_rate: float = 0.2
    next_review: int = 0
    cooldown: int = 0
    can_act: bool = False
    L_budget: float = 0.0
    L_live: float = 0.0


# ========= Reference market =========

@dataclass
class ReferenceMarket:
    m: float
    mu: float
    sigma: float
    kappa: float
    xi: float = 0.0
    def diffuse_once(self) -> None:
        z = np.random.normal()
        self.m *= math.exp(self.mu - 0.5 * self.sigma**2 + self.sigma * z)
        self.m = max(1e-12, self.m)
    def apply_impact(self, delta_a_cex_signed: float) -> float:
        impact = self.kappa * math.copysign(abs(delta_a_cex_signed)**(1.0 + self.xi),
                                            delta_a_cex_signed)
        self.m = max(1e-12, self.m + impact)
        return impact


# ========= Builders =========

def build_empty_pool() -> Tuple[V3Pool, float]:
    f = 0.003
    g = np.sqrt(1.0001)
    m0 = 2000.0
    S0 = math.sqrt(m0)
    base_s = S0 / math.sqrt(g)
    pool = V3Pool(g=g, base_s=base_s, tick=0, S=S0, f=f, liquidity_net={}, tick_spacing=10)
    return pool, m0

def minted_amounts_at_S(L: float, sa: float, sb: float, S: float) -> Tuple[float, float]:
    if S <= sa:
        return L * (1 / sa - 1 / sb), 0.0
    elif S >= sb:
        return 0.0, L * (sb - sa)
    else:
        return L * (1 / S - 1 / sb), L * (S - sa)

def bootstrap_initial_binomial_hill_sharded(
    pool: V3Pool,
    ref: ReferenceMarket,
    LPs: List[LPAgent],
    N: int = 400,
    L_total: float = 70_000.0,
    num_seed_lps: int = 20,
    seed_lp_id_base: int = 10_000,
    seed_mint_prob: float = 0.0,
    tau: int = 20,
) -> List[LPAgent]:
    center_tick = pool._snap(pool.tick)
    S_entry = pool.S

    # seed LPs with staggered clocks
    seed_LPs: List[LPAgent] = []
    for j in range(num_seed_lps):
        sid = seed_lp_id_base + j
        lp = LPAgent(id=sid, mintProb=seed_mint_prob, is_active_narrow=False)
        lp.review_rate = 1.0 / max(1, tau)
        lp.next_review = int(np.random.geometric(lp.review_rate))
        lp.cooldown = 0
        lp.can_act = False
        seed_LPs.append(lp)

    # precompute tick weights
    tick_specs: List[Tuple[int, float]] = []
    denom = float(2 ** N)
    for k in range(N + 1):
        w = math.comb(N, k) / denom
        L_i = w * L_total
        if L_i < 1e-9:
            continue
        rel = k - (N // 2)
        lower = center_tick + rel * pool.tick_spacing
        tick_specs.append((lower, L_i))

    # round-robin assign to seed LPs
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
    for lp in seed_LPs:
        LPs.append(lp)
    return seed_LPs


# ========= Mempool types =========

@dataclass(order=True)
class Order:
    sort_index: Tuple[float,int] = field(init=False, repr=False)
    gas_fee: float
    arrival_seq: int
    kind: str
    agent_id: int
    side: Optional[str] = None
    amount_in: float = 0.0
    min_out: float = 0.0
    slippage_bps: float = 50.0
    L: float = 0.0
    lower_tick: int = 0
    upper_tick: int = 0
    pos_idx: Optional[int] = None
    width_ticks: Optional[int] = None
    def __post_init__(self):
        self.sort_index = (-self.gas_fee, self.arrival_seq)

class Mempool:
    def __init__(self):
        self.orders: List[Order] = []
        self.seq = 0
    def append(self, order: Order) -> None:
        order.arrival_seq = self.seq
        order.__post_init__()
        self.seq += 1
        self.orders.append(order)
    def sort_gas_desc_fifo(self) -> None:
        self.orders.sort()
    def drain(self) -> List[Order]:
        out = self.orders
        self.orders = []
        return out


# ========= Simulation with mempool =========

def simulate_with_mempool(
    T_blocks: int = 120,
    B: int = 12,
    seed: int = 7,
    verbose_blocks: int = 5,
    use_fee_band: bool = True,
    p_trade: float = 0.6,
    p_lp_narrow: float = 0.7,
    p_lp_wide: float = 0.15,
    N_LP: int = 12,
    tau: int = 20,
    w_min_ticks: int = 25,
    psi_ticks: int = 300,
    binom_n: int = 8,
    binom_p: float = 0.5,
    mint_mu: float = 0.01,
    mint_sigma: float = 0.02,
    theta_TP: float = 0.003,
    theta_SL: float = 0.01,
    evolve_initial_hill: bool = True,
    initial_binom_N: int = 400,
    initial_total_L: float = 70_000.0,
    k_out: int = 2,
    visualize: bool = True,
):
    np.random.seed(seed); random.seed(seed)

    pool, m0 = build_empty_pool()
    ref = ReferenceMarket(m=m0, mu=0.0, sigma=0.02, kappa=1e-3)

    # LP population (non-seeds)
    LPs: List[LPAgent] = []
    for i in range(N_LP):
        is_narrow = random.random() < p_lp_narrow
        mintProb = p_lp_narrow if is_narrow else p_lp_wide
        lp = LPAgent(id=i, mintProb=mintProb, is_active_narrow=is_narrow)
        lp.review_rate = 1.0 / max(1, tau)
        lp.next_review = int(np.random.geometric(lp.review_rate))
        lp.cooldown = 0
        lp.can_act = False
        LPs.append(lp)

    # Seed initial hill as **owned positions** (sharded), or static if disabled
    if evolve_initial_hill:
        bootstrap_initial_binomial_hill_sharded(
            pool, ref, LPs,
            N=initial_binom_N, L_total=initial_total_L,
            num_seed_lps=20, seed_lp_id_base=10_000,
            seed_mint_prob=0.0, tau=tau
        )
    else:
        # (fallback) static hill (no ownership) – not recommended
        center_tick = pool._snap(pool.tick)
        denom = float(2 ** initial_binom_N)
        for k in range(initial_binom_N + 1):
            w = math.comb(initial_binom_N, k) / denom
            L_i = w * initial_total_L
            if L_i < 1e-9: continue
            rel = k - (initial_binom_N // 2)
            lower = center_tick + rel * pool.tick_spacing
            pool.add_liquidity_range(lower, lower + pool.tick_spacing, L_i)
        pool.recompute_active_L()

    # give every LP budgets
    L_SCALE = initial_total_L / max(1, N_LP)
    for lp in LPs:
        if lp.L_budget <= 0.0:
            lp.L_budget = 2.0 * L_SCALE
        if lp.L_live < 0.0:
            lp.L_live = 0.0

    bidx = BoundaryIndex(pool.liquidity_net)

    # Heuristic windows (token1 notionals, absolute)
    arb_abs_win = deque(maxlen=tau)
    tot_abs_win = deque(maxlen=tau)

    # -------- block-level recorders ----------
    P_series_block, M_series_block = [], []
    band_lo_post_block, band_hi_post_block = [], []
    L_end_block = []
    trader_y_series_block, arb_y_series_block = [], []
    liq_history: List[Dict[int, float]] = []
    tick_history: List[int] = []
    L_first_arb_block: List[Optional[float]] = []

    # -------- micro-time recorders ----------
    micro_P, micro_M = [], []
    micro_band_lo, micro_band_hi = [], []
    micro_band_lo_pre, micro_band_hi_pre = [], []
    micro_L = []
    micro_L_pre_block = []
    micro_L_pre_trader = []
    micro_L_pre_arb_eff = []
    micro_trader_y, micro_arb_y = [], []
    micro_deltaL_mint, micro_deltaL_burn = [], []
    micro_X_active, micro_Y_active = [], []

    # event markers at micro-time
    trader_steps, trader_dirs = [], []
    arb_steps, arb_dirs = [], []
    lp_mint_micro, lp_mint_sizes = [], []
    lp_burn_micro, lp_burn_sizes = [], []

    # ----- Helpers -----

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

    def ensure_liquidity(direction: str) -> bool:
        # execution-time bridge (mutates)
        if pool.L_active > EPS_LIQ:
            return True
        curr_tick = tick_from_S(pool.S)
        if direction == "up":
            nb = bidx.next_up(curr_tick)
            if nb is None:
                return False
            pool.tick = pool._snap(nb)
            pool.S = pool.s_lower(pool.tick)
            pool.recompute_active_L()
            return pool.L_active > EPS_LIQ
        else:
            pb = bidx.prev_down(curr_tick)
            if pb is None:
                return False
            pool.tick = pool._snap(pb - pool.tick_spacing)
            pool.S = pool.s_upper(pool.tick)
            pool.recompute_active_L()
            return pool.L_active > EPS_LIQ

    # ---- PURE (non-mutating) quotes with desert bridging ----

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

        # local snapshot
        tick_loc = tick_from_S(pool.S)
        S_loc = pool.S
        L_loc = _active_L_at_tick_local(tick_loc)

        # down-bridge desert for X->Y
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

        # up-bridge desert for Y->X
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

    # ---- fast band spans for arbitrage ----

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

    def swap_exact_to_target(target_price: float, direction: str) -> Tuple[float, float, float, float, float]:
        target_S = math.sqrt(max(1e-18, target_price))
        if pool.L_active <= EPS_LIQ:
            need = "up" if target_S > pool.S else "down"
            if need != direction:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            if not ensure_liquidity(direction):
                return 0.0, 0.0, 0.0, 0.0, 0.0
        total_in = total_out = fee_x = fee_y = 0.0
        L_first = 0.0
        if direction == "up":
            while pool.L_active > 0 and pool.S < target_S - EPS_BOUNDARY:
                S_hi = pool.s_upper()
                L_before = pool.L_active
                dy, dx, f = fast_span_up(S_hi, target_S)
                if dy > 0 and L_first == 0.0:
                    L_first = L_before
                fee_y += f; total_in += dy; total_out += dx
                if pool.S >= target_S - EPS_BOUNDARY:
                    break
                pool._cross_up_once()
                if pool.L_active <= 0:
                    break
            return total_in, total_out, fee_x, fee_y, L_first
        else:
            while pool.L_active > 0 and pool.S > target_S + EPS_BOUNDARY:
                S_lo = pool.s_lower()
                L_before = pool.L_active
                dx, dy, f = fast_span_down(S_lo, target_S)
                if dx > 0 and L_first == 0.0:
                    L_first = L_before
                fee_x += f; total_in += dx; total_out += dy
                if pool.S <= target_S + EPS_BOUNDARY:
                    break
                pool._cross_down_once()
                if pool.L_active <= 0:
                    break
            return total_in, total_out, fee_x, fee_y, L_first

    def arbitrage_to_target() -> Tuple[float, float, Optional[str], float, float, float]:
        P = pool.price
        r = pool.r
        if use_fee_band:
            lo, hi = ref.m * r, ref.m / r
            if P < lo * (1 - 1e-9):
                dy_in, dx_out, fx, fy, Lff = swap_exact_to_target(lo, "up")
                return dy_in, dx_out, ("up" if dy_in > 0 else None), fx, fy, Lff
            if P > hi * (1 + 1e-9):
                dx_in, dy_out, fx, fy, Lff = swap_exact_to_target(hi, "down")
                return dx_in, 0.0, ("down" if dx_in > 0 else None), fx, fy, Lff
            return 0.0, 0.0, None, 0.0, 0.0, 0.0
        else:
            target = ref.m
            if abs(P - target) / max(target, 1e-12) < 1e-9:
                return 0.0, 0.0, None, 0.0, 0.0, 0.0
            direction = "up" if P < target else "down"
            ain, aout, fx, fy, Lff = swap_exact_to_target(target, direction)
            return (ain, aout if direction == "up" else 0.0, direction, fx, fy, Lff)

    def reserves_in_active_tick() -> Tuple[float, float]:
        if pool.L_active <= EPS_LIQ:
            return 0.0, 0.0
        sa, sb = pool.s_lower(), pool.s_upper()
        S_eff = min(max(pool.S, sa), sb)
        x = pool.L_active * max(0.0, 1.0 / S_eff - 1.0 / sb)
        y = pool.L_active * max(0.0, S_eff - sa)
        return x, y

    # ---- main loop over blocks ----

    for t in range(T_blocks):
        r = pool.r
        block_mempool = Mempool()

        # step-level windows for width heuristic
        V_arb_tau = sum(arb_abs_win) if arb_abs_win else 0.0
        V_tot_tau = sum(tot_abs_win) if tot_abs_win else 1e-12
        informed_ratio = V_arb_tau / max(V_tot_tau, 1e-12)
        eps_noise = np.random.binomial(binom_n, binom_p) - binom_n * binom_p
        w_ticks = int(round(w_min_ticks + psi_ticks * informed_ratio + abs(eps_noise)))

        # decrement cooldowns + advance review clocks (per block)
        due_lps: List[int] = []
        for j, lp in enumerate(LPs):
            if lp.cooldown > 0:
                lp.cooldown -= 1
                lp.next_review = max(1, lp.next_review - 1)
                continue
            lp.next_review -= 1
            if lp.next_review <= 0:
                due_lps.append(j)
                lp.next_review = int(np.random.geometric(lp.review_rate))

        # distribute due LPs across micro-steps in this block (asynchronous)
        schedule_map: Dict[int, List[int]] = {b: [] for b in range(B)}
        random.shuffle(due_lps)
        for idx, j in enumerate(due_lps):
            schedule_map[idx % B].append(j)

        def gas_sample() -> float:
            return max(1.0, np.random.lognormal(mean=3.2, sigma=0.4))

        # ---- submission routines (run per micro-step) ----

        def submit_LPs(b_micro: int):
            # Only LPs scheduled for this micro-step may submit
            act_set = set(schedule_map.get(b_micro, []))
            # burns (TP/SL)
            for j in act_set:
                lp = LPs[j]
                to_burn = []
                for i_pos, pos in enumerate(lp.positions):
                    pnl = pos.PnL_y(pool.S, ref.m)
                    if pnl >= theta_TP * pos.hodl0_value_y or pnl <= -theta_SL * pos.hodl0_value_y:
                        to_burn.append(i_pos)
                for i_pos in reversed(to_burn):
                    block_mempool.append(Order(gas_fee=gas_sample(), arrival_seq=0, kind="burn",
                                               agent_id=lp.id, pos_idx=i_pos))
            # re-centers (only check out-of-range once per block to avoid overcount)
            if b_micro == 0:
                for j in act_set:
                    lp = LPs[j]
                    to_recenters: List[int] = []
                    for i_pos, pos in enumerate(lp.positions):
                        in_rng = pos.in_range(pool.tick)
                        out_steps = getattr(pos, "out_steps", 0)
                        out_steps = 0 if in_rng else out_steps + 1
                        setattr(pos, "out_steps", out_steps)
                        if lp.is_active_narrow and out_steps >= k_out:
                            to_recenters.append(i_pos)
                    for i_pos in reversed(to_recenters):
                        pos = lp.positions[i_pos]
                        width = pos.upper - pos.lower
                        block_mempool.append(Order(gas_fee=gas_sample(), arrival_seq=0, kind="recenter",
                                                   agent_id=lp.id, pos_idx=i_pos, width_ticks=width))
            # mints (subject to cooldown + budget)
            for j in act_set:
                lp = LPs[j]
                if lp.cooldown > 0:
                    continue
                if random.random() < lp.mintProb:
                    half_w = max(
                        pool.tick_spacing,
                        pool.tick_spacing * int(max(1, round(w_ticks / pool.tick_spacing)))
                    )
                    X = abs(np.random.normal(mint_mu, mint_sigma))
                    want = X * L_SCALE
                    cap_step = 0.25 * lp.L_budget
                    cap_left = max(0.0, lp.L_budget - lp.L_live)
                    L_new = max(0.0, min(want, cap_step, cap_left))
                    if L_new <= 0:
                        continue
                    center = pool._snap(pool.tick)
                    lower, upper = center - half_w, center + half_w
                    if upper <= lower:
                        upper = lower + pool.tick_spacing
                    block_mempool.append(Order(gas_fee=gas_sample(), arrival_seq=0, kind="mint",
                                               agent_id=lp.id, L=L_new, lower_tick=lower, upper_tick=upper))

        def submit_trader():
            if random.random() >= p_trade:
                return
            side = random.choice(["X_to_Y", "Y_to_X"])
            P_pre = pool.price
            m_now = ref.m

            if side == "X_to_Y":
                # size
                clip_x = 0.0015 * max(pool.L_active, 1e-12) / max(1e-12, pool.S)
                dx = abs(np.random.lognormal(mean=-2.0, sigma=1.0))  # * clip_x (optional)
                if dx <= 0:
                    return
                # profitability gate vs CEX: dy_dex > dx * m
                dy_quote = quote_x_to_y(dx)
                if dy_quote <= dx * m_now:
                    return
                min_out = (1.0 - 0.005) * dy_quote  # 50 bps slippage tol
                block_mempool.append(Order(gas_fee=gas_sample(), arrival_seq=0, kind="swap_x_to_y",
                                           agent_id=-1, side="X_to_Y", amount_in=dx, min_out=min_out))
            else:
                clip_y = 0.0015 * max(pool.L_active, 1e-12) * pool.S
                dy = abs(np.random.lognormal(mean=-2.0, sigma=1.0))  # * clip_y (optional)
                if dy <= 0:
                    return
                # profitability gate vs CEX: value_dex > value_cex
                dx_quote = quote_y_to_x(dy)
                if dx_quote * P_pre <= dy:
                    return
                min_out = (1.0 - 0.005) * dx_quote
                block_mempool.append(Order(gas_fee=gas_sample(), arrival_seq=0, kind="swap_y_to_x",
                                           agent_id=-1, side="Y_to_X", amount_in=dy, min_out=min_out))

        def submit_arbitrageur():
            P = pool.price
            lo, hi = ref.m * r, ref.m / r
            if use_fee_band and (P < lo * (1 - 1e-9) or P > hi * (1 + 1e-9)):
                block_mempool.append(Order(gas_fee=1e6, arrival_seq=0, kind="arb_to_band", agent_id=-2))

        # ----- Submission window (CEX micro-time): record with DEX flat -----
        ref_m_start = ref.m
        P_block_const = pool.price
        L_block_const = pool.L_active
        band_lo_pre_val = ref_m_start * r
        band_hi_pre_val = ref_m_start / r

        for b in range(B):
            actors = [
                lambda b=b: submit_LPs(b),  # pass b by value
                submit_trader,
                submit_arbitrageur
            ]
            random.shuffle(actors)
            for act in actors:
                act()
            ref.diffuse_once()
            micro_M.append(ref.m)
            micro_P.append(P_block_const)
            micro_band_lo.append(ref.m * r)
            micro_band_hi.append(ref.m / r)
            micro_band_lo_pre.append(band_lo_pre_val)
            micro_band_hi_pre.append(band_hi_pre_val)
            micro_L.append(L_block_const)
            micro_L_pre_block.append(L_block_const)
            micro_L_pre_trader.append(L_block_const)
            micro_L_pre_arb_eff.append(np.nan)
            micro_trader_y.append(0.0)
            micro_arb_y.append(0.0)
            micro_deltaL_mint.append(0.0)
            micro_deltaL_burn.append(0.0)
            x_now, y_now = reserves_in_active_tick()
            micro_X_active.append(x_now)
            micro_Y_active.append(y_now)

        # ----- Block sealing (validation) -----
        block_mempool.sort_gas_desc_fifo()
        orders = block_mempool.drain()

        trader_y_block = 0.0
        arb_y_block = 0.0
        delta_a_cex_block = 0.0
        minted_this_block = 0.0
        burned_this_block = 0.0
        L_first_this_block: Optional[float] = None

        v_idx = t * (B + 1) + B  # micro-time index of validation tick

        for o in orders:
            if o.kind == "swap_y_to_x":
                # pre-check with pure quote against min_out
                dx_quote = quote_y_to_x(o.amount_in)
                if dx_quote + 1e-12 < o.min_out:
                    continue
                if pool.L_active <= 0 and not ensure_liquidity("up"):
                    continue
                tick_snap = pool.tick; L_snap = pool.L_active
                dy_pre, dx_out, fee_y = pool.swap_y_to_x(o.amount_in)
                if dy_pre <= 0:
                    continue
                allocate_fees("y", fee_y, tick_snap, L_snap)
                trader_y_block += dy_pre
                trader_steps.append(t); trader_dirs.append("up")

            elif o.kind == "swap_x_to_y":
                dy_quote = quote_x_to_y(o.amount_in)
                if dy_quote + 1e-12 < o.min_out:
                    continue
                if pool.L_active <= 0 and not ensure_liquidity("down"):
                    continue
                tick_snap = pool.tick; L_snap = pool.L_active
                dx_pre, dy_out, fee_x = pool.swap_x_to_y(o.amount_in)
                if dx_pre <= 0:
                    continue
                allocate_fees("x", fee_x, tick_snap, L_snap)
                trader_y_block += -pool.price * dx_pre
                trader_steps.append(t); trader_dirs.append("down")

            elif o.kind == "mint":
                sa, sb = pool.s_lower(o.lower_tick), pool.s_upper(o.lower_tick)
                amt0, amt1 = minted_amounts_at_S(o.L, sa, sb, pool.S)
                lp = next((lp for lp in LPs if lp.id == o.agent_id), None)
                if lp is None:
                    continue
                pos = Position(owner=lp.id, lower=o.lower_tick, upper=o.upper_tick, L=o.L,
                               sa=sa, sb=sb, amt0_init=amt0, amt1_init=amt1,
                               hodl0_value_y=amt0 * ref.m + amt1)
                pool.add_liquidity_range(o.lower_tick, o.upper_tick, o.L)
                pool.recompute_active_L()
                bidx.mark_dirty()
                lp.positions.append(pos)
                minted_this_block += o.L
                lp_mint_micro.append(v_idx); lp_mint_sizes.append(o.L)
                lp.L_live = getattr(lp, "L_live", 0.0) + o.L

            elif o.kind == "burn":
                lp = next((lp for lp in LPs if lp.id == o.agent_id), None)
                if lp is None:
                    continue
                if o.pos_idx is None or not (0 <= o.pos_idx < len(lp.positions)):
                    continue
                pos = lp.positions.pop(o.pos_idx)
                pool.add_liquidity_range(pos.lower, pos.upper, -pos.L)
                pool.recompute_active_L()
                bidx.mark_dirty()
                burned_this_block += pos.L
                lp_burn_micro.append(v_idx); lp_burn_sizes.append(pos.L)
                lp.cooldown = np.random.randint(3, 9)  # 3–8 blocks cooldown
                lp.L_live = max(0.0, getattr(lp, "L_live", 0.0) - pos.L)

            elif o.kind == "recenter":
                lp = next((lp for lp in LPs if lp.id == o.agent_id), None)
                if lp is None:
                    continue
                if o.pos_idx is None or not (0 <= o.pos_idx < len(lp.positions)):
                    continue
                old = lp.positions.pop(o.pos_idx)
                pool.add_liquidity_range(old.lower, old.upper, -old.L)
                pool.recompute_active_L()
                bidx.mark_dirty()
                width = o.width_ticks if o.width_ticks is not None else (old.upper - old.lower)
                center = pool._snap(pool.tick)
                lower = pool._snap(center - (width // 2))
                upper = lower + width
                sa, sb = pool.s_lower(lower), pool.s_upper(lower)
                amt0, amt1 = minted_amounts_at_S(old.L, sa, sb, pool.S)
                newpos = Position(owner=lp.id, lower=lower, upper=upper, L=old.L,
                                  sa=sa, sb=sb, amt0_init=amt0, amt1_init=amt1,
                                  hodl0_value_y=amt0 * ref.m + amt1)
                pool.add_liquidity_range(lower, upper, old.L)
                pool.recompute_active_L()
                bidx.mark_dirty()
                lp.positions.append(newpos)
                minted_this_block += old.L
                lp_mint_micro.append(v_idx); lp_mint_sizes.append(old.L)
                lp.cooldown = np.random.randint(3, 9)  # reposition triggers cooldown too

            elif o.kind == "arb_to_band":
                in_used, a_out_from_dex, dir_arb, fee_x_arb, fee_y_arb, L_first = arbitrage_to_target()
                if in_used > 0 and dir_arb is not None:
                    if L_first_this_block is None:
                        L_first_this_block = L_first
                    if dir_arb == "up":
                        delta_a_cex_block += -a_out_from_dex      # sell A on CEX
                        arb_y_block += +in_used
                        if fee_y_arb > 0:
                            allocate_fees("y", fee_y_arb, pool.tick, L_first)
                        arb_steps.append(t); arb_dirs.append("up")
                    else:
                        delta_a_cex_block += +in_used             # buy A on CEX
                        arb_y_block += -pool.price * in_used
                        if fee_x_arb > 0:
                            allocate_fees("x", fee_x_arb, pool.tick, L_first)
                        arb_steps.append(t); arb_dirs.append("down")

        # CEX permanent impact at validation
        ref.apply_impact(delta_a_cex_block)

        # ---- block-level logs ----
        P_series_block.append(pool.price)
        M_series_block.append(ref.m)
        band_lo_post_block.append(ref.m * r)
        band_hi_post_block.append(ref.m / r)
        L_end_block.append(pool.L_active)
        trader_y_series_block.append(trader_y_block)
        arb_y_series_block.append(arb_y_block)
        L_first_arb_block.append(L_first_this_block)
        liq_history.append(dict(pool.liquidity_net))
        tick_history.append(pool.tick)

        # update windows
        arb_abs_win.append(abs(arb_y_block))
        tot_abs_win.append(abs(arb_y_block) + abs(trader_y_block))

        if t < verbose_blocks:
            print(f"[block={t:03d}] DEX={pool.price:.4f} | CEX={ref.m:.4f} | "
                  f"traderY={trader_y_block:.2f} | arbY={arb_y_block:.2f} | "
                  f"L={pool.L_active:.2f} | Δa_cex={delta_a_cex_block:.4f} | "
                  f"+ΔL={minted_this_block:.1f} -ΔL={burned_this_block:.1f}")

        # ---- validation tick on micro-time (append with DEX jump + event payloads) ----
        micro_M.append(ref.m)
        micro_P.append(pool.price)
        micro_band_lo.append(ref.m * r)
        micro_band_hi.append(ref.m / r)
        micro_band_lo_pre.append(band_lo_pre_val)
        micro_band_hi_pre.append(band_hi_pre_val)
        micro_L.append(pool.L_active)
        micro_L_pre_block.append(L_block_const)
        micro_L_pre_trader.append(L_block_const)
        micro_L_pre_arb_eff.append(L_first_this_block if L_first_this_block is not None else np.nan)
        micro_trader_y.append(trader_y_block)
        micro_arb_y.append(arb_y_block)
        micro_deltaL_mint.append(minted_this_block)
        micro_deltaL_burn.append(-burned_this_block)
        x_now, y_now = reserves_in_active_tick()
        micro_X_active.append(x_now)
        micro_Y_active.append(y_now)

    # ---- Visualization (micro-time panels) ----
    if visualize:
        x = np.arange(len(micro_P))
        mint_bar = np.array(micro_deltaL_mint)
        L_end_micro = np.array(micro_L)
        L_pre_block_micro = np.array(micro_L_pre_block)
        L_pre_trader_micro = np.array(micro_L_pre_trader)
        L_pre_arb_eff_micro = np.array(micro_L_pre_arb_eff)
        val_idx = [t * (B + 1) + B for t in range(T_blocks)]

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
            5, 1, figsize=(12, 12), sharex=True,
            gridspec_kw={"height_ratios": [3, 1.2, 1, 1, 1]}
        )

        off = (50 / 1e4) * np.array(micro_P)

        # Panel 1: price + bands
        ax1.fill_between(x, micro_band_lo, micro_band_hi, color="lightgray", alpha=0.35, label="No-arb fee band (micro)")
        ax1.plot(x, micro_band_lo_pre, color="#888", linestyle=":", linewidth=1.2, label="Band low (block start)")
        ax1.plot(x, micro_band_hi_pre, color="#888", linestyle=":", linewidth=1.2, label="Band high (block start)")
        ax1.plot(x, micro_P, label="DEX price P (step-like)", linewidth=2)
        ax1.plot(x, micro_M, "--", label="CEX price m (GBM + impact at validation)", linewidth=1.6)
        for v in val_idx:
            ax1.axvline(v, color="k", alpha=0.08, lw=1)

        # Trader markers
        if trader_steps:
            t_up = [t_ for t_, d in zip(trader_steps, trader_dirs) if d == "up"]
            t_dn = [t_ for t_, d in zip(trader_steps, trader_dirs) if d == "down"]
            t_up_idx = [ti * (B + 1) + B for ti in t_up]
            t_dn_idx = [ti * (B + 1) + B for ti in t_dn]
            if t_up_idx:
                ax1.scatter(t_up_idx, (np.array(micro_P)[t_up_idx] - off[t_up_idx]),
                            marker="*", color="#008B8B", s=50, label="Trader Y→X (↑)")
            if t_dn_idx:
                ax1.scatter(t_dn_idx, (np.array(micro_P)[t_dn_idx] - off[t_dn_idx]),
                            marker="*", color="#20B2AA", s=50, label="Trader X→Y (↓)")

        # Arb markers
        if arb_steps:
            arb_abs = np.array([abs(arb_y_series_block[s]) for s in range(len(arb_y_series_block))])
            max_abs = float(max(1e-12, np.max(arb_abs)))
            def scale(a): return 30 + 120 * (a / max_abs)
            up_idx = [s * (B + 1) + B for s, d in zip(arb_steps, arb_dirs) if d == "up"]
            dn_idx = [s * (B + 1) + B for s, d in zip(arb_steps, arb_dirs) if d == "down"]
            if up_idx:
                ax1.scatter(up_idx, (np.array(micro_P)[up_idx] + off[up_idx]),
                            marker="^", color="green",
                            s=[scale(abs(arb_y_series_block[s])) for s in [ui // (B + 1) for ui in up_idx]],
                            label="Arb (↑ to band)")
            if dn_idx:
                ax1.scatter(dn_idx, (np.array(micro_P)[dn_idx] + off[dn_idx]),
                            marker="v", color="red",
                            s=[scale(abs(arb_y_series_block[s])) for s in [di // (B + 1) for di in dn_idx]],
                            label="Arb (↓ to band)")

        # LP markers
        if lp_mint_micro:
            maxL = max(1e-12, max(lp_mint_sizes + (lp_burn_sizes if lp_burn_sizes else [])))
            def scaleL(L): return 30 + 120 * (L / maxL)
            ax1.scatter(lp_mint_micro, (np.array(micro_P)[lp_mint_micro] + 2 * off[lp_mint_micro]),
                        marker="s", facecolors="none", edgecolors="#6a0dad",
                        s=[scaleL(L) for L in lp_mint_sizes], label="LP mint/center")
        if lp_burn_micro:
            maxL = max(1e-12, max((lp_mint_sizes if lp_mint_sizes else []) + lp_burn_sizes))
            def scaleL2(L): return 30 + 120 * (L / maxL)
            ax1.scatter(lp_burn_micro, (np.array(micro_P)[lp_burn_micro] - 2 * off[lp_burn_micro]),
                        marker="x", color="#ff8c00",
                        s=[scaleL2(L) for L in lp_burn_sizes], label="LP burn")

        ax1.set_ylabel("Price (Y per X)")
        ax1.set_title("CEX vs DEX — micro-time (DEX flat within blocks, jumps at validation)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(ncol=2, fontsize=9)
        ax1.margins(y=0.14)

        # Panel 2: notionals
        ax2.axhline(0.0, color="k", lw=1.0, alpha=0.3)
        ax2.bar(x, micro_trader_y, width=0.8, alpha=0.5, label="Trader notional (token1, signed)")
        ax2.plot(x, micro_arb_y, label="Arbitrage notional (token1, signed)", lw=1.8)
        ax2.set_ylabel("Notional (token1, signed)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9, loc="upper left")

        # Panel 3: liquidity traces
        ax3.plot(x, L_end_micro, lw=1.8, label="Active L (end, step-like)")
        ax3.plot(x, L_pre_block_micro, lw=1.0, ls="--", label="Active L (block start)")
        ax3.plot(x, L_pre_trader_micro, lw=1.0, ls=":", label="Active L (before trader)")
        ax3.plot(x, L_pre_arb_eff_micro, lw=1.2, ls="-.", label="Active L (before arb, effective)")
        ax3.set_ylabel("Active L")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9, loc="upper left")
        for v in val_idx:
            if L_end_micro[v] <= 1e-9:
                ax3.axvspan(v - 0.5, v + 0.5, color="red", alpha=0.05, lw=0)

        # Panel 4: ΔL at validation
        ax4.axhline(0.0, color="k", lw=1.0, alpha=0.3)
        ax4.bar(x, [mint_bar[i] if i in val_idx else 0.0 for i in range(len(x))],
                width=0.8, alpha=0.6, label="LP mint/center ΔL (>0)", color="#6a0dad")
        ax4.bar(x, [micro_deltaL_burn[i] if i in val_idx else 0.0 for i in range(len(x))],
                width=0.8, alpha=0.6, label="LP burn ΔL (<0)", color="#ff8c00")
        ax4.set_ylabel("ΔL at validation")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9, loc="upper left")

        # Panel 5: active-band reserves
        ax5.plot(x, np.array(micro_X_active) * np.array(micro_P), lw=1.8, label="token0 value in active band (≈ token1)")
        ax5.plot(x, np.array(micro_Y_active), lw=1.8, label="token1 in active band (Y)")
        ax5.set_xlabel("CEX updates (micro-steps)")
        ax5.set_ylabel("Active-band reserves")
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=9, loc="upper left")
        for v in val_idx:
            if L_end_micro[v] <= 1e-9:
                ax5.axvspan(v - 0.5, v + 0.5, color="red", alpha=0.05, lw=0)

        plt.tight_layout()
        plt.savefig("simulation_mempool.png", dpi=300)
        plt.show()

    return {
        # micro-time (primary)
        "micro_P": np.array(micro_P),
        "micro_M": np.array(micro_M),
        "micro_band_lo": np.array(micro_band_lo),
        "micro_band_hi": np.array(micro_band_hi),
        "micro_band_lo_pre": np.array(micro_band_lo_pre),
        "micro_band_hi_pre": np.array(micro_band_hi_pre),
        "micro_L": np.array(micro_L),
        "micro_L_pre_block": np.array(micro_L_pre_block),
        "micro_L_pre_trader": np.array(micro_L_pre_trader),
        "micro_L_pre_arb_eff": np.array(micro_L_pre_arb_eff),
        "micro_trader_y": np.array(micro_trader_y),
        "micro_arb_y": np.array(micro_arb_y),
        "micro_deltaL_mint": np.array(micro_deltaL_mint),
        "micro_deltaL_burn": np.array(micro_deltaL_burn),
        "micro_X_active": np.array(micro_X_active),
        "micro_Y_active": np.array(micro_Y_active),
        "lp_mint_micro": lp_mint_micro,
        "lp_mint_sizes": lp_mint_sizes,
        "lp_burn_micro": lp_burn_micro,
        "lp_burn_sizes": lp_burn_sizes,
        # block-level (secondary)
        "DEX_price_block": np.array(P_series_block),
        "CEX_price_block": np.array(M_series_block),
        "band_lo_post_block": np.array(band_lo_post_block),
        "band_hi_post_block": np.array(band_hi_post_block),
        "L_active_end_block": np.array(L_end_block),
        "trader_notional_y_block": np.array(trader_y_series_block),
        "arb_notional_y_block": np.array(arb_y_series_block),
        "L_first_arb_block": np.array([x if x is not None else np.nan for x in L_first_arb_block]),
        "liq_history": liq_history,
        "tick_history": tick_history,
        "B": B,
        "T_blocks": T_blocks,
    }


# ========= Optional GIF maker =========

def make_liquidity_gif(
    liq_history: List[Dict[int, float]],
    tick_history: List[int],
    out_path: str = "liquidity_evolution.gif",
    fps: int = 10,
    dpi: int = 120,
    pad_frac: float = 0.05,
    downsample_every: int = 1,
):
    assert len(liq_history) == len(tick_history), "Mismatched histories."
    if downsample_every > 1:
        liq_history = liq_history[::downsample_every]
        tick_history = tick_history[::downsample_every]
    all_boundaries = set()
    for snap in liq_history:
        all_boundaries.update(k for k, v in snap.items() if abs(v) > EPS_LIQ)
    if not all_boundaries:
        all_boundaries = {0}
    tmin = min(all_boundaries) - 5
    tmax = max(all_boundaries) + 5
    boundaries = np.arange(tmin, tmax + 1, dtype=int)
    tick_axis = boundaries[:-1]
    L_frames = []
    ymax = 1e-12
    for snap in liq_history:
        delta = np.zeros_like(boundaries, dtype=float)
        for k, dL in snap.items():
            if tmin <= k <= tmax:
                delta[k - tmin] += dL
        L_per_tick = np.cumsum(delta)[:-1]
        L_frames.append(L_per_tick)
        ymax = max(ymax, float(np.max(L_per_tick)))
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(tick_axis, L_frames[0], width=1.0, align="edge", color="#4C78A8")
    tick_line = ax.axvline(tick_history[0], color="crimson", lw=2, alpha=0.9, label="Active tick")
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(0.0, ymax * (1.0 + pad_frac))
    ax.set_xlabel("Tick")
    ax.set_ylabel("Active liquidity per tick")
    ax.set_title("Liquidity vs Tick — evolution")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.8))
    def update(frame_idx: int):
        L = L_frames[frame_idx]
        for rect, h in zip(bars, L):
            rect.set_height(float(h))
        tick_line.set_xdata([tick_history[frame_idx], tick_history[frame_idx]])
        txt.set_text(f"step = {frame_idx * downsample_every}")
        return (*bars, tick_line, txt)
    anim = animation.FuncAnimation(fig, update, frames=len(L_frames), blit=False)
    writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"[GIF] wrote {out_path}")


if __name__ == "__main__":
    out = simulate_with_mempool(
        T_blocks=60, B=12, seed=42, verbose_blocks=5, visualize=True,
        use_fee_band=True, p_trade=0.8, p_lp_narrow=0.6, p_lp_wide=0.4,
        N_LP=60, tau=20, w_min_ticks=10, psi_ticks=200,
        binom_n=10, binom_p=0.5, mint_mu=0.05, mint_sigma=0.01,
        theta_TP=0.10, theta_SL=0.15, evolve_initial_hill=True,
        initial_binom_N=500, initial_total_L=250_000, k_out=10
    )
