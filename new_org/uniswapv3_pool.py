"""
Uniswap v3 pool implementation with spacing-aware tick management.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Callable
from bisect import bisect_right, bisect_left

from utils import EPS_LIQ, EPS_BOUNDARY, EPS_LIQ2


# =============================================================================
# Pool (Uniswap v3–style, spacing-aware)
# =============================================================================

@dataclass
class V3Pool:
    """
    Minimal Uniswap v3 pool state for a single asset pair, **spacing-aware**.

    Grid & price:
      • S := sqrt(P) where P is token1 per token0. Grid ratio g (>1) is geometric in S.
      • Tick i has lower boundary s_i = base_s · g^i and upper s_{i+Δ} with Δ = tick_spacing.
      • The active **band** is [tick, tick+tick_spacing). Active liquidity L_active is the
        prefix-sum of `liquidity_net` up to `tick`.

    Liquidity book:
      • `liquidity_net[k]` stores the ΔL applied when crossing **upward** through boundary k.
      • `recompute_active_L()` rebuilds L_active robustly from liquidity_net.

    Fees & swaps:
      • Fee f is **on input**; r = 1 - f is the retained fraction used to move S.
        Fees are accrued **per band segment**: if a swap consumes input Δq_eff in a band,
        the fee charged there is Δq_eff·(1/r - 1) and is allocated to LPs pro-rata by
        the band's active L at the *start* of that segment.
      • Swaps consume liquidity only where it exists. If L_active == 0 ("desert"), price
        does not move. The engine outside the raw swap functions can **bridge** to the next
        initialized band and then swap. See `swap_exact_to_target(...)` and `ensure_liquidity(...)`.

    Invariants:
      • Price only changes via swaps; L_active only changes via (re)mint/burn.
      • We clamp tiny drifts to zero with EPS_* guards to avoid numerical ghosts.
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

    def _active_L_at_tick(self, tick_i: int) -> float:
        L = 0.0
        for k, dL in self.liquidity_net.items():
            if k <= tick_i:
                L += dL
        return L

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

    def quote_x_to_y(self, dx_in: float, bidx: "BoundaryIndex") -> float:
        """Return the expected token1 out for an X→Y swap without mutating state."""
        if dx_in <= 0:
            return 0.0

        r = self.r
        dx_eff = dx_in * r

        tick_loc = self._snap(self.tick)
        S_loc = self.S
        L_loc = self._active_L_at_tick(tick_loc)

        if L_loc <= EPS_LIQ:
            pb = bidx.prev_down(tick_loc)
            if pb is None:
                return 0.0
            tick_loc = self._snap(pb - self.tick_spacing)
            S_loc = self.s_upper(tick_loc)
            L_loc = self._active_L_at_tick(tick_loc)
            if L_loc <= EPS_LIQ:
                return 0.0

        dy_out = 0.0
        while dx_eff > EPS_LIQ and L_loc > EPS_LIQ:
            S_lo = self.s_lower(tick_loc)
            if S_loc <= S_lo + EPS_BOUNDARY:
                S_loc = S_lo
                tick_loc -= self.tick_spacing
                L_loc = self._active_L_at_tick(tick_loc)
                continue
            dx_to = L_loc * (1.0 / S_lo - 1.0 / S_loc)
            if dx_eff < dx_to - EPS_BOUNDARY:
                S_new = 1.0 / (1.0 / S_loc + dx_eff / L_loc)
                dy = L_loc * (S_new - S_loc)
                dy_out += -dy
                dx_eff = 0.0
            else:
                dy = L_loc * (S_lo - S_loc)
                dy_out += -dy
                dx_eff -= dx_to
                S_loc = S_lo
                tick_loc -= self.tick_spacing
                L_loc = self._active_L_at_tick(tick_loc)
        return max(0.0, dy_out)

    def quote_y_to_x(self, dy_in: float, bidx: "BoundaryIndex") -> float:
        """Return the expected token0 out for a Y→X swap without mutating state."""
        if dy_in <= 0:
            return 0.0

        r = self.r
        dy_eff = dy_in * r

        tick_loc = self._snap(self.tick)
        S_loc = self.S
        L_loc = self._active_L_at_tick(tick_loc)

        if L_loc <= EPS_LIQ:
            nb = bidx.next_up(tick_loc)
            if nb is None:
                return 0.0
            tick_loc = self._snap(nb)
            S_loc = self.s_lower(tick_loc)
            L_loc = self._active_L_at_tick(tick_loc)
            if L_loc <= EPS_LIQ:
                return 0.0

        dx_out = 0.0
        while dy_eff > EPS_LIQ and L_loc > EPS_LIQ:
            S_hi = self.s_upper(tick_loc)
            if S_loc >= S_hi - EPS_BOUNDARY:
                S_loc = S_hi
                tick_loc += self.tick_spacing
                L_loc = self._active_L_at_tick(tick_loc)
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
                tick_loc += self.tick_spacing
                L_loc = self._active_L_at_tick(tick_loc)
        return max(0.0, dx_out)


# =============================================================================
# Sparse boundary index (for fast next boundary lookups)
# =============================================================================

class BoundaryIndex:
    """
    Sparse index over boundaries with non-zero `liquidity_net` entries.

    Purpose:
      • O(log B) lookup for the next initialized boundary upward/downward from a tick.
      • Used by non-mutating quotes and by the "desert-bridging" logic so we can find
        the next band that actually has liquidity without touching pool state.

    Contract:
      • Call `mark_dirty()` whenever `liquidity_net` changes; reads auto-refresh lazily.
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

