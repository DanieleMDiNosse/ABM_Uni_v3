"""
Main simulation runner for the ABM model.
"""
from __future__ import annotations

import argparse
import math
import random
import inspect
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional, Callable, Set
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

try:
    import mpld3  # type: ignore
except ImportError:
    mpld3 = None

HTML_SAVE_WARNING_EMITTED = False


def _save_html(fig: plt.Figure, html_path: Path, source: str) -> None:
    """Save an interactive HTML version of a Matplotlib figure if mpld3 is available."""
    global HTML_SAVE_WARNING_EMITTED
    if mpld3 is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        mpld3.save_html(fig, str(html_path))
    elif not HTML_SAVE_WARNING_EMITTED:
        print(f"[{source}] mpld3 not installed; skipping interactive HTML exports.")
        HTML_SAVE_WARNING_EMITTED = True

# Import from new module structure
from utils import (
    build_empty_pool,
    bootstrap_initial_binomial_hill_sharded,
    minted_amounts_at_S,
    ReferenceMarket,
    EWMA,
    next_numbered_path,
    EPS_LIQ,
    EPS_LIQ2,
    EPS_BOUNDARY,
    EPS_PRICE_CHANGE,
    TICK_LN,
    clamp,
    TITLE_FONT_SIZE,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    make_liquidity_gif,
    load_simulation_parameters,
)
from agents import (
    LPAgent,
    Position,
    lp_token0_exposure,
    lp_mark_to_market_y,
    lp_wealth_y,
)
from uniswapv3_pool import V3Pool, BoundaryIndex


# =============================================================================
# Simulation
# =============================================================================

@dataclass
class TraderStepAccumulator:
    notional_y: float = 0.0
    pnl: float = 0.0
    execs: int = 0
    dx_in: float = 0.0
    dx_out: float = 0.0
    dy_in: float = 0.0
    dy_out: float = 0.0

    def record_swap(
        self,
        *,
        dx_in: float = 0.0,
        dx_out: float = 0.0,
        dy_in: float = 0.0,
        dy_out: float = 0.0,
    ) -> None:
        """Track token flows for later PnL settlement."""
        self.dx_in += dx_in
        self.dx_out += dx_out
        self.dy_in += dy_in
        self.dy_out += dy_out

    def settle(self, m_settle: float) -> None:
        """
        Value accumulated flows versus the provided CEX price.
        Positive result means net token1 profit.
        """
        self.pnl = (self.dy_out - self.dy_in) + (self.dx_out - self.dx_in) * m_settle

def simulate(
    block_size: int = 10,
    T: int = 120,
    seed: int = 7,
    cex_mu: float = 0.0,
    cex_sigma: float = 0.02,
    p_trade: float = 0.6, 
    noise_floor: float = 0.0,
    p_lp_narrow: float = 0.7,
    p_lp_wide: float = 0.15,
    passive_lp_share: float = 0.0,
    passive_mint_prob: float = 0.1,
    passive_burn_prob: float = 0.05,
    passive_width_ticks: int = 500,
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
    # --- slippage ---
    slippage_tolerance: float = 0.01,
    # --- other params ---
    mint_mu: float = 0.01,
    mint_sigma: float = 0.02,
    theta_TP: float = 0.003,
    theta_SL: float = 0.01,
    initial_binom_N: int = 400,
    initial_total_L: float = 70_000.0,
    k_out: int = 2,
    visualize: bool = True,
    skip_step: int = 0,
    # --- Dynamic fee controller (new) ---
    fee_mode: str = "volatility",      # "static" | "volatility" 
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
    active_wide_lp_enabled: bool = True,
):
    valid_fee_modes = {"static", "volatility", "toxicity"}
    if fee_mode not in valid_fee_modes:
        raise ValueError(f"Invalid fee_mode '{fee_mode}'. Expected one of {sorted(valid_fee_modes)}.")

    slippage_tolerance = clamp(slippage_tolerance, 0.0, 1.0)
    """
    Run a Step-1 ABM with a Uniswap v3–style pool.

    - noise_floor (float in [0,1]): with this probability, the noise trader executes on the DEX even if the DEX quote fails the relative-value check.
    Run a Step-1 ABM of a Uniswap v3–style pool with noise traders, a band-targeting
    arbitrageur, and adaptive LPs. **Actor order is randomized each step.**
    """
    initial_params = dict(locals())
    np.random.seed(seed)
    random.seed(seed)
    passive_share = max(0.0, min(1.0, passive_lp_share))
    share_narrow_default = 0.7
    share_narrow_eff = min(share_narrow_default, max(0.0, 1.0 - passive_share))
    if not active_wide_lp_enabled:
        share_narrow_eff = max(0.0, 1.0 - passive_share)

    # --- Build pool + reference market + LP agents ----------------------------
    pool, m0 = build_empty_pool()
    ref = ReferenceMarket(m=m0, mu=cex_mu, sigma=cex_sigma, kappa=1e-3)

    LPs: List[LPAgent] = []
    for i in range(N_LP):
        r = random.random()
        is_passive = r < passive_share
        is_narrow = False
        if not is_passive and r < passive_share + share_narrow_eff:
            is_narrow = True
        if not active_wide_lp_enabled and not is_passive:
            is_narrow = True
        mintProb = passive_mint_prob if is_passive else (p_lp_narrow if is_narrow else p_lp_wide)
        LPs.append(
            LPAgent(
                id=i,
                mintProb=mintProb,
                is_active_narrow=is_narrow,
                is_passive=is_passive,
            )
        )
        lp = LPs[-1]
        lp.review_rate = 1.0 / max(1, tau)
        lp.next_review = int(np.random.geometric(lp.review_rate))
        lp.cooldown = 0
        lp.can_act = False

    # Distribute initial_total_L across LPs (each gets ~equal share)
    L_SCALE = initial_total_L / max(1, N_LP)
    for lp in LPs:
        lp.L_budget = 2.0 * L_SCALE   # each LP can deploy up to ~2× their fair share
        lp.L_live = 0.0               # tracked across mints/burns

    bootstrap_initial_binomial_hill_sharded(
        pool, ref, LPs,
        N=initial_binom_N,
        L_total=initial_total_L,
        num_seed_lps=20,
        seed_lp_id_base=10_000,
        seed_mint_prob=0.0,
        tau=tau,
        plot=False
    )

    # ensure budgets exist for every LP, including the just-appended seed
    for lp in LPs:
        if lp.L_budget <= 0.0:
            lp.L_budget = 2.0 * L_SCALE
        if lp.L_live < 0.0:
            lp.L_live = 0.0

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
    # --- Block-start target band (arb_ref_m) ---
    band_lo_target, band_hi_target = [], []
    # --- Micro-time traces (for block_size > 1 visualization) ---
    micro_steps, M_micro, P_micro = [], [], []
    # --- PnL recorders ---
    trader_pnl_steps = []       # realized per-step PnL (token1)
    arb_pnl_steps = []          # realized per-step PnL (token1)
    lp_pnl_total_series = []    # cumulative hedged PnL (fees - rebal) across all LPs
    lp_pnl_active_series = []   # cumulative hedged PnL for active (narrow) LPs
    lp_pnl_passive_series = []  # cumulative hedged PnL for passive LPs
    lp_pnl_wide_series = []     # cumulative hedged PnL for active wide LPs
    lp_rebal_total_series = []  # cumulative rebalancing PnL (benchmark) across LPs
    lp_rebal_active_series = []
    lp_rebal_passive_series = []
    lp_rebal_wide_series = []
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
    lp_wallet_active_series = []   # active narrow LPs
    lp_wallet_passive_series = []  # passive LPs
    lp_wallet_wide_series = []     # active wide LPs
    lp_wealth_active_series = []
    lp_wealth_passive_series = []
    lp_wealth_wide_series = []
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
    prev_m_for_vol = ref.m

    # ------------------ LVR rebalancer helpers ------------------
    REBAL_EPS = 1e-18

    def _ensure_rebalancer_initialized(lp: LPAgent, M_now: float, S_now: float) -> None:
        rb = lp.rebalancer
        if rb.initialized:
            return
        rb.reset()
        x_target = lp_token0_exposure(lp, S_now)
        rb.x_prev = x_target
        rb.cash_y = -x_target * M_now
        rb.cumulative_R = 0.0
        rb.last_M = M_now
        wealth_now = lp_wealth_y(lp, S_now, M_now)
        rb.last_wealth_y = wealth_now
        rb.last_cumulative_R = 0.0
        rb.hedged_pnl_cum = 0.0
        rb.initialized = True

    def _rebalance_lp_to_target(lp: LPAgent, M_now: float, S_now: float) -> None:
        _ensure_rebalancer_initialized(lp, M_now, S_now)
        rb = lp.rebalancer
        x_target = lp_token0_exposure(lp, S_now)
        dx = x_target - rb.x_prev
        if abs(dx) > REBAL_EPS:
            rb.cash_y -= dx * M_now
            rb.x_prev = x_target
        rb.last_M = M_now

    def _rebalance_subset(lp_subset: List[LPAgent], M_now: float, S_now: float) -> None:
        if not lp_subset:
            return
        for lp in lp_subset:
            _rebalance_lp_to_target(lp, M_now, S_now)

    def _rebalance_by_ids(lp_ids: Set[int], M_now: float, S_now: float) -> None:
        if not lp_ids:
            return
        id_lookup = lp_ids
        for lp in LPs:
            if lp.id in id_lookup:
                _rebalance_lp_to_target(lp, M_now, S_now)

    def _rebalance_all(M_now: float, S_now: float) -> None:
        for lp in LPs:
            _rebalance_lp_to_target(lp, M_now, S_now)

    def _accrue_price_move(lp: LPAgent, M_new: float) -> None:
        rb = lp.rebalancer
        if not rb.initialized:
            rb.last_M = M_new
            return
        delta = M_new - rb.last_M
        if abs(delta) > 0.0:
            rb.cumulative_R += rb.x_prev * delta
            rb.last_M = M_new

    def _broadcast_price_move(M_new: float) -> None:
        for lp in LPs:
            _accrue_price_move(lp, M_new)

    # Initialize rebalancers to match current exposures before the simulation loop
    _rebalance_all(ref.m, pool.S)

    # ------------------ Helpers ------------------
    def allocate_fees(token: str, fee_amt: float, tick_snapshot: int, L_snapshot: float) -> None:
        if fee_amt <= 0 or L_snapshot <= 0:
            return
        touched_lp_ids: Set[int] = set()
        for lp in LPs:
            for pos in lp.positions:
                if pos.in_range(tick_snapshot):
                    share = pos.L / L_snapshot
                    if token == "x":
                        delta_fee0 = share * fee_amt
                        pos.fees0 += delta_fee0
                        if delta_fee0 != 0.0:
                            touched_lp_ids.add(pos.owner)
                    else:
                        pos.fees1 += share * fee_amt
        if touched_lp_ids:
            _rebalance_by_ids(touched_lp_ids, ref.m, pool.S)

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
        _rebalance_lp_to_target(lp, ref.m, pool.S)

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

        lp.cooldown = np.random.randint(3, 9)  # 3–8 steps of "hands off"
        lp.L_live = max(0.0, getattr(lp, "L_live", 0.0) - pos.L)
        _rebalance_lp_to_target(lp, ref.m, pool.S)


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

        # --- Desert bridge (peek → optionally apply) ---
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


    def arbitrage_to_target(arb_ref_m: float) -> Tuple[float, float, float, Optional[str], float, float, float]:
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
        # Block-invariant arrival probabilities
        if block_size > 1:
            p_trade_micro = 1.0 - (1.0 - p_trade)**(1.0 / block_size)
            noise_floor_micro = 1.0 - (1.0 - noise_floor)**(1.0 / block_size)
        else:
            p_trade_micro = p_trade
            noise_floor_micro = noise_floor

        lo, hi = arb_ref_m * r, arb_ref_m / r
        if P < lo * (1 - 1e-9):
            # up: returns (dy_in, dx_out, fee_x=0, fee_y, L_first)
            dy_in, dx_out, fx, fy, Lff = swap_exact_to_target(lo, "up", fee_cb=allocate_fees)
            return dy_in, dx_out, 0.0, ("up" if dy_in > 0 else None), fx, fy, Lff
        if P > hi * (1 + 1e-9):
            # down: returns (dx_in, dy_out, fee_x, fee_y=0, L_first)
            dx_in, dy_out, fx, fy, Lff = swap_exact_to_target(hi, "down", fee_cb=allocate_fees)
            return dx_in, 0.0, dy_out, ("down" if dx_in > 0 else None), fx, fy, Lff
        return 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0

    # Block-invariant arrival probabilities (computed once before loop)
    if block_size > 1:
        p_trade_micro = 1.0 - (1.0 - p_trade)**(1.0 / block_size)
        noise_floor_micro = 1.0 - (1.0 - noise_floor)**(1.0 / block_size)
    else:
        p_trade_micro = p_trade
        noise_floor_micro = noise_floor

    total_noise_swaps_executed = 0
    total_noise_swaps_skipped = 0
    total_smart_swaps_executed = 0
    total_smart_swaps_skipped = 0

    # ------------------ Main loop ------------------
    for t in range(T):
        arb_ref_m = ref.m  # default; overridden per-block to end-of-block CEX
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

        # Start-of-step rebalance benchmark update (predictable integrand)
        _rebalance_all(ref.m, pool.S)

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
        # ---------------------------------------------------------------------

        # --- Per-step accumulators (so we can randomize actor order) ---
        trader_y_this = 0.0
        arb_y_this = 0.0
        trader_pnl_this = 0.0
        arb_pnl_this = 0.0
        _trader_execs = 0
        _arb_execs = 0
        # Split per-actor accumulators
        sr_acc = TraderStepAccumulator()
        noise_acc = TraderStepAccumulator()
        arb_acc = TraderStepAccumulator()
        delta_a_cex_this = 0.0
        L_pre_trader_this = np.nan
        L_pre_arb_eff_this = np.nan
        dir_arb_this: Optional[str] = None

        
        # --- Mempool structures (for block_size > 1) ---
        mempool_orders = []
        
        # ----- Non-mutating Uni v3 quotes (spacing-aware, can bridge deserts) -----
        def maybe_enqueue_smart_router_intent(m_now: float):
            """Enqueue a smart-router swap intent if DEX output is competitive vs CEX (theta_T)."""
            if random.random() >= p_trade_micro:
                return
            side = random.choice(["X_to_Y", "Y_to_X"])
            if side == "X_to_Y":
                dx = float(np.exp(np.random.normal(loc=trader_mean, scale=trader_sigma)))
                if dx <= 0.0:
                    return
                initial_quote = pool.quote_x_to_y(dx, bidx)
                if initial_quote <= 0.0:
                    return
                # best-ex vs CEX: compare dy_out to dx * m_now (value in token1)
                if initial_quote < theta_T * dx * m_now:
                    return
                min_output = max(0.0, initial_quote * (1.0 - slippage_tolerance))
                mempool_orders.append({
                    'type': 'swap',
                    'agent': 'smart',
                    'side': 'X_to_Y',
                    'amount': dx,
                    'unit': 'dx',
                    'm_submit': m_now,
                    'min_output': min_output,
                })
            else:
                dy = float(np.exp(np.random.normal(loc=trader_mean, scale=trader_sigma)))
                if dy <= 0.0:
                    return
                initial_quote = pool.quote_y_to_x(dy, bidx)
                if initial_quote <= 0.0:
                    return
                # best-ex vs CEX: compare dx_out to dy / m_now (value in token0)
                if initial_quote < theta_T * dy / max(m_now, 1e-18):
                    return
                min_output = max(0.0, initial_quote * (1.0 - slippage_tolerance))
                mempool_orders.append({
                    'type': 'swap',
                    'agent': 'smart',
                    'side': 'Y_to_X',
                    'amount': dy,
                    'unit': 'dy',
                    'm_submit': m_now,
                    'min_output': min_output,
                })

        def maybe_enqueue_noise_trader_intent(m_now: float):
            """Enqueue a noise swap intent (no best-ex check)."""
            if random.random() >= noise_floor_micro:
                return
            side = random.choice(["X_to_Y", "Y_to_X"])
            if side == "X_to_Y":
                dx = float(np.exp(np.random.normal(loc=trader_mean, scale=trader_sigma)))
                if dx > 0.0:
                    initial_quote = pool.quote_x_to_y(dx, bidx)
                    if initial_quote <= 0.0:
                        return
                    min_output = max(0.0, initial_quote * (1.0 - slippage_tolerance))
                    mempool_orders.append({
                        'type': 'swap',
                        'agent': 'noise',
                        'side': 'X_to_Y',
                        'amount': dx,
                        'unit': 'dx',
                        'm_submit': m_now,
                        'min_output': min_output,
                    })
            else:
                dy = float(np.exp(np.random.normal(loc=trader_mean, scale=trader_sigma)))
                if dy > 0.0:
                    initial_quote = pool.quote_y_to_x(dy, bidx)
                    if initial_quote <= 0.0:
                        return
                    min_output = max(0.0, initial_quote * (1.0 - slippage_tolerance))
                    mempool_orders.append({
                        'type': 'swap',
                        'agent': 'noise',
                        'side': 'Y_to_X',
                        'amount': dy,
                        'unit': 'dy',
                        'm_submit': m_now,
                        'min_output': min_output,
                    })

        def execute_mempool_orders():
            nonlocal trader_y_this, sr_acc, noise_acc
            nonlocal total_noise_swaps_executed, total_noise_swaps_skipped
            nonlocal total_smart_swaps_executed, total_smart_swaps_skipped
            P_pre_exec = pool.price
            def _exec_one(o):
                nonlocal P_pre_exec, trader_y_this, sr_acc, noise_acc
                nonlocal total_noise_swaps_executed, total_noise_swaps_skipped
                nonlocal total_smart_swaps_executed, total_smart_swaps_skipped
                P_pre_exec = pool.price
                # Handle LP intents (they don't have 'agent' or 'side')
                typ = o.get('type')
                if typ in ('lp_burn','lp_mint','lp_recenter'):
                    lp = next((x for x in LPs if x.id == o.get('lp_id')), None)
                    if lp is None:
                        return
                    if typ == 'lp_burn':
                        idx = None
                        for i, pos in enumerate(lp.positions):
                            if pos.lower == o.get('lower') and pos.upper == o.get('upper') and abs(pos.L - float(o.get('L', 0.0))) < 1e-12:
                                idx = i; break
                        if idx is None:
                            return
                        burn_any(lp, idx)
                        return
                    if typ == 'lp_mint':
                        lower = int(o.get('lower')); upper = int(o.get('upper')); L_new = float(o.get('L', 0.0))
                        if upper <= lower or L_new <= 0.0:
                            return
                        sa, sb = pool.s_lower(lower), pool.s_upper(upper)
                        amt0, amt1 = minted_amounts_at_S(L_new, sa, sb, pool.S)
                        pos = Position(owner=lp.id, lower=lower, upper=upper, L=L_new, sa=sa, sb=sb,
                                        amt0_init=amt0, amt1_init=amt1, hodl0_value_y=amt0 * ref.m + amt1)
                        pool.add_liquidity_range(lower, upper, L_new)
                        pool.recompute_active_L(); bidx.mark_dirty()
                        lp.positions.append(pos)
                        mint_steps.append(t); mint_sizes.append(L_new); mint_widths.append(upper - lower)
                        with open(verbose_log_path_str, 'a') as f:
                            f.write(f"[t={t:03d}] LP{lp.id} MINT L={L_new:.4f} [{lower},{upper}) | L_active={pool.L_active:.4f}\n")
                        lp.L_live = getattr(lp, 'L_live', 0.0) + L_new
                        _rebalance_lp_to_target(lp, ref.m, pool.S)
                        return
                    if typ == 'lp_recenter':
                        idx = None
                        for i, pos in enumerate(lp.positions):
                            if pos.lower == o.get('old_lower') and pos.upper == o.get('old_upper') and abs(pos.L - float(o.get('old_L', 0.0))) < 1e-12:
                                idx = i; break
                        if idx is not None:
                            burn_any(lp, idx)
                        lower = int(o.get('new_lower')); upper = int(o.get('new_upper')); L_new = float(o.get('new_L', 0.0))
                        if upper <= lower or L_new <= 0.0:
                            return
                        sa, sb = pool.s_lower(lower), pool.s_upper(upper)
                        amt0, amt1 = minted_amounts_at_S(L_new, sa, sb, pool.S)
                        pos = Position(owner=lp.id, lower=lower, upper=upper, L=L_new, sa=sa, sb=sb,
                                        amt0_init=amt0, amt1_init=amt1, hodl0_value_y=amt0 * ref.m + amt1)
                        pool.add_liquidity_range(lower, upper, L_new)
                        pool.recompute_active_L(); bidx.mark_dirty()
                        lp.positions.append(pos)
                        mint_steps.append(t); mint_sizes.append(L_new); mint_widths.append(upper - lower)
                        with open(verbose_log_path_str, 'a') as f:
                            f.write(f"[t={t:03d}] LP{lp.id} RECENTER L={L_new:.4f} [{lower},{upper}) | L_active={pool.L_active:.4f}\n")
                        _rebalance_lp_to_target(lp, ref.m, pool.S)
                        return
                if o.get('side') == 'X_to_Y':
                    min_output = o.get('min_output')
                    if min_output is not None:
                        final_quote = pool.quote_x_to_y(o['amount'], bidx)
                        if final_quote < min_output:
                            agent = o.get('agent')
                            if agent == 'smart':
                                total_smart_swaps_skipped += 1
                            elif agent == 'noise':
                                total_noise_swaps_skipped += 1
                            with open(verbose_log_path_str, 'a') as f:
                                f.write(
                                    f"[t={t:03d}] {agent or 'N/A'} swap X_to_Y SKIPPED (slippage). "
                                    f"final_quote={final_quote:.4f} < min_output={min_output:.4f}\n"
                                )
                            return
                    prev_tick, prev_S = pool.tick, pool.S
                    bridged = False
                    if pool.L_active <= EPS_LIQ:
                        ok, new_tick, new_S, _ = ensure_liquidity('down')
                        if not ok: return
                        pool.tick, pool.S = new_tick, new_S
                        pool.recompute_active_L()
                        bridged = True
                    used_dx_pre, dy_out_real, fee_x = pool.swap_x_to_y(o['amount'], fee_cb=allocate_fees)
                    if used_dx_pre <= EPS_LIQ:
                        if bridged:
                            pool.tick, pool.S = prev_tick, prev_S
                            pool.recompute_active_L()
                        return
                    agent = o.get('agent')
                    if agent == 'smart':
                        trader_steps.append(t); trader_dirs.append('down')
                        sr_acc.notional_y += -P_pre_exec * used_dx_pre
                        trader_y_this += -P_pre_exec * used_dx_pre
                        sr_acc.record_swap(dx_in=used_dx_pre, dy_out=dy_out_real)
                        sr_acc.execs += int(used_dx_pre > 0)
                        total_smart_swaps_executed += int(used_dx_pre > 0)
                    elif agent == 'noise':
                        trader_steps.append(t); trader_dirs.append('down')
                        noise_acc.notional_y += -P_pre_exec * used_dx_pre
                        trader_y_this += -P_pre_exec * used_dx_pre
                        noise_acc.record_swap(dx_in=used_dx_pre, dy_out=dy_out_real)
                        noise_acc.execs += int(used_dx_pre > 0)
                        total_noise_swaps_executed += int(used_dx_pre > 0)
                else:
                    min_output = o.get('min_output')
                    if min_output is not None:
                        final_quote = pool.quote_y_to_x(o['amount'], bidx)
                        if final_quote < min_output:
                            agent = o.get('agent')
                            if agent == 'smart':
                                total_smart_swaps_skipped += 1
                            elif agent == 'noise':
                                total_noise_swaps_skipped += 1
                            with open(verbose_log_path_str, 'a') as f:
                                f.write(
                                    f"[t={t:03d}] {agent or 'N/A'} swap Y_to_X SKIPPED (slippage). "
                                    f"final_quote={final_quote:.4f} < min_output={min_output:.4f}\n"
                                )
                            return
                    prev_tick, prev_S = pool.tick, pool.S
                    bridged = False
                    if pool.L_active <= EPS_LIQ:
                        ok, new_tick, new_S, _ = ensure_liquidity('up')
                        if not ok: return
                        pool.tick, pool.S = new_tick, new_S
                        pool.recompute_active_L()
                        bridged = True
                    used_dy_pre, dx_out_real, fee_y = pool.swap_y_to_x(o['amount'], fee_cb=allocate_fees)
                    if used_dy_pre <= EPS_LIQ:
                        if bridged:
                            pool.tick, pool.S = prev_tick, prev_S
                            pool.recompute_active_L()
                        return
                    agent = o.get('agent')
                    if agent == 'smart':
                        trader_steps.append(t); trader_dirs.append('up')
                        sr_acc.notional_y += +used_dy_pre
                        trader_y_this += +used_dy_pre
                        sr_acc.record_swap(dy_in=used_dy_pre, dx_out=dx_out_real)
                        sr_acc.execs += int(used_dy_pre > 0)
                        total_smart_swaps_executed += int(used_dy_pre > 0)
                    elif agent == 'noise':
                        trader_steps.append(t); trader_dirs.append('up')
                        noise_acc.notional_y += +used_dy_pre
                        trader_y_this += +used_dy_pre
                        noise_acc.record_swap(dy_in=used_dy_pre, dx_out=dx_out_real)
                        noise_acc.execs += int(used_dy_pre > 0)
                        total_noise_swaps_executed += int(used_dy_pre > 0)
            n_noise = sum(1 for o in mempool_orders if o.get('agent')=='noise')
            n_smart = sum(1 for o in mempool_orders if o.get('agent')=='smart')
            order_book = list(mempool_orders)
            random.shuffle(order_book)
            with open(verbose_log_path_str, 'a') as f:
                f.write(f"[t={t:03d}] MEMPOOL before P={P_pre_exec:.4f} | n_orders={len(order_book)} (smart={n_smart}, noise={n_noise})\n")
            for o in order_book:
                _exec_one(o)
            with open(verbose_log_path_str, 'a') as f:
                f.write(f"[t={t:03d}] MEMPOOL after P={pool.price:.4f}\n")
            mempool_orders.clear()

        def execute_trader(agent_label: str, probability: float, accumulator: TraderStepAccumulator, enforce_best_ex: bool) -> None:
            nonlocal L_pre_trader_this, trader_y_this, _trader_execs
            nonlocal total_noise_swaps_executed, total_noise_swaps_skipped
            nonlocal total_smart_swaps_executed, total_smart_swaps_skipped

            if random.random() >= probability:
                return

            side = random.choice(["X_to_Y", "Y_to_X"])
            L_pre_trader_this = pool.L_active
            P_pre = pool.price
            m_now = ref.m

            if side == "X_to_Y":
                dx = np.exp(np.random.normal(loc=trader_mean, scale=trader_sigma))
                if dx <= 0.0:
                    return
                initial_quote = pool.quote_x_to_y(dx, bidx)
                if initial_quote <= 0.0:
                    return
                if enforce_best_ex:
                    if initial_quote < theta_T * dx * m_now:
                        return
                min_output = max(0.0, initial_quote * (1.0 - slippage_tolerance))
                final_quote = pool.quote_x_to_y(dx, bidx)
                if final_quote < min_output:
                    if agent_label == "smart":
                        total_smart_swaps_skipped += 1
                    elif agent_label == "noise":
                        total_noise_swaps_skipped += 1
                    with open(verbose_log_path_str, "a") as f:
                        f.write(
                            f"[t={t:03d}] {agent_label} swap X_to_Y SKIPPED (slippage). "
                            f"final_quote={final_quote:.4f} < min_output={min_output:.4f}\n"
                        )
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

                used_dx_pre, dy_out_real, _ = pool.swap_x_to_y(dx, fee_cb=allocate_fees)
                if used_dx_pre <= EPS_LIQ:
                    if bridged:
                        pool.tick, pool.S = prev_tick, prev_S
                        pool.recompute_active_L()
                    return

                trader_steps.append(t)
                trader_dirs.append("down")

                delta_y = -P_pre * used_dx_pre
                accumulator.notional_y += delta_y
                trader_y_this += delta_y

                accumulator.record_swap(dx_in=used_dx_pre, dy_out=dy_out_real)

                executed = int(used_dx_pre > 0)
                accumulator.execs += executed
                _trader_execs += executed
                if agent_label == "smart":
                    total_smart_swaps_executed += executed
                elif agent_label == "noise":
                    total_noise_swaps_executed += executed

            else:
                dy = np.exp(np.random.normal(loc=trader_mean, scale=trader_sigma))
                if dy <= 0.0:
                    return
                initial_quote = pool.quote_y_to_x(dy, bidx)
                if initial_quote <= 0.0:
                    return
                if enforce_best_ex:
                    dx_cex = dy / max(m_now, 1e-18)
                    if initial_quote < theta_T * dx_cex:
                        return
                min_output = max(0.0, initial_quote * (1.0 - slippage_tolerance))
                final_quote = pool.quote_y_to_x(dy, bidx)
                if final_quote < min_output:
                    if agent_label == "smart":
                        total_smart_swaps_skipped += 1
                    elif agent_label == "noise":
                        total_noise_swaps_skipped += 1
                    with open(verbose_log_path_str, "a") as f:
                        f.write(
                            f"[t={t:03d}] {agent_label} swap Y_to_X SKIPPED (slippage). "
                            f"final_quote={final_quote:.4f} < min_output={min_output:.4f}\n"
                        )
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

                used_dy_pre, dx_out_real, _ = pool.swap_y_to_x(dy, fee_cb=allocate_fees)
                if used_dy_pre <= EPS_LIQ:
                    if bridged:
                        pool.tick, pool.S = prev_tick, prev_S
                        pool.recompute_active_L()
                    return

                trader_steps.append(t)
                trader_dirs.append("up")

                accumulator.notional_y += used_dy_pre
                trader_y_this += used_dy_pre

                accumulator.record_swap(dy_in=used_dy_pre, dx_out=dx_out_real)

                executed = int(used_dy_pre > 0)
                accumulator.execs += executed
                _trader_execs += executed
                if agent_label == "smart":
                    total_smart_swaps_executed += executed
                elif agent_label == "noise":
                    total_noise_swaps_executed += executed

        # --- Actor routines (closures) ---
        def act_LPs():
            # ----- burns (TP/SL) -----
            for lp in LPs:
                if hasattr(lp, "can_act") and not lp.can_act:
                    continue
                if lp.is_passive:
                    if lp.positions and random.random() < passive_burn_prob:
                        burn_any(lp, len(lp.positions) - 1)
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
                if lp.is_passive:
                    if lp.positions or random.random() >= passive_mint_prob:
                        continue
                    width_ticks = max(passive_width_ticks, pool.tick_spacing)
                    n_bands = max(1, int(round(width_ticks / pool.tick_spacing)))
                else:
                    if random.random() >= lp.mintProb:
                        continue
                    n_bands = max(1, int(round(w_ticks / pool.tick_spacing)))
                X = abs(np.random.normal(mint_mu, mint_sigma))
                try:
                    _L_SCALE = L_SCALE
                except NameError:
                    _L_SCALE = initial_total_L / max(1, N_LP)
                want = X * _L_SCALE
                cap_step = 0.25 * getattr(lp, 'L_budget', want)
                cap_left = max(0.0, getattr(lp, 'L_budget', want) - getattr(lp, 'L_live', 0.0))
                L_new = max(0.0, min(want, cap_step, cap_left))
                if L_new <= 0.0:
                    continue
                S_now = pool.S
                sps = pool.tick_spacing
                nb = n_bands
                denom = (1.0 + (pool.g ** (nb * sps + sps)))
                if denom <= 0.0:
                    denom = 1.0
                lower_real = math.log((2.0 * S_now / pool.base_s) / denom, pool.g)
                lower = pool._snap(int(round(lower_real)))
                upper = lower + nb * sps
                if upper <= lower:
                    upper = lower + pool.tick_spacing
                mempool_orders.append({'type':'lp_mint','lp_id': lp.id,'lower': lower,'upper': upper,'L': L_new})

            pool.recompute_active_L()
            if -1e-9 < pool.L_active < 0.0:
                pool.L_active = 0.0

        def act_smart_router():
            execute_trader("smart", p_trade_micro, sr_acc, True)

        def act_noise_trader():
            execute_trader("noise", noise_floor, noise_acc, False)

        def act_arbitrageur():
            nonlocal arb_y_this, L_pre_arb_eff_this, dir_arb_this, delta_a_cex_this, _arb_execs

            in_used, x_out_from_dex, y_out_from_dex, dir_arb, fee_x_arb, fee_y_arb, L_first = arbitrage_to_target(arb_ref_m)
            delta_a_cex_this = 0.0
            if in_used > 0 and dir_arb is not None:
                L_pre_arb_eff_this = L_first
                dir_arb_this = dir_arb
                arb_steps.append(t); arb_dirs.append(dir_arb)

                if dir_arb == "up":
                    # DEX cheap: buy A on DEX (A out), sell A on CEX @ m_now
                    delta_a_cex_this = -x_out_from_dex
                    arb_y_this = +in_used
                    arb_acc.record_swap(dy_in=in_used, dx_out=x_out_from_dex)
                    _arb_execs += int(in_used > 0)
                    # Fees already allocated per span via fee_cb
                else:
                    # DEX expensive: sell A on DEX (A in), buy A on CEX @ m_now
                    delta_a_cex_this = +in_used
                    arb_y_this = -pool.price * in_used
                    arb_acc.record_swap(dx_in=in_used, dy_out=y_out_from_dex)
                    _arb_execs += int(in_used > 0)
                    # Fees already allocated per span via fee_cb

        # --- async LP micro-scheduler: A → trader → B → arb → C ---

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

        # ===================== BLOCK SCHEDULING & ORDER =====================
        # Non-block mode (block_size == 1): A -> Smart+Noise -> B -> Arb -> C.
        # Block mode (block_size > 1):
        #   - Snapshot CEX at block start: arb_ref_m
        #   - Micro-steps: diffuse-only; enqueue intents with p_trade_micro/noise_floor_micro
        #   - Boundary order (current): Arb -> (populate+execute mempool)  (LPs act via mempool)
        # =====================================================================
        # run the schedule
        if block_size == 1:
            # target band uses current ref.m (end-of-step ≈ current) in non-block mode
            band_lo_target.append(ref.m * r)
            band_hi_target.append(ref.m / r)
            _enable(bucketA)
            act_LPs()

            _enable(bucketB)
            act_smart_router()
            act_noise_trader()
            act_LPs()

            _enable(bucketC)
            act_arbitrageur()
            act_LPs()
        else:
            arb_ref_m_start = ref.m  # block-start CEX snapshot (diagnostic only; arb targets end-of-block)
            # prepare micro-time arrays: keep DEX price stale within the block
            _micro_start = len(P_micro)
            with open(verbose_log_path_str, 'a') as f:
                f.write(f"[t={t:03d}] BLOCK start m={arb_ref_m_start:.4f}\n")
            P_micro.extend([pool.price] * block_size)
            micro_steps.extend([t * block_size + k for k in range(block_size)])
            for _k in range(block_size):
                maybe_enqueue_smart_router_intent(ref.m)
                maybe_enqueue_noise_trader_intent(ref.m)
                ref.diffuse_only()
                _broadcast_price_move(ref.m)
                M_micro.append(ref.m)

            # --- Arbitrage before mempool execution ---
            arb_ref_m = ref.m  # end-of-block CEX price (post-diffusion, pre-arb impact)
            band_lo_target.append(arb_ref_m * r)
            band_hi_target.append(arb_ref_m / r)
            act_arbitrageur()

            # --- Include LP intents in the mempool (shuffled with traders) ---
            # Allow due LPs to act this block
            _enable(due)
            # Burns (TP/SL)
            for lp_idx in due:
                lp = LPs[lp_idx]
                if not lp.can_act:
                    continue
                if lp.is_passive:
                    if lp.positions and random.random() < passive_burn_prob:
                        pos = lp.positions[-1]
                        mempool_orders.append({'type':'lp_burn','lp_id': lp.id,'lower': pos.lower,'upper': pos.upper,'L': pos.L})
                    continue
                to_burn = []
                for i, pos in enumerate(lp.positions):
                    pnl = pos.PnL_y(pool.S, ref.m)
                    if pnl >= theta_TP * pos.hodl0_value_y or pnl <= -theta_SL * pos.hodl0_value_y:
                        to_burn.append(i)
                for i in reversed(to_burn):
                    pos = lp.positions[i]
                    mempool_orders.append({'type':'lp_burn','lp_id': lp.id,'lower': pos.lower,'upper': pos.upper,'L': pos.L})

            # Recenter (narrow LPs that have been out-of-range for >= k_out)
            for lp_idx in due:
                lp = LPs[lp_idx]
                if not lp.can_act:
                    continue
                to_recenters = []
                for i, pos in enumerate(lp.positions):
                    in_rng = pos.in_range(pool.tick)
                    out_steps = getattr(pos, "out_steps", 0)
                    out_steps = 0 if in_rng else out_steps + 1
                    setattr(pos, "out_steps", out_steps)
                    if lp.is_active_narrow and out_steps >= k_out:
                        to_recenters.append(i)
                for i in reversed(to_recenters):
                    pos = lp.positions[i]
                    width_ticks = pos.upper - pos.lower
                    n_bands = max(1, int(round(width_ticks / pool.tick_spacing)))
                    S_now = pool.S
                    sps = pool.tick_spacing
                    nb = n_bands
                    denom = (1.0 + (pool.g ** (nb * sps + sps)))
                    if denom <= 0.0:
                        denom = 1.0
                    lower_real = math.log((2.0 * S_now / pool.base_s) / denom, pool.g)
                    lower = pool._snap(int(round(lower_real)))
                    upper = lower + nb * sps
                    mempool_orders.append({'type':'lp_recenter','lp_id': lp.id,
                                           'old_lower': pos.lower,'old_upper': pos.upper,'old_L': pos.L,
                                           'new_lower': lower,'new_upper': upper,'new_L': pos.L})

            # New mints (probabilistic; respect budget/cooldown)
            for lp_idx in due:
                lp = LPs[lp_idx]
                if not lp.can_act or getattr(lp, "cooldown", 0) > 0:
                    continue
                if lp.is_passive:
                    if lp.positions or random.random() >= passive_mint_prob:
                        continue
                    width_ticks = max(passive_width_ticks, pool.tick_spacing)
                    n_bands = max(1, int(round(width_ticks / pool.tick_spacing)))
                else:
                    if random.random() >= lp.mintProb:
                        continue
                    n_bands = max(1, int(round(w_ticks / pool.tick_spacing)))
                X = abs(np.random.normal(mint_mu, mint_sigma))
                try:
                    _L_SCALE = L_SCALE
                except NameError:
                    _L_SCALE = initial_total_L / max(1, N_LP)
                want = X * _L_SCALE
                cap_step = 0.25 * getattr(lp, 'L_budget', want)
                cap_left = max(0.0, getattr(lp, 'L_budget', want) - getattr(lp, 'L_live', 0.0))
                L_new = max(0.0, min(want, cap_step, cap_left))
                if L_new <= 0.0:
                    continue
                S_now = pool.S
                sps = pool.tick_spacing
                nb = n_bands
                denom = (1.0 + (pool.g ** (nb * sps + sps)))
                if denom <= 0.0:
                    denom = 1.0
                lower_real = math.log((2.0 * S_now / pool.base_s) / denom, pool.g)
                lower = pool._snap(int(round(lower_real)))
                upper = lower + nb * sps
                if upper <= lower:
                    upper = lower + pool.tick_spacing
                mempool_orders.append({'type':'lp_mint','lp_id': lp.id,'lower': lower,'upper': upper,'L': L_new})

            # Execute all mempool intents (traders + LPs) in random order
            L_pre_trader_this = pool.L_active
            execute_mempool_orders()

            # update last micro-sample of DEX price to reflect mempool executions
            if block_size > 1:
                P_micro[_micro_start + block_size - 1] = pool.price
        # disable everyone for next step
        _enable([])

        # ---- CEX update  ----
        if block_size == 1:
            ref.step(delta_a_cex_this)
        else:
            ref.apply_impact_only(delta_a_cex_this)
        _broadcast_price_move(ref.m)

        settlement_m = ref.m
        sr_acc.settle(settlement_m)
        noise_acc.settle(settlement_m)
        trader_pnl_this = sr_acc.pnl + noise_acc.pnl
        arb_acc.settle(settlement_m)
        arb_pnl_this = arb_acc.pnl

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
        _val_x = x_e * pool.price
        _val_y = y_e
        _den = max(1e-12, (_val_x + _val_y))
        fee_imb_series.append((_val_y - _val_x) / _den)

        band_lo_post.append(ref.m * r)
        band_hi_post.append(ref.m / r)
        L_end.append(pool.L_active)
        # ---- PnL bookkeeping ----
        trader_pnl_steps.append(trader_pnl_this)
        arb_pnl_steps.append(arb_pnl_this)
        trader_exec_count.append(_trader_execs)
        arb_exec_count.append(_arb_execs)
        lp_total = 0.0             # cumulative hedged PnL (fees - rebal)
        lp_total_active = 0.0      # active narrow LPs
        lp_total_passive = 0.0     # passive LPs
        lp_total_wide = 0.0        # active wide LPs
        lp_rebal_total = 0.0
        lp_rebal_active = 0.0
        lp_rebal_passive = 0.0
        lp_rebal_wide = 0.0
        lp_wallet_total = 0.0
        lp_wallet_active = 0.0
        lp_wallet_passive = 0.0
        lp_wallet_wide = 0.0
        lp_wealth_total = 0.0
        lp_wealth_active = 0.0
        lp_wealth_passive = 0.0
        lp_wealth_wide = 0.0
        for lp in LPs:
            wallet_y = getattr(lp, 'wallet_y', 0.0)
            lp_wallet_total += wallet_y
            rb = lp.rebalancer
            _ensure_rebalancer_initialized(lp, ref.m, pool.S)
            wealth_now = lp_wealth_y(lp, pool.S, ref.m)
            delta_rebal = rb.cumulative_R - rb.last_cumulative_R
            delta_wealth = wealth_now - rb.last_wealth_y
            hedged_step = delta_wealth - delta_rebal
            rb.hedged_pnl_cum += hedged_step
            rb.last_wealth_y = wealth_now
            rb.last_cumulative_R = rb.cumulative_R
            rb.last_M = ref.m

            lp_total += rb.hedged_pnl_cum
            lp_rebal_total += rb.cumulative_R
            lp_wealth_total += wealth_now

            if lp.is_passive:
                lp_total_passive += rb.hedged_pnl_cum
                lp_rebal_passive += rb.cumulative_R
                lp_wallet_passive += wallet_y
                lp_wealth_passive += wealth_now
            elif lp.is_active_narrow:
                lp_total_active += rb.hedged_pnl_cum
                lp_rebal_active += rb.cumulative_R
                lp_wallet_active += wallet_y
                lp_wealth_active += wealth_now
            else:
                lp_total_wide += rb.hedged_pnl_cum
                lp_rebal_wide += rb.cumulative_R
                lp_wallet_wide += wallet_y
                lp_wealth_wide += wealth_now
        lp_pnl_total_series.append(lp_total)
        lp_pnl_active_series.append(lp_total_active)
        lp_pnl_passive_series.append(lp_total_passive)
        lp_pnl_wide_series.append(lp_total_wide)
        lp_rebal_total_series.append(lp_rebal_total)
        lp_rebal_active_series.append(lp_rebal_active)
        lp_rebal_passive_series.append(lp_rebal_passive)
        lp_rebal_wide_series.append(lp_rebal_wide)

        # Wealth accounting (wallet + open PnL)
        lp_wallet_series.append(lp_wallet_total)
        lp_wallet_active_series.append(lp_wallet_active)
        lp_wallet_passive_series.append(lp_wallet_passive)
        lp_wallet_wide_series.append(lp_wallet_wide)
        lp_wealth_series.append(lp_wealth_total)
        lp_wealth_active_series.append(lp_wealth_active)
        lp_wealth_passive_series.append(lp_wealth_passive)
        lp_wealth_wide_series.append(lp_wealth_wide)
        # store per-step trader/arb details (now that order is randomized)
        trader_y_series.append(trader_y_this)
        arb_y_series.append(arb_y_this)
        L_pre_trader.append(L_pre_trader_this)

        sr_y_series.append(sr_acc.notional_y)
        noise_y_series.append(noise_acc.notional_y)
        sr_pnl_steps.append(sr_acc.pnl)
        noise_pnl_steps.append(noise_acc.pnl)
        sr_exec_count.append(sr_acc.execs)
        noise_exec_count.append(noise_acc.execs)
        L_pre_arb_eff.append(L_pre_arb_eff_this)

        price_moved = abs(P_after - P_before) > EPS_PRICE_CHANGE
        had_fill = (abs(trader_y_this) > 0) or (abs(arb_y_this) > 0)
        had_L_event = (t in mint_steps) or (t in burn_steps)
        if price_moved and not (had_fill or had_L_event):
            raise RuntimeError(
                f"DEX price changed at t={t} without swap or LP ΔL. L_active={pool.L_active:.4f}. Change {abs(P_after - P_before)}"
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

    summary_lines = [
        "# Run summary",
        f"total_mints = {len(mint_steps)}",
        f"total_burns = {len(burn_steps)}",
        f"total_noise_trader_swaps = {total_noise_swaps_executed}",
        f"noise_trader_swaps_rejected_slippage = {total_noise_swaps_skipped}",
        f"total_smart_router_swaps = {total_smart_swaps_executed}",
        f"smart_router_swaps_rejected_slippage = {total_smart_swaps_skipped}",
        "----------------------------------------------------------\n",
    ]

    verbose_path = Path(verbose_log_path_str)
    try:
        original_text = verbose_path.read_text()
    except FileNotFoundError:
        original_text = ""
    verbose_path.write_text("\n".join(summary_lines) + original_text)

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
    sr_y_series = np.array(sr_y_series)
    noise_y_series = np.array(noise_y_series)
    lp_pnl_total_series = np.array(lp_pnl_total_series)
    lp_pnl_active_series = np.array(lp_pnl_active_series)
    lp_pnl_passive_series = np.array(lp_pnl_passive_series)
    lp_pnl_wide_series = np.array(lp_pnl_wide_series)
    lp_rebal_total_series = np.array(lp_rebal_total_series)
    lp_rebal_active_series = np.array(lp_rebal_active_series)
    lp_rebal_passive_series = np.array(lp_rebal_passive_series)
    lp_rebal_wide_series = np.array(lp_rebal_wide_series)
    lp_wallet_series = np.array(lp_wallet_series)
    lp_wallet_active_series = np.array(lp_wallet_active_series)
    lp_wallet_passive_series = np.array(lp_wallet_passive_series)
    lp_wallet_wide_series = np.array(lp_wallet_wide_series)
    lp_wealth_series = np.array(lp_wealth_series)
    lp_wealth_active_series = np.array(lp_wealth_active_series)
    lp_wealth_passive_series = np.array(lp_wealth_passive_series)
    lp_wealth_wide_series = np.array(lp_wealth_wide_series)
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
    lp_wealth_wide_series_v = lp_wealth_wide_series[s0:]
    lp_pnl_total_series_v = lp_pnl_total_series[s0:]
    lp_pnl_active_series_v = lp_pnl_active_series[s0:]
    lp_pnl_passive_series_v = lp_pnl_passive_series[s0:]
    lp_pnl_wide_series_v = lp_pnl_wide_series[s0:]
    lp_rebal_total_series_v = lp_rebal_total_series[s0:]
    lp_rebal_active_series_v = lp_rebal_active_series[s0:]
    lp_rebal_passive_series_v = lp_rebal_passive_series[s0:]
    lp_rebal_wide_series_v = lp_rebal_wide_series[s0:]
    fee_series_v = fee_series[s0:]
    fee_sigma_series_v = fee_sigma_series[s0:]
    fee_basis_ticks_series_v = fee_basis_ticks_series[s0:]
    fee_imb_series_v = fee_imb_series[s0:]
    fee_signal_series_v = fee_signal_series[s0:]
    arb_y_v = np.array(arb_y_series)[s0:]
    sr_y_v = sr_y_series[s0:]
    noise_y_v = noise_y_series[s0:]
    
    if visualize:
        # ΔL per step (aggregate)
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
        _png_dir = _out_dir / "png"
        _html_dir = _out_dir / "html"
        _png_dir.mkdir(parents=True, exist_ok=True)
        _html_dir.mkdir(parents=True, exist_ok=True)
        _prefix = f"abm_fee_{fee_mode}_{cex_sigma}"

        def _scale_marker_sizes(values: List[float], min_pts: float = 4.0, max_pts: float = 16.0) -> List[float]:
            if not values:
                return []
            cleaned = [max(float(v), 0.0) for v in values]
            if not any(cleaned):
                return [min_pts] * len(cleaned)
            # Dampen extremes with sqrt scaling
            dampened = [math.sqrt(v) for v in cleaned]
            max_dampened = max(dampened)
            if max_dampened <= 0:
                return [min_pts] * len(cleaned)
            return [
                min_pts + (max_pts - min_pts) * (dv / max_dampened)
                for dv in dampened
            ]

        def _plot_variable_markers(
            ax: plt.Axes,
            xs: List[int],
            ys: List[float],
            values: List[float],
            marker: str,
            color: str,
            label: Optional[str] = None,
            facecolor: Optional[str] = None,
            edgecolor: Optional[str] = None,
            min_size: float = 1.0,
            max_size: float = 4.0,
        ) -> None:
            if not xs:
                return
            sizes_pts = _scale_marker_sizes(values, min_size, max_size)
            grouped: Dict[float, Tuple[List[int], List[float]]] = {}
            for x_val, y_val, size in zip(xs, ys, sizes_pts):
                size_key = round(size, 2)
                if size_key not in grouped:
                    grouped[size_key] = ([], [])
                grouped[size_key][0].append(x_val)
                grouped[size_key][1].append(y_val)
            for idx, (size_key, (gx, gy)) in enumerate(grouped.items()):
                use_label = label if idx == 0 else None
                ax.plot(
                    gx,
                    gy,
                    linestyle="None",
                    marker=marker,
                    markersize=size_key,
                    markerfacecolor=facecolor if facecolor is not None else color,
                    markeredgecolor=edgecolor if edgecolor is not None else color,
                    label=use_label,
                )

        # Common helper to save + tidy
        def _save_fig(fig, name):
            fig.tight_layout()
            png_path = _png_dir / f"{_prefix}_{name}.png"
            fig.savefig(png_path, dpi=150)
            html_path = _html_dir / f"{_prefix}_{name}.html"
            _save_html(fig, html_path, "simulate")
            plt.close(fig)
        
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
            up_vals = [abs(arb_y_series[s]) for s in up]
            up_xs = arr.tolist()
            up_ys = [P_series[s] + 1.00 * off_y[s] for s in up]
            _plot_variable_markers(
                ax,
                up_xs,
                up_ys,
                up_vals,
                marker="^",
                color="green",
                label="Arb (↑ to target)",
                min_size=1.0,
                max_size=4.0,
            )
        if dn:
            arr = np.array(dn, dtype=int)
            dn_vals = [abs(arb_y_series[s]) for s in dn]
            dn_xs = arr.tolist()
            dn_ys = [P_series[s] + 1.60 * off_y[s] for s in dn]
            _plot_variable_markers(
                ax,
                dn_xs,
                dn_ys,
                dn_vals,
                marker="v",
                color="red",
                label="Arb (↓ to target)",
                min_size=1.0,
                max_size=4.0,
            )

        # LP markers (circles/crosses), stacked higher than arb
        if len(mint_steps) + len(burn_steps) > 0:
            if mint_steps:
                mint_points = [(step, size) for step, size in zip(mint_steps, mint_sizes) if step >= s0]
                if mint_points:
                    mint_xs = [step for step, _ in mint_points]
                    mint_vals = [size for _, size in mint_points]
                    mint_ys = [P_series[step] + 2.40 * off_y[step] for step in mint_xs]
                    _plot_variable_markers(
                        ax,
                        mint_xs,
                        mint_ys,
                        mint_vals,
                        marker="o",
                        color="#6a0dad",
                        facecolor="none",
                        edgecolor="#6a0dad",
                        label="LP mint/center",
                        min_size=1.0,
                        max_size=4.0,
                    )
            if burn_steps:
                burn_points = [(step, size) for step, size in zip(burn_steps, burn_sizes) if step >= s0]
                if burn_points:
                    burn_xs = [step for step, _ in burn_points]
                    burn_vals = [size for _, size in burn_points]
                    burn_ys = [P_series[step] + 3.20 * off_y[step] for step in burn_xs]
                    _plot_variable_markers(
                        ax,
                        burn_xs,
                        burn_ys,
                        burn_vals,
                        marker="x",
                        color="#ff8c00",
                        label="LP burn",
                        min_size=1.0,
                        max_size=4.0,
                    )

        ax.set_xlim(steps_v[0]-0.5, steps_v[-1]+0.5)
        ax.set_ylabel("Price (token1 per token0)", fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("Step", fontsize=LABEL_FONT_SIZE)
        ax.set_title("CEX vs DEX Price", fontsize=TITLE_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        # Show target band (start-of-block) as dotted lines so you can see what arb targets
        band_lo_target_v = np.array(band_lo_target)[s0:]
        band_hi_target_v = np.array(band_hi_target)[s0:]
        ax.plot(steps_v, band_lo_target_v, linestyle='--', linewidth=1.0, alpha=0.6,
                label='Target band (end-of-block) lo')
        ax.plot(steps_v, band_hi_target_v, linestyle='--', linewidth=1.0, alpha=0.6,
                label='Target band (end-of-block) hi')
        ax.legend(ncol=2, fontsize=LEGEND_FONT_SIZE)
        ax.margins(y=0.14)
        _save_fig(fig1, "1_price")

        # ----- 1b) Micro-time price panel (only meaningful if block_size>1) -----
        if block_size > 1 and len(M_micro) == len(P_micro) == len(micro_steps) and len(micro_steps) > 0:
            fig1b, ax = plt.subplots(figsize=(15, 3.2))
            ax.plot(micro_steps, P_micro, label="DEX price (micro)", linewidth=1.2)
            ax.plot(micro_steps, M_micro, "--", label="CEX price (micro)", linewidth=1.0)
            ax.set_title("Micro-time CEX vs DEX (within blocks)", fontsize=TITLE_FONT_SIZE-2)
            ax.set_xlabel("Micro step", fontsize=LABEL_FONT_SIZE-1)
            ax.set_ylabel("Price", fontsize=LABEL_FONT_SIZE-1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=LEGEND_FONT_SIZE-1)
            _save_fig(fig1b, "1b_price_micro")
        # ----- 2) Notionals -----
        fig2, ax = plt.subplots(figsize=(15, 3.6))
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.3)
        ax.plot(steps_v, sr_y_v, lw=1.8, label="Smart router notional (token1, signed)")
        ax.plot(steps_v, noise_y_v, lw=1.8, linestyle="--", label="Noise trader notional (token1, signed)")
        ax.plot(steps_v, arb_y_v, lw=1.8, linestyle="-.", label="Arbitrageur notional (token1, signed)")
        ax.set_ylabel("Notional (token1, signed)", fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left")
        _save_fig(fig2, "2_notional")

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

        # ----- 4) L per step -----
        fig4, ax = plt.subplots(figsize=(15, 3.6))
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.3)
        ax.bar(steps_v, mint_step_sum_v, width=0.8, alpha=0.6, label="LP mint/center L (>0)", color="#6a0dad")
        ax.bar(steps_v, -burn_step_sum_v, width=0.8, alpha=0.6, label="LP burn L (<0)", color="#ff8c00")
        ax.set_ylabel("L per step", fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left")
        _save_fig(fig4, "4_L_per_step")

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

        # ----- 6) PnL panel -----
        fig6, ax = plt.subplots(figsize=(15, 3.6))
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.3)
        ax.plot(steps_v, sr_pnl_cum_v, lw=1.8, label="Smart Router cumulative PnL (token1)")
        ax.plot(
            steps_v,
            noise_pnl_cum_v,
            lw=1.5,
            linestyle="--",
            color="#2ca02c",
            label="Noise trader cumulative PnL (token1)",
        )
        ax.plot(steps_v, arb_pnl_cum_v, lw=1.8, label="Arbitrageur cumulative PnL (token1)")
        # ax.plot(
        #     steps_v,
        #     lp_pnl_total_series_v,
        #     lw=1.8,
        #     color="#8c564b",
        #     label="LP cumulative hedged PnL (token1)",
        # )
        # ax.plot(
        #     steps_v,
        #     lp_rebal_total_series_v,
        #     lw=1.2,
        #     linestyle="--",
        #     color="#bcbd22",
        #     label="LP cumulative rebal PnL (token1)",
        # )
        ax.plot(
            steps_v,
            lp_pnl_active_series_v,
            lw=1.5,
            linestyle="--",
            color="#8c564b",
            label="Active narrow LP Fee-LVR",
        )
        if active_wide_lp_enabled and np.any(np.abs(lp_pnl_wide_series_v) > 1e-12):
            ax.plot(
                steps_v,
                lp_pnl_wide_series_v,
                lw=1.5,
                linestyle="-.",
                color="#bcbd22",
                label="Active wide LP Fee-LVR",
            )
        ax.plot(
            steps_v,
            lp_pnl_passive_series_v,
            lw=1.5,
            linestyle=":",
            color="#9467bd",
            label="Passive LP Fee-LVR",
        )
        # ax.plot(
        #     steps_v,
        #     lp_wealth_active_series_v,
        #     lw=1.5,
        #     linestyle="--",
        #     label="Active narrow LP wealth (wallet+open)",
        # )
        # if active_wide_lp_enabled and np.any(np.abs(lp_wealth_wide_series_v) > 1e-12):
        #     ax.plot(
        #         steps_v,
        #         lp_wealth_wide_series_v,
        #         lw=1.5,
        #         linestyle="-.",
        #         label="Active wide LP wealth (wallet+open)",
        #     )
        # ax.plot(
        #     steps_v,
        #     lp_wealth_passive_series_v,
        #     lw=1.5,
        #     linestyle=":",
        #     label="Passive LP wealth (wallet+open)",
        # )
        ax.set_ylabel("Token1 value", fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("Step", fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc="upper left")
        _save_fig(fig6, "6_pnl")

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
        "lp_pnl_wide": lp_pnl_wide_series.tolist(),
        "lp_rebal_total_series": lp_rebal_total_series.tolist(),
        "lp_rebal_active_series": lp_rebal_active_series.tolist(),
        "lp_rebal_passive_series": lp_rebal_passive_series.tolist(),
        "lp_rebal_wide_series": lp_rebal_wide_series.tolist(),
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
        "lp_wallet_wide_series": lp_wallet_wide_series.tolist(),
        "lp_wealth_active_series": lp_wealth_active_series.tolist(),
        "lp_wealth_passive_series": lp_wealth_passive_series.tolist(),
        "lp_wealth_wide_series": lp_wealth_wide_series.tolist(),
        "arb_exec_count": arb_exec_count,
        "active_wide_lp_enabled": active_wide_lp_enabled,
    }


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ABM Uni v3 simulation.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file containing a complete 'simulate' section.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    scenario_label, params = load_simulation_parameters(config_path, simulate_func=simulate)

    print(f"[config] {config_path}")
    print(f"[scenario] {scenario_label}")

    out = simulate(**params)

    skip_steps = int(params.get("skip_step", 0))
    dex_prices = np.array(out["DEX_price"][skip_steps:])
    dex_returns = np.diff(np.log(dex_prices))

    max_lag = 15
    autocorr = [np.corrcoef(dex_returns[:-lag], dex_returns[lag:])[0, 1] for lag in range(1, max_lag + 1)]

    # Plot autocorrelation of DEX log-returns
    lags = np.arange(1, max_lag + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lags, autocorr, width=0.6, color="#1f77b4")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("DEX Log-Return Autocorrelation")
    ax.set_xticks(lags)
    ax.grid(True, axis="y", alpha=0.3)

    results_root = Path("abm_results")
    png_dir = results_root / "png"
    html_dir = results_root / "html"
    png_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png_path = png_dir / f"autocorr_{scenario_label}.png"
    fig.savefig(png_path, dpi=150)
    html_path = html_dir / f"autocorr_{scenario_label}.html"
    _save_html(fig, html_path, "autocorr")
    plt.close(fig)

    # make liquidity GIF
    # make_liquidity_gif(
    # liq_history=out["liq_history"],
    # tick_history=out["tick_history"],
    # base_s=out["grid_base_s"],
    # g=out["grid_g"],
    # out_path=f"abm_results/liquidity_evolution_{fee_mode}.gif",
    # fps=20,
    # dpi=120,
    # pad_frac=0.05,
    # downsample_every=5,
    # center_line=True,
    # )
