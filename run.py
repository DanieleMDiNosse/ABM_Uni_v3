"""
Main simulation runner for the ABM model.
X (token0) is like ETH and Y (token1) is like USDC.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import inspect
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional, Callable, Set
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY_STATIC_WARNING_EMITTED = False
_DEFAULT_GRID_STYLE = dict(showgrid=True, gridcolor="#e1e1e1", gridwidth=1)


def save_plotly_figure(
    fig: go.Figure,
    png_path: Path,
    html_path: Path,
    source: str = "plot",
    *,
    width: int = 1400,
    height: int = 900,
    scale: float = 1.0,
) -> None:
    global PLOTLY_STATIC_WARNING_EMITTED
    """Persist a Plotly figure as both HTML and PNG (if Kaleido is available)."""
    fig.update_xaxes(**_DEFAULT_GRID_STYLE)
    fig.update_yaxes(**_DEFAULT_GRID_STYLE)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        fig.write_image(str(png_path), width=width, height=height, scale=scale)
    except Exception as exc:  # pragma: no cover - depends on kaleido availability
        if not PLOTLY_STATIC_WARNING_EMITTED:
            print(f"[{source}] Warning: could not export Plotly PNGs ({exc})")
            PLOTLY_STATIC_WARNING_EMITTED = True

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
from collections import defaultdict

from uniswapv3_pool import V3Pool


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
    block_time: int,
    T: int,
    seed: int,
    cex_mu: float,
    cex_sigma: float,
    p_trade: float, 
    noise_floor: float,
    p_lp_narrow: float,
    passive_lp_share: float,
    passive_mint_prob: float,
    passive_burn_prob: float,
    passive_width_ticks: int,
    N_LP: int,
    tau: int,
    # --- LP width via EWMA(B_t) + binomial noise ---
    w_min_ticks: int,
    w_max_ticks: int,
    basis_half_life: int,   # steps
    slope_s: float,        # ticks per (basis-in-ticks)
    # --- binomial noise parameters (already present; now used) ---
    binom_n: int,
    binom_p: float,
    # --- lognormal noise parameters (new; not yet used) ---
    trader_mean: float,
    trader_sigma: float,
    theta_T: float,
    # --- slippage ---
    slippage_tolerance: float,
    # --- other params ---
    mint_mu: float,
    mint_sigma: float,
    theta_TP: float,
    theta_SL: float,
    initial_binom_N: int,
    initial_total_L: float,
    k_out_min: int,
    k_out_max: int,
    visualize: bool,
    skip_step: int,
    # --- Dynamic fee controller (new) ---
    fee_mode: str,      # "static" | "volatility" 
    f0: float,             # baseline fee (e.g., 30 bps)
    f_min: float,         # 5 bps
    f_max: float,           # 200 bps safety cap
    fee_half_life: int,       # EWMA half-life (steps) for signals
    k_sigma: float,         # adds ~k_sigma * EWMA(|logret|) to fee
    k_basis: float,         # fee per tick of dislocation (basis in ticks)
    # k_imb: float = 0.002,          # fee += k_imb * |imbalance|, imbalance in [0,1]
    fee_step_bps_min: float, # do not change fee unless ≥ 0.5 bps move
    fee_step_bps_max: float, # max step per update (bps)
    fee_cooldown: int,         # blocks between fee changes (hysteresis)
):
    valid_fee_modes = {"static", "volatility", "toxicity"}
    if fee_mode not in valid_fee_modes:
        raise ValueError(f"Invalid fee_mode '{fee_mode}'. Expected one of {sorted(valid_fee_modes)}.")
    if k_out_min <= 0 or k_out_max <= 0:
        raise ValueError("k_out_min and k_out_max must be positive integers.")
    if k_out_min > k_out_max:
        raise ValueError("k_out_min cannot exceed k_out_max.")

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

    # --- Build pool + reference market + LP agents ----------------------------
    pool, m0 = build_empty_pool()
    ref = ReferenceMarket(m=m0, mu=cex_mu, sigma=cex_sigma, kappa=1e-3)

    LPs: List[LPAgent] = []
    for i in range(N_LP):
        r = random.random()
        is_passive = r < passive_share
        is_narrow = not is_passive
        mintProb = passive_mint_prob if is_passive else p_lp_narrow
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
        lp.k_out_threshold = random.randint(k_out_min, k_out_max)

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
        plot=False,
        seed_is_passive=True,
    )

    # ensure budgets exist for every LP, including the just-appended seed
    for lp in LPs:
        if lp.L_budget <= 0.0:
            lp.L_budget = 2.0 * L_SCALE
        if lp.L_live < 0.0:
            lp.L_live = 0.0
        if not hasattr(lp, "k_out_threshold"):
            lp.k_out_threshold = random.randint(k_out_min, k_out_max)

    lp_lookup: Dict[int, LPAgent] = {lp.id: lp for lp in LPs}
    positions_by_tick: Dict[int, List[Position]] = defaultdict(list)

    def _register_position(pos: Position) -> None:
        slots = tuple(range(pos.lower, pos.upper, pool.tick_spacing))
        pos.tick_slots = slots
        for tick_val in slots:
            positions_by_tick[tick_val].append(pos)

    def _unregister_position(pos: Position) -> None:
        for tick_val in getattr(pos, "tick_slots", ()):
            bucket = positions_by_tick.get(tick_val)
            if not bucket:
                continue
            try:
                bucket.remove(pos)
            except ValueError:
                continue
            if not bucket:
                positions_by_tick.pop(tick_val, None)
        pos.tick_slots = tuple()

    def _assert_active_liquidity_state(label: str) -> None:
        """Runtime guard to ensure pool.L_active agrees with liquidity_net."""
        prefix_L = pool.bidx.active_liquidity_at_tick(pool.tick)

        # If both representations are numerically tiny, snap to zero and skip.
        if abs(pool.L_active) <= EPS_LIQ2 and abs(prefix_L) <= EPS_LIQ2:
            pool.L_active = 0.0
            return

        # Treat significantly negative active liquidity as a real bug.
        underflow_tol = 100.0 * EPS_LIQ2
        if pool.L_active < -underflow_tol:
            raise AssertionError(
                f"L_active underflow ({label}): {pool.L_active}"
            )

        # Require close agreement between cached and prefix-sum views.
        tolerance = max(
            underflow_tol,
            1e-9 * max(1.0, abs(prefix_L), abs(pool.L_active)),
        )
        if abs(prefix_L - pool.L_active) > tolerance:
            raise AssertionError(
                f"L_active mismatch ({label}) tick={pool.tick} active={pool.L_active} prefix={prefix_L}"
            )

        # If we have meaningful active liquidity, ensure price lies inside the band.
        if pool.L_active > EPS_LIQ:
            sa = pool.s_lower()
            sb = pool.s_upper()
            band_scale = max(1.0, abs(sa), abs(sb), abs(pool.S))
            boundary_tol = EPS_BOUNDARY * band_scale
            if pool.S < sa - boundary_tol or pool.S > sb + boundary_tol:
                raise AssertionError(
                    f"Price S={pool.S} outside active band during {label}: tick={pool.tick} band=[{sa},{sb}]"
                )
            # Snap tiny floating-point drift so pool.S stays on the boundary.
            if pool.S < sa:
                pool.S = sa
            elif pool.S > sb:
                pool.S = sb

    for lp in LPs:
        for pos in lp.positions:
            _register_position(pos)

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
    mint_is_passive: List[bool] = []
    burn_is_passive: List[bool] = []
    mint_widths = []
    w_ticks_series: List[int] = []
    w_unclipped_series: List[float] = []
    w_noise_series: List[float] = []
    liq_history: List[Dict[int, float]] = []
    tick_history: List[int] = []
    delta_a_cex_series = []
    # --- Block-start target band (arb_ref_m) ---
    band_lo_target, band_hi_target = [], []
    # --- Micro-time traces (for block_time > 1 visualization) ---
    micro_steps, M_micro, P_micro = [], [], []
    micro_valid_steps, micro_valid_prices = [], []
    micro_counter = 0
    # --- PnL recorders ---
    trader_pnl_steps = []       # realized per-step PnL (token1)
    arb_pnl_steps = []          # realized per-step PnL (token1)
    lp_pnl_total_series = []    # cumulative hedged PnL (fees - rebal) across all LPs
    lp_pnl_active_series = []   # cumulative hedged PnL for active (narrow) LPs
    lp_pnl_passive_series = []  # cumulative hedged PnL for passive LPs
    lp_rebal_total_series = []  # cumulative rebalancing PnL (benchmark) across LPs
    lp_rebal_active_series = []
    lp_rebal_passive_series = []
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
    os.makedirs("abm_results/logs", exist_ok=True)
    verbose_log_path = next_numbered_path(Path(f"abm_results/logs/verbose_steps_{fee_mode}"))
    verbose_log_path_str = str(verbose_log_path)

    verbose_log = open(verbose_log_path_str, "a")
    LOG_BUFFER_LIMIT = 10_000
    log_buffer: List[str] = []

    def buffer_log(msg: str) -> None:
        """Accumulate log lines before flushing to disk."""
        log_buffer.append(msg)
        if len(log_buffer) >= LOG_BUFFER_LIMIT:
            verbose_log.write("".join(log_buffer))
            log_buffer.clear()

    def flush_log_buffer() -> None:
        """Write the buffered log entries to disk."""
        if log_buffer:
            verbose_log.write("".join(log_buffer))
            log_buffer.clear()

    buffer_log("# Simulation parameters\n")
    for key in sorted(initial_params):
        buffer_log(f"{key} = {initial_params[key]}\n")
    buffer_log("\n")


    # --- LP wealth recorders (new) ---
    lp_wallet_series = []      # realized wallet (token1)
    lp_wealth_series = []      # wallet + open PnL (token1)
    lp_wallet_active_series = []   # active narrow LPs
    lp_wallet_passive_series = []  # passive LPs
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

    def _rebalance_by_ids(lp_ids: Set[int], M_now: float, S_now: float) -> None:
        if not lp_ids:
            return
        for lp_id in lp_ids:
            lp = lp_lookup.get(lp_id)
            if lp is not None:
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
        if fee_amt <= 0:
            return
        bucket = positions_by_tick.get(tick_snapshot)
        if not bucket:
            return
        total_L = math.fsum(pos.L for pos in bucket if pos.L > 0.0)
        if total_L <= 0.0:
            return
        touched_lp_ids: Set[int] = set()
        for pos in bucket:
            share = pos.L / total_L
            if share <= 0.0:
                continue
            if token == "x":
                delta_fee0 = share * fee_amt
                pos.fees0 += delta_fee0
                if delta_fee0 != 0.0:
                    touched_lp_ids.add(pos.owner)
            else:
                pos.fees1 += share * fee_amt
                if fee_amt != 0.0:
                    touched_lp_ids.add(pos.owner)
        if touched_lp_ids:
            _rebalance_by_ids(touched_lp_ids, ref.m, pool.S)

    def burn_any(lp: LPAgent, idx: int) -> None:
        pos = lp.positions.pop(idx)
        # Realize PnL into LP wallet at burn time (fees + IL vs floating HODL)
        realized_y = pos.PnL_y(pool.S, ref.m)
        lp.wallet_y = getattr(lp, 'wallet_y', 0.0) + float(realized_y)
        _unregister_position(pos)
        pool.add_liquidity_range(pos.lower, pos.upper, -pos.L)
        _assert_active_liquidity_state("lp_burn")

        burn_steps.append(t)
        burn_sizes.append(pos.L)
        burn_is_passive.append(bool(lp.is_passive))

        buffer_log(
            f"[t={t:03d}] LP{lp.id} BURN L={pos.L:.4f} [{pos.lower},{pos.upper}) | "
            f"L_active={pool.L_active:.4f} | tick={pool.tick}\n"
        )

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
        return dx_pre, dy_out, fee_x

    def swap_exact_to_target(target_price: float, direction: str, fee_cb: Optional[Callable[[str, float, int, float], None]] = None) -> Tuple[float, float, float]:
        target_S = math.sqrt(max(1e-18, target_price))

        total_in = total_out = 0.0
        L_first = 0.0
        if pool.L_active <= EPS_LIQ:
            return 0.0, 0.0, 0.0

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
                total_in += dy
                total_out += dx
                if pool.S >= target_S - EPS_BOUNDARY:
                    break
                pool._cross_up_once()
                _assert_active_liquidity_state("swap_exact_to_target:cross_up")
                if pool.L_active <= 0:
                    break
            _assert_active_liquidity_state("swap_exact_to_target:up_end")
            return total_in, total_out, L_first

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
                total_in += dx
                total_out += dy
                if pool.S <= target_S + EPS_BOUNDARY:
                    break
                pool._cross_down_once()
                _assert_active_liquidity_state("swap_exact_to_target:cross_down")
                if pool.L_active <= 0:
                    break
            _assert_active_liquidity_state("swap_exact_to_target:down_end")
            return total_in, total_out, L_first


    def arbitrage_to_target(arb_ref_m: float) -> Tuple[float, float, float, Optional[str], float]:
        """
        Returns:
        in_used        = total input amount into the DEX (dy for 'up', dx for 'down')
        x_out_from_dex = token A out from the DEX (dx_out for 'up'; 0.0 for 'down')
        y_out_from_dex = token B out from the DEX (0.0 for 'up'; dy_out for 'down')
        direction      = 'up' or 'down' or None
        L_first
        """
        P = pool.price
        r = pool.r
        lo, hi = arb_ref_m * r, arb_ref_m / r
        if P < lo * (1 - 1e-9):
            # up: returns (dy_in, dx_out, 0.0, direction, L_first)
            dy_in, dx_out, Lff = swap_exact_to_target(lo, "up", fee_cb=allocate_fees)
            return dy_in, dx_out, 0.0, ("up" if dy_in > 0 else None), Lff
        if P > hi * (1 + 1e-9):
            # down: returns (dx_in, 0.0, dy_out, direction, L_first)
            dx_in, dy_out, Lff = swap_exact_to_target(hi, "down", fee_cb=allocate_fees)
            return dx_in, 0.0, dy_out, ("down" if dx_in > 0 else None), Lff
        return 0.0, 0.0, 0.0, None, 0.0

    # Per-micro-step arrival probabilities (used directly)
    p_trade_micro = p_trade
    noise_floor_micro = noise_floor

    total_noise_swaps_executed = 0
    total_noise_swaps_skipped = 0
    total_smart_swaps_executed = 0
    total_smart_swaps_skipped = 0
    smart_swaps_x_to_y = 0
    smart_swaps_y_to_x = 0
    noise_swaps_x_to_y = 0
    noise_swaps_y_to_x = 0
    # Track the last validated DEX state (end of previous block)
    validated_S = pool.S
    validated_tick = pool.tick
    validated_cex = ref.m
    agent_S_ref = validated_S
    agent_tick_ref = validated_tick
    cex_ref_for_agents = validated_cex
    def _baseline_quote_x_to_y(dx: float) -> float:
        if dx <= 0.0:
            return 0.0
        return dx * pool.r * pool.price

    def _baseline_quote_y_to_x(dy: float) -> float:
        if dy <= 0.0:
            return 0.0
        return (dy * pool.r) / max(pool.price, 1e-18)

    def _draw_trader_notional() -> float:
        """Sample a token1 notional for trader orders (shared across directions)."""
        return float(np.exp(np.random.normal(loc=trader_mean, scale=trader_sigma)))

    # ------------------ Main loop ------------------
    for t in tqdm(range(T), desc="Simulating ABM", unit=" step"):
        agent_S_ref = validated_S
        agent_tick_ref = validated_tick
        cex_ref_for_agents = validated_cex
        arb_ref_m = cex_ref_for_agents  # default snapshot (end of previous block)
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
        w_ticks_series.append(w_ticks)
        w_unclipped_series.append(w_unclipped)
        w_noise_series.append(noise_ticks)
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

        
        # --- Mempool structures (for block_time > 1) ---
        mempool_orders = []
        
        # ----- Non-mutating Uni v3 quotes (spacing-aware, can bridge deserts) -----
        def maybe_enqueue_smart_router_intent(m_now: float):
            """Enqueue a smart-router swap intent if DEX output is competitive vs CEX (theta_T)."""
            if random.random() >= p_trade_micro:
                return
            side = random.choice(["X_to_Y", "Y_to_X"])
            if side == "X_to_Y":
                notional_y = _draw_trader_notional()
                if notional_y <= 0.0:
                    return
                price_snapshot = max(m_now, 1e-18)
                dx = notional_y / price_snapshot
                if dx <= 0.0:
                    return
                initial_quote = pool.quote_x_to_y(dx)
                if initial_quote <= 0.0:
                    return
                # best-ex vs CEX: compare dy_out to dx * m_now (value in token1)
                if initial_quote < theta_T * dx * m_now:
                    return
                baseline_quote = _baseline_quote_x_to_y(dx)
                min_output = max(0.0, baseline_quote * (1.0 - slippage_tolerance))
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
                dy = _draw_trader_notional()
                if dy <= 0.0:
                    return
                initial_quote = pool.quote_y_to_x(dy)
                if initial_quote <= 0.0:
                    return
                # best-ex vs CEX: compare dx_out to dy / m_now (value in token0)
                if initial_quote < theta_T * dy / max(m_now, 1e-18):
                    return
                baseline_quote = _baseline_quote_y_to_x(dy)
                min_output = max(0.0, baseline_quote * (1.0 - slippage_tolerance))
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
                notional_y = _draw_trader_notional()
                if notional_y > 0.0:
                    price_snapshot = max(m_now, 1e-18)
                    dx = notional_y / price_snapshot
                    if dx > 0.0:
                        initial_quote = pool.quote_x_to_y(dx)
                        if initial_quote <= 0.0:
                            return
                        baseline_quote = _baseline_quote_x_to_y(dx)
                        min_output = max(0.0, baseline_quote * (1.0 - slippage_tolerance))
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
                dy = _draw_trader_notional()
                if dy > 0.0:
                    initial_quote = pool.quote_y_to_x(dy)
                    if initial_quote <= 0.0:
                        return
                    baseline_quote = _baseline_quote_y_to_x(dy)
                    min_output = max(0.0, baseline_quote * (1.0 - slippage_tolerance))
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
            nonlocal micro_counter
            P_pre_exec = pool.price
            executed_smart_swaps = 0
            executed_noise_swaps = 0
            executed_lp_events = 0

            def _exec_one(o):
                nonlocal P_pre_exec, trader_y_this, sr_acc, noise_acc
                nonlocal total_noise_swaps_executed, total_noise_swaps_skipped
                nonlocal total_smart_swaps_executed, total_smart_swaps_skipped
                nonlocal smart_swaps_x_to_y, smart_swaps_y_to_x
                nonlocal noise_swaps_x_to_y, noise_swaps_y_to_x
                nonlocal executed_smart_swaps, executed_noise_swaps, executed_lp_events
                nonlocal arb_y_this, L_pre_arb_eff_this, dir_arb_this, delta_a_cex_this, _arb_execs
                P_pre_exec = pool.price
                tick_before_exec = pool.tick
                typ = o.get('type')
                def _record_micro(price_before_local: float) -> None:
                    nonlocal micro_counter
                    if block_time > 1 and abs(pool.price - price_before_local) > EPS_PRICE_CHANGE:
                        micro_steps.append(micro_counter)
                        P_micro.append(pool.price)
                        M_micro.append(ref.m)
                        micro_counter += 1

                # Handle LP intents (they don't have 'agent' or 'side')
                if typ in ('lp_burn','lp_mint','lp_recenter'):
                    lp = lp_lookup.get(o.get('lp_id'))
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
                        executed_lp_events += 1
                        return
                    if typ == 'lp_mint':
                        lower = int(o.get('lower')); upper = int(o.get('upper')); L_new = float(o.get('L', 0.0))
                        if upper <= lower or L_new <= 0.0:
                            return
                        sa, sb = pool.s_lower(lower), pool.s_upper(upper)
                        amt0, amt1 = minted_amounts_at_S(L_new, sa, sb, agent_S_ref)
                        pos = Position(owner=lp.id, lower=lower, upper=upper, L=L_new, sa=sa, sb=sb,
                                        amt0_init=amt0, amt1_init=amt1, hodl0_value_y=amt0 * cex_ref_for_agents + amt1)
                        pool.add_liquidity_range(lower, upper, L_new)
                        lp.positions.append(pos)
                        _register_position(pos)
                        _assert_active_liquidity_state("lp_mint_mempool")
                        mint_steps.append(t); mint_sizes.append(L_new); mint_widths.append(upper - lower); mint_is_passive.append(bool(lp.is_passive))
                        buffer_log(f"[t={t:03d}] LP{lp.id} MINT L={L_new:.4f} [{lower},{upper}) | L_active={pool.L_active:.4f} | tick={pool.tick}\n")
                        lp.L_live = getattr(lp, 'L_live', 0.0) + L_new
                        _rebalance_lp_to_target(lp, ref.m, pool.S)
                        executed_lp_events += 1
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
                        amt0, amt1 = minted_amounts_at_S(L_new, sa, sb, agent_S_ref)
                        pos = Position(owner=lp.id, lower=lower, upper=upper, L=L_new, sa=sa, sb=sb,
                                        amt0_init=amt0, amt1_init=amt1, hodl0_value_y=amt0 * cex_ref_for_agents + amt1)
                        pool.add_liquidity_range(lower, upper, L_new)
                        lp.positions.append(pos)
                        _register_position(pos)
                        _assert_active_liquidity_state("lp_recenter_mempool")
                        mint_steps.append(t); mint_sizes.append(L_new); mint_widths.append(upper - lower); mint_is_passive.append(bool(lp.is_passive))
                        buffer_log(f"[t={t:03d}] LP{lp.id} RECENTER L={L_new:.4f} [{lower},{upper}) | L_active={pool.L_active:.4f} | tick={pool.tick}\n")
                        _rebalance_lp_to_target(lp, ref.m, pool.S)
                        executed_lp_events += 1
                        return

                # Handle arbitrage intent (executes before other swaps in a block)
                if typ == 'arb':
                    arb_ref = float(o.get('arb_ref_m', ref.m))
                    price_before = pool.price
                    tick_before = pool.tick
                    in_used, x_out_from_dex, y_out_from_dex, dir_arb, L_first = arbitrage_to_target(arb_ref)
                    delta_a_cex_this = 0.0
                    if in_used > 0 and dir_arb is not None:
                        L_pre_arb_eff_this = L_first
                        dir_arb_this = dir_arb
                        arb_steps.append(t); arb_dirs.append(dir_arb)

                        if dir_arb == "up":
                            # DEX cheap: buy token0 on DEX, sell on CEX
                            delta_a_cex_this = -x_out_from_dex
                            arb_y_this = +in_used
                            arb_acc.record_swap(dy_in=in_used, dx_out=x_out_from_dex)
                            buffer_log(
                                f"[t={t:03d}] arb swap up dy_in={in_used:.6f} dx_out={x_out_from_dex:.6f} "
                                f"| price {price_before:.4f}->{pool.price:.4f} | tick {tick_before}->{pool.tick}\n"
                            )
                        else:
                            # DEX expensive: sell token0 on DEX, buy on CEX
                            delta_a_cex_this = +in_used
                            arb_y_this = -pool.price * in_used
                            arb_acc.record_swap(dx_in=in_used, dy_out=y_out_from_dex)
                            buffer_log(
                                f"[t={t:03d}] arb swap down dx_in={in_used:.6f} dy_out={y_out_from_dex:.6f} "
                                f"| price {price_before:.4f}->{pool.price:.4f} | tick {tick_before}->{pool.tick}\n"
                            )
                        _record_micro(price_before)
                        _arb_execs += int(in_used > 0)
                    return

                if o.get('side') == 'X_to_Y':
                    min_output = o.get('min_output')
                    if min_output is not None:
                        final_quote = pool.quote_x_to_y(o['amount'])
                        if final_quote <= min_output:
                            agent = o.get('agent')
                            if agent == 'smart':
                                total_smart_swaps_skipped += 1
                            elif agent == 'noise':
                                total_noise_swaps_skipped += 1
                            buffer_log(
                                f"[t={t:03d}] {agent or 'N/A'} swap X_to_Y SKIPPED (slippage). "
                                f"final_quote={final_quote:.4f} <= min_output={min_output:.4f} | tick={tick_before_exec}\n"
                            )
                            return
                    if pool.L_active <= EPS_LIQ:
                        return
                    used_dx_pre, dy_out_real, fee_x = pool.swap_x_to_y(o['amount'], fee_cb=allocate_fees)
                    _assert_active_liquidity_state("mempool_swap_x_to_y")
                    if used_dx_pre <= EPS_LIQ:
                        return
                    executed = int(used_dx_pre > 0)
                    agent = o.get('agent')
                    if agent == 'smart':
                        trader_steps.append(t); trader_dirs.append('down')
                        sr_acc.notional_y += -P_pre_exec * used_dx_pre
                        trader_y_this += -P_pre_exec * used_dx_pre
                        sr_acc.record_swap(dx_in=used_dx_pre, dy_out=dy_out_real)
                        sr_acc.execs += int(used_dx_pre > 0)
                        total_smart_swaps_executed += executed
                        smart_swaps_x_to_y += int(used_dx_pre > 0)
                        executed_smart_swaps += executed
                        buffer_log(
                            f"[t={t:03d}] smart swap X_to_Y EXEC dx={used_dx_pre:.6f} dy_out={dy_out_real:.6f} "
                            f"| price {P_pre_exec:.4f}->{pool.price:.4f} | tick {tick_before_exec}->{pool.tick}\n"
                        )
                    elif agent == 'noise':
                        trader_steps.append(t); trader_dirs.append('down')
                        noise_acc.notional_y += -P_pre_exec * used_dx_pre
                        trader_y_this += -P_pre_exec * used_dx_pre
                        noise_acc.record_swap(dx_in=used_dx_pre, dy_out=dy_out_real)
                        noise_acc.execs += int(used_dx_pre > 0)
                        total_noise_swaps_executed += executed
                        noise_swaps_x_to_y += int(used_dx_pre > 0)
                        executed_noise_swaps += executed
                        buffer_log(
                            f"[t={t:03d}] noise swap X_to_Y EXEC dx={used_dx_pre:.6f} dy_out={dy_out_real:.6f} "
                            f"| price {P_pre_exec:.4f}->{pool.price:.4f} | tick {tick_before_exec}->{pool.tick}\n"
                        )
                    _record_micro(P_pre_exec)
                else:
                    min_output = o.get('min_output')
                    if min_output is not None:
                        final_quote = pool.quote_y_to_x(o['amount'])
                        if final_quote <= min_output:
                            agent = o.get('agent')
                            if agent == 'smart':
                                total_smart_swaps_skipped += 1
                            elif agent == 'noise':
                                total_noise_swaps_skipped += 1
                            buffer_log(
                                f"[t={t:03d}] {agent or 'N/A'} swap Y_to_X SKIPPED (slippage). "
                                f"final_quote={final_quote:.4f} <= min_output={min_output:.4f} | tick={tick_before_exec}\n"
                            )
                            return
                    if pool.L_active <= EPS_LIQ:
                        return
                    used_dy_pre, dx_out_real, fee_y = pool.swap_y_to_x(o['amount'], fee_cb=allocate_fees)
                    _assert_active_liquidity_state("mempool_swap_y_to_x")
                    if used_dy_pre <= EPS_LIQ:
                        return
                    executed = int(used_dy_pre > 0)
                    agent = o.get('agent')
                    if agent == 'smart':
                        trader_steps.append(t); trader_dirs.append('up')
                        sr_acc.notional_y += +used_dy_pre
                        trader_y_this += +used_dy_pre
                        sr_acc.record_swap(dy_in=used_dy_pre, dx_out=dx_out_real)
                        sr_acc.execs += int(used_dy_pre > 0)
                        total_smart_swaps_executed += executed
                        smart_swaps_y_to_x += int(used_dy_pre > 0)
                        executed_smart_swaps += executed
                        buffer_log(
                            f"[t={t:03d}] smart swap Y_to_X EXEC dy={used_dy_pre:.6f} dx_out={dx_out_real:.6f} "
                            f"| price {P_pre_exec:.4f}->{pool.price:.4f} | tick {tick_before_exec}->{pool.tick}\n"
                        )
                    elif agent == 'noise':
                        trader_steps.append(t); trader_dirs.append('up')
                        noise_acc.notional_y += +used_dy_pre
                        trader_y_this += +used_dy_pre
                        noise_acc.record_swap(dy_in=used_dy_pre, dx_out=dx_out_real)
                        noise_acc.execs += int(used_dy_pre > 0)
                        total_noise_swaps_executed += executed
                        noise_swaps_y_to_x += int(used_dy_pre > 0)
                        executed_noise_swaps += executed
                        buffer_log(
                            f"[t={t:03d}] noise swap Y_to_X EXEC dy={used_dy_pre:.6f} dx_out={dx_out_real:.6f} "
                            f"| price {P_pre_exec:.4f}->{pool.price:.4f} | tick {tick_before_exec}->{pool.tick}\n"
                        )
                    _record_micro(P_pre_exec)
            order_book = list(mempool_orders)
            # Ensure arbitrage intents execute first, then shuffle the rest
            arb_orders = [o for o in order_book if o.get('type') == 'arb']
            non_arb_orders = [o for o in order_book if o.get('type') != 'arb']
            random.shuffle(non_arb_orders)
            order_book = arb_orders + non_arb_orders
            tick_before_orders = pool.tick
            buffer_log(
                f"[t={t:03d}] MEMPOOL before P={P_pre_exec:.4f} | tick={tick_before_orders} | "
                f"n_orders={len(order_book)}\n"
            )
            for o in order_book:
                _exec_one(o)
            buffer_log(
                f"[t={t:03d}] MEMPOOL after P={pool.price:.4f} | tick={pool.tick} | "
                f"smart_exec={executed_smart_swaps} | noise_exec={executed_noise_swaps} | "
                f"lp_events={executed_lp_events}\n"
            )
            mempool_orders.clear()

        def execute_trader(agent_label: str, probability: float, accumulator: TraderStepAccumulator, enforce_best_ex: bool, m_reference: float) -> None:
            nonlocal L_pre_trader_this, trader_y_this, _trader_execs
            nonlocal total_noise_swaps_executed, total_noise_swaps_skipped
            nonlocal total_smart_swaps_executed, total_smart_swaps_skipped
            nonlocal smart_swaps_x_to_y, smart_swaps_y_to_x
            nonlocal noise_swaps_x_to_y, noise_swaps_y_to_x

            if random.random() >= probability:
                return

            side = random.choice(["X_to_Y", "Y_to_X"])
            L_pre_trader_this = pool.L_active
            P_pre = pool.price
            tick_before = pool.tick

            if side == "X_to_Y":
                notional_y = _draw_trader_notional()
                if notional_y <= 0.0:
                    return
                price_snapshot = max(m_reference, 1e-18)
                dx = notional_y / price_snapshot
                if dx <= 0.0:
                    return
                initial_quote = pool.quote_x_to_y(dx)
                if initial_quote <= 0.0:
                    return
                if enforce_best_ex:
                    if initial_quote < theta_T * dx * m_reference:
                        return
                baseline_quote = _baseline_quote_x_to_y(dx)
                min_output = max(0.0, baseline_quote * (1.0 - slippage_tolerance))
                final_quote = pool.quote_x_to_y(dx)
                if final_quote < min_output:
                    if agent_label == "smart":
                        total_smart_swaps_skipped += 1
                    elif agent_label == "noise":
                        total_noise_swaps_skipped += 1
                    buffer_log(
                        f"[t={t:03d}] {agent_label} swap X_to_Y SKIPPED (slippage). "
                        f"final_quote={final_quote:.4f} < min_output={min_output:.4f} | tick={tick_before}\n"
                    )
                    return

                if pool.L_active <= EPS_LIQ:
                    return

                used_dx_pre, dy_out_real, _ = pool.swap_x_to_y(dx, fee_cb=allocate_fees)
                _assert_active_liquidity_state("trader_swap_x_to_y")
                if used_dx_pre <= EPS_LIQ:
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
                    smart_swaps_x_to_y += executed
                    if executed:
                        buffer_log(
                            f"[t={t:03d}] smart swap X_to_Y EXEC dx={used_dx_pre:.6f} dy_out={dy_out_real:.6f} "
                            f"| price {P_pre:.4f}->{pool.price:.4f} | tick {tick_before}->{pool.tick}\n"
                        )
                elif agent_label == "noise":
                    total_noise_swaps_executed += executed
                    noise_swaps_x_to_y += executed
                    if executed:
                        buffer_log(
                            f"[t={t:03d}] noise swap X_to_Y EXEC dx={used_dx_pre:.6f} dy_out={dy_out_real:.6f} "
                            f"| price {P_pre:.4f}->{pool.price:.4f} | tick {tick_before}->{pool.tick}\n"
                        )

            else:
                dy = _draw_trader_notional()
                if dy <= 0.0:
                    return
                initial_quote = pool.quote_y_to_x(dy)
                if initial_quote <= 0.0:
                    return
                if enforce_best_ex:
                    dx_cex = dy / max(m_reference, 1e-18)
                    if initial_quote < theta_T * dx_cex:
                        return
                baseline_quote = _baseline_quote_y_to_x(dy)
                min_output = max(0.0, baseline_quote * (1.0 - slippage_tolerance))
                final_quote = pool.quote_y_to_x(dy)
                if final_quote < min_output:
                    if agent_label == "smart":
                        total_smart_swaps_skipped += 1
                    elif agent_label == "noise":
                        total_noise_swaps_skipped += 1
                    buffer_log(
                        f"[t={t:03d}] {agent_label} swap Y_to_X SKIPPED (slippage). "
                        f"final_quote={final_quote:.4f} < min_output={min_output:.4f} | tick={tick_before}\n"
                    )
                    return

                if pool.L_active <= EPS_LIQ:
                    return

                used_dy_pre, dx_out_real, _ = pool.swap_y_to_x(dy, fee_cb=allocate_fees)
                _assert_active_liquidity_state("trader_swap_y_to_x")
                if used_dy_pre <= EPS_LIQ:
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
                    smart_swaps_y_to_x += executed
                    if executed:
                        buffer_log(
                            f"[t={t:03d}] smart swap Y_to_X EXEC dy={used_dy_pre:.6f} dx_out={dx_out_real:.6f} "
                            f"| price {P_pre:.4f}->{pool.price:.4f} | tick {tick_before}->{pool.tick}\n"
                        )
                elif agent_label == "noise":
                    total_noise_swaps_executed += executed
                    noise_swaps_y_to_x += executed
                    if executed:
                        buffer_log(
                            f"[t={t:03d}] noise swap Y_to_X EXEC dy={used_dy_pre:.6f} dx_out={dx_out_real:.6f} "
                            f"| price {P_pre:.4f}->{pool.price:.4f} | tick {tick_before}->{pool.tick}\n"
                        )

        # --- Actor routines (closures) ---
        def act_LPs():
            # ----- burns (TP/SL) -----
            for lp in LPs:
                if hasattr(lp, "can_act") and not lp.can_act:
                    continue
                if lp.is_passive:
                    if lp.positions and random.random() < passive_burn_prob:
                        burn_idx = random.randrange(len(lp.positions))
                        burn_any(lp, burn_idx)
                    continue
                to_burn = []
                for i, pos in enumerate(lp.positions):
                    pnl = pos.PnL_y(agent_S_ref, ref.m)
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
                    in_rng = pos.in_range(agent_tick_ref)
                    out_steps = getattr(pos, "out_steps", 0)
                    out_steps = 0 if in_rng else out_steps + 1
                    setattr(pos, "out_steps", out_steps)
                    k_out_thresh = getattr(lp, "k_out_threshold", k_out_max)
                    if lp.is_active_narrow and out_steps >= k_out_thresh:
                        to_recenters.append(i)

                for i in reversed(to_recenters):
                    pos = lp.positions[i]
                    width = pos.upper - pos.lower
                    L_same = pos.L
                    burn_any(lp, i)
                    # Center around current sqrt price S (approximately), not the snapped active band.
                    n_bands = max(1, int(round(width / pool.tick_spacing)))
                    S_now = agent_S_ref
                    s = pool.tick_spacing
                    nb = n_bands
                    denom = (1.0 + (pool.g ** (nb * s + s)))
                    if denom <= 0.0:
                        denom = 1.0
                    lower_real = math.log((2.0 * S_now / pool.base_s) / denom, pool.g)
                    lower = pool._snap(int(round(lower_real)))
                    upper = lower + nb * s
                    sa, sb = pool.s_lower(lower), pool.s_upper(upper)
                    amt0, amt1 = minted_amounts_at_S(L_same, sa, sb, agent_S_ref)
                    newpos = Position(
                        owner=lp.id, lower=lower, upper=upper, L=L_same, sa=sa, sb=sb,
                        amt0_init=amt0, amt1_init=amt1, hodl0_value_y=amt0 * cex_ref_for_agents + amt1,
                    )
                    pool.add_liquidity_range(lower, upper, L_same)
                    lp.positions.append(newpos)
                    _register_position(newpos)
                    _assert_active_liquidity_state("lp_recenter_active")
                    mint_steps.append(t); mint_sizes.append(L_same); mint_widths.append(upper - lower); mint_is_passive.append(bool(lp.is_passive))
                    buffer_log(
                        f"[t={t:03d}] LP{lp.id} RECENTER L={L_same:.4f} [{lower},{upper}) | "
                        f"L_active={pool.L_active:.4f} | tick={pool.tick}\n"
                    )

            # ----- probabilistic mints (blocked during cooldown) -----
            for lp in LPs:
                if hasattr(lp, "can_act") and not lp.can_act:
                    continue
                if getattr(lp, "cooldown", 0) > 0:
                    continue
                if lp.is_passive:
                    if random.random() >= passive_mint_prob:
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
                S_now = agent_S_ref
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

            if -1e-9 < pool.L_active < 0.0:
                pool.L_active = 0.0

        def act_smart_router():
            # Single smart-router trader per step (probabilistic arrival)
            execute_trader("smart", p_trade_micro, sr_acc, True, cex_ref_for_agents)

        def act_noise_trader():
            # Single noise trader per step (probabilistic arrival)
            execute_trader("noise", noise_floor_micro, noise_acc, False, cex_ref_for_agents)

        def act_arbitrageur():
            nonlocal arb_y_this, L_pre_arb_eff_this, dir_arb_this, delta_a_cex_this, _arb_execs

            in_used, x_out_from_dex, y_out_from_dex, dir_arb, L_first = arbitrage_to_target(arb_ref_m)
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
        # Non-block mode (block_time == 1): A -> Smart+Noise -> B -> Arb -> C.
        # Block mode (block_time > 1):
        #   - Snapshot CEX at block start: arb_ref_m
        #   - Micro-steps: diffuse-only; enqueue intents with p_trade_micro/noise_floor_micro
        #   - Boundary order (current): Arb -> (populate+execute mempool)  (LPs act via mempool)
        # =====================================================================
        # run the schedule
        if block_time == 1:
            # target band uses validated CEX snapshot in non-block mode
            target_band_m = cex_ref_for_agents
            band_lo_target.append(target_band_m * r)
            band_hi_target.append(target_band_m / r)
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
            # prepare micro-time arrays (event-time logging)
            buffer_log(f"[t={t:03d}] BLOCK start m={arb_ref_m_start:.4f} due_lp={len(due)}\n")
            micro_steps.append(micro_counter)
            P_micro.append(pool.price)
            M_micro.append(ref.m)
            micro_counter += 1
            for _k in range(block_time):
                maybe_enqueue_smart_router_intent(cex_ref_for_agents)
                maybe_enqueue_noise_trader_intent(cex_ref_for_agents)
                ref.diffuse_only()
                _broadcast_price_move(ref.m)
                micro_steps.append(micro_counter)
                P_micro.append(pool.price)
                M_micro.append(ref.m)
                micro_counter += 1

            # --- Arbitrage intent (executes first in mempool) ---
            arb_ref_m = cex_ref_for_agents  # snapshot from end of previous block
            target_band_m = cex_ref_for_agents
            band_lo_target.append(target_band_m * r)
            band_hi_target.append(target_band_m / r)
            mempool_orders.append({'type': 'arb', 'arb_ref_m': arb_ref_m})

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
                        pos = random.choice(lp.positions)
                        mempool_orders.append({'type':'lp_burn','lp_id': lp.id,'lower': pos.lower,'upper': pos.upper,'L': pos.L})
                    continue
                to_burn = []
                for i, pos in enumerate(lp.positions):
                    pnl = pos.PnL_y(agent_S_ref, ref.m)
                    if pnl >= theta_TP * pos.hodl0_value_y or pnl <= -theta_SL * pos.hodl0_value_y:
                        to_burn.append(i)
                for i in reversed(to_burn):
                    pos = lp.positions[i]
                    mempool_orders.append({'type':'lp_burn','lp_id': lp.id,'lower': pos.lower,'upper': pos.upper,'L': pos.L})

            # Recenter (narrow LPs that have been out-of-range past their threshold)
            for lp_idx in due:
                lp = LPs[lp_idx]
                if not lp.can_act:
                    continue
                to_recenters = []
                for i, pos in enumerate(lp.positions):
                    in_rng = pos.in_range(agent_tick_ref)
                    out_steps = getattr(pos, "out_steps", 0)
                    out_steps = 0 if in_rng else out_steps + 1
                    setattr(pos, "out_steps", out_steps)
                    k_out_thresh = getattr(lp, "k_out_threshold", k_out_max)
                    if lp.is_active_narrow and out_steps >= k_out_thresh:
                        to_recenters.append(i)
                for i in reversed(to_recenters):
                    pos = lp.positions[i]
                    width_ticks = pos.upper - pos.lower
                    n_bands = max(1, int(round(width_ticks / pool.tick_spacing)))
                    S_now = agent_S_ref
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
                    if random.random() >= passive_mint_prob:
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
                S_now = agent_S_ref
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
            if block_time > 1 and micro_steps:
                micro_valid_steps.append(micro_steps[-1])
                micro_valid_prices.append(pool.price)

        # disable everyone for next step
        _enable([])

        # ---- CEX update  ----
        if block_time == 1:
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
        lp_rebal_total = 0.0
        lp_rebal_active = 0.0
        lp_rebal_passive = 0.0
        lp_wallet_total = 0.0
        lp_wallet_active = 0.0
        lp_wallet_passive = 0.0
        lp_wealth_total = 0.0
        lp_wealth_active = 0.0
        lp_wealth_passive = 0.0
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
        lp_pnl_total_series.append(lp_total)
        lp_pnl_active_series.append(lp_total_active)
        lp_pnl_passive_series.append(lp_total_passive)
        lp_rebal_total_series.append(lp_rebal_total)
        lp_rebal_active_series.append(lp_rebal_active)
        lp_rebal_passive_series.append(lp_rebal_passive)

        # Wealth accounting (wallet + open PnL)
        lp_wallet_series.append(lp_wallet_total)
        lp_wallet_active_series.append(lp_wallet_active)
        lp_wallet_passive_series.append(lp_wallet_passive)
        lp_wealth_series.append(lp_wealth_total)
        lp_wealth_active_series.append(lp_wealth_active)
        lp_wealth_passive_series.append(lp_wealth_passive)
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
        f"L={pool.L_active:.4f} | tick={pool.tick} | w_ticks={w_ticks}"
        )
        buffer_log(log_line + "\n")

        liq_history.append(dict(pool.liquidity_net))
        tick_history.append(pool.tick)

        validated_S = pool.S
        validated_tick = pool.tick
        validated_cex = ref.m

    summary_lines = [
        "# Run summary",
        f"total_mints = {len(mint_steps)}",
        f"total_burns = {len(burn_steps)}",
        f"total_noise_trader_swaps = {total_noise_swaps_executed}",
        f"noise_trader_swaps_rejected_slippage = {total_noise_swaps_skipped}",
        f"total_smart_router_swaps = {total_smart_swaps_executed}",
        f"smart_router_swaps_rejected_slippage = {total_smart_swaps_skipped}",
        f"smart_router_swaps_X_to_Y (price down) = {smart_swaps_x_to_y}",
        f"smart_router_swaps_Y_to_X (price up) = {smart_swaps_y_to_x}",
        f"noise_trader_swaps_X_to_Y (price down) = {noise_swaps_x_to_y}",
        f"noise_trader_swaps_Y_to_X (price up) = {noise_swaps_y_to_x}",
        "----------------------------------------------------------\n",
    ]

    flush_log_buffer()
    verbose_log.flush()
    verbose_log.close()
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
    lp_rebal_total_series = np.array(lp_rebal_total_series)
    lp_rebal_active_series = np.array(lp_rebal_active_series)
    lp_rebal_passive_series = np.array(lp_rebal_passive_series)
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
    band_lo_post_v = band_lo_post[s0:]
    band_hi_post_v = band_hi_post[s0:]
    L_end_v = L_end[s0:]
    L_pre_step_v = L_pre_step[s0:]
    L_pre_trader_v = L_pre_trader[s0:]
    L_pre_arb_eff_v = L_pre_arb_eff[s0:]
    arb_pnl_cum_v = arb_pnl_cum[s0:]
    sr_pnl_cum_v = sr_pnl_cum[s0:]
    noise_pnl_cum_v = noise_pnl_cum[s0:]
    lp_pnl_active_series_v = lp_pnl_active_series[s0:]
    lp_pnl_passive_series_v = lp_pnl_passive_series[s0:]
    fee_series_v = fee_series[s0:]
    fee_sigma_series_v = fee_sigma_series[s0:]
    fee_basis_ticks_series_v = fee_basis_ticks_series[s0:]
    fee_signal_series_v = fee_signal_series[s0:]
    arb_y_v = np.array(arb_y_series)[s0:]
    sr_y_v = sr_y_series[s0:]
    noise_y_v = noise_y_series[s0:]
    w_ticks_series_v = np.array(w_ticks_series)[s0:] if w_ticks_series else np.array([])
    w_unclipped_series_v = np.array(w_unclipped_series)[s0:] if w_unclipped_series else np.array([])
    w_noise_series_v = np.array(w_noise_series)[s0:] if w_noise_series else np.array([])
    
    if visualize:
        # ΔL per step (split by LP type)
        mint_step_sum_passive = np.zeros_like(P_series)
        mint_step_sum_active = np.zeros_like(P_series)
        n_steps = len(P_series)
        for s, L, is_passive in zip(mint_steps, mint_sizes, mint_is_passive):
            if 0 <= s < n_steps:
                target = mint_step_sum_passive if is_passive else mint_step_sum_active
                target[s] += L
        burn_step_sum_passive = np.zeros_like(P_series)
        burn_step_sum_active = np.zeros_like(P_series)
        for s, L, is_passive in zip(burn_steps, burn_sizes, burn_is_passive):
            if 0 <= s < n_steps:
                target = burn_step_sum_passive if is_passive else burn_step_sum_active
                target[s] += L

        mint_step_sum_passive_v = mint_step_sum_passive[s0:]
        mint_step_sum_active_v = mint_step_sum_active[s0:]
        burn_step_sum_passive_v = burn_step_sum_passive[s0:]
        burn_step_sum_active_v = burn_step_sum_active[s0:]

        from pathlib import Path as _Path

        _out_dir = _Path("abm_results")
        _png_dir = _out_dir / "png"
        _html_dir = _out_dir / "html"
        _png_dir.mkdir(parents=True, exist_ok=True)
        _html_dir.mkdir(parents=True, exist_ok=True)
        _prefix = f"abm_fee_{fee_mode}_{cex_sigma}"

        total_steps = max(1, len(steps) - s0)

        def _save_plotly(name: str, fig: go.Figure) -> None:
            suffix = f"{name}_steps{total_steps}"
            save_plotly_figure(
                fig,
                _png_dir / f"{_prefix}_{suffix}.png",
                _html_dir / f"{_prefix}_{suffix}.html",
                "simulate",
            )

        band_lo_target_v = np.array(band_lo_target)[s0:]
        band_hi_target_v = np.array(band_hi_target)[s0:]
        steps_list = steps_v.tolist()

        # ----- 1) Price panel -----
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=steps_list,
                y=band_lo_post_v,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=steps_list,
                y=band_hi_post_v,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(180,180,180,0.35)",
                line=dict(width=0),
                name="No-arb fee band",
                hoverinfo="skip",
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=steps_list,
                y=P_series_v,
                mode="lines",
                name="DEX price Pₜ",
                line=dict(width=2),
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=steps_list,
                y=M_series_v,
                mode="lines",
                name="CEX price mₜ",
                line=dict(width=1.6, dash="dash"),
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=steps_list,
                y=band_lo_target_v,
                mode="lines",
                name="Target band lo",
                line=dict(width=1, dash="dot"),
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=steps_list,
                y=band_hi_target_v,
                mode="lines",
                name="Target band hi",
                line=dict(width=1, dash="dot"),
            )
        )
        fig1.update_layout(
            template="plotly_white",
            title="CEX vs DEX Price",
            xaxis_title="Step",
            yaxis_title="Price (token1 per token0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        _save_plotly("1_price", fig1)

        # ----- 1b) Micro-time price panel -----
        if block_time > 1 and len(M_micro) == len(P_micro) == len(micro_steps) and len(micro_steps) > 0:
            fig1b = go.Figure()
            fig1b.add_trace(
                go.Scatter(x=micro_steps, y=P_micro, mode="lines", name="DEX price (micro)", line=dict(width=1.2))
            )
            fig1b.add_trace(
                go.Scatter(
                    x=micro_steps,
                    y=M_micro,
                    mode="lines",
                    name="CEX price (micro)",
                    line=dict(width=1.0, dash="dash"),
                )
            )
            if micro_valid_steps:
                fig1b.add_trace(
                    go.Scatter(
                        x=micro_valid_steps,
                        y=micro_valid_prices,
                        mode="markers",
                        name="Validated DEX price",
                        marker=dict(color="#d62728", size=6),
                    )
                )
            fig1b.update_layout(
                template="plotly_white",
                title="Micro-time CEX vs DEX (within blocks)",
                xaxis_title="Event time",
                yaxis_title="Price",
            )
            _save_plotly("1b_price_micro", fig1b)

        # ----- 2) Notionals -----
        fig2 = go.Figure()
        fig2.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"))
        fig2.add_trace(go.Scatter(x=steps_list, y=sr_y_v, mode="lines", name="Smart router (token1)"))
        fig2.add_trace(
            go.Scatter(
                x=steps_list,
                y=noise_y_v,
                mode="lines",
                name="Noise trader (token1)",
                line=dict(dash="dash"),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=steps_list,
                y=arb_y_v,
                mode="lines",
                name="Arbitrageur (token1)",
                line=dict(dash="dot"),
            )
        )
        fig2.update_layout(
            template="plotly_white",
            title="Trader Notionals",
            xaxis_title="Step",
            yaxis_title="Notional (token1, signed)",
        )
        _save_plotly("2_notional", fig2)

        # helper for zero-liquidity shading
        def _zero_liquidity_shapes():
            shapes = []
            for s_idx, L_val in zip(steps_v, L_end_v):
                if L_val <= 1e-9:
                    shapes.append(
                        dict(
                            type="rect",
                            x0=float(s_idx) - 0.5,
                            x1=float(s_idx) + 0.5,
                            y0=0,
                            y1=1,
                            yref="paper",
                            fillcolor="rgba(255,0,0,0.06)",
                            line=dict(width=0),
                        )
                    )
            return shapes

        # ----- 3) Liquidity traces -----
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=steps_list, y=L_end_v, mode="lines", name="Active L (end of step)", line=dict(width=1.8)))
        fig3.add_trace(
            go.Scatter(
                x=steps_list,
                y=L_pre_step_v,
                mode="lines",
                name="Active L (start of step)",
                line=dict(width=1.0, dash="dash"),
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=steps_list,
                y=L_pre_trader_v,
                mode="lines",
                name="Active L (before trader)",
                line=dict(width=1.0, dash="dot"),
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=steps_list,
                y=L_pre_arb_eff_v,
                mode="lines",
                name="Active L (before arb)",
                line=dict(width=1.2, dash="dashdot"),
            )
        )
        fig3.update_layout(
            template="plotly_white",
            title="Active Liquidity",
            xaxis_title="Step",
            yaxis_title="Active L",
            shapes=_zero_liquidity_shapes(),
        )
        _save_plotly("3_activeL", fig3)

        # ----- 4) L per step (passive vs active) -----
        fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Passive LPs", "Active LPs"))
        fig4.add_trace(
            go.Bar(x=steps_list, y=mint_step_sum_passive_v, name="Mint / recenter L", marker_color="#6a0dad"),
            row=1,
            col=1,
        )
        fig4.add_trace(
            go.Bar(x=steps_list, y=-burn_step_sum_passive_v, name="Burn L", marker_color="#ff8c00"),
            row=1,
            col=1,
        )
        fig4.add_trace(
            go.Bar(x=steps_list, y=mint_step_sum_active_v, showlegend=False, marker_color="#6a0dad"),
            row=2,
            col=1,
        )
        fig4.add_trace(
            go.Bar(x=steps_list, y=-burn_step_sum_active_v, showlegend=False, marker_color="#ff8c00"),
            row=2,
            col=1,
        )
        fig4.update_layout(
            template="plotly_white",
            title="ΔL per Step",
            barmode="relative",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        fig4.update_yaxes(title_text="ΔL per step", row=1, col=1)
        fig4.update_yaxes(title_text="ΔL per step", row=2, col=1)
        fig4.update_xaxes(title_text="Step", row=2, col=1)
        _save_plotly("4_L_per_step", fig4)

        # ----- 5) Active-band reserves -----
        fig5 = go.Figure()
        fig5.add_trace(
            go.Scatter(
                x=steps_list,
                y=X_active_end_v * P_series_v,
                mode="lines",
                name="token0 value in active band",
                line=dict(width=1.8),
            )
        )
        fig5.add_trace(
            go.Scatter(
                x=steps_list,
                y=Y_active_end_v,
                mode="lines",
                name="token1 in active band",
                line=dict(width=1.8),
            )
        )
        fig5.update_layout(
            template="plotly_white",
            title="Active-band Reserves",
            xaxis_title="Step",
            yaxis_title="Token1 units",
            shapes=_zero_liquidity_shapes(),
        )
        _save_plotly("5_active_reserves", fig5)

        # ----- 6b) LP mint width signal -----
        if len(w_ticks_series_v) > 0:
            width_baseline_v = w_unclipped_series_v - w_noise_series_v
            fig6b = go.Figure()
            fig6b.add_trace(
                go.Scatter(
                    x=steps_list,
                    y=width_baseline_v,
                    mode="lines",
                    name="Baseline width",
                    line=dict(width=1.4),
                )
            )
            fig6b.add_trace(
                go.Scatter(
                    x=steps_list,
                    y=w_unclipped_series_v,
                    mode="lines",
                    name="Baseline + noise",
                    line=dict(width=1.2, dash="dash"),
                )
            )
            fig6b.add_trace(
                go.Scatter(
                    x=steps_list,
                    y=w_ticks_series_v,
                    mode="lines",
                    name="Final width",
                    line=dict(width=1.6, dash="dashdot"),
                )
            )
            fig6b.update_layout(
                template="plotly_white",
                title="LP Mint Width Signal",
                xaxis_title="Step",
                yaxis_title="Width (ticks)",
            )
            _save_plotly("6b_mint_width_signal", fig6b)

        # ----- 7) PnL panel -----
        fig6 = go.Figure()
        fig6.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"))
        fig6.add_trace(go.Scatter(x=steps_list, y=sr_pnl_cum_v, mode="lines", name="Smart router PnL"))
        fig6.add_trace(
            go.Scatter(
                x=steps_list,
                y=noise_pnl_cum_v,
                mode="lines",
                name="Noise trader PnL",
                line=dict(dash="dash"),
            )
        )
        fig6.add_trace(go.Scatter(x=steps_list, y=arb_pnl_cum_v, mode="lines", name="Arbitrageur PnL"))
        fig6.add_trace(
            go.Scatter(
                x=steps_list,
                y=lp_pnl_active_series_v,
                mode="lines",
                name="Active narrow LP Fee-LVR",
                line=dict(dash="dash"),
            )
        )
        fig6.add_trace(
            go.Scatter(
                x=steps_list,
                y=lp_pnl_passive_series_v,
                mode="lines",
                name="Passive LP Fee-LVR",
                line=dict(dash="dot"),
            )
        )
        fig6.update_layout(
            template="plotly_white",
            title="Agent PnL (token1)",
            xaxis_title="Step",
            yaxis_title="Token1 value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        _save_plotly("6_pnl", fig6)

        # ----- 8) Fee panel + controller signal -----
        fig7 = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        fig7.add_trace(
            go.Scatter(x=steps_list, y=fee_series_v, mode="lines", name="Fee", line=dict(width=1.8)),
            row=1,
            col=1,
            secondary_y=False,
        )
        if fee_mode == "volatility":
            secondary_vals = fee_sigma_series_v
            secondary_label = "σ̂ (abs log-return)"
        elif fee_mode == "toxicity":
            secondary_vals = fee_basis_ticks_series_v
            secondary_label = "Basis (ticks)"
        else:
            secondary_vals = fee_signal_series_v
            secondary_label = "Controller signal"
        fig7.add_trace(
            go.Scatter(
                x=steps_list,
                y=secondary_vals,
                mode="lines",
                name=secondary_label,
                line=dict(width=1.2, dash="dash"),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
        fig7.update_layout(
            template="plotly_white",
            title="Fee & Controller Signal",
            xaxis_title="Step",
        )
        fig7.update_yaxes(title_text="Fee", secondary_y=False)
        fig7.update_yaxes(title_text=secondary_label, secondary_y=True)
        _save_plotly("7_fee", fig7)

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
        "lp_rebal_total_series": lp_rebal_total_series.tolist(),
        "lp_rebal_active_series": lp_rebal_active_series.tolist(),
        "lp_rebal_passive_series": lp_rebal_passive_series.tolist(),
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
    autocorr_fig = go.Figure()
    autocorr_fig.add_trace(go.Bar(x=lags, y=autocorr, name="Autocorr"))
    autocorr_fig.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dash"))
    autocorr_fig.update_layout(
        template="plotly_white",
        title="DEX Log-Return Autocorrelation",
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
    )

    results_root = Path("abm_results")
    png_dir = results_root / "png"
    html_dir = results_root / "html"
    png_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    total_steps = max(1, len(dex_prices))
    png_path = png_dir / f"autocorr_{scenario_label}_steps{total_steps}.png"
    html_path = html_dir / f"autocorr_{scenario_label}_steps{total_steps}.html"
    save_plotly_figure(autocorr_fig, png_path, html_path, "autocorr")

    # make liquidity GIF
    # make_liquidity_gif(
    # liq_history=out["liq_history"],
    # tick_history=out["tick_history"],
    # base_s=out["grid_base_s"],
    # g=out["grid_g"],
    # out_path=f"abm_results/liquidity_evolution_{scenario_label}_{params['cex_sigma']}_{params['T']}.gif",
    # fps=20,
    # dpi=120,
    # pad_frac=0.05,
    # downsample_every=10,
    # center_line=True,
    # )
