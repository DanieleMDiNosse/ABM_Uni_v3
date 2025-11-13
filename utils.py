"""
Utility functions, constants, and helper classes for the ABM simulation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
import yaml
import inspect

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


# def lcm(a: int, b: int) -> int:
#     """Least common multiple (for aligning width to w_min and grid)."""
#     return abs(a * b) // math.gcd(a, b) if a and b else 0


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

    Let λ = exp(-ln 2 / half_life_steps). On update with observation x_t:
        v_t = λ v_{t-1} + (1 - λ) x_t

    We use this to smooth the *fee-adjusted absolute basis* B_t that drives the LP
    width rule (see simulate(): Eq. (10)(12) in the PDF for the model-side notation).
    """
    def __init__(self, half_life_steps: int, init: float = 0.0):
        # decay λ so that value halves every 'half_life_steps'
        self.lambda_ = math.exp(-math.log(2.0) / max(1, half_life_steps))
        self.v = init

    def update(self, x: float) -> float:
        self.v = self.lambda_ * self.v + (1.0 - self.lambda_) * x
        return self.v


# =============================================================================
# Reference market (CEX)
# =============================================================================

@dataclass
class ReferenceMarket:
    m: float            # CEX price of token A in token B (B per A)
    mu: float           # drift (per step) of log-returns
    sigma: float        # vol (per step) of log-returns
    kappa: float        # impact scale (price units per A^(1+xi))
    xi: float = 0.0     # impact exponent (xi = 0 => linear in |Δa|)

    def step(self, delta_a_cex_signed: float) -> float:
        """
        Apply permanent, additive impact from the CEX trade in token A units,
        then diffuse via GBM. Returns the impact applied (for debugging).
        """
        impact = self.apply_impact_only(delta_a_cex_signed)
        self.diffuse_only()
        return impact

    def apply_impact_only(self, delta_a_cex_signed: float) -> float:
        """
        Apply the permanent impact component without diffusion. Returns the impact used.
        """
        impact = self.kappa * math.copysign(
            abs(delta_a_cex_signed) ** (1.0 + self.xi),
            delta_a_cex_signed,
        )
        self.m = max(1e-12, self.m + impact)
        return impact

    def diffuse_only(self) -> float:
        """Diffuse the reference price via GBM without additional impact."""
        z = np.random.normal()
        self.m *= math.exp(self.mu - 0.5 * self.sigma**2 + self.sigma * z)
        self.m = max(1e-12, self.m)
        return self.m


# =============================================================================
# Builders (for pool initialization)
# =============================================================================

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


def build_empty_pool():
    """Build an empty pool with no initial liquidity."""
    from uniswapv3_pool import V3Pool
    f = 0.003
    g = np.sqrt(1.0001)
    m0 = 2000.0
    S0 = math.sqrt(m0)
    base_s = S0 / math.sqrt(g)
    pool = V3Pool(g=g, base_s=base_s, tick=0, S=S0, f=f, liquidity_net={}, tick_spacing=10)
    return pool, m0


def add_static_binomial_hill(
    pool,
    N: int = 400,
    L_total: float = 70_000.0,
    min_L_per_tick: float = 1e-9,
    plot: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> None:
    """Add a static binomial hill of liquidity to the pool."""
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
        ticks.append(lower)
        L_vals.append(L_i)

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


def bootstrap_initial_binomial_hill_sharded(
    pool,
    ref: ReferenceMarket,
    LPs: List,
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
) -> List:
    """
    Split the binomial hill across `num_seed_lps` seed LPs so burns are staggered.
    Each seed LP has its own review clock; all have mintProb=0 and is_active_narrow=False.
    """
    from agents import LPAgent, Position
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


# =============================================================================
# Visualization
# =============================================================================

def make_liquidity_gif(
    liq_history: List[Dict[int, float]],
    tick_history: List[int],
    base_s: float,
    g: float,
    out_path: str = "abm_results/liquidity_evolution.gif",
    fps: int = 10,
    dpi: int = 120,
    pad_frac: float = 0.05,
    downsample_every: int = 1,
    center_line: bool = True,
):
    """
    Animate liquidity per 1-tick bin, with the x-axis in **price** (P = S^2).

    Bars:
      left edge  = P_lower(i) = (base_s * g**i)^2
      width      = ΔP(i) = P_lower(i) * (g**2 - 1)
      height     = active liquidity in that 1-tick bin

    Vertical line:
      at the active band's **center** price (default) or at the lower edge.
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
        return P_lo * (g if center_line else 1.0)        # center = geometric mean => ×g

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
    ax.set_title("Liquidity vs Price — evolution", fontsize=TITLE_FONT_SIZE)
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
    print(f"[GIF] wrote {out_path}")


# =============================================================================
# Configuration loading
# =============================================================================

def load_simulation_parameters(config_path: Path, simulate_func=None) -> Tuple[str, Dict[str, Any]]:
    """
    Load simulation parameters from a YAML configuration file.

    The configuration must contain a `simulate` mapping with every parameter
    accepted by `simulate`. An optional `scenario` key can be provided for
    labeling outputs; if omitted, the fee mode is used as the label.
    """
    if simulate_func is None:
        from .run import simulate as simulate_func
    
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configuration file: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)

    if not isinstance(config_data, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")

    params = config_data.get("simulate")
    if not isinstance(params, dict):
        raise ValueError(f"'simulate' section missing in {config_path}")

    scenario_fee_mode = config_data.get("fee_mode")
    if scenario_fee_mode is not None:
        fee_mode_param = params.get("fee_mode")
        if fee_mode_param is not None and fee_mode_param != scenario_fee_mode:
            raise ValueError(
                "Conflicting 'fee_mode' definitions between top-level config and simulate() parameters."
            )
        params = dict(params)
        params["fee_mode"] = scenario_fee_mode
    else:
        params = dict(params)

    signature = inspect.signature(simulate_func)
    expected_keys = set(signature.parameters)
    missing_keys = [name for name in signature.parameters if name not in params]
    if missing_keys:
        raise ValueError(f"Missing simulate parameters in {config_path}: {missing_keys}")

    extra_keys = sorted(set(params) - expected_keys)
    if extra_keys:
        raise ValueError(f"Unexpected keys in 'simulate' section: {extra_keys}")

    scenario_label = params.get("fee_mode")
    if scenario_label is None:
        raise ValueError(
            f"Missing 'fee_mode' in {config_path}. Provide it either at the top level or inside the simulate() parameters."
        )

    return str(scenario_label), dict(params)

