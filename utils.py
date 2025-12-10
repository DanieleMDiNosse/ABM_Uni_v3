"""
Utility functions, constants, and helper classes for the ABM simulation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
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
# Treat extremely small active liquidity as zero to avoid desert bridging/instability.
EPS_LIQ = 1e-4         # active liquidity ~ zero for swaps
EPS_PRICE_CHANGE = 1e-10
EPS_BOUNDARY = 1e-12
EPS_LIQ2 = 1e-4        # active liquidity ~ zero for LP

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
    sigma_mode: str = "static"   # "static" | "regime" | "noisy_sine" | "heston"
    sigma_low: Optional[float] = None
    sigma_high: Optional[float] = None
    p_LL: float = 1.0
    p_HH: float = 1.0
    regime_state: str = "L"
    sigma_sine_amp: Optional[float] = None
    sigma_sine_period: int = 10_000
    sigma_sine_noise: float = 0.0
    sigma_sine_phase: float = 0.0
    sigma_floor: float = 0.0
    # Heston-style stochastic volatility parameters (variance process)
    heston_kappa: Optional[float] = None     # mean reversion speed of variance
    heston_theta: Optional[float] = None     # long-run variance level
    heston_sigma_v: Optional[float] = None   # vol of variance
    heston_rho: Optional[float] = None       # corr between price and variance shocks
    heston_v0: Optional[float] = None        # initial variance; if None fall back to sigma^2
    _sigma_step: int = field(init=False, default=0, repr=False)
    _sigma_sine_center: float = field(init=False, default=0.0, repr=False)
    _sigma_sine_amp_eff: float = field(init=False, default=0.0, repr=False)
    _heston_v: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self):
        """
        Normalize and validate volatility regime settings.
        """
        mode = (self.sigma_mode or "static").lower()
        valid_modes = {"static", "regime", "regime_switch", "regime_switching", "noisy_sine", "heston"}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid sigma_mode '{self.sigma_mode}'. Use 'static', 'regime', 'noisy_sine', or 'heston'."
            )
        if mode in {"regime", "regime_switch", "regime_switching"}:
            self.sigma_mode = "regime"
        else:
            self.sigma_mode = mode

        if self.sigma_mode == "regime":
            if self.sigma_low is None or self.sigma_high is None:
                raise ValueError("sigma_low and sigma_high must be provided for regime-switching sigma.")
            if self.sigma_high <= self.sigma_low:
                raise ValueError(f"sigma_high must exceed sigma_low (got {self.sigma_low} >= {self.sigma_high}).")
            if not (0.0 <= self.p_LL <= 1.0) or not (0.0 <= self.p_HH <= 1.0):
                raise ValueError("p_LL and p_HH must be probabilities in [0, 1].")
            self.regime_state = "H" if str(self.regime_state).upper().startswith("H") else "L"
            self.sigma = self._sigma_for_state(self.regime_state)
        elif self.sigma_mode == "noisy_sine":
            self.sigma_floor = max(0.0, float(self.sigma_floor))
            self._sigma_sine_center = max(self.sigma_floor, float(self.sigma))
            self.sigma_sine_period = max(1, int(self.sigma_sine_period))
            self.sigma_sine_noise = max(0.0, float(self.sigma_sine_noise))
            self.sigma_sine_phase = float(self.sigma_sine_phase)
            self.regime_state = "S"
            if self.sigma_sine_amp is None:
                if self.sigma_low is not None and self.sigma_high is not None:
                    self._sigma_sine_center = max(
                        self.sigma_floor,
                        0.5 * (float(self.sigma_low) + float(self.sigma_high)),
                    )
                    self._sigma_sine_amp_eff = 0.5 * abs(float(self.sigma_high) - float(self.sigma_low))
                else:
                    self._sigma_sine_amp_eff = 0.5 * self._sigma_sine_center
            else:
                if self.sigma_sine_amp < 0:
                    raise ValueError("sigma_sine_amp must be non-negative.")
                self._sigma_sine_amp_eff = float(self.sigma_sine_amp)
            self._sigma_step = 0
            self.sigma = self._sigma_from_sine(advance=False)
        elif self.sigma_mode == "heston":
            # Validate required Heston parameters.
            missing = []
            if self.heston_kappa is None:
                missing.append("heston_kappa")
            if self.heston_theta is None:
                missing.append("heston_theta")
            if self.heston_sigma_v is None:
                missing.append("heston_sigma_v")
            if self.heston_rho is None:
                missing.append("heston_rho")
            if missing:
                raise ValueError(
                    f"Heston sigma_mode requires parameters: {', '.join(missing)}."
                )
            if self.heston_kappa <= 0.0:
                raise ValueError("heston_kappa must be positive for Heston sigma_mode.")
            if self.heston_theta is None or self.heston_theta < 0.0:
                raise ValueError("heston_theta must be non-negative for Heston sigma_mode.")
            if self.heston_sigma_v is None or self.heston_sigma_v < 0.0:
                raise ValueError("heston_sigma_v must be non-negative for Heston sigma_mode.")
            if not (-1.0 <= float(self.heston_rho) <= 1.0):
                raise ValueError("heston_rho must be in [-1, 1] for Heston sigma_mode.")

            # Initialize variance v_0. Prefer explicit heston_v0, fall back to sigma^2.
            if self.heston_v0 is not None:
                v0 = float(self.heston_v0)
                if v0 <= 0.0:
                    raise ValueError("heston_v0 must be positive for Heston sigma_mode.")
            else:
                if self.sigma <= 0.0:
                    raise ValueError(
                        "sigma must be positive when using sigma_mode='heston' without explicit heston_v0."
                    )
                v0 = float(self.sigma) ** 2
            self._heston_v = max(1e-18, v0)
            # Keep sigma in sync with sqrt(variance) for downstream consumers (plots, etc.).
            self.sigma = math.sqrt(self._heston_v)
            # Regime-related attributes are unused in Heston mode.
            self.regime_state = "H"
            self.sigma_low = None
            self.sigma_high = None
        else:
            # Static mode: keep provided sigma and ignore regime params
            self.regime_state = "L"
            self.sigma_low = None
            self.sigma_high = None

    @property
    def regime_enabled(self) -> bool:
        return self.sigma_mode == "regime"

    def _sigma_for_state(self, state: str) -> float:
        if self.regime_enabled:
            return self.sigma_low if state == "L" else self.sigma_high  # type: ignore[arg-type]
        return self.sigma

    def _transition_regime(self) -> None:
        """Advance the Markov chain and update the active sigma."""
        if not self.regime_enabled:
            return
        current = self.regime_state
        draw = random.random()
        if current == "L":
            self.regime_state = "L" if draw < self.p_LL else "H"
        else:
            self.regime_state = "H" if draw < self.p_HH else "L"
        self.sigma = self._sigma_for_state(self.regime_state)

    def _sigma_from_sine(self, advance: bool = True) -> float:
        """
        Generate sigma from a noisy sine wave, optionally advancing the counter.
        """
        if self.sigma_mode != "noisy_sine":
            return self.sigma
        step = self._sigma_step + (1 if advance else 0)
        phase = self.sigma_sine_phase + 2.0 * math.pi * step / self.sigma_sine_period
        noise = np.random.normal(scale=self.sigma_sine_noise) if self.sigma_sine_noise > 0 else 0.0
        sigma_raw = self._sigma_sine_center + self._sigma_sine_amp_eff * math.sin(phase) + noise
        sigma_new = max(self.sigma_floor, sigma_raw)
        if advance:
            self._sigma_step += 1
        return sigma_new

    def _update_sigma(self) -> None:
        if self.sigma_mode == "regime":
            self._transition_regime()
        elif self.sigma_mode == "noisy_sine":
            self.sigma = self._sigma_from_sine()
        # Heston mode updates sigma together with the variance process in diffuse_only.

    def _diffuse_heston(self) -> None:
        """
        One discrete-time Heston step over Δt = 1 (per micro-step).

        Variance:
            v_{t+1} = max(0, v_t + κ(θ - v_t)Δt + σ_v sqrt(max(v_t, 0)) sqrt(Δt) z_1)

        Price:
            log M_{t+1} = log M_t + (μ - 0.5 v_t)Δt
                           + sqrt(max(v_t, 0)) * (ρ z_1 + sqrt(1-ρ²) z_2) * sqrt(Δt)
        """
        # Guard: this should only be called in Heston mode, but keep it robust.
        if self.sigma_mode != "heston":
            # Fallback to GBM update if misused.
            z = np.random.normal()
            self.m *= math.exp(self.mu - 0.5 * self.sigma**2 + self.sigma * z)
            self.m = max(1e-12, self.m)
            return

        v_t = max(0.0, float(self._heston_v))
        dt = 1.0
        # Two independent standard normals
        z1 = np.random.normal()
        z2 = np.random.normal()
        sqrt_v_t = math.sqrt(max(v_t, 0.0))

        # Variance update (full truncation to keep v >= 0)
        kappa_v = float(self.heston_kappa)  # type: ignore[arg-type]
        theta_v = float(self.heston_theta)  # type: ignore[arg-type]
        sigma_v = float(self.heston_sigma_v)  # type: ignore[arg-type]
        dv = kappa_v * (theta_v - v_t) * dt + sigma_v * sqrt_v_t * math.sqrt(dt) * z1
        v_next = max(0.0, v_t + dv)
        self._heston_v = v_next
        sigma_eff = math.sqrt(max(v_next, 0.0))
        self.sigma = sigma_eff

        # Correlated shock for price
        rho = float(self.heston_rho)  # type: ignore[arg-type]
        rho = max(-1.0, min(1.0, rho))
        z_price = rho * z1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * z2

        log_m = math.log(max(self.m, 1e-18))
        log_m_next = log_m + (self.mu - 0.5 * v_t) * dt + sqrt_v_t * math.sqrt(dt) * z_price
        self.m = max(1e-12, math.exp(log_m_next))

    def step(self, delta_a_cex_signed: float) -> float:
        """
        Apply permanent, additive impact from the CEX trade in token A units,
        then diffuse via GBM or Heston dynamics. Returns the impact applied (for debugging).
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
        """Diffuse the reference price via GBM/Heston without additional impact."""
        if self.sigma_mode == "heston":
            self._diffuse_heston()
        else:
            self._update_sigma()
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
    seed_is_passive: bool = True,
) -> List:
    """
    Split the binomial hill across `num_seed_lps` seed LPs so burns are staggered.
    Each seed LP has its own review clock; all have mintProb=0, is_active_narrow=False,
    and optionally behave as passive LPs (seed_is_passive=True).
    """
    from agents import LPAgent, Position
    assert num_seed_lps >= 1
    center_tick = pool._snap(pool.tick)
    S_entry = pool.S

    # prepare seed LPs
    seed_LPs: List[LPAgent] = []
    for j in range(num_seed_lps):
        sid = seed_lp_id_base + j
        lp = LPAgent(
            id=sid,
            mintProb=seed_mint_prob,
            is_active_narrow=False,
            is_passive=seed_is_passive,
            is_seed=True,
        )
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

def scenario_output_root(config_path: Path, base_dir: Path | str | None = None) -> Path:
    """
    Derive the per-scenario output directory from a YAML config path.

    For a config like `.../sigma_sine_fee_volatility.yml` this returns
    `abm_results/scenarios/sigma_sine_fee_volatility` (by default) and ensures
    the directory exists.
    """
    if base_dir is None:
        base_dir = Path("abm_results") / "scenarios"
    base_dir = Path(base_dir)
    scenario_name = Path(config_path).stem
    out_root = base_dir / scenario_name
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


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
    missing_keys = []
    for name, param in signature.parameters.items():
        if name in params:
            continue
        if param.default is inspect._empty:
            missing_keys.append(name)
        else:
            # populate optional parameters with their default if omitted in config
            params[name] = param.default
    if missing_keys:
        raise ValueError(f"Missing simulate parameters in {config_path}: {missing_keys}")

    extra_keys = sorted(set(params) - set(signature.parameters))
    if extra_keys:
        raise ValueError(f"Unexpected keys in 'simulate' section: {extra_keys}")

    scenario_label = params.get("fee_mode")
    if scenario_label is None:
        raise ValueError(
            f"Missing 'fee_mode' in {config_path}. Provide it either at the top level or inside the simulate() parameters."
        )

    return str(scenario_label), dict(params)

