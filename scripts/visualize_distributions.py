"""
Visualize the stochastic components and distributions used in the ABM simulation.

Outputs Plotly figures (HTML, and PNG if kaleido is installed) for:
- Initial binomial-hill liquidity profile (deterministic initialization, not sampled).
- Trader-side Bernoulli draw.
- Binomial noise term in the LP width rule.
- Realized active-LP width distribution from the clipped/snap-to-grid width rule.
- Log-normal trader notional distribution.
- Capped log-normal LP wallet utilization distribution (η = min(1, Z)).
- Geometric distribution for LP review clocks.
- Uniform distribution for LP out-of-range recenter thresholds (k_out_threshold).
- Discrete-uniform LP post-burn cooldown.
- Poisson arrival distributions:
  - traders: intents per block
  - LPs: target mint/burn counts per block
- Effective Bernoulli distribution for JIT (Jiter) attempt blocks.

By default this script mirrors the parameters in `abm_results/scenarios/test.yml`.
You can point it at any run-style scenario YAML (the same config used by `scripts/run.py`)
to visualize the implied distributions for that scenario.
"""

import argparse
import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow `python scripts/visualize_distributions.py ...` to work from any CWD by
# ensuring the repo root (parent of `scripts/`) is on `sys.path` so `import core`
# succeeds.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.utils import EWMA, ReferenceMarket, TICK_LN, build_empty_pool, scenario_output_root
from core.artifacts import (
    build_run_manifest,
    make_unique_dir,
    safe_tag,
    snapshot_file,
    write_csv_rows,
    write_json,
)


# =============================================================================
# Global hyperparameters (aligned with abm_results/scenarios/test.yml by default)
# =============================================================================

DEFAULT_CONFIG_PATH = Path("abm_results") / "scenarios" / "test.yml"
PLOTLY_STATIC_WARNING_EMITTED = False
_DEFAULT_GRID_STYLE = dict(showgrid=True, gridcolor="#e1e1e1", gridwidth=1)
LP_COOLDOWN_BLOCK_MIN = 3
LP_COOLDOWN_BLOCK_MAX = 8


@dataclass(frozen=True)
class DistributionParams:
    """
    Parameters needed to reproduce distribution visualizations.

    Parameters
    ----------
    seed : int
        RNG seed used for both NumPy and Python's `random`.
    n_steps : int
        Number of reference-market steps to simulate.
    initial_price : float
        Starting reference price m_0 (token1 per token0).
    cex_mu : float
        Per-step drift of log price.
    cex_sigma : float
        Base per-step log-return volatility.
    cex_sigma_mode : str
        One of {"static", "heston"}.
    cex_heston_kappa, cex_heston_theta, cex_heston_sigma_v, cex_heston_rho, cex_heston_v0 : float | None
        Heston stochastic volatility parameters (variance process).
    initial_binom_N : int
        Binomial hill depth used to seed initial liquidity.
    initial_total_L : float
        Total initial liquidity to distribute across ticks.
    min_L_per_tick : float
        Minimum per-tick liquidity to include in the histogram.
    w_min_ticks, w_max_ticks : int
        Minimum/maximum narrow-LP widths in tick units after snapping and clipping.
    basis_half_life : int
        EWMA half-life used for the volatility signal inside the width rule.
    slope_s : float
        Width sensitivity in ticks per unit of EWMA volatility (expressed in tick units).
    binom_n, binom_p : int, float
        Binomial noise parameters for the LP width rule.
    trader_mean, trader_sigma : float
        Trader notional log-normal parameters (in log space).
    mint_mu, mint_sigma : float
        LP mint-scale log-normal parameters (in log space).
    tau : int
        Legacy mean LP review interval in blocks. `scripts/run.py` coerces this
        parameter to an integer before constructing the geometric review clock.
    tau_seconds : float | None
        Optional real-time LP review interval in seconds (micro-step = 1 second). When set,
        it overrides `tau` for scheduling semantics in the main simulator.
    k_out_min, k_out_max : int
        Inclusive bounds for the discrete uniform draw of LP out-of-range
        recenter thresholds.
    block_time : int
        Micro-steps per block (used for per-micro-step Poisson visualizations).
    smart_trades_per_block, noise_trades_per_block : float
        Legacy expected trader intents per block.
    smart_trades_per_second, noise_trades_per_second : float | None
        Optional real-time Poisson intensities per second (micro-step). When set, they override
        the legacy per-block knobs in the main simulator.
    narrow_mints_per_block, passive_mints_per_block, passive_burns_per_block : float
        Legacy expected LP event targets per block.
    narrow_mints_per_second, passive_mints_per_second, passive_burns_per_second : float | None
        Optional real-time LP target intensities per second. When set, expected targets per block
        scale with `block_time`.
    p_jit : float
        Bernoulli arrival probability per block for the JIT searcher.
    N_jit : int
        JIT target count enable/disable knob (kept for config alignment).
    liquidity_perc_jit : float
        JIT target liquidity share enable/disable knob (kept for config alignment).
    n_samples : int
        Monte Carlo sample size for histogram-based distributions.

    Returns
    -------
    DistributionParams
        Frozen dataclass instance with validated, simulation-aligned defaults.

    Notes
    -----
    These parameters are intentionally a *subset* of `scripts/run.py`'s `simulate()` inputs:
    only what is needed to visualize the stochastic primitives.
    """

    seed: int = 7
    n_steps: int = 10_000
    initial_price: float = 2_000.0
    cex_mu: float = 0.0
    cex_sigma: float = 0.00015
    cex_sigma_mode: str = "heston"
    cex_heston_kappa: Optional[float] = 1.0
    cex_heston_theta: Optional[float] = 1.0e-8
    cex_heston_sigma_v: Optional[float] = 0.001
    cex_heston_rho: Optional[float] = -0.5
    cex_heston_v0: Optional[float] = 2.25e-8
    initial_binom_N: int = 450
    initial_total_L: float = 500_000.0
    min_L_per_tick: float = 1e-9
    w_min_ticks: int = 10
    w_max_ticks: int = 100
    basis_half_life: int = 1
    slope_s: float = 1.0
    binom_n: int = 70
    binom_p: float = 0.5
    trader_mean: float = 2.5
    trader_sigma: float = 1.5
    mint_mu: float = -1.0
    mint_sigma: float = 1.5
    tau: int = 5
    tau_seconds: Optional[float] = None
    k_out_min: int = 10
    k_out_max: int = 20
    block_time: int = 5
    smart_trades_per_block: float = 0.8
    noise_trades_per_block: float = 0.8
    smart_trades_per_second: Optional[float] = None
    noise_trades_per_second: Optional[float] = None
    narrow_mints_per_block: float = 0.5
    passive_mints_per_block: float = 0.5
    passive_burns_per_block: float = 0.5
    narrow_mints_per_second: Optional[float] = None
    passive_mints_per_second: Optional[float] = None
    passive_burns_per_second: Optional[float] = None
    p_jit: float = 0.0
    N_jit: int = 1
    liquidity_perc_jit: float = 0.90
    n_samples: int = 500_000


# =============================================================================
# Heston reference market simulation
# =============================================================================


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
    """
    Persist a Plotly figure as both HTML and PNG (if Kaleido is available).

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to write.
    png_path : pathlib.Path
        Target PNG output path.
    html_path : pathlib.Path
        Target HTML output path.
    source : str
        Label used in warning messages.
    width, height : int
        PNG export dimensions.
    scale : float
        PNG export scale multiplier.

    Returns
    -------
    None

    Notes
    -----
    This mirrors `scripts/run.py`'s output convention: always write HTML; attempt PNG
    export and emit a single warning if Kaleido is missing.
    """
    global PLOTLY_STATIC_WARNING_EMITTED
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


def _normalize_sigma_mode(mode: str) -> str:
    mode_norm = (mode or "static").lower()
    if mode_norm not in {"static", "heston"}:
        raise ValueError(
            f"Invalid cex_sigma_mode '{mode}'. Expected one of ['heston', 'static']."
        )
    return mode_norm


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def load_scenario_config(config_path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Load a run-style scenario YAML and return its label and `simulate` parameters.

    Parameters
    ----------
    config_path : pathlib.Path
        Path to a YAML scenario file containing a top-level `simulate` mapping.

    Returns
    -------
    scenario_label : str
        Scenario label used for filenames (prefers top-level `fee_mode`).
    simulate_params : dict[str, Any]
        Mapping of inputs passed to `scripts/run.py`'s `simulate()` function.

    Notes
    -----
    This intentionally does *not* validate that all `simulate()` parameters are
    present; this script only needs a subset for distribution visualization.
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configuration file: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)

    if not isinstance(config_data, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")

    simulate_params = config_data.get("simulate")
    if not isinstance(simulate_params, dict):
        raise ValueError(f"'simulate' section missing in {config_path}")

    # Match `core.utils.load_simulation_parameters` behavior: prefer top-level fee_mode.
    fee_mode = config_data.get("fee_mode")
    if fee_mode is not None:
        simulate_fee_mode = simulate_params.get("fee_mode")
        if simulate_fee_mode is not None and simulate_fee_mode != fee_mode:
            raise ValueError("Conflicting 'fee_mode' between top-level and simulate() parameters.")
        scenario_label = str(fee_mode)
    else:
        scenario_label = str(simulate_params.get("fee_mode", config_path.stem))

    return scenario_label, dict(simulate_params)


def build_distribution_params(
    simulate_params: Dict[str, Any],
    *,
    n_steps: int,
    n_samples: int,
) -> DistributionParams:
    """
    Convert a `simulate()` parameter mapping into `DistributionParams`.

    Parameters
    ----------
    simulate_params : dict[str, Any]
        Run parameters (typically loaded from `abm_results/scenarios/*.yml`).
    n_steps : int
        Number of reference-market steps to simulate for the path plot.
    n_samples : int
        Monte Carlo sample size for histogram distributions.

    Returns
    -------
    DistributionParams
        Parsed parameters with conservative defaults for missing keys.

    Notes
    -----
    `scripts/run.py` uses `build_empty_pool()` for `m0`; here we default to 2000.0 and
    allow overriding via `initial_price` if ever added to configs.
    """
    seed = int(simulate_params.get("seed", DistributionParams.seed))
    block_time = int(simulate_params.get("block_time", DistributionParams.block_time))
    initial_price = float(simulate_params.get("initial_price", DistributionParams.initial_price))
    tau_raw = simulate_params.get("tau")

    cex_mu = float(simulate_params.get("cex_mu", DistributionParams.cex_mu))
    cex_sigma = float(simulate_params.get("cex_sigma", DistributionParams.cex_sigma))
    cex_sigma_mode = str(simulate_params.get("cex_sigma_mode", DistributionParams.cex_sigma_mode))

    return DistributionParams(
        seed=seed,
        n_steps=max(1, int(n_steps)),
        initial_price=initial_price,
        cex_mu=cex_mu,
        cex_sigma=cex_sigma,
        cex_sigma_mode=_normalize_sigma_mode(cex_sigma_mode),
        cex_heston_kappa=_coerce_optional_float(simulate_params.get("cex_heston_kappa", DistributionParams.cex_heston_kappa)),
        cex_heston_theta=_coerce_optional_float(simulate_params.get("cex_heston_theta", DistributionParams.cex_heston_theta)),
        cex_heston_sigma_v=_coerce_optional_float(simulate_params.get("cex_heston_sigma_v", DistributionParams.cex_heston_sigma_v)),
        cex_heston_rho=_coerce_optional_float(simulate_params.get("cex_heston_rho", DistributionParams.cex_heston_rho)),
        cex_heston_v0=_coerce_optional_float(simulate_params.get("cex_heston_v0", DistributionParams.cex_heston_v0)),
        initial_binom_N=int(simulate_params.get("initial_binom_N", DistributionParams.initial_binom_N)),
        initial_total_L=float(simulate_params.get("initial_total_L", DistributionParams.initial_total_L)),
        min_L_per_tick=float(simulate_params.get("min_L_per_tick", DistributionParams.min_L_per_tick)),
        w_min_ticks=int(simulate_params.get("w_min_ticks", DistributionParams.w_min_ticks)),
        w_max_ticks=int(simulate_params.get("w_max_ticks", DistributionParams.w_max_ticks)),
        basis_half_life=int(simulate_params.get("basis_half_life", DistributionParams.basis_half_life)),
        slope_s=float(simulate_params.get("slope_s", DistributionParams.slope_s)),
        binom_n=int(simulate_params.get("binom_n", DistributionParams.binom_n)),
        binom_p=float(simulate_params.get("binom_p", DistributionParams.binom_p)),
        trader_mean=float(simulate_params.get("trader_mean", DistributionParams.trader_mean)),
        trader_sigma=float(simulate_params.get("trader_sigma", DistributionParams.trader_sigma)),
        mint_mu=float(simulate_params.get("mint_mu", DistributionParams.mint_mu)),
        mint_sigma=float(simulate_params.get("mint_sigma", DistributionParams.mint_sigma)),
        tau=int(DistributionParams.tau if tau_raw is None else tau_raw),
        tau_seconds=_coerce_optional_float(simulate_params.get("tau_seconds", DistributionParams.tau_seconds)),
        k_out_min=int(simulate_params.get("k_out_min", DistributionParams.k_out_min)),
        k_out_max=int(simulate_params.get("k_out_max", DistributionParams.k_out_max)),
        block_time=max(1, block_time),
        smart_trades_per_block=float(
            DistributionParams.smart_trades_per_block
            if simulate_params.get("smart_trades_per_block") is None
            else simulate_params.get("smart_trades_per_block")
        ),
        noise_trades_per_block=float(
            DistributionParams.noise_trades_per_block
            if simulate_params.get("noise_trades_per_block") is None
            else simulate_params.get("noise_trades_per_block")
        ),
        smart_trades_per_second=_coerce_optional_float(simulate_params.get("smart_trades_per_second", DistributionParams.smart_trades_per_second)),
        noise_trades_per_second=_coerce_optional_float(simulate_params.get("noise_trades_per_second", DistributionParams.noise_trades_per_second)),
        narrow_mints_per_block=float(
            DistributionParams.narrow_mints_per_block
            if simulate_params.get("narrow_mints_per_block") is None
            else simulate_params.get("narrow_mints_per_block")
        ),
        passive_mints_per_block=float(
            DistributionParams.passive_mints_per_block
            if simulate_params.get("passive_mints_per_block") is None
            else simulate_params.get("passive_mints_per_block")
        ),
        passive_burns_per_block=float(
            DistributionParams.passive_burns_per_block
            if simulate_params.get("passive_burns_per_block") is None
            else simulate_params.get("passive_burns_per_block")
        ),
        narrow_mints_per_second=_coerce_optional_float(simulate_params.get("narrow_mints_per_second", DistributionParams.narrow_mints_per_second)),
        passive_mints_per_second=_coerce_optional_float(simulate_params.get("passive_mints_per_second", DistributionParams.passive_mints_per_second)),
        passive_burns_per_second=_coerce_optional_float(simulate_params.get("passive_burns_per_second", DistributionParams.passive_burns_per_second)),
        p_jit=float(simulate_params.get("p_jit", DistributionParams.p_jit)),
        N_jit=int(simulate_params.get("N_jit", DistributionParams.N_jit)),
        liquidity_perc_jit=float(simulate_params.get("liquidity_perc_jit", DistributionParams.liquidity_perc_jit)),
        n_samples=max(1, int(n_samples)),
    )


def simulate_reference_market_path(params: DistributionParams) -> Tuple[List[float], List[float]]:
    """
    Simulate a reference market path using the same volatility modes as `scripts/run.py`.

    Parameters
    ----------
    params : DistributionParams
        Scenario-aligned parameters (seed, mu, sigma_mode, and related settings).

    Returns
    -------
    prices : list[float]
        Reference price series m_t (length params.n_steps + 1).
    vols : list[float]
        Volatility series sigma_t (per-step log-return volatility; in Heston mode,
        this is sqrt(variance), length params.n_steps + 1).

    Notes
    -----
    In the main simulator, `build_empty_pool()` pins the initial price to ~2000.
    Here we default to the same starting value and focus on matching the
    `ReferenceMarket` configuration and update rule.

    Examples
    --------
    >>> p = DistributionParams(seed=1, n_steps=3, cex_sigma_mode="static", cex_sigma=1e-4)
    >>> prices, vols = simulate_reference_market_path(p)
    >>> len(prices) == len(vols)
    True
    """
    np.random.seed(int(params.seed))
    random.seed(int(params.seed))

    sigma_mode = _normalize_sigma_mode(params.cex_sigma_mode)
    sigma_for_ref = float(params.cex_sigma)

    ref = ReferenceMarket(
        m=float(params.initial_price),
        mu=float(params.cex_mu),
        sigma=float(sigma_for_ref),
        kappa=1e-3,
        sigma_mode=sigma_mode,
        heston_kappa=params.cex_heston_kappa,
        heston_theta=params.cex_heston_theta,
        heston_sigma_v=params.cex_heston_sigma_v,
        heston_rho=params.cex_heston_rho,
        heston_v0=params.cex_heston_v0,
    )

    prices: List[float] = [ref.m]
    vols: List[float] = [ref.sigma]

    for _ in range(int(params.n_steps)):
        ref.diffuse_only()
        prices.append(ref.m)
        vols.append(ref.sigma)

    return prices, vols


# =============================================================================
# Distribution helpers
# =============================================================================


def _add_hist_with_mean(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    data: np.ndarray,
    x_label: str,
    y_label: str,
    color: str,
    log_y: bool = False,
    nbins: int = 50,
    histnorm: str = "probability density",
) -> None:
    data = np.asarray(data, dtype=float)

    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=nbins,
            histnorm=histnorm,
            marker_color=color,
            opacity=0.85,
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title_text=x_label, row=row, col=col, showgrid=True, gridcolor="lightgray", gridwidth=1)
    fig.update_yaxes(title_text=y_label, type="log" if log_y else "linear", row=row, col=col, showgrid=True, gridcolor="lightgray", gridwidth=1)


def _add_discrete_pmf(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    support: np.ndarray,
    probabilities: np.ndarray,
    x_label: str,
    y_label: str,
    color: str,
    tick_text: Optional[List[str]] = None,
    show_mean: bool = False,
) -> None:
    """
    Add a discrete probability mass function as a bar chart.

    Parameters
    ----------
    fig : go.Figure
        Target Plotly figure.
    row, col : int
        Subplot coordinates.
    support : np.ndarray
        Discrete support values (shape (k,)).
    probabilities : np.ndarray
        Probabilities for each support value (shape (k,)); should sum to ~1.
    x_label, y_label : str
        Axis labels.
    color : str
        Bar color.

    Returns
    -------
    None

    Notes
    -----
    Used for small-support discrete distributions (e.g., Bernoulli, UniformInt).
    """
    x = np.asarray(support, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if x.ndim != 1 or p.ndim != 1 or x.size != p.size:
        raise ValueError("support and probabilities must be 1D arrays of the same length.")
    p_sum = float(np.sum(p))
    if not np.isfinite(p_sum) or p_sum <= 0.0:
        raise ValueError("probabilities must sum to a positive finite value.")
    p = p / p_sum
    mean_val = float(np.sum(x * p))

    fig.add_trace(
        go.Bar(x=x, y=p, marker_color=color, opacity=0.85, showlegend=False),
        row=row,
        col=col,
    )
    if show_mean:
        fig.add_vline(
            x=mean_val,
            line_width=1,
            line_dash="dash",
            line_color="black",
            row=row,
            col=col,
        )
    fig.update_xaxes(title_text=x_label, row=row, col=col, showgrid=True, gridcolor="lightgray", gridwidth=1)
    if tick_text is not None:
        if len(tick_text) != int(x.size):
            raise ValueError("tick_text must match the support size.")
        fig.update_xaxes(
            tickmode="array",
            tickvals=x.tolist(),
            ticktext=list(tick_text),
            row=row,
            col=col,
        )
    fig.update_yaxes(title_text=y_label, row=row, col=col, showgrid=True, gridcolor="lightgray", gridwidth=1)


def sample_initial_liquidity(
    N: int,
    L_total: float,
    min_L_per_tick: float,
) -> np.ndarray:
    """
    Build the initial binomial-hill liquidity levels per tick.

    Parameters
    ----------
    N : int
        Binomial depth (N+1 tick bands).
    L_total : float
        Total liquidity to distribute across ticks.
    min_L_per_tick : float
        Minimum liquidity per tick to include.

    Returns
    -------
    np.ndarray
        Positive liquidity values per eligible tick (shape (n_ticks,)).

    Notes
    -----
    Matches the binomial weight construction used in
    utils.bootstrap_initial_binomial_hill_sharded. This is a deterministic
    initialization profile, not a stochastic sample.

    Examples
    --------
    >>> L = sample_initial_liquidity(N=4, L_total=100.0, min_L_per_tick=0.0)
    >>> L.ndim == 1
    True
    """
    L_vals: List[float] = []
    denom = float(2**N)
    for k in range(N + 1):
        w = math.comb(N, k) / denom
        L_i = w * L_total
        if L_i >= min_L_per_tick:
            L_vals.append(L_i)
    return np.asarray(L_vals, dtype=float)


def sample_trader_side(n_samples: int = 500_000) -> np.ndarray:
    """
    Sample trader directions used by smart and noise traders.

    Parameters
    ----------
    n_samples : int
        Number of side draws.

    Returns
    -------
    np.ndarray
        Integer-coded directions in {0, 1}, where 0 = X_to_Y and 1 = Y_to_X.

    Notes
    -----
    `scripts/run.py` uses `random.choice(["X_to_Y", "Y_to_X"])`, which is a
    symmetric Bernoulli draw on the two directions.
    """
    return np.random.binomial(1, 0.5, size=n_samples).astype(float)


def sample_width_noise(
    n: int,
    p: float,
    tick_spacing: int = 1,
    n_samples: int = 500_000,
) -> np.ndarray:
    """
    Sample the binomial noise term used in the LP width rule.

    Parameters
    ----------
    n : int
        Number of binomial trials.
    p : float
        Success probability in (0, 1).
    tick_spacing : int
        Tick spacing used to scale the noise into tick units.
    n_samples : int
        Number of Monte Carlo samples to draw.

    Returns
    -------
    np.ndarray
        Mean-zero noise in tick units (shape (n_samples,)).

    Notes
    -----
    In scripts/run.py, noise_ticks = (K - n * p) * tick_spacing.

    Examples
    --------
    >>> noise = sample_width_noise(n=2, p=0.5, tick_spacing=10, n_samples=5)
    >>> noise.shape[0] == 5
    True
    """
    if n <= 0 or not (0.0 < p < 1.0):
        return np.zeros(n_samples, dtype=float)
    spacing = max(1, int(tick_spacing))
    K = np.random.binomial(n, p, size=n_samples)
    return ((K - n * p) * spacing).astype(float)


def sample_trader_notional(
    mean_log: float,
    sigma_log: float,
    n_samples: int = 500_000,
) -> np.ndarray:
    """
    Sample trader notionals following the log-normal model.

    Parameters
    ----------
    mean_log : float
        Mean of log-notional (natural log).
    sigma_log : float
        Standard deviation of log-notional.
    n_samples : int
        Number of samples.

    Returns
    -------
    np.ndarray
        Positive notionals in token1 units (shape (n_samples,)).

    Notes
    -----
    Matches scripts/run.py: exp(N(mean_log, sigma_log)).

    Examples
    --------
    >>> x = sample_trader_notional(mean_log=0.0, sigma_log=0.1, n_samples=3)
    >>> x.shape[0]
    3
    """
    return np.exp(np.random.normal(loc=mean_log, scale=sigma_log, size=n_samples))


def sample_lp_mint_scale(
    mean_log: float,
    sigma_log: float,
    n_samples: int = 500_000,
) -> np.ndarray:
    """
    Sample the LP wallet utilization factor used in mint attempts.

    Parameters
    ----------
    mean_log : float
        Mean of log-scale (natural log).
    sigma_log : float
        Standard deviation of log-scale.
    n_samples : int
        Number of samples.

    Returns
    -------
    np.ndarray
        Wallet utilization factors η in [0, 1] (shape (n_samples,)).

    Notes
    -----
    Mirrors `scripts/run.py`'s `_draw_wallet_utilization_factor`:

        Z ~ LogNormal(mean_log, sigma_log)
        η = min(1, Z)

    The cap introduces a point mass at η = 1.

    Examples
    --------
    >>> s = sample_lp_mint_scale(mean_log=0.0, sigma_log=0.1, n_samples=2)
    >>> s.shape[0] == 2
    True
    """
    z = np.random.lognormal(mean=mean_log, sigma=sigma_log, size=n_samples)
    valid = np.isfinite(z) & (z > 0.0)
    eta = np.where(valid, np.minimum(1.0, z), 0.0)
    return eta.astype(float)


def sample_review_intervals(
    tau: float,
    n_samples: int = 500_000,
) -> np.ndarray:
    """
    Sample geometric inter-review intervals for LP review clocks.

    Parameters
    ----------
    tau : float
        Effective mean review interval after applying the simulator's coercions
        and clamps (p = 1 / max(1, tau)).
    n_samples : int
        Number of samples.

    Returns
    -------
    np.ndarray
        Geometric draws on {1, 2, ...} (shape (n_samples,)).

    Notes
    -----
    Matches the geometric draw used by `scripts/run.py` once the review-rate
    parameter has been converted into an effective mean interval.

    Examples
    --------
    >>> r = sample_review_intervals(tau=5, n_samples=4)
    >>> r.shape[0] == 4
    True
    """
    p = 1.0 / max(1.0, float(tau))
    return np.random.geometric(p, size=n_samples).astype(float)


def sample_post_burn_cooldown(
    *,
    block_time: int,
    tau_seconds: Optional[float],
    n_samples: int = 500_000,
) -> np.ndarray:
    """
    Sample LP cooldown durations after a burn.

    Parameters
    ----------
    block_time : int
        Seconds (micro-steps) per block.
    tau_seconds : float | None
        If set, cooldowns are scaled into seconds by multiplying the sampled
        block count by `block_time`, matching the simulator.
    n_samples : int
        Number of cooldown draws.

    Returns
    -------
    np.ndarray
        Cooldown durations in blocks (legacy mode) or seconds (real-time mode).

    Notes
    -----
    Mirrors `scripts/run.py`:

        cooldown_blocks ~ UniformInt[3, 8]
        cooldown = cooldown_blocks * block_time  if tau_seconds is set
        cooldown = cooldown_blocks               otherwise
    """
    cooldown_blocks = np.random.randint(
        LP_COOLDOWN_BLOCK_MIN,
        LP_COOLDOWN_BLOCK_MAX + 1,
        size=n_samples,
    ).astype(float)
    if tau_seconds is not None:
        return cooldown_blocks * float(max(1, int(block_time)))
    return cooldown_blocks


def _effective_review_interval_and_unit(params: DistributionParams) -> Tuple[float, str]:
    """
    Return the simulator-implied mean review interval and its unit label.

    Parameters
    ----------
    params : DistributionParams
        Scenario-aligned visualization parameters.

    Returns
    -------
    tuple[float, str]
        Effective geometric mean interval and unit label (`"blocks"` or `"s"`).

    Notes
    -----
    This mirrors `scripts/run.py`:

    - legacy mode: `review_rate = 1 / max(1, int(tau))`
    - real-time mode: `review_rate = min(1, 1 / max(1e-12, tau_seconds))`

    The returned mean is `1 / review_rate`, which is what the geometric draw
    actually uses inside the simulator.
    """
    if params.tau_seconds is not None:
        review_rate = min(1.0, 1.0 / max(1e-12, float(params.tau_seconds)))
        return 1.0 / review_rate, "s"
    review_rate = 1.0 / max(1, int(params.tau))
    return 1.0 / review_rate, "blocks"


def _effective_jit_attempt_probability(params: DistributionParams) -> float:
    """
    Return the simulator-implied JIT attempt probability per block.

    Parameters
    ----------
    params : DistributionParams
        Scenario-aligned visualization parameters.

    Returns
    -------
    float
        Effective Bernoulli probability in [0, 1].

    Notes
    -----
    In `scripts/run.py`, JIT planning is skipped unless all of the following
    are true:

    - `p_jit > 0`
    - `N_jit > 0`
    - `liquidity_perc_jit > 0`

    `N_jit` no longer increases the target count, but it still acts as an
    enable/disable knob and therefore affects whether Bernoulli attempts occur.
    """
    p_jit = max(0.0, min(1.0, float(params.p_jit)))
    if p_jit <= 0.0:
        return 0.0
    if int(params.N_jit) <= 0:
        return 0.0
    if float(params.liquidity_perc_jit) <= 0.0:
        return 0.0
    return p_jit


def sample_lp_widths_from_reference_prices(
    prices: np.ndarray,
    *,
    basis_half_life: int,
    w_min_ticks: int,
    w_max_ticks: int,
    slope_s: float,
    binom_n: int,
    binom_p: float,
    tick_spacing: int,
) -> np.ndarray:
    """
    Sample realized active-LP widths from a reference-price path.

    Parameters
    ----------
    prices : np.ndarray
        Reference prices `m_t` with shape `(n_steps + 1,)`.
    basis_half_life : int
        EWMA half-life for the volatility signal.
    w_min_ticks, w_max_ticks : int
        Minimum and maximum allowed widths in tick units.
    slope_s : float
        Width sensitivity to EWMA volatility measured in tick units.
    binom_n, binom_p : int, float
        Binomial-noise parameters.
    tick_spacing : int
        Pool tick spacing used for grid snapping.

    Returns
    -------
    np.ndarray
        Realized widths in ticks with shape `(n_steps,)`.

    Notes
    -----
    This follows the exact width rule in `scripts/run.py`:

        vol_hat_t = EWMA(|log m_t - log m_{t-1}|)
        noise_t = (K_t - n p) * tick_spacing
        w_t = clip_round(w_min + slope_s * vol_hat_t / TICK_LN + noise_t)

    where `clip_round` snaps to the tick grid and clamps to `[w_min_ticks, w_max_ticks]`.
    """
    prices_arr = np.asarray(prices, dtype=float)
    if prices_arr.ndim != 1:
        raise ValueError("prices must be a 1D array.")
    if prices_arr.size <= 1:
        return np.asarray([], dtype=float)

    spacing = max(1, int(tick_spacing))
    ewma_width = EWMA(half_life_steps=max(1, int(basis_half_life)))
    prev_m_for_width = float(prices_arr[0])
    noise_ticks = sample_width_noise(
        n=int(binom_n),
        p=float(binom_p),
        tick_spacing=spacing,
        n_samples=int(prices_arr.size - 1),
    )

    min_bands = max(1, (int(w_min_ticks) + spacing - 1) // spacing)
    max_bands = max(1, int(w_max_ticks) // spacing)
    snapped_min = min_bands * spacing
    snapped_max = max_bands * spacing

    widths: List[float] = []
    for idx, m_now in enumerate(prices_arr[1:]):
        try:
            log_m_now = math.log(max(float(m_now), 1e-18))
            log_m_prev = math.log(max(prev_m_for_width, 1e-18))
            vol_obs = abs(log_m_now - log_m_prev)
        except ValueError:
            vol_obs = 0.0
        prev_m_for_width = float(m_now)
        vol_hat = ewma_width.update(vol_obs)
        vol_in_ticks = vol_hat / TICK_LN
        w_unclipped = float(w_min_ticks) + float(slope_s) * vol_in_ticks + float(noise_ticks[idx])
        w_ticks = int(round(w_unclipped / spacing)) * spacing
        w_ticks = max(snapped_min, min(w_ticks, snapped_max))
        widths.append(float(w_ticks))

    return np.asarray(widths, dtype=float)


def sample_poisson_per_micro_step(per_block_rate: float, block_time: int, n_samples: int) -> np.ndarray:
    """
    Sample Poisson arrivals per micro-step.

    Parameters
    ----------
    per_block_rate : float
        Expected arrivals per block.
    block_time : int
        Number of micro-steps per block.
    n_samples : int
        Number of samples.

    Returns
    -------
    np.ndarray
        Poisson counts per micro-step (shape (n_samples,)).

    Notes
    -----
    Uses lambda = per_block_rate / max(1, block_time), matching scripts/run.py.

    Examples
    --------
    >>> x = sample_poisson_per_micro_step(2.0, 4, n_samples=3)
    >>> x.shape[0] == 3
    True
    """
    B = max(1, int(block_time))
    lam = max(0.0, float(per_block_rate)) / B
    return np.random.poisson(lam, size=n_samples).astype(float)


def sample_poisson_per_block(per_block_rate: float, n_samples: int) -> np.ndarray:
    """
    Sample Poisson arrivals per block.

    Parameters
    ----------
    per_block_rate : float
        Expected arrivals per block.
    n_samples : int
        Number of samples.

    Returns
    -------
    np.ndarray
        Poisson counts per block (shape (n_samples,)).

    Notes
    -----
    Uses lambda = per_block_rate, matching scripts/run.py.

    Examples
    --------
    >>> x = sample_poisson_per_block(2.0, n_samples=3)
    >>> x.shape[0] == 3
    True
    """
    lam = max(0.0, float(per_block_rate))
    return np.random.poisson(lam, size=n_samples).astype(float)


# =============================================================================
# Plotly figures
# =============================================================================


def plot_reference_market_paths(prices: List[float], vols: List[float], *, sigma_mode: str) -> go.Figure:
    """
    Build a Plotly panel with the reference market path and histograms.

    Parameters
    ----------
    prices : list[float]
        Reference price series m_t.
    vols : list[float]
        Volatility series sigma_t (sqrt variance).
    sigma_mode : str
        Volatility mode label (either "static" or "heston").

    Returns
    -------
    go.Figure
        2x2 subplot figure with path and histogram panels.

    Notes
    -----
    Log-returns are computed as diff(log(prices)).

    Examples
    --------
    >>> fig = plot_reference_market_paths([1.0, 1.1], [0.1, 0.1], sigma_mode="static")
    >>> isinstance(fig, go.Figure)
    True
    """
    steps = np.arange(len(prices))
    prices_arr = np.asarray(prices, dtype=float)
    vols_arr = np.asarray(vols, dtype=float)
    returns = np.diff(np.log(prices_arr))

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Reference price path (sigma_mode={sigma_mode})",
            "Empirical log-returns",
            "Volatility path (σ_t)",
            "Empirical volatility",
        ),
    )

    fig.add_trace(
        go.Scatter(x=steps, y=prices_arr, mode="lines", line=dict(color="#1f77b4"), showlegend=False),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Price (m_t)", row=1, col=1, showgrid=True, gridcolor="lightgray", gridwidth=1)

    _add_hist_with_mean(
        fig,
        row=1,
        col=2,
        data=returns,
        x_label="Return",
        y_label="Density",
        color="#2ca02c",
        log_y=True,
        histnorm="probability density",
    )

    fig.add_trace(
        go.Scatter(x=steps, y=vols_arr, mode="lines", line=dict(color="#ff7f0e"), showlegend=False),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Step", row=2, col=1, showgrid=True, gridcolor="lightgray", gridwidth=1)
    fig.update_yaxes(title_text="Volatility σ_t", row=2, col=1, showgrid=True, gridcolor="lightgray", gridwidth=1)

    _add_hist_with_mean(
        fig,
        row=2,
        col=2,
        data=vols_arr,
        x_label="Volatility",
        y_label="Density",
        color="#d62728",
        log_y=True,
        histnorm="probability density",
    )

    fig.update_layout(template="plotly_white", height=650, width=1100, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_distribution_suite(
    params: DistributionParams,
    *,
    reference_prices: Optional[List[float]] = None,
) -> go.Figure:
    """
    Build a Plotly figure with core distribution histograms.

    Parameters
    ----------
    params : DistributionParams
        Scenario-aligned distribution parameters.

    Returns
    -------
    go.Figure
        3x3 subplot figure covering deterministic initialization plus the main
        stochastic primitives sampled by the simulator.

    Notes
    -----
    Uses build_empty_pool to obtain tick_spacing for width noise scaling.

    Examples
    --------
    >>> fig = plot_distribution_suite(DistributionParams(n_steps=10, n_samples=1000))
    >>> isinstance(fig, go.Figure)
    True
    """
    # Offset the seed so each panel stays deterministic but independent.
    np.random.seed(int(params.seed) + 1)

    pool, _ = build_empty_pool()
    tick_spacing = pool.tick_spacing
    prices_for_width = reference_prices
    if prices_for_width is None:
        prices_for_width, _ = simulate_reference_market_path(params)

    L_vals = sample_initial_liquidity(
        N=int(params.initial_binom_N),
        L_total=float(params.initial_total_L),
        min_L_per_tick=float(params.min_L_per_tick),
    )
    width_noise = sample_width_noise(
        n=int(params.binom_n),
        p=float(params.binom_p),
        tick_spacing=tick_spacing,
        n_samples=int(params.n_samples),
    )
    sampled_widths = sample_lp_widths_from_reference_prices(
        np.asarray(prices_for_width, dtype=float),
        basis_half_life=int(params.basis_half_life),
        w_min_ticks=int(params.w_min_ticks),
        w_max_ticks=int(params.w_max_ticks),
        slope_s=float(params.slope_s),
        binom_n=int(params.binom_n),
        binom_p=float(params.binom_p),
        tick_spacing=int(tick_spacing),
    )
    trader_notionals = sample_trader_notional(
        mean_log=float(params.trader_mean),
        sigma_log=float(params.trader_sigma),
        n_samples=int(params.n_samples),
    )
    wallet_utilization = sample_lp_mint_scale(
        mean_log=float(params.mint_mu),
        sigma_log=float(params.mint_sigma),
        n_samples=int(params.n_samples),
    )
    review_interval_mean, tau_unit = _effective_review_interval_and_unit(params)
    review_intervals = sample_review_intervals(
        tau=review_interval_mean,
        n_samples=int(params.n_samples),
    )
    cooldowns = sample_post_burn_cooldown(
        block_time=int(params.block_time),
        tau_seconds=params.tau_seconds,
        n_samples=int(params.n_samples),
    )
    cooldown_unit = "s" if params.tau_seconds is not None else "blocks"

    k_out_min = int(params.k_out_min)
    k_out_max = int(params.k_out_max)
    if k_out_min <= 0 or k_out_max <= 0:
        raise ValueError("k_out_min and k_out_max must be positive integers.")
    if k_out_min > k_out_max:
        raise ValueError("k_out_min cannot exceed k_out_max.")
    k_out_support = np.arange(k_out_min, k_out_max + 1, dtype=float)
    k_out_pmf = np.full_like(k_out_support, 1.0 / float(k_out_support.size), dtype=float)

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "Initial liquidity profile per tick (deterministic binomial hill)",
            "Trader side (Bernoulli 0.5 / 0.5)",
            "Trader notional distribution (log-normal)",
            f"LP width noise (Binomial, tick_spacing={tick_spacing})",
            f"Sampled narrow-LP width (clip[{int(params.w_min_ticks)},{int(params.w_max_ticks)}], snap={tick_spacing})",
            "LP wallet utilization η = min(1, Z)",
            f"LP review intervals (Geometric, mean≈{review_interval_mean:g} {tau_unit})",
            f"Out-of-range recenter threshold (UniformInt[{k_out_min},{k_out_max}])",
            f"Post-burn cooldown (UniformInt[{LP_COOLDOWN_BLOCK_MIN},{LP_COOLDOWN_BLOCK_MAX}] {cooldown_unit})",
        ),
    )

    _add_hist_with_mean(
        fig,
        row=1,
        col=1,
        data=L_vals,
        x_label="Liquidity L_i",
        y_label="Density",
        color="#1f77b4",
        log_y=True,
        histnorm="probability density",
    )
    _add_discrete_pmf(
        fig,
        row=1,
        col=2,
        support=np.asarray([0.0, 1.0], dtype=float),
        probabilities=np.asarray([0.5, 0.5], dtype=float),
        x_label="Direction",
        y_label="Probability",
        color="#17becf",
        tick_text=["X_to_Y", "Y_to_X"],
        show_mean=False,
    )
    _add_hist_with_mean(
        fig,
        row=1,
        col=3,
        data=trader_notionals,
        x_label="Notional (token1 units)",
        y_label="Density",
        color="#2ca02c",
        log_y=True,
        histnorm="probability density",
    )
    _add_hist_with_mean(
        fig,
        row=2,
        col=1,
        data=width_noise,
        x_label="Noise (ticks)",
        y_label="Probability",
        color="#ff7f0e",
        log_y=False,
        histnorm="probability",
    )
    _add_hist_with_mean(
        fig,
        row=2,
        col=2,
        data=sampled_widths,
        x_label="Width (ticks)",
        y_label="Probability",
        color="#bcbd22",
        log_y=False,
        histnorm="probability",
    )
    _add_hist_with_mean(
        fig,
        row=2,
        col=3,
        data=wallet_utilization,
        x_label="Utilization factor η",
        y_label="Probability",
        color="#d62728",
        log_y=False,
        histnorm="probability",
    )
    _add_hist_with_mean(
        fig,
        row=3,
        col=1,
        data=review_intervals,
        x_label=f"Review interval ({tau_unit})",
        y_label="Probability",
        color="#9467bd",
        log_y=True,
        histnorm="probability",
    )

    _add_discrete_pmf(
        fig,
        row=3,
        col=2,
        support=k_out_support,
        probabilities=k_out_pmf,
        x_label="k_out_threshold (steps)",
        y_label="Probability",
        color="#7f7f7f",
    )
    _add_hist_with_mean(
        fig,
        row=3,
        col=3,
        data=cooldowns,
        x_label=f"Cooldown ({cooldown_unit})",
        y_label="Probability",
        color="#8c564b",
        log_y=False,
        histnorm="probability",
    )
    fig.update_layout(template="plotly_white", height=1000, width=1500, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_arrival_distributions(params: DistributionParams) -> go.Figure:
    """
    Build a Plotly figure with Poisson arrival distributions.

    Parameters
    ----------
    params : DistributionParams
        Scenario-aligned distribution parameters.

    Returns
    -------
    go.Figure
        3x3 subplot figure covering trader, LP, and JIT arrival counts.

    Notes
    -----
    The main simulator supports two compatible arrival-rate parameterizations:

    - Legacy per-block: provide `*_per_block` and the simulator uses per-micro-step intensity
      `λ_micro = λ_block / block_time`, so the block sum is `Poisson(λ_block)`.
    - Real-time per-second: provide `*_per_second` and the simulator uses `λ_micro = λ_second`
      (micro-step = 1 second), so the block sum is `Poisson(block_time * λ_second)`.
    - JIT attempts occur only when the searcher is enabled, i.e. when
      `p_jit > 0`, `N_jit > 0`, and `liquidity_perc_jit > 0`.

    Examples
    --------
    >>> fig = plot_arrival_distributions(DistributionParams(n_steps=10, n_samples=1000))
    >>> isinstance(fig, go.Figure)
    True
    """
    np.random.seed(int(params.seed) + 2)

    B = max(1, int(params.block_time))
    n_samples = int(params.n_samples)

    if params.smart_trades_per_second is not None:
        smart_lambda_micro = max(0.0, float(params.smart_trades_per_second))
        smart_lambda_block = smart_lambda_micro * float(B)
        smart_micro = np.random.poisson(smart_lambda_micro, size=n_samples).astype(float)
    else:
        smart_lambda_block = max(0.0, float(params.smart_trades_per_block))
        smart_lambda_micro = smart_lambda_block / float(B)
        smart_micro = sample_poisson_per_micro_step(smart_lambda_block, B, n_samples)

    if params.noise_trades_per_second is not None:
        noise_lambda_micro = max(0.0, float(params.noise_trades_per_second))
        noise_lambda_block = noise_lambda_micro * float(B)
        noise_micro = np.random.poisson(noise_lambda_micro, size=n_samples).astype(float)
    else:
        noise_lambda_block = max(0.0, float(params.noise_trades_per_block))
        noise_lambda_micro = noise_lambda_block / float(B)
        noise_micro = sample_poisson_per_micro_step(noise_lambda_block, B, n_samples)

    narrow_lambda_block = (
        max(0.0, float(params.narrow_mints_per_second)) * float(B)
        if params.narrow_mints_per_second is not None
        else max(0.0, float(params.narrow_mints_per_block))
    )
    passive_mints_lambda_block = (
        max(0.0, float(params.passive_mints_per_second)) * float(B)
        if params.passive_mints_per_second is not None
        else max(0.0, float(params.passive_mints_per_block))
    )
    passive_burns_lambda_block = (
        max(0.0, float(params.passive_burns_per_second)) * float(B)
        if params.passive_burns_per_second is not None
        else max(0.0, float(params.passive_burns_per_block))
    )

    smart_block = sample_poisson_per_block(float(smart_lambda_block), n_samples)
    noise_block = sample_poisson_per_block(float(noise_lambda_block), n_samples)
    narrow_mints = sample_poisson_per_block(float(narrow_lambda_block), n_samples)
    passive_mints = sample_poisson_per_block(float(passive_mints_lambda_block), n_samples)
    passive_burns = sample_poisson_per_block(float(passive_burns_lambda_block), n_samples)

    p_jit = _effective_jit_attempt_probability(params)
    jit_support = np.asarray([0.0, 1.0])
    jit_pmf = np.asarray([1.0 - p_jit, p_jit], dtype=float)

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            f"Smart intents / micro-step (1s) (λ={float(smart_lambda_micro):.3g})",
            f"Noise intents / micro-step (1s) (λ={float(noise_lambda_micro):.3g})",
            f"Effective JIT attempt / block (Bernoulli p={p_jit:.3g})",
            f"Smart intents / block (λ={float(smart_lambda_block):g})",
            f"Noise intents / block (λ={float(noise_lambda_block):g})",
            f"Narrow mint targets / block (λ={float(narrow_lambda_block):g})",
            f"Passive mint targets / block (λ={float(passive_mints_lambda_block):g})",
            f"Passive burn targets / block (λ={float(passive_burns_lambda_block):g})",
            "",
        ),
    )

    _add_hist_with_mean(fig, row=1, col=1, data=smart_micro, x_label="Count", y_label="Probability", color="#2ca02c", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=1, col=2, data=noise_micro, x_label="Count", y_label="Probability", color="#d62728", log_y=True, histnorm="probability")

    _add_hist_with_mean(fig, row=2, col=1, data=smart_block, x_label="Count", y_label="Probability", color="#2ca02c", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=2, col=2, data=noise_block, x_label="Count", y_label="Probability", color="#d62728", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=2, col=3, data=narrow_mints, x_label="Count", y_label="Probability", color="#9467bd", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=3, col=1, data=passive_mints, x_label="Count", y_label="Probability", color="#8c564b", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=3, col=2, data=passive_burns, x_label="Count", y_label="Probability", color="#e377c2", log_y=True, histnorm="probability")

    _add_discrete_pmf(
        fig,
        row=1,
        col=3,
        support=jit_support,
        probabilities=jit_pmf,
        x_label="Attempt (0/1)",
        y_label="Probability",
        color="#7f7f7f",
    )
    fig.update_xaxes(visible=False, row=3, col=3)
    fig.update_yaxes(visible=False, row=3, col=3)
    fig.update_layout(template="plotly_white", height=980, width=1200, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def _normal_cdf(x: float) -> float:
    """
    Compute the standard normal CDF Φ(x).

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        Φ(x) in [0, 1].

    Notes
    -----
    Uses the error-function identity:

        Φ(x) = 0.5 * (1 + erf(x / sqrt(2))).
    """
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _run_self_check(params: DistributionParams) -> None:
    """
    Run lightweight, deterministic sanity checks for distribution alignment.

    Parameters
    ----------
    params : DistributionParams
        Scenario-aligned distribution parameters.

    Returns
    -------
    None

    Notes
    -----
    These checks are intended as guardrails for paper figures:
    - Wallet utilization factor η is capped at 1.
    - The point mass at η=1 matches the closed-form probability P(Z>=1).
    - Trader sides are symmetric Bernoulli draws.
    - k_out thresholds form a valid discrete uniform distribution.
    - Burn cooldowns have the correct support in block or second units.
    - Binomial width noise has approximately zero mean.
    - Sampled narrow-LP widths stay on-grid and within the configured clip range.
    """
    n = int(min(50_000, max(10_000, params.n_samples // 50)))
    np.random.seed(int(params.seed) + 12_345)

    # --- Wallet utilization: η = min(1, Z), Z ~ LogNormal(mu, sigma) ---
    eta = sample_lp_mint_scale(float(params.mint_mu), float(params.mint_sigma), n_samples=n)
    if eta.size == 0:
        raise AssertionError("wallet utilization check failed: empty sample.")
    if float(np.min(eta)) < -1e-12 or float(np.max(eta)) > 1.0 + 1e-12:
        raise AssertionError("wallet utilization check failed: η outside [0, 1].")

    sigma = float(params.mint_sigma)
    mu = float(params.mint_mu)
    if sigma <= 0.0:
        expected_mass_at_one = 1.0 if mu >= 0.0 else 0.0
    else:
        # P(Z >= 1) = P(N(mu, sigma) >= 0) = Φ(mu / sigma)
        expected_mass_at_one = _normal_cdf(mu / sigma)
    observed_mass_at_one = float(np.mean(np.isclose(eta, 1.0)))
    if abs(observed_mass_at_one - expected_mass_at_one) > 0.02:
        raise AssertionError(
            "wallet utilization check failed: mass at 1 mismatch "
            f"(observed={observed_mass_at_one:.3f}, expected={expected_mass_at_one:.3f})."
        )

    # --- Trader side: symmetric Bernoulli via random.choice(["X_to_Y", "Y_to_X"]) ---
    trader_side = sample_trader_side(n_samples=n)
    if trader_side.size == 0:
        raise AssertionError("trader side check failed: empty sample.")
    if not np.all(np.isin(trader_side, [0.0, 1.0])):
        raise AssertionError("trader side check failed: support is not {0, 1}.")
    if abs(float(np.mean(trader_side)) - 0.5) > 0.02:
        raise AssertionError("trader side check failed: imbalance exceeds tolerance.")

    # --- k_out threshold: discrete uniform on [k_out_min, k_out_max] ---
    k_out_min = int(params.k_out_min)
    k_out_max = int(params.k_out_max)
    if k_out_min <= 0 or k_out_max <= 0 or k_out_min > k_out_max:
        raise AssertionError("k_out threshold check failed: invalid bounds.")

    # --- Burn cooldown: UniformInt[3, 8] in blocks or seconds ---
    cooldowns = sample_post_burn_cooldown(
        block_time=int(params.block_time),
        tau_seconds=params.tau_seconds,
        n_samples=n,
    )
    expected_support = np.arange(
        LP_COOLDOWN_BLOCK_MIN,
        LP_COOLDOWN_BLOCK_MAX + 1,
        dtype=float,
    )
    if params.tau_seconds is not None:
        expected_support = expected_support * float(max(1, int(params.block_time)))
    observed_support = np.unique(cooldowns)
    if not np.array_equal(observed_support, expected_support):
        raise AssertionError(
            "cooldown check failed: unexpected support "
            f"(observed={observed_support.tolist()}, expected={expected_support.tolist()})."
        )

    # --- Width noise: mean-zero binomial term in tick units ---
    pool, _ = build_empty_pool()
    width_noise = sample_width_noise(
        n=int(params.binom_n),
        p=float(params.binom_p),
        tick_spacing=int(pool.tick_spacing),
        n_samples=n,
    )
    if abs(float(np.mean(width_noise))) > float(pool.tick_spacing):
        raise AssertionError("width noise check failed: mean too far from zero.")

    width_params = replace(params, n_steps=int(min(2_500, max(100, params.n_steps))))
    prices, _ = simulate_reference_market_path(width_params)
    sampled_widths = sample_lp_widths_from_reference_prices(
        np.asarray(prices, dtype=float),
        basis_half_life=int(params.basis_half_life),
        w_min_ticks=int(params.w_min_ticks),
        w_max_ticks=int(params.w_max_ticks),
        slope_s=float(params.slope_s),
        binom_n=int(params.binom_n),
        binom_p=float(params.binom_p),
        tick_spacing=int(pool.tick_spacing),
    )
    if sampled_widths.size == 0:
        raise AssertionError("width-rule check failed: empty sampled-width series.")
    if np.any(np.mod(sampled_widths, float(pool.tick_spacing)) != 0.0):
        raise AssertionError("width-rule check failed: widths are off-grid.")
    min_allowed = max(
        pool.tick_spacing,
        ((int(params.w_min_ticks) + pool.tick_spacing - 1) // pool.tick_spacing) * pool.tick_spacing,
    )
    max_allowed = max(pool.tick_spacing, (int(params.w_max_ticks) // pool.tick_spacing) * pool.tick_spacing)
    if float(np.min(sampled_widths)) < float(min_allowed) or float(np.max(sampled_widths)) > float(max_allowed):
        raise AssertionError("width-rule check failed: sampled widths exceed clip bounds.")


def main() -> None:
    """
    Generate and write all distribution figures to disk.

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    Notes
    -----
    Writes to the same scenario output convention as `scripts/run.py`:
    `abm_results/scenarios/<scenario_name>/distributions/<run_id>/{html,png}`.

    Examples
    --------
    >>> main()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description="Visualize ABM distribution primitives from a scenario YAML.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to a run-style scenario YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DistributionParams.n_steps,
        help="Reference market steps to simulate (default: 10000).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DistributionParams.n_samples,
        help="Monte Carlo samples for histogram distributions (default: 500000).",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run quick distribution sanity checks before exporting figures.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    scenario_label, simulate_params = load_scenario_config(config_path)
    params = build_distribution_params(simulate_params, n_steps=int(args.n_steps), n_samples=int(args.n_samples))

    if bool(args.self_check):
        _run_self_check(params)

    scenario_root = scenario_output_root(config_path)
    out_root_base = scenario_root / "distributions"
    out_root_base.mkdir(parents=True, exist_ok=True)
    run_id_base = safe_tag(f"{scenario_label}_steps{int(args.n_steps)}_samples{int(args.n_samples)}")
    out_root = make_unique_dir(out_root_base / run_id_base)

    snapshot_file(config_path, out_root / "config_snapshot.yml")
    manifest = build_run_manifest(script="visualize_distributions", run_id=out_root.name, config_path=config_path)
    write_json(out_root / "metadata.json", {**manifest.to_dict(), "scenario_label": str(scenario_label)})
    write_csv_rows(
        out_root / "summary.csv",
        [
            {
                "run_id": out_root.name,
                "config_path": str(config_path),
                "scenario_label": str(scenario_label),
                "seed": int(getattr(params, "seed", 0)),
                "n_steps": int(getattr(params, "n_steps", 0)),
                "n_samples": int(getattr(params, "n_samples", 0)),
                "cex_sigma_mode": str(getattr(params, "cex_sigma_mode", "")),
                "cex_sigma": float(getattr(params, "cex_sigma", float("nan"))),
            }
        ],
    )

    html_dir = out_root / "html"
    png_dir = out_root / "png"

    prices, _ = simulate_reference_market_path(params)
    fig_distributions = plot_distribution_suite(params, reference_prices=prices)
    fig_arrivals = plot_arrival_distributions(params)
    save_plotly_figure(
        fig_distributions,
        png_dir / f"distributions_core_{scenario_label}.png",
        html_dir / f"distributions_core_{scenario_label}.html",
        "distributions_core",
        width=1600,
        height=1150,
        scale=1.0,
    )
    save_plotly_figure(
        fig_arrivals,
        png_dir / f"distributions_arrivals_{scenario_label}.png",
        html_dir / f"distributions_arrivals_{scenario_label}.html",
        "distributions_arrivals",
        width=1400,
        height=1050,
        scale=1.0,
    )
    print(f"[distributions] wrote: {out_root}")


if __name__ == "__main__":
    main()
