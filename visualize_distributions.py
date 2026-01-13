"""
Visualize the stochastic components and distributions used in the ABM simulation.

Outputs Plotly figures (HTML, and PNG if kaleido is installed) for:
- Initial binomial-hill liquidity distribution.
- Binomial noise term in the LP width rule.
- Log-normal trader notional distribution.
- Log-normal LP mint-size scale distribution.
- Geometric distribution for LP review clocks.
- Poisson arrival distributions:
  - traders: intents per block
  - LPs: target mint/burn counts per block
- Heston-style reference market path (price, volatility, and return/volatility histograms).
"""

import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import ReferenceMarket, build_empty_pool


# =============================================================================
# Global hyperparameters (aligned with abm_results/scenarios/test.yml by default)
# =============================================================================

SEED = 7

# Heston reference market
N_STEPS = 10_000
INITIAL_PRICE = 2_000.0
DRIFT = 0.0  # per-step drift of log price
IMPACT_KAPPA = 1e-3  # permanent impact scale (unused here)
CEX_SIGMA = 0.00015  # per-step log-return volatility (used when HESTON_V0 is None)
HESTON_KAPPA = 1.0
HESTON_THETA = 1.0e-8
HESTON_SIGMA_V = 0.001
HESTON_RHO = -0.5
HESTON_V0 = 2.25e-8

# Initial liquidity binomial hill
INITIAL_BINOM_N = 450
INITIAL_TOTAL_L = 500_000.0
MIN_L_PER_TICK = 1e-9

# Width noise (binomial in ticks)
BINOM_N = 70
BINOM_P = 0.5

# Trader notional log-normal parameters (log space)
TRADER_MEAN = 2.5
TRADER_SIGMA = 1.5

# LP mint size log-normal parameters (log space)
MINT_MU = 2.5
MINT_SIGMA = 1.5

# LP review clock (geometric) parameter: mean inter-review time ≈ tau
TAU = 5.0

# Arrival distributions (Poisson)
#   - traders: per block N ~ Poisson(trades_per_block)
#   - LP targets: per block N ~ Poisson(target_per_block)
BLOCK_TIME = 5
SMART_TRADES_PER_BLOCK = 1
NOISE_TRADES_PER_BLOCK = 1
NARROW_MINTS_PER_BLOCK = 1
PASSIVE_MINTS_PER_BLOCK = 1
PASSIVE_BURNS_PER_BLOCK = 1

# Generic sample size for Monte Carlo histograms
N_SAMPLES = 500_000

# Output directory for figures
RESULTS_DIR = Path(__file__).resolve().parent / "abm_results"


# =============================================================================
# Heston reference market simulation
# =============================================================================


def simulate_heston_path() -> Tuple[List[float], List[float]]:
    """
    Simulate a Heston-style reference market path.

    Parameters
    ----------
    None.

    Returns
    -------
    prices : list[float]
        Reference price series m_t (length N_STEPS + 1).
    vols : list[float]
        Volatility series sigma_t (sqrt variance, length N_STEPS + 1).

    Notes
    -----
    Uses global constants (SEED, N_STEPS, DRIFT, HESTON_*) to mirror run.py.
    The initial volatility equals sqrt(HESTON_V0) when provided; otherwise it
    uses CEX_SIGMA.

    Examples
    --------
    >>> prices, vols = simulate_heston_path()
    >>> len(prices) == len(vols)
    True
    """
    np.random.seed(SEED)
    random.seed(SEED)

    sigma0 = math.sqrt(HESTON_V0) if HESTON_V0 is not None else float(CEX_SIGMA)
    ref = ReferenceMarket(
        m=INITIAL_PRICE,
        mu=DRIFT,
        sigma=sigma0,
        kappa=IMPACT_KAPPA,
        sigma_mode="heston",
        heston_kappa=HESTON_KAPPA,
        heston_theta=HESTON_THETA,
        heston_sigma_v=HESTON_SIGMA_V,
        heston_rho=HESTON_RHO,
        heston_v0=HESTON_V0,
    )

    prices: List[float] = [ref.m]
    vols: List[float] = [ref.sigma]  # sigma is sqrt of variance v_t

    for _ in range(N_STEPS):
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
    mean_val = float(np.mean(data)) if data.size else 0.0

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
    fig.add_vline(
        x=mean_val,
        line_width=1,
        line_dash="dash",
        line_color="black",
        row=row,
        col=col,
    )
    # fig.add_annotation(
    #     x=mean_val,
    #     y=0.98,
    #     xref="x",
    #     yref="y domain",
    #     # text=f"Mean: {mean_val:.3g}",
    #     showarrow=False,
    #     xanchor="left",
    #     yanchor="top",
    #     bgcolor="rgba(255,255,255,0.90)",
    #     bordercolor="black",
    #     borderwidth=1,
    #     font=dict(color="black", size=12),
    #     row=row,
    #     col=col,
    # )
    fig.update_xaxes(title_text=x_label, row=row, col=col, showgrid=True, gridcolor="lightgray", gridwidth=1)
    fig.update_yaxes(title_text=y_label, type="log" if log_y else "linear", row=row, col=col, showgrid=True, gridcolor="lightgray", gridwidth=1)


def sample_initial_liquidity(
    N: int = INITIAL_BINOM_N,
    L_total: float = INITIAL_TOTAL_L,
    min_L_per_tick: float = MIN_L_PER_TICK,
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
    utils.bootstrap_initial_binomial_hill_sharded.

    Examples
    --------
    >>> L = sample_initial_liquidity(N=4, L_total=100.0)
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


def sample_width_noise(
    n: int = BINOM_N,
    p: float = BINOM_P,
    tick_spacing: int = 1,
    n_samples: int = N_SAMPLES,
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
    In run.py, noise_ticks = (K - n * p) * tick_spacing.

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
    mean_log: float = TRADER_MEAN,
    sigma_log: float = TRADER_SIGMA,
    n_samples: int = N_SAMPLES,
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
    Matches run.py: exp(N(mean_log, sigma_log)).

    Examples
    --------
    >>> x = sample_trader_notional(mean_log=0.0, sigma_log=0.1, n_samples=3)
    >>> x.shape[0]
    3
    """
    return np.exp(np.random.normal(loc=mean_log, scale=sigma_log, size=n_samples))


def sample_lp_mint_scale(
    mean_log: float = MINT_MU,
    sigma_log: float = MINT_SIGMA,
    n_samples: int = N_SAMPLES,
) -> np.ndarray:
    """
    Sample LP mint-size scale factors prior to caps.

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
        Positive scale factors (shape (n_samples,)).

    Notes
    -----
    Mirrors run.py where z ~ LogNormal(mint_mu, mint_sigma) is later truncated
    by min(1, z).

    Examples
    --------
    >>> s = sample_lp_mint_scale(mean_log=0.0, sigma_log=0.1, n_samples=2)
    >>> s.shape[0] == 2
    True
    """
    return np.random.lognormal(mean=mean_log, sigma=sigma_log, size=n_samples)


def sample_review_intervals(
    tau: float = TAU,
    n_samples: int = N_SAMPLES,
) -> np.ndarray:
    """
    Sample geometric inter-review intervals for LP review clocks.

    Parameters
    ----------
    tau : float
        Mean review interval in steps (p = 1 / max(1, tau)).
    n_samples : int
        Number of samples.

    Returns
    -------
    np.ndarray
        Geometric draws on {1, 2, ...} (shape (n_samples,)).

    Notes
    -----
    Matches run.py: next_review ~ Geometric(1 / max(1, tau)).

    Examples
    --------
    >>> r = sample_review_intervals(tau=5, n_samples=4)
    >>> r.shape[0] == 4
    True
    """
    p = 1.0 / max(1.0, float(tau))
    return np.random.geometric(p, size=n_samples).astype(float)


def sample_poisson_per_micro_step(per_block_rate: float, block_time: int, n_samples: int = N_SAMPLES) -> np.ndarray:
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
    Uses lambda = per_block_rate / max(1, block_time), matching run.py.

    Examples
    --------
    >>> x = sample_poisson_per_micro_step(2.0, 4, n_samples=3)
    >>> x.shape[0] == 3
    True
    """
    B = max(1, int(block_time))
    lam = max(0.0, float(per_block_rate)) / B
    return np.random.poisson(lam, size=n_samples).astype(float)


def sample_poisson_per_block(per_block_rate: float, n_samples: int = N_SAMPLES) -> np.ndarray:
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
    Uses lambda = per_block_rate, matching run.py.

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


def plot_heston_paths(prices: List[float], vols: List[float]) -> go.Figure:
    """
    Build a Plotly panel with the Heston path and histograms.

    Parameters
    ----------
    prices : list[float]
        Reference price series m_t.
    vols : list[float]
        Volatility series sigma_t (sqrt variance).

    Returns
    -------
    go.Figure
        2x2 subplot figure with path and histogram panels.

    Notes
    -----
    Log-returns are computed as diff(log(prices)).

    Examples
    --------
    >>> fig = plot_heston_paths([1.0, 1.1], [0.1, 0.1])
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
            "Heston price path",
            "Empirical log-returns",
            "Volatility path",
            "Empirical volatility (σ_t)",
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


def plot_distribution_suite() -> go.Figure:
    """
    Build a Plotly figure with core distribution histograms.

    Parameters
    ----------
    None.

    Returns
    -------
    go.Figure
        2x3 subplot figure covering liquidity, width noise, notionals, mint scale,
        and review intervals.

    Notes
    -----
    Uses build_empty_pool to obtain tick_spacing for width noise scaling.

    Examples
    --------
    >>> fig = plot_distribution_suite()
    >>> isinstance(fig, go.Figure)
    True
    """
    np.random.seed(SEED + 1)

    pool, _ = build_empty_pool()
    tick_spacing = pool.tick_spacing

    L_vals = sample_initial_liquidity()
    width_noise = sample_width_noise(tick_spacing=tick_spacing)
    trader_notionals = sample_trader_notional()
    lp_scale = sample_lp_mint_scale()
    review_intervals = sample_review_intervals()

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Initial liquidity per tick (binomial hill)",
            f"LP width noise (Binomial, tick_spacing={tick_spacing})",
            "Trader notional distribution (log-normal)",
            "LP mint scale X (log-normal)",
            f"LP review intervals (Geometric, mean≈{TAU:g})",
            "",
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
    _add_hist_with_mean(
        fig,
        row=1,
        col=2,
        data=width_noise,
        x_label="Noise (ticks)",
        y_label="Probability",
        color="#ff7f0e",
        log_y=False,
        histnorm="probability",
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
        data=lp_scale,
        x_label="Scale factor X",
        y_label="Density",
        color="#d62728",
        log_y=True,
        histnorm="probability density",
    )
    _add_hist_with_mean(
        fig,
        row=2,
        col=2,
        data=review_intervals,
        x_label="Steps between reviews",
        y_label="Probability",
        color="#9467bd",
        log_y=True,
        histnorm="probability",
    )

    fig.update_xaxes(visible=False, row=2, col=3, showgrid=True, gridcolor="lightgray", gridwidth=1)
    fig.update_yaxes(visible=False, row=2, col=3, showgrid=True, gridcolor="lightgray", gridwidth=1)
    fig.update_layout(template="plotly_white", height=750, width=1400, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_arrival_distributions() -> go.Figure:
    """
    Build a Plotly figure with Poisson arrival distributions.

    Parameters
    ----------
    None.

    Returns
    -------
    go.Figure
        2x4 subplot figure covering trader and LP arrival counts.

    Notes
    -----
    Trader arrivals are shown per block only.

    Examples
    --------
    >>> fig = plot_arrival_distributions()
    >>> isinstance(fig, go.Figure)
    True
    """
    np.random.seed(SEED + 2)

    smart_block = sample_poisson_per_block(SMART_TRADES_PER_BLOCK)
    noise_block = sample_poisson_per_block(NOISE_TRADES_PER_BLOCK)
    narrow_mints = sample_poisson_per_block(NARROW_MINTS_PER_BLOCK)
    passive_mints = sample_poisson_per_block(PASSIVE_MINTS_PER_BLOCK)
    passive_burns = sample_poisson_per_block(PASSIVE_BURNS_PER_BLOCK)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            f"Smart intents / block (λ={SMART_TRADES_PER_BLOCK})",
            f"Noise intents / block (λ={NOISE_TRADES_PER_BLOCK})",
            f"Narrow mint targets / block (λ={NARROW_MINTS_PER_BLOCK})",
            f"Passive mint targets / block (λ={PASSIVE_MINTS_PER_BLOCK})",
            f"Passive burn targets / block (λ={PASSIVE_BURNS_PER_BLOCK})",
            "",
        ),
    )

    _add_hist_with_mean(fig, row=1, col=1, data=smart_block, x_label="Count", y_label="Probability", color="#2ca02c", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=1, col=2, data=noise_block, x_label="Count", y_label="Probability", color="#d62728", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=1, col=3, data=narrow_mints, x_label="Count", y_label="Probability", color="#9467bd", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=2, col=1, data=passive_mints, x_label="Count", y_label="Probability", color="#8c564b", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=2, col=2, data=passive_burns, x_label="Count", y_label="Probability", color="#e377c2", log_y=True, histnorm="probability")

    fig.update_xaxes(visible=False, row=2, col=3, showgrid=True, gridcolor="lightgray", gridwidth=1)
    fig.update_yaxes(visible=False, row=2, col=3, showgrid=True, gridcolor="lightgray", gridwidth=1)
    fig.update_layout(template="plotly_white", height=750, width=1200, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def _write_plotly(fig: go.Figure, stem: str) -> None:
    html_path = RESULTS_DIR / f"{stem}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
        fig.write_image(RESULTS_DIR / f"{stem}.png", scale=2)
    except Exception:
        pass


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
    Writes HTML files to RESULTS_DIR and PNGs when Kaleido is available.

    Examples
    --------
    >>> main()  # doctest: +SKIP
    """
    prices, vols = simulate_heston_path()
    fig_heston = plot_heston_paths(prices, vols)
    fig_distributions = plot_distribution_suite()
    fig_arrivals = plot_arrival_distributions()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_plotly(fig_heston, "heston_paths")
    _write_plotly(fig_distributions, "distribution_suite")
    _write_plotly(fig_arrivals, "arrival_distributions")


if __name__ == "__main__":
    main()
