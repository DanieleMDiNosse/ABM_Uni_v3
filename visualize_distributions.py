"""
Visualize the stochastic components and distributions used in the ABM simulation.

Outputs Plotly figures (HTML, and PNG if kaleido is installed) for:
- Initial binomial-hill liquidity distribution.
- Binomial noise term in the LP width rule.
- Log-normal trader notional distribution.
- Log-normal LP mint-size scale distribution.
- Geometric distribution for LP review clocks.
- Poisson arrival distributions:
  - traders: intents per micro-step and per block
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
IMPACT_KAPPA = 0.0  # permanent impact scale (unused here)
HESTON_KAPPA = 0.1
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
TRADER_MEAN = 3
TRADER_SIGMA = 1.5

# LP mint size log-normal parameters (log space)
MINT_MU = 2.5
MINT_SIGMA = 1.5

# LP review clock (geometric) parameter: mean inter-review time ≈ tau
TAU = 10.0

# Arrival distributions (Poisson)
# In block mode (block_time = B > 1):
#   - traders: per micro-step N_k ~ Poisson(trades_per_block / B)
#   - LP targets: per block N ~ Poisson(target_per_block)
BLOCK_TIME = 5
SMART_TRADES_PER_BLOCK = 2.0
NOISE_TRADES_PER_BLOCK = 20.0
NARROW_MINTS_PER_BLOCK = 10.0
PASSIVE_MINTS_PER_BLOCK = 5.0
PASSIVE_BURNS_PER_BLOCK = 2.0

# Generic sample size for Monte Carlo histograms
N_SAMPLES = 500_000

# Output directory for figures
RESULTS_DIR = Path(__file__).resolve().parent / "abm_results"


# =============================================================================
# Heston reference market simulation
# =============================================================================


def simulate_heston_path() -> Tuple[List[float], List[float]]:
    """
    Run the same Heston diffusion as the simulator and return price and volatility series.
    """
    np.random.seed(SEED)
    random.seed(SEED)

    ref = ReferenceMarket(
        m=INITIAL_PRICE,
        mu=DRIFT,
        sigma=math.sqrt(HESTON_V0),
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
    Binomial-hill per-tick liquidity levels used for initial seed LP bootstrap.
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
    n_samples: int = N_SAMPLES,
) -> np.ndarray:
    """
    Sample the binomial noise term used in the LP width rule (tick-spacing units).
    """
    if n <= 0 or not (0.0 < p < 1.0):
        return np.zeros(n_samples, dtype=float)
    K = np.random.binomial(n, p, size=n_samples)
    return (K - n * p).astype(float)


def sample_trader_notional(
    mean_log: float = TRADER_MEAN,
    sigma_log: float = TRADER_SIGMA,
    n_samples: int = N_SAMPLES,
) -> np.ndarray:
    """
    Sample trader notionals following the log-normal model used in run.py.
    """
    return np.exp(np.random.normal(loc=mean_log, scale=sigma_log, size=n_samples))


def sample_lp_mint_scale(
    mean_log: float = MINT_MU,
    sigma_log: float = MINT_SIGMA,
    n_samples: int = N_SAMPLES,
) -> np.ndarray:
    """
    Sample the LP mint-size scale factor X ~ LogNormal(mint_mu, mint_sigma)
    before budget and cooldown caps are applied.
    """
    return np.random.lognormal(mean=mean_log, sigma=sigma_log, size=n_samples)


def sample_review_intervals(
    tau: float = TAU,
    n_samples: int = N_SAMPLES,
) -> np.ndarray:
    """
    Sample geometric inter-review intervals used for LP review clocks.

    In the simulator, p = 1 / tau and next_review ~ Geometric(p) with support {1,2,...}.
    """
    p = 1.0 / max(1.0, float(tau))
    return np.random.geometric(p, size=n_samples).astype(float)


def sample_poisson_per_micro_step(per_block_rate: float, block_time: int, n_samples: int = N_SAMPLES) -> np.ndarray:
    """
    Sample Poisson arrivals per micro-step with intensity per_block_rate / block_time.
    """
    B = max(1, int(block_time))
    lam = max(0.0, float(per_block_rate)) / B
    return np.random.poisson(lam, size=n_samples).astype(float)


def sample_poisson_per_block(per_block_rate: float, n_samples: int = N_SAMPLES) -> np.ndarray:
    """
    Sample Poisson arrivals per block with intensity per_block_rate.
    """
    lam = max(0.0, float(per_block_rate))
    return np.random.poisson(lam, size=n_samples).astype(float)


# =============================================================================
# Plotly figures
# =============================================================================


def plot_heston_paths(prices: List[float], vols: List[float]) -> go.Figure:
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
    np.random.seed(SEED + 1)

    pool, _ = build_empty_pool()
    tick_spacing = pool.tick_spacing

    L_vals = sample_initial_liquidity()
    width_noise = sample_width_noise()
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
        x_label="Noise (tick-spacing units)",
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
    np.random.seed(SEED + 2)

    smart_micro = sample_poisson_per_micro_step(SMART_TRADES_PER_BLOCK, BLOCK_TIME)
    noise_micro = sample_poisson_per_micro_step(NOISE_TRADES_PER_BLOCK, BLOCK_TIME)
    smart_block = sample_poisson_per_block(SMART_TRADES_PER_BLOCK)
    noise_block = sample_poisson_per_block(NOISE_TRADES_PER_BLOCK)
    narrow_mints = sample_poisson_per_block(NARROW_MINTS_PER_BLOCK)
    passive_mints = sample_poisson_per_block(PASSIVE_MINTS_PER_BLOCK)
    passive_burns = sample_poisson_per_block(PASSIVE_BURNS_PER_BLOCK)

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=(
            f"Smart intents / micro-step (λ={SMART_TRADES_PER_BLOCK}/{BLOCK_TIME})",
            f"Noise intents / micro-step (λ={NOISE_TRADES_PER_BLOCK}/{BLOCK_TIME})",
            f"Smart intents / block (λ={SMART_TRADES_PER_BLOCK})",
            f"Noise intents / block (λ={NOISE_TRADES_PER_BLOCK})",
            f"Narrow mint targets / block (λ={NARROW_MINTS_PER_BLOCK})",
            f"Passive mint targets / block (λ={PASSIVE_MINTS_PER_BLOCK})",
            f"Passive burn targets / block (λ={PASSIVE_BURNS_PER_BLOCK})",
            "",
        ),
    )

    _add_hist_with_mean(fig, row=1, col=1, data=smart_micro, x_label="Count", y_label="Probability", color="#1f77b4", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=1, col=2, data=noise_micro, x_label="Count", y_label="Probability", color="#ff7f0e", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=1, col=3, data=smart_block, x_label="Count", y_label="Probability", color="#2ca02c", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=1, col=4, data=noise_block, x_label="Count", y_label="Probability", color="#d62728", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=2, col=1, data=narrow_mints, x_label="Count", y_label="Probability", color="#9467bd", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=2, col=2, data=passive_mints, x_label="Count", y_label="Probability", color="#8c564b", log_y=True, histnorm="probability")
    _add_hist_with_mean(fig, row=2, col=3, data=passive_burns, x_label="Count", y_label="Probability", color="#e377c2", log_y=True, histnorm="probability")

    fig.update_xaxes(visible=False, row=2, col=4, showgrid=True, gridcolor="lightgray", gridwidth=1)
    fig.update_yaxes(visible=False, row=2, col=4, showgrid=True, gridcolor="lightgray", gridwidth=1)
    fig.update_layout(template="plotly_white", height=750, width=1600, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def _write_plotly(fig: go.Figure, stem: str) -> None:
    html_path = RESULTS_DIR / f"{stem}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
        fig.write_image(RESULTS_DIR / f"{stem}.png", scale=2)
    except Exception:
        pass


def main() -> None:
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
