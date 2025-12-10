"""
Visualize the stochastic components and distributions used in the ABM simulation:

- Initial binomial-hill liquidity distribution.
- Binomial noise term in the LP width rule.
- Log-normal trader notional distribution.
- Log-normal LP mint-size scale distribution.
- Geometric distribution for LP review clocks.
- Heston-style reference market path (price, volatility, and return/volatility histograms).
"""

import math
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from utils import ReferenceMarket, build_empty_pool


# =============================================================================
# Global hyperparameters (aligned with tests/test.yml by default)
# =============================================================================

SEED = 7

# Heston reference market
N_STEPS = 10_000
INITIAL_PRICE = 2_000.0
DRIFT = 0.0          # per-step drift of log price
IMPACT_KAPPA = 0.0   # permanent impact scale (unused here, keep for parity)
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
    vols: List[float] = [ref.sigma]  # sigma is sqrt of the variance v_t

    for _ in range(N_STEPS):
        ref.diffuse_only()  # uses ReferenceMarket._diffuse_heston under the hood
        prices.append(ref.m)
        vols.append(ref.sigma)

    return prices, vols


def plot_heston_paths(prices: List[float], vols: List[float]) -> plt.Figure:
    """
    Plot Heston price/volatility paths and their empirical distributions.
    """
    steps = np.arange(len(prices))
    returns = np.diff(np.log(prices))  # empirical log-returns

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    ax_price, ax_ret = axes[0]
    ax_vol, ax_vol_hist = axes[1]

    # Price path
    ax_price.plot(steps, prices, color="C0")
    ax_price.set_ylabel("Price (m_t)")
    ax_price.set_title("Heston price path")

    # Volatility path
    ax_vol.plot(steps, vols, color="C1")
    ax_vol.set_ylabel("Volatility σ_t (sqrt variance)")
    ax_vol.set_xlabel("Step")
    ax_vol.set_title("Volatility path")

    # Log-return histogram
    ax_ret.hist(returns, bins=50, density=True, color="C2", alpha=0.8)
    mean_ret = float(np.mean(returns))
    ax_ret.axvline(
        mean_ret,
        color="k",
        linestyle="--",
        linewidth=1,
        label=f"Mean return: {mean_ret:.2e}",
    )
    ax_ret.set_title("Empirical log-returns")
    ax_ret.set_xlabel("Return")
    ax_ret.set_ylabel("Density")
    ax_ret.set_yscale("log")
    ax_ret.legend()

    # Volatility histogram
    ax_vol_hist.hist(vols, bins=50, density=True, color="C3", alpha=0.8)
    mean_vol = float(np.mean(vols))
    ax_vol_hist.axvline(
        mean_vol,
        color="k",
        linestyle="--",
        linewidth=1,
        label=fr"Mean $σ_t$: {mean_vol:.5f}",
    )
    ax_vol_hist.set_title("Empirical volatility (σ_t)")
    ax_vol_hist.set_xlabel("Volatility")
    ax_vol_hist.set_ylabel("Density")
    ax_vol_hist.set_yscale("log")
    ax_vol_hist.legend()

    fig.tight_layout()
    return fig


# =============================================================================
# Distribution helpers
# =============================================================================

def plot_hist_with_mean(
    data: np.ndarray,
    ax: plt.Axes,
    title: str,
    xlabel: str,
    log_y: bool = False,
    color: str = "C0",
) -> None:
    """
    Plot a 1D histogram with a vertical line at the sample mean.
    """
    data = np.asarray(data)
    ax.hist(data, bins=50, density=True, color=color, alpha=0.8)
    mean_val = float(np.mean(data))
    ax.axvline(
        mean_val,
        color="k",
        linestyle="--",
        linewidth=1,
        label=f"Mean: {mean_val:.3g}",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    if log_y:
        ax.set_yscale("log")
    ax.legend()


def sample_initial_liquidity(
    N: int = INITIAL_BINOM_N,
    L_total: float = INITIAL_TOTAL_L,
    min_L_per_tick: float = MIN_L_PER_TICK,
) -> np.ndarray:
    """
    Sample the per-tick liquidity levels implied by the binomial-hill bootstrap
    used for initial seed LPs.

    We reproduce the same binomial weights as in `bootstrap_initial_binomial_hill_sharded`,
    but only care about the L_i values (not the tick locations or LP assignment).
    """
    L_vals: List[float] = []
    denom = float(2 ** N)
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
    Sample the binomial noise term used in the LP width rule, measured in ticks.

    In the simulator the noise term is:
        noise_ticks = (K - n p) * tick_spacing
    where K ~ Bin(n, p). Here we normalize by tick_spacing so the values are in
    "tick-spacing units"; the mean is still 0.
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


def plot_distribution_suite() -> plt.Figure:
    """
    Plot all static distributions used in the simulation in a single figure.
    """
    # Ensure deterministic samples separate from Heston path
    np.random.seed(SEED + 1)

    # Grab a representative tick spacing (for documentation only)
    pool, _ = build_empty_pool()
    tick_spacing = pool.tick_spacing

    # Sample distributions
    L_vals = sample_initial_liquidity()
    width_noise = sample_width_noise()
    trader_notionals = sample_trader_notional()
    lp_scale = sample_lp_mint_scale()
    review_intervals = sample_review_intervals()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax_init, ax_width, ax_trader, ax_lp, ax_review, ax_empty = axes.flatten()

    # Initial liquidity (binomial hill)
    plot_hist_with_mean(
        L_vals,
        ax_init,
        "Initial liquidity per tick (binomial hill)",
        "Liquidity L_i",
        log_y=True,
        color="C0",
    )

    # Width noise (binomial)
    plot_hist_with_mean(
        width_noise,
        ax_width,
        f"Width noise in ticks (Bin(n={BINOM_N}, p={BINOM_P}))",
        "Noise (tick-spacing units)",
        log_y=False,
        color="C1",
    )
    ax_width.set_title(
        f"LP width noise (Binomial, tick_spacing={tick_spacing})",
    )

    # Trader notional (log-normal)
    plot_hist_with_mean(
        trader_notionals,
        ax_trader,
        "Trader notional distribution (log-normal)",
        "Notional (token1 units)",
        log_y=True,
        color="C2",
    )

    # LP mint scale (log-normal)
    plot_hist_with_mean(
        lp_scale,
        ax_lp,
        "LP mint scale X (log-normal)",
        "Scale factor X",
        log_y=True,
        color="C3",
    )

    # LP review clock intervals (geometric)
    plot_hist_with_mean(
        review_intervals,
        ax_review,
        f"LP review intervals (Geometric, mean≈{TAU:g})",
        "Steps between reviews",
        log_y=True,
        color="C4",
    )

    # Hide unused axis
    ax_empty.axis("off")

    fig.tight_layout()
    return fig


def main() -> None:
    prices, vols = simulate_heston_path()
    fig_heston = plot_heston_paths(prices, vols)
    fig_distributions = plot_distribution_suite()

    # Persist figures for reuse in documentation or reports
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_heston.savefig(RESULTS_DIR / "heston_paths.png", dpi=200)
    fig_distributions.savefig(RESULTS_DIR / "distribution_suite.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
