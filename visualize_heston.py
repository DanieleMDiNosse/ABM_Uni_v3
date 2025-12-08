"""
Visualize the Heston dynamics used in the simulation by running the same
ReferenceMarket diffusion and plotting price and volatility paths.
"""

import math
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import ReferenceMarket


# --------------------
# Hyperparameters
# --------------------
SEED = 7
N_STEPS = 10_000
INITIAL_PRICE = 2_000.0
DRIFT = 0.0  # per-step drift of log price
IMPACT_KAPPA = 0.0  # permanent impact scale (unused here, keep for parity)

# Heston variance process parameters
HESTON_KAPPA = 0.1       # mean reversion velocity
HESTON_THETA = 1.0e-8    # long run variance (vol=sqrt(var))
HESTON_SIGMA_V = 0.001    # volatility of variance
HESTON_RHO = -0.5         # correlation between price and variance shocks
HESTON_V0 = 2.25e-8       # initial variance; sqrt gives the initial volatility


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


def plot_paths(prices: List[float], vols: List[float]) -> None:
    steps = range(len(prices))
    returns = np.diff(np.log(prices))  # empirical log-returns
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    ax_price, ax_ret = axes[0]
    ax_vol, ax_vol_hist = axes[1]

    ax_price.plot(steps, prices, color="C0")
    ax_price.set_ylabel("Price (m_t)")
    ax_price.set_title("Heston price path")

    ax_vol.plot(steps, vols, color="C1")
    ax_vol.set_ylabel("Volatility σ_t (sqrt variance)")
    ax_vol.set_xlabel("Step")
    ax_vol.set_title("Volatility path")

    ax_ret.hist(returns, bins=50, density=True, color="C2", alpha=0.8)
    ax_ret.axvline(0.0, color="k", linestyle="--", linewidth=1, label="Zero")
    ax_ret.set_title("Empirical log-returns")
    ax_ret.set_xlabel("Return")
    ax_ret.set_ylabel("Density")
    ax_ret.set_yscale('log')
    ax_ret.legend()

    ax_vol_hist.hist(vols, bins=50, density=True, color="C3", alpha=0.8)
    ax_vol_hist.axvline(np.mean(vols), color="k", linestyle="--", linewidth=1, label=fr"Mean $σ_t$: {np.mean(vols):.5f}")
    ax_vol_hist.set_title("Empirical volatility (σ_t)")
    ax_vol_hist.set_xlabel("Volatility")
    ax_vol_hist.set_ylabel("Density")
    ax_vol_hist.set_yscale('log')
    ax_vol_hist.legend()

    fig.tight_layout()
    plt.show()


def main() -> None:
    prices, vols = simulate_heston_path()
    plot_paths(prices, vols)


if __name__ == "__main__":
    main()
