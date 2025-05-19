"""
ABM Uniswap‑v3 toy simulation — v1.7
===================================
* Single‑tick pool (equivalent to a Uniswap‑v2 constant‑product AMM).
* Retail traders only hit the AMM when its payout is **no worse than θ** below
  the CEX payout; they never move the CEX price.
* Arbitrageurs compute the **optimal trade size** that brings the pool price
  onto the fee‑adjusted CEX price and execute the opposite leg on the CEX,
  pushing the reference price with a nonlinear impact function.
* Reserves, prices, no‑arb bands and arbitrage markers are plotted.
"""

from __future__ import annotations
import sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng()
EPS = 1e-18  # numerical safety floor

# ────────────────────────── AMM pool ──────────────────────────
class UniswapV3Pool:
    """Single‑tick Uniswap‑v3 pool (behaves like v2)."""

    def __init__(self, price: float, liquidity: float, fee_rate: float):
        self.sqrtP = math.sqrt(max(price, EPS))
        self.L = liquidity          # active liquidity
        self.fee_rate = fee_rate    # τ

    # --- swaps --------------------------------------------------------------
    def swap_a_for_b(self, delta_a: float) -> float:  # token‑A in → B out
        eta = 1 - self.fee_rate
        delta_a_net = delta_a * eta
        denom = self.L + max(delta_a_net * self.sqrtP, EPS)
        sqrtP_tgt = max((self.L * self.sqrtP) / denom, EPS)
        delta_b = self.L * (self.sqrtP - sqrtP_tgt)
        self.sqrtP = sqrtP_tgt
        return delta_b

    def swap_b_for_a(self, delta_b: float) -> float:  # token‑B in → A out
        eta = 1 - self.fee_rate
        delta_b_net = delta_b * eta
        sqrtP_tgt = max(self.sqrtP + delta_b_net / self.L, EPS)
        delta_a = self.L * (1 / self.sqrtP - 1 / sqrtP_tgt)
        self.sqrtP = sqrtP_tgt
        return delta_a

    # --- state helpers ------------------------------------------------------
    @property
    def price(self) -> float:
        return self.sqrtP ** 2

    @property
    def reserve_a(self) -> float:
        return self.L / self.sqrtP

    @property
    def reserve_b(self) -> float:
        return self.L * self.sqrtP

# ─────────────────────── reference market (CEX) ─────────────────────────
class ReferenceMarket:
    def __init__(self, price: float, kappa: float, xi: float, sigma: float, mu: float):
        self.price = max(price, 1e-9)
        self.kappa, self.xi = kappa, xi      # impact parameters
        self.sigma, self.mu = sigma, mu      # diffusion parameters

    def trade_impact(self, delta_a: float):
        impact = self.kappa * math.copysign(abs(delta_a) ** (1 + self.xi), delta_a)
        self.price = max(self.price + impact, 1e-9)

    def diffuse(self):
        self.price *= math.exp(self.sigma * rng.normal() + self.mu)
        if not math.isfinite(self.price) or self.price <= 0:
            self.price = 1e-9

# ───────────────────────── retail trader ───────────────────────────────
class Trader:
    """Compares AMM payout with CEX payout; trades only on the AMM."""

    def __init__(self, Qa0: float, Qb0: float, theta_tr: float, fee_rate: float):
        self.Qa, self.Qb = Qa0, Qb0
        self.theta = theta_tr     # tolerance (0.005 → 0.5 % worse acceptable)
        self.fee_rate = fee_rate

    def act(self, pool: UniswapV3Pool, ref: ReferenceMarket):
        sell_a = rng.random() < 0.5
        if sell_a:
            delta_a = abs(rng.normal(0.01, 0.02)) * self.Qa
            if delta_a < EPS:
                return
            # AMM quote (Δb out)
            eta = 1 - self.fee_rate
            L, s = pool.L, pool.sqrtP
            s_tgt = (L * s) / (L + delta_a * eta * s)
            dex_b = L * (s - s_tgt)
            # CEX quote (Δb out)
            cex_b = ref.price * delta_a
            if dex_b >= cex_b * (1 - self.theta):  # AMM no worse than θ
                pool.swap_a_for_b(delta_a)
                self.Qa -= delta_a
                self.Qb += dex_b
        else:  # sell B
            delta_b = abs(rng.normal(0.01, 0.02)) * self.Qb
            if delta_b < EPS:
                return
            eta = 1 - self.fee_rate
            L, s = pool.L, pool.sqrtP
            s_tgt = s + (delta_b * eta) / L
            dex_a = L * (1 / s - 1 / s_tgt)
            cex_a = delta_b / ref.price
            if dex_a >= cex_a * (1 - self.theta):
                pool.swap_b_for_a(delta_b)
                self.Qb -= delta_b
                self.Qa += dex_a

# ────────────────────────── arbitrageur ───────────────────────────────
class Arbitrageur:
    def __init__(self, fee_rate: float):
        self.fee_rate = fee_rate

    # helper volumes to move price to target √P*
    def _a_in_to_target(self, pool: UniswapV3Pool, sqrtP_tgt: float) -> float:
        s, L = pool.sqrtP, pool.L
        return L * (s - sqrtP_tgt) / (s * sqrtP_tgt)

    def _b_in_to_target(self, pool: UniswapV3Pool, sqrtP_tgt: float) -> float:
        s, L = pool.sqrtP, pool.L
        return L * (sqrtP_tgt - s)

    def act(self, pool: UniswapV3Pool, ref: ReferenceMarket) -> int:
        """Return +1 (arb sell), −1 (arb buy), 0 (idle)."""
        eta = 1 - self.fee_rate
        upper = ref.price / eta
        lower = ref.price * eta
        P = pool.price

        # Pool overpriced → send A to pool, buy same A on CEX (Δa positive)
        if P > upper * (1 + 1e-12):
            sqrtP_tgt = math.sqrt(upper)
            delta_a_net = self._a_in_to_target(pool, sqrtP_tgt)
            if delta_a_net <= 0:
                return 0
            ref.trade_impact(+delta_a_net)                 # buy A on CEX
            pool.swap_a_for_b(delta_a_net / eta)
            return 1

        # Pool under‑priced → send B to pool, receive A, then sell A on CEX
        if P < lower * (1 - 1e-12):
            sqrtP_tgt = math.sqrt(lower)
            delta_b_net = self._b_in_to_target(pool, sqrtP_tgt)
            if delta_b_net <= 0:
                return 0
            delta_a_out = pool.swap_b_for_a(delta_b_net / eta)
            ref.trade_impact(-delta_a_out)                 # sell A on CEX
            return -1
        return 0

# ───────────────────────── simulation loop ────────────────────────────

def simulate(T: int = 2_000,
             pool_price: float = 1.0,
             pool_liquidity: float = 1_000_000.0,
             fee_rate: float = 0.003,
             num_traders: int = 100,
             num_arbs: int = 5):
    pool = UniswapV3Pool(pool_price, pool_liquidity, fee_rate)
    ref  = ReferenceMarket(pool_price, kappa=1e-6, xi=0.1, sigma=0.01, mu=0.0)

    traders = [Trader(10_000.0, 10_000.0 * pool_price, 0.005, fee_rate)
               for _ in range(num_traders)]
    arbs = [Arbitrageur(fee_rate) for _ in range(num_arbs)]

    eta = 1 - fee_rate
    hist = np.zeros((T, 7))

    for t in range(T):
        # retail flow (AMM only)
        for tr in traders:
            tr.act(pool, ref)

        # arbitrage (may impact CEX)
        signal = 0
        for arb in arbs:
            sgn = arb.act(pool, ref)
            if sgn != 0:
                signal = sgn

        upper, lower = ref.price / eta, ref.price * eta
        hist[t] = [pool.price, ref.price, upper, lower, signal, pool.reserve_a, pool.reserve_b]

        ref.diffuse()

    df = pd.DataFrame(hist, columns=["pool_price", "ref_price", "upper_band", "lower_band", "arb_signal", "reserve_a", "reserve_b"])
    df["t"] = df.index
    return df

# ─────────────────────────────── main & viz ───────────────────────────
if __name__ == "__main__":
    try:
        steps = int(sys.argv[1])
    except (IndexError, ValueError):
        steps = 2_000

    df = simulate(T=steps)
    print(df.tail())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["t"], df["pool_price"], label="Pool (AMM)")
    ax.plot(df["t"], df["ref_price"], alpha=0.4, label="Reference (CEX)")
    ax.plot(df["t"], df["upper_band"], "--", color="black", linewidth=0.8, label="No-arb band")
    ax.plot(df["t"], df["lower_band"], "--", color="black", linewidth=0.8)

    sells = df[df["arb_signal"] == 1]
    buys  = df[df["arb_signal"] == -1]
    ax.scatter(sells["t"], sells["pool_price"], color="red", marker="v", s=45, label="Arb sell")
    ax.scatter(buys["t"], buys["pool_price"], color="green", marker="^", s=45, label="Arb buy")

    ax.set_xlabel("Time step")
    ax.set_ylabel("Price")
    ax.set_title("ABM Uniswap-v3 — Only Arbitrage Impacts CEX & Reserves")
    ax.legend()
    plt.tight_layout()
    plt.show()