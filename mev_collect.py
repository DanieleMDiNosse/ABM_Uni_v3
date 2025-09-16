#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MEV collector for a single Uniswap v3 pool
==========================================

What this script does
---------------------
• Scans a per-pool, per-event dataset (swaps/mints/burns) and **detects three MEV patterns** in a
  same-block, single-pool, *strict* sense:
  1) **Pure JIT**: `mint(att) → victim swaps (≠att) → burn(att)` (contiguous)  
  2) **Classical Sandwich**: `front(att) → victim swaps (dir D, ≠att) → back(att, dir ¬D)` (contiguous)  
     **JIT-Sandwich** variant is also detected: `front → mint → victim swaps → burn → back`
  3) **Reverse Back-run**: `victim swap by A` immediately followed by `swap by B` in opposite direction

• For each pattern, it **augments rows with Section 3 theory**—normalized sizes (σ, ε), direction-aware
  price-impact I(·), viability thresholds (e.g., σ_min for back-runs), and **profit ceilings** (optimal
  Π⋆_br, sandwich π⋆ under slippage γ, JIT bribe ceilings)—all exactly as derived in
  Section 3. Formulas/eq. numbers below refer to that Section. :contentReference[oaicite:0]{index=0}

• **Outputs computation-ready CSVs** in `--outdir`:
  - `jit_cycles_tidy.csv`
  - `sandwich_attacks_tidy.csv`
  - `reverse_backruns_tidy.csv`
  Each already contains the raw anchors (tx hashes, directions, amounts) **plus** the theory fields.

Key theory used (Section 3 of the PDF)
--------------------------------------
• Fee fraction f (tier), retention r≔1−f; native price P₀≔y₀/x₀; virtual reserves `x₀=L/√P`, `y₀=L√P`.  
• Direction-aware price impact of a victim of normalized size σ (Eq. 8 and mirror):  
  X→Y:  I = 1/(1+rσ)² − 1;  Y→X: I = (1+rσ)² − 1.  
• **Back-run** threshold and optimum: σ_min (Eq. 13), Π⋆_br(σ) (Eq. 15).  
• **Sandwich** feasibility cap ε_max(σ,γ,r) from slippage (Eq. 27) and optimal π⋆(σ; r, γ) (Eqs. 29–35).  
• **JIT** ceiling m_max^JIT = S·f − G_JIT (Sec. 3.1). :contentReference[oaicite:1]{index=1}

Inputs & schema
---------------
Accepts CSV/Parquet/Pickle with standard Uniswap v3 pool logs. Column names are auto-resolved if present:

Required:
  • event type:   `eventType|event`  (values like 'swap','mint','burn')
  • block:        `blockNumber`
  • log index:    `logIndex`
  • tx hash:      `transactionHash`
  • pool deltas:  `amount0`, `amount1`  (Uniswap v3 semantics)

Optional (strongly recommended when available):
  • origins:      `origin|owner|sender`
  • mint range:   `tickLower`, `tickUpper`
  • gas metrics:  `gasUsed`, `effectiveGasPrice|gasPrice`
  • pre-state:    `L_before`, `sqrt_before|sqrtPriceX96_before`, `x_before`, `y_before`
  • event state:  `liquidityAfter_event|L_after|liquidity`, `sqrtPriceX96_event|sqrt_after`
  • ticks:        `tick_before`, `tick_event|tick_after`, `tick_after`  (for diagnostics)

CLI
---
    --in PATH                Input CSV/Parquet
    --outdir DIR             Output directory (default: ./mev_out)
    --n-jobs N               Worker processes (-1 = all cores)
    --chunk-size K           Blocks per chunk (default: 64)
    --min-victims M          Min victim swaps for a sandwich (default: 1)
    --fee-bps BPS            Pool fee tier in bps (e.g., 5, 30, 100). If omitted, defaults to 5 bps.
    --gamma G                Victim slippage tolerance γ for sandwich theory (default: 0.01 = 1%)
    --grid-npoints N         Grid resolution for sandwich π⋆ under γ (default: 128)
    --quiet                  Less verbose
    --recompute              Ignore existing CSVs and recompute

Sign conventions & evaluation units
-----------------------------------
• **Uniswap v3 swap semantics (pool deltas):**  
  X→Y (token0→token1): `amount0>0`, `amount1<0`;  Y→X: `amount0<0`, `amount1>0`.  
• **Trader flow = −(pool delta)/r**. Signs and normalizations are **direction-aware**, and profits are
  **valued in token0 units** at P₀ unless noted. Many “native” normalizations pick the direction base
  (token0 for X→Y, token1 for Y→X), exactly as in Section 3. :contentReference[oaicite:2]{index=2}

-----------------------------------------
Common variables (present across outputs)
-----------------------------------------
Raw anchors & state
  • `block_number`                    – Block number of the pattern
  • `pattern_type`                    – 'Pure JIT' | 'Classical' | 'JIT-Sandwich' | 'ReverseBackrun'
  • `L0`, `sqrtP0_Q96`                – Active liquidity and √price (Q96) **before** the victim/front
  • `x0`, `y0`                        – Virtual reserves mapped from (L0, √P₀): x₀=L/√P, y₀=L√P
  • `S_net_token0`, `S_net_token1`    – Victim **net** pool deltas (after fee)
  • `victim_dir`                      – 'x2y' or 'y2x' (direction-aware base)
  • `gas_used`, `gas_price`           – Arrays with gas metrics for the whole sequence (order-aligned)

Theory (direction-aware, Section 3)
  • `fee_fraction` (f), `r` (=1−f)    – Fee and retention
  • `sigma_net`, `sigma_gross`        – Victim size S normalized by native base (after/before fee)
  • `sigma0_net|gross`, `sigma1_net|gross` – Side-specific normalized sizes
  • `I_theory`                        – Theoretical price impact of the victim (Eq. 8 and mirror)
  (All three strategies carry these shared fields.) :contentReference[oaicite:3]{index=3}

---------------------------------
JIT-specific variables (Pure JIT)
---------------------------------
Anchors
  • `origin`                          – JIT LP address
  • `mint_tick_lower`, `mint_tick_upper` – Provision range
  • `mint_tx`, `burn_tx`              – Anchoring txs; `victim_txs` (list) between them

Ceilings & diagnostics
  • `mmax_JIT_per_x0`                 – JIT bribe ceiling normalized by x₀ (= f · σ_gross)
  • `mmax_JIT_token0`                 – JIT bribe ceiling in token0 units (= f·S_gross, Y→X converted by P₀)
(From Sec. 3.1; bribe ceiling `S·f−G_JIT`; gas conversion left to analysis.) :contentReference[oaicite:4]{index=4}

---------------------------------------------
Sandwich-specific variables (Classical / JIT-Sandwich)
---------------------------------------------
Anchors & amounts
  • `origin`                          – Attacker address (both legs share the same origin)
  • `front_dir`, `back_dir`           – Swap directions ('swap_x2y'/'swap_y2x')
  • `front_a0`, `front_a1`            – Front-run pool deltas
  • `back_a0`,  `back_a1`             – Back-run pool deltas
  • `front_tx`, `back_tx`             – Attacker leg tx hashes
  • `mint_tx`, `burn_tx`              – Present only for JIT-Sandwich
  • `victim_txs` (list)               – Contiguous victim swaps between attacker legs
  • `P_victim_pre`, `P_victim_post`   – Prices before first victim and after last victim
  • `tx_sequence`                     – (Post-save convenience) compact ordered tx bundle

Realized PnL (token0) & sizes
  • `profit_token0`, `profit_per_x0`  – Trader PnL valued at P₀ (token1 converted by P₀); per-x₀ version
  • `eps_net`, `eps_gross`            – Front size ε normalized by native base (after/before fee)
  • `sigma_*`                         – Victim normalized sizes as in “Common”

Theory comparisons
  • `I_measured`                      – Empirical victim impact: P_post/P_pre − 1
  • `sigma_min_backrun`               – Back-run viability threshold (Eq. 13)
  • `br_pi_star_token0`               – Optimal back-run Π⋆_br(σ) in token0 (Eq. 15; native→token0 via P₀)
  • `br_pi_star_per_x0`               – Π⋆_br normalized by x₀
  • `sand_pi_obs_token0`              – Observed self-funded sandwich π(ε,σ; r)·base, in token0 (Eq. 29)
  • `sand_pi_obs_per_x0`              – Observed π normalized by x₀
  • `sand_pi_star_token0`             – Optimal sandwich profit under γ (grid over ε≤ε_max; Eqs. 27, 33–35)
  • `sand_pi_star`                    – Optimal sandwich profit **per x₀** (normalized)
  • `eps_max`                         – Slippage cap ε_max(σ,γ,r) (Eq. 27)
  • `gamma_used`                      – γ used for that row
  • `sigma_star_br_vs_jit_phi0`       – Closed-form σ⋆ crossover (JIT vs back-run) for φ=0 (Eq. 21)
(Section 3.2–3.3; self-funded sandwich matches your derivation exactly.) :contentReference[oaicite:5]{index=5}

---------------------------------------
Reverse Back-run–specific variables
---------------------------------------
Anchors & amounts
  • `victim_origin`, `arb_origin`     – Distinct wallets A (victim) and B (arbitrageur)
  • `victim_tx`, `back_tx`            – Pair of adjacent, opposite-direction swaps (same block)
  • `P_victim_pre`, `P_victim_post`, `P_back_post` – Prices around the pair
  • `back_a0`, `back_a1`              – Back-run pool deltas (net)

Observed PnL & ceilings (token0)
  • `profit_obs_token0`, `profit_obs_per_x0` – Trader PnL of the back-run leg (valued at P₀)
  • `br_pi_star_token0`, `br_pi_star_per_x0` – Optimal Π⋆_br(σ) from theory (Eq. 15)
  • `mmax_JIT_token0`                 – Comparable JIT ceiling for the same victim flow (token0 units)

Diagnostics
  • `I_measured`                      – Empirical victim impact sign check
  • `viable_sigma`                    – True if σ_native_gross ≥ σ_min_backrun (Eq. 13)
  • `reverted`                        – Price moved back toward P₀ after back-run (coarse target check)
  • `near_target`                     – |P_back_post − r·P₀| ≤ 0.5%·P₀ (Eq. 14 target proximity)
  • `reverse_label`                   – One of:
       'subthreshold' (σ below σ_min), 'wrong_sign' (victim impact sign off),
       'no_reversion' (didn’t revert toward P₀), 'candidate' (passes checks)
(Section 3.2; observed vs optimal back-run economics.) :contentReference[oaicite:6]{index=6}

Notes
-----
• All profits and bribe ceilings reported by this script are **pre-gas in native theory**; any gas costs
  you pass (and arrays the script carries through) let you net them out downstream.
• Where Y→X results must be expressed in token0, conversions use P₀ as in the derivation.  
• When pre-state fields are missing, the script conservatively falls back to adjacent event-state values.
• Direction inference is **robust to missing `eventType` labels** via (amount0, amount1) signs.

References
----------
All formulas, symbols, and thresholds are taken verbatim from Section 3:
JIT (Sec. 3.1), Back-run (Sec. 3.2, Eqs. 8, 13, 15), Sandwich (Sec. 3.3, Eqs. 27–35). :contentReference[oaicite:7]{index=7}
"""


from __future__ import annotations

# Limit BLAS threads before numpy/pandas import
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import argparse
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

Q96 = 2 ** 96

# ---------------------------------------------------------------------
# Helpers: schema & numeric accessors
# ---------------------------------------------------------------------

@dataclass
class Schema:
    col_event: str
    col_block: str
    col_log_index: str
    col_txhash: str
    col_origin: Optional[str]
    col_amount0: str
    col_amount1: str
    col_amount: Optional[str]
    col_tick_lower: Optional[str]
    col_tick_upper: Optional[str]
    col_gas_used: Optional[str]
    col_gas_price: Optional[str]
    col_L_before: Optional[str]
    col_sqrt_before: Optional[str]
    col_x_before: Optional[str]
    col_y_before: Optional[str]
    col_L_after: Optional[str]
    col_sqrt_event: Optional[str]
    col_tick_before: Optional[str]
    col_tick_event: Optional[str]
    col_tick_after: Optional[str]


def resolve_schema(df: pd.DataFrame) -> Schema:
    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_event = pick('eventType', 'event')
    col_block = pick('blockNumber')
    col_log_index = pick('logIndex')
    col_txhash = pick('transactionHash')
    col_origin = pick('origin', 'owner', 'sender')

    col_amount0 = pick('amount0')
    col_amount1 = pick('amount1')
    col_amount = pick('liquidityDelta')

    col_tick_lower = pick('tickLower')
    col_tick_upper = pick('tickUpper')

    col_gas_used = pick('gasUsed')
    col_gas_price = pick('effectiveGasPrice', 'gasPrice')

    col_L_before = pick('L_before')
    col_sqrt_before = pick('sqrt_before', 'sqrtPriceX96_before')
    col_x_before = pick('x_before')
    col_y_before = pick('y_before')

    col_L_after = pick('liquidityAfter_event', 'L_after', 'liquidity')
    col_sqrt_event = pick('sqrtPriceX96_event', 'sqrt_after')

    col_tick_before = pick('tick_before')
    col_tick_event = pick('tick_event', 'tick_after')
    col_tick_after = pick('tick_after')

    for must in [col_event, col_block, col_log_index, col_txhash, col_amount0, col_amount1]:
        if must is None:
            raise ValueError("Missing a required column — check your dataset has eventType,event/logIndex/blockNumber/transactionHash, amount0/amount1.")

    return Schema(
        col_event, col_block, col_log_index, col_txhash, col_origin, col_amount0,
        col_amount1, col_amount, col_tick_lower, col_tick_upper, col_gas_used, col_gas_price,
        col_L_before, col_sqrt_before, col_x_before, col_y_before, col_L_after,
        col_sqrt_event, col_tick_before, col_tick_event, col_tick_after
    )


def to_num(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    # strings of ints/decimals
    try:
        if isinstance(x, str) and x.strip().startswith("0x"):
            return float(int(x, 16))
        return float(x)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------
# Uniswap v3 math helpers (v2-like virtual reserves)
# ---------------------------------------------------------------------

def sqrtp_q96_to_float(sqrt_q96: float) -> float:
    """Convert a Q96 sqrt price to a float sqrt(P)."""
    return float(sqrt_q96) / Q96


def price_from_sqrtq96(sqrt_q96: float) -> float:
    sp = sqrtp_q96_to_float(sqrt_q96)
    return sp * sp


def virtual_xy_from_L_sqrt(L: float, sqrt_q96: float) -> Tuple[float, float]:
    """Map (L, √P_Q96) -> (x, y) via x = L/√P, y = L*√P."""
    sp = sqrtp_q96_to_float(sqrt_q96)
    if sp <= 0 or np.isnan(sp) or np.isnan(L):
        return (np.nan, np.nan)
    return (L / sp, L * sp)


# ---------------------------------------------------------------------
# Section 3 formulas
#   r = 1 - f
#   For X→Y: I = 1/(1+rσ_x)^2 - 1
#   For Y→X: I = (1+rσ_y)^2 - 1
# ---------------------------------------------------------------------

def price_impact_x2y(r: float, sigma_x_gross: float) -> float:
    return 1.0 / (1.0 + r * sigma_x_gross) ** 2 - 1.0


def price_impact_y2x(r: float, sigma_y_gross: float) -> float:
    return (1.0 + r * sigma_y_gross) ** 2 - 1.0


def backrun_sigma_min(r: float) -> float:
    # Eq. (13): σ ≥ (1/√r - 1)/r  (same numeric value for the mirrored case)
    return (1.0 / math.sqrt(r) - 1.0) / r


def backrun_opt_dy_and_profit(x0: float, y0: float, r: float, sigma_gross_x: float) -> float:
    """Return Π*_br measured in token X units (profit in token X).
    Uses the closed-form profit (Eq. 15) in Section 3.
    """
    u = 1.0 + r * sigma_gross_x
    Pi_start = x0 * (math.sqrt(r) * u - 1.0)**2 / (r*u) if (u and u > 0) else np.nan
    return Pi_start

def backrun_opt_profit_base(base_reserve: float, r: float, sigma_native_gross: float) -> float:
    """
    Eq. (15), but generic: returns optimal back-run profit in *native base units*.
    - If victim is X→Y, base is x0 and the result is in token0 units.
    - If victim is Y→X, base is y0 and the result is in token1 units.
    """
    u = 1.0 + r * sigma_native_gross
    if not (u > 0):
        return float("nan")
    return base_reserve * (math.sqrt(r) * u - 1.0) ** 2 / (r * u)


def eps_max_under_slippage(sigma_gross: float, gamma: float, r: float) -> float:
    if gamma >= 1.0:
        return float("inf")
    if gamma < 0:
        gamma = 0.0
    A = (r * sigma_gross) ** 2
    B = 4.0 * (1.0 + r * sigma_gross) / (1.0 - gamma)
    return max(0.0, ((-r * sigma_gross + math.sqrt(A + B)) / 2.0 - 1.0) / r)


def sandwich_profit_normalized(eps_gross: float, sigma_gross: float, r: float) -> float:
    # Eq. (29) normalized π(ε) = [ r^2 ε (1 + rε + rσ)^2 / ( (1 + rε) + r^2 ε (1 + rε + rσ) ) ] - ε
    if eps_gross < 0:
        return np.nan
    A = r * r * eps_gross * (1.0 + r * eps_gross + r * sigma_gross) ** 2
    B = (1.0 + r * eps_gross) + r * r * eps_gross * (1.0 + r * eps_gross + r * sigma_gross)
    if B == 0:
        return np.nan
    return A / B - eps_gross


def sandwich_profit_star(sigma_gross: float, r: float, gamma: float, grid_n: int = 128) -> Tuple[float, float]:
    """Return (ε* (gross), π*(σ; r, γ)) using a simple grid search up to ε_max.
    Sufficient for empirical comparisons; avoids solving the quartic.
    """
    eps_max = eps_max_under_slippage(sigma_gross, gamma, r)
    if not np.isfinite(eps_max) or eps_max <= 0:
        return (0.0, 0.0)
    xs = np.linspace(0.0, eps_max, max(8, int(grid_n)))
    vals = [sandwich_profit_normalized(e, sigma_gross, r) for e in xs]
    j = int(np.nanargmax(vals))
    return (float(xs[j]), float(vals[j]))


def sigma_star_backrun_vs_jit(f: float, phi: float, r: float) -> Optional[float]:
    """Solve Eq. (21) for σ* with g = f - φ (X→Y base). Returns σ* or None if degenerate."""
    g = f - phi
    a = (r - g)
    b = (g - 2.0 * math.sqrt(r))
    c = 1.0
    disc = b * b - 4.0 * a * c
    if a == 0 or disc < 0:
        return None
    u_pos = (-b + math.sqrt(disc)) / (2.0 * a)
    sigma = (u_pos - 1.0) / r
    return sigma


# ---------------------------------------------------------------------
# Block slicer & multiprocessing
# ---------------------------------------------------------------------

_GDF: Optional[pd.DataFrame] = None
_GSCHEMA: Optional[Schema] = None


def _pool_init(df: pd.DataFrame, schema: Schema):
    global _GDF, _GSCHEMA
    _GDF = df
    _GSCHEMA = schema


def _build_block_slices(df: pd.DataFrame, schema: Schema) -> List[Tuple[int, int, int]]:
    blocks = df[schema.col_block].astype(np.int64).to_numpy()
    uniq = np.unique(blocks)
    order = np.argsort(blocks)
    left = np.searchsorted(blocks[order], uniq, side='left')
    right = list(left[1:]) + [len(df)]
    return list(zip(uniq.tolist(), left.tolist(), right))


def _chunkify(slices: List[Tuple[int, int, int]], k: int) -> List[List[Tuple[int, int, int]]]:
    if k <= 1:
        return [[s] for s in slices]
    return [slices[i:i + k] for i in range(0, len(slices), k)]


# ---------------------------------------------------------------------
# Core detectors (operate on the raw df; *no column renaming*)
# ---------------------------------------------------------------------

def _read_block_arrays(sub: pd.DataFrame, schema: Schema) -> Dict[str, Any]:
    ev = sub[schema.col_event].astype(str).str.lower().to_numpy()
    a0 = sub[schema.col_amount0].apply(to_num).to_numpy()
    a1 = sub[schema.col_amount1].apply(to_num).to_numpy()
    tx = sub[schema.col_txhash].astype(str).to_numpy()
    gas_used = sub[schema.col_gas_used].apply(to_num).to_numpy() if schema.col_gas_used else np.full(len(sub), np.nan)
    gas_price = sub[schema.col_gas_price].apply(to_num).to_numpy() if schema.col_gas_price else np.full(len(sub), np.nan)
    amt = sub[schema.col_amount].apply(to_num).to_numpy() if schema.col_amount else np.full(len(sub), np.nan)

    if schema.col_origin:
        origins = sub[schema.col_origin].astype(str).str.lower().to_numpy()
    else:
        origins = np.array([None] * len(sub), dtype=object)

    tlo = sub[schema.col_tick_lower].apply(to_num).to_numpy() if schema.col_tick_lower else np.full(len(sub), np.nan)
    thi = sub[schema.col_tick_upper].apply(to_num).to_numpy() if schema.col_tick_upper else np.full(len(sub), np.nan)

    # pre-state preferred if present
    Lb = sub[schema.col_L_before].apply(to_num).to_numpy() if schema.col_L_before else np.full(len(sub), np.nan)
    sqb = sub[schema.col_sqrt_before].apply(to_num).to_numpy() if schema.col_sqrt_before else np.full(len(sub), np.nan)
    xb = sub[schema.col_x_before].apply(to_num).to_numpy() if schema.col_x_before else np.full(len(sub), np.nan)
    yb = sub[schema.col_y_before].apply(to_num).to_numpy() if schema.col_y_before else np.full(len(sub), np.nan)

    # fallback to event-time state
    La = sub[schema.col_L_after].apply(to_num).to_numpy() if schema.col_L_after else np.full(len(sub), np.nan)
    sqe = sub[schema.col_sqrt_event].apply(to_num).to_numpy() if schema.col_sqrt_event else np.full(len(sub), np.nan)

    return dict(ev=ev, a0=a0, a1=a1, tx=tx, origins=origins, tlo=tlo, thi=thi,
                gas_used=gas_used, gas_price=gas_price, amt=amt,
                Lb=Lb, sqb=sqb, xb=xb, yb=yb, La=La, sqe=sqe)


def _dir_from_amounts(a0: float, a1: float) -> Optional[str]:
    # Uniswap v3 semantics: pool deltas; X→Y has amount0>0, amount1<0; Y→X vice-versa
    if np.isnan(a0) or np.isnan(a1):
        return None
    if a0 > 0 and a1 < 0:
        return 'swap_x2y'
    if a0 < 0 and a1 > 0:
        return 'swap_y2x'
    return None

def detect_reverse_backrun_in_block(sub: pd.DataFrame, schema: Schema) -> List[Dict[str, Any]]:
    """
    Reverse back-run arbitrage (strict, single-pool, same-block adjacency)

    Pattern:
      • Victim swap by wallet A in direction e (strictly a single swap).
      • Immediately next event is a swap by wallet B in the opposite direction (the back-run).
      • A != B.

    Implementation details:
      • We require *adjacency* inside the same block: the back-run must be the very next swap
        after the victim swap (no intervening events). This captures the canonical "reverse trade"
        arbitrage where a searcher back-runs the price dislocation created by A.
      • Uniswap v3 event semantics are pool deltas: amount0 > 0 & amount1 < 0 means X→Y,
        amount0 < 0 & amount1 > 0 means Y→X (token0≡X, token1≡Y).
      • We DO NOT pre-filter on “return > fee” here. Instead, we compute σ and the theoretical
        back-run profit ceiling Π*_br later (augmenter) using the Section 3 formulas so you can
        cut by σ ≥ σ_min or |I| > f in analysis. (σ_min from Eq. (13); Π*_br from Eq. (15).) :contentReference[oaicite:2]{index=2}
      • To avoid classifying sandwich endings as reverse arbitrage, we’ll filter out any candidates
        whose back-run TX matches a sandwich back-run TX *after* sandwich collection completes
        (see main()).

    Output (per row):
      block_number, pattern_type='ReverseBackrun',
      victim_origin, arb_origin,
      victim_dir ('x2y' or 'y2x'),
      victim_tx, back_tx,
      L0, sqrtP0_Q96, x0, y0,
      P_victim_pre, P_victim_post,
      S_net_token0, S_net_token1 (victim pool deltas),
      back_a0, back_a1              (back-run pool deltas),
      gas_used, gas_price           (victim+backrun, as a 2-element list).

    Notes:
      • Theoretical/observed profits are added later by augment_reverse_backruns(), following
        Section 3: I(σ) (Eq. 8), σ_min (Eq. 13), Π*_br (Eq. 15). :contentReference[oaicite:3]{index=3}
      • Style/columns are consistent with detect_jit_in_block() / detect_sandwich_in_block() for
        tidy downstream analysis. :contentReference[oaicite:4]{index=4}
    """
    arr = _read_block_arrays(sub, schema)
    n = len(sub)
    rows: List[Dict[str, Any]] = []

    for i in range(n - 1):
        # victim must be a swap
        is_swap_i = (arr['ev'][i] == 'swap') or arr['ev'][i].startswith('swap_') or (_dir_from_amounts(arr['a0'][i], arr['a1'][i]) is not None)
        if not is_swap_i:
            continue

        # immediately next must be a swap in the opposite direction by a different origin
        j = i + 1
        is_swap_j = (arr['ev'][j] == 'swap') or arr['ev'][j].startswith('swap_') or (_dir_from_amounts(arr['a0'][j], arr['a1'][j]) is not None)
        if not is_swap_j:
            continue

        dir_i = _dir_from_amounts(arr['a0'][i], arr['a1'][i])
        dir_j = _dir_from_amounts(arr['a0'][j], arr['a1'][j])
        if dir_i is None or dir_j is None:
            continue
        if dir_i == dir_j:  # must be opposite
            continue

        origin_i = arr['origins'][i]
        origin_j = arr['origins'][j]
        if origin_i is None or origin_j is None or origin_i == origin_j:
            continue

        # Pre-victim pool state & prices
        L0  = arr['Lb'][i]  if not np.isnan(arr['Lb'][i])  else (arr['La'][i-1]  if i>0 else np.nan)
        SQ0 = arr['sqb'][i] if not np.isnan(arr['sqb'][i]) else (arr['sqe'][i-1] if i>0 else np.nan)
        x0, y0 = (arr['xb'][i], arr['yb'][i]) if not np.isnan(arr['xb'][i]) else virtual_xy_from_L_sqrt(L0, SQ0)

        P_pre  = price_from_sqrtq96(SQ0) if np.isfinite(SQ0) else np.nan
        P_post = price_from_sqrtq96(arr['sqe'][i]) if np.isfinite(arr['sqe'][i]) else np.nan
        P_back_post = price_from_sqrtq96(arr['sqe'][j]) if np.isfinite(arr['sqe'][j]) else np.nan

        rows.append(dict(
            block_number=int(sub.iloc[0][schema.col_block]),
            pattern_type='ReverseBackrun',
            victim_origin=origin_i,
            arb_origin=origin_j,
            victim_dir=('x2y' if dir_i == 'swap_x2y' else 'y2x'),
            victim_tx=arr['tx'][i],
            back_tx=arr['tx'][j],
            L0=L0, sqrtP0_Q96=SQ0, x0=x0, y0=y0,
            P_victim_pre=P_pre, P_victim_post=P_post, P_back_post=P_back_post,
            S_net_token0=float(arr['a0'][i]),
            S_net_token1=float(arr['a1'][i]),
            back_a0=float(arr['a0'][j]),
            back_a1=float(arr['a1'][j]),
            gas_used=[arr['gas_used'][i], arr['gas_used'][j]],
            gas_price=[arr['gas_price'][i], arr['gas_price'][j]],
        ))

    return rows


def detect_jit_in_block(sub: pd.DataFrame, schema: Schema) -> List[Dict[str, Any]]:
    """
    Pure JIT (strict):
      mint(attacker) -> victim swaps (any direction, NOT attacker), all consecutive -> burn(attacker)
    No other actions allowed between anchors. Assumes single-pool input (as in this script).
    """
    arr = _read_block_arrays(sub, schema)
    n = len(sub)
    rows: List[Dict[str, Any]] = []

    i = 0
    while i < n:
        if arr['ev'][i] == 'mint':
            attacker = arr['origins'][i]
            lower    = arr['tlo'][i]
            upper    = arr['thi'][i]
            mint_idx = i

            # --- victims must be consecutive swaps by NON-attacker (any direction) ---
            swaps_idx: List[int] = []
            j = i + 1
            while j < n:
                # victim swap? (swap event & not attacker)
                is_swap = (arr['ev'][j] == 'swap') or arr['ev'][j].startswith('swap_') or (_dir_from_amounts(arr['a0'][j], arr['a1'][j]) is not None)
                if is_swap and arr['origins'][j] != attacker:
                    swaps_idx.append(j)
                    j += 1
                    continue
                break  # first non-victim ends the contiguous victim span

            # need ≥1 victim and next must be the burn by the same attacker (same range), immediately
            if swaps_idx and j < n and arr['ev'][j] == 'burn' and arr['origins'][j] == attacker and arr['tlo'][j] == lower and arr['thi'][j] == upper:
                burn_idx = j

                first_swap = swaps_idx[0]
                last_swap  = swaps_idx[-1]

                # pre-victim state (use state at first swap; fallback to mint post-state)
                L0  = arr['Lb'][first_swap]  if not np.isnan(arr['Lb'][first_swap])  else arr['La'][mint_idx]
                SQ0 = arr['sqb'][first_swap] if not np.isnan(arr['sqb'][first_swap]) else arr['sqe'][mint_idx]
                x0, y0 = (arr['xb'][first_swap], arr['yb'][first_swap]) if not np.isnan(arr['xb'][first_swap]) else virtual_xy_from_L_sqrt(L0, SQ0)

                vict_a0 = float(np.nansum(arr['a0'][first_swap:last_swap+1]))
                vict_a1 = float(np.nansum(arr['a1'][first_swap:last_swap+1]))
                mint_amount = float(arr['amt'][mint_idx]) if not np.isnan(arr['amt'][mint_idx]) else np.nan
                burn_amount = float(arr['amt'][burn_idx]) if not np.isnan(arr['amt'][burn_idx]) else np.nan

                rows.append(dict(
                    block_number=int(sub.iloc[0][schema.col_block]),
                    pattern_type='Pure JIT',
                    origin=attacker,
                    mint_tick_lower=lower,
                    mint_tick_upper=upper,
                    mint_tx=arr['tx'][mint_idx],
                    burn_tx=arr['tx'][burn_idx],
                    victim_txs=[arr['tx'][k] for k in swaps_idx],
                    L0=L0, sqrtP0_Q96=SQ0, x0=x0, y0=y0,
                    S_net_token0=vict_a0, S_net_token1=vict_a1,
                    mint_amount=mint_amount,
                    burn_amount=burn_amount,
                    gas_used=[arr['gas_used'][mint_idx]] + [arr['gas_used'][k] for k in swaps_idx] + [arr['gas_used'][burn_idx]],
                    gas_price=[arr['gas_price'][mint_idx]] + [arr['gas_price'][k] for k in swaps_idx] + [arr['gas_price'][burn_idx]],
                ))
                i = burn_idx + 1
                continue

        i += 1

    return rows



def detect_sandwich_in_block(sub: pd.DataFrame, schema: Schema, min_victims: int) -> List[Dict[str, Any]]:
    """
    Sandwich (strict, single-pool):
      Classical:     front swap(att) -> victim swaps(dir D, not att), all consecutive -> back swap(att, opposite dir)
      JIT-sandwich:  front swap(att) -> mint(att) -> victim swaps(dir D, not att), all consecutive -> burn(att) -> back swap(att, opposite dir)
    No other actions allowed between anchors; both attacker legs share the same origin.
    """
    arr = _read_block_arrays(sub, schema)
    n = len(sub)
    rows: List[Dict[str, Any]] = []

    i = 0
    while i < n:
        # front-run candidate
        is_swap = (arr['ev'][i] == 'swap') or arr['ev'][i].startswith('swap_') or (_dir_from_amounts(arr['a0'][i], arr['a1'][i]) is not None)
        if not is_swap:
            i += 1
            continue

        front_dir = _dir_from_amounts(arr['a0'][i], arr['a1'][i]) or ('swap_x2y' if (arr['a0'][i] > 0) else 'swap_y2x')
        back_dir  = 'swap_x2y' if front_dir == 'swap_y2x' else 'swap_y2x'
        attacker  = arr['origins'][i]

        # pre-front pool state (unchanged fields)
        L0  = arr['Lb'][i]  if not np.isnan(arr['Lb'][i])  else (arr['La'][i-1]  if i>0 else np.nan)
        SQ0 = arr['sqb'][i] if not np.isnan(arr['sqb'][i]) else (arr['sqe'][i-1] if i>0 else np.nan)
        x0, y0 = (arr['xb'][i], arr['yb'][i]) if not np.isnan(arr['xb'][i]) else virtual_xy_from_L_sqrt(L0, SQ0)

        # ==========================
        # A) JIT-sandwich (strict)
        # ==========================
        mint_idx = i + 1 if (i + 1 < n and arr['ev'][i + 1] == 'mint' and arr['origins'][i + 1] == attacker) else None
        if mint_idx is not None:
            # consecutive victims after mint
            victims, victim_idx, victim_hashes = [], [], []
            k = mint_idx + 1
            while k < n:
                is_swap_k = (arr['ev'][k] == 'swap') or arr['ev'][k].startswith('swap_') or (_dir_from_amounts(arr['a0'][k], arr['a1'][k]) is not None)
                if is_swap_k and (_dir_from_amounts(arr['a0'][k], arr['a1'][k]) == front_dir) and (arr['origins'][k] != attacker):
                    victims.append([arr['a0'][k], arr['a1'][k]])
                    victim_idx.append(k)
                    victim_hashes.append(arr['tx'][k])
                    k += 1
                    continue
                break  # end of the strictly consecutive victim window

            # need ≥ min_victims victims and next must be burn(attacker), then next must be back-run(attacker, back_dir)
            if len(victims) >= min_victims and k < n and arr['ev'][k] == 'burn' and arr['origins'][k] == attacker:
                burn_idx = k
                back_idx = k + 1 if (k + 1 < n and (_dir_from_amounts(arr['a0'][k + 1], arr['a1'][k + 1]) == back_dir) and (arr['origins'][k + 1] == attacker)) else None
                if back_idx is not None:
                    SQv0 = arr['sqe'][i]                 # price right after front-run
                    Pv0  = price_from_sqrtq96(SQv0) if np.isfinite(SQv0) else np.nan
                    SQv1 = arr['sqe'][victim_idx[-1]]    # after last victim
                    Pv1  = price_from_sqrtq96(SQv1) if np.isfinite(SQv1) else np.nan
                    vict_a0 = float(np.nansum(arr['a0'][victim_idx]))
                    vict_a1 = float(np.nansum(arr['a1'][victim_idx]))
                    mint_amount = float(arr['amt'][mint_idx]) if not np.isnan(arr['amt'][mint_idx]) else np.nan
                    burn_amount = float(arr['amt'][burn_idx]) if not np.isnan(arr['amt'][burn_idx]) else np.nan

                    rows.append(dict(
                        block_number=int(sub.iloc[0][schema.col_block]),
                        pattern_type='JIT-Sandwich',
                        origin=attacker,
                        front_dir=front_dir,
                        front_a0=float(arr['a0'][i]),
                        front_a1=float(arr['a1'][i]),
                        back_dir=back_dir,
                        back_a0=float(arr['a0'][back_idx]),
                        back_a1=float(arr['a1'][back_idx]),
                        mint_tx=arr['tx'][mint_idx],
                        burn_tx=arr['tx'][burn_idx],
                        front_tx=arr['tx'][i],
                        back_tx=arr['tx'][back_idx],
                        victim_txs=victim_hashes,
                        L0=L0, sqrtP0_Q96=SQ0, x0=x0, y0=y0,
                        P_victim_pre=Pv0, P_victim_post=Pv1,
                        S_net_token0=vict_a0, S_net_token1=vict_a1,
                        mint_amount=mint_amount,
                        burn_amount=burn_amount,
                        gas_used=[arr['gas_used'][i], arr['gas_used'][mint_idx]] + [arr['gas_used'][k] for k in victim_idx] + [arr['gas_used'][burn_idx], arr['gas_used'][back_idx]],
                        gas_price=[arr['gas_price'][i], arr['gas_price'][mint_idx]] + [arr['gas_price'][k] for k in victim_idx] + [arr['gas_price'][burn_idx], arr['gas_price'][back_idx]],
                    ))
                    i = back_idx + 1
                    continue  # finished this pattern

        # ==========================
        # B) Classical (strict)
        # ==========================
        victims, victim_idx, victim_hashes = [], [], []
        k = i + 1
        while k < n:
            is_swap_k = (arr['ev'][k] == 'swap') or arr['ev'][k].startswith('swap_') or (_dir_from_amounts(arr['a0'][k], arr['a1'][k]) is not None)
            if is_swap_k and (_dir_from_amounts(arr['a0'][k], arr['a1'][k]) == front_dir) and (arr['origins'][k] != attacker):
                victims.append([arr['a0'][k], arr['a1'][k]])
                victim_idx.append(k)
                victim_hashes.append(arr['tx'][k])
                k += 1
                continue
            break  # end of strictly consecutive victim window

        if len(victims) >= min_victims:
            # next must be the back-run swap by the SAME attacker in opposite direction
            back_idx = k if (k < n and (_dir_from_amounts(arr['a0'][k], arr['a1'][k]) == back_dir) and (arr['origins'][k] == attacker)) else None
            if back_idx is not None:
                SQv0 = arr['sqe'][i]               # after front-run
                Pv0  = price_from_sqrtq96(SQv0) if np.isfinite(SQv0) else np.nan
                SQv1 = arr['sqe'][victim_idx[-1]]  # after last victim
                Pv1  = price_from_sqrtq96(SQv1) if np.isfinite(SQv1) else np.nan
                vict_a0 = float(np.nansum(arr['a0'][victim_idx]))
                vict_a1 = float(np.nansum(arr['a1'][victim_idx]))

                rows.append(dict(
                    block_number=int(sub.iloc[0][schema.col_block]),
                    pattern_type='Classical',
                    origin=attacker,
                    front_dir=front_dir,
                    front_a0=float(arr['a0'][i]),
                    front_a1=float(arr['a1'][i]),
                    back_dir=back_dir,
                    back_a0=float(arr['a0'][back_idx]),
                    back_a1=float(arr['a1'][back_idx]),
                    front_tx=arr['tx'][i],
                    back_tx=arr['tx'][back_idx],
                    victim_txs=victim_hashes,
                    L0=L0, sqrtP0_Q96=SQ0, x0=x0, y0=y0,
                    P_victim_pre=Pv0, P_victim_post=Pv1,
                    S_net_token0=vict_a0, S_net_token1=vict_a1,
                    mint_amount=np.nan,
                    burn_amount=np.nan,
                    gas_used=[arr['gas_used'][i]] + [arr['gas_used'][k] for k in victim_idx] + [arr['gas_used'][back_idx]],
                    gas_price=[arr['gas_price'][i]] + [arr['gas_price'][k] for k in victim_idx] + [arr['gas_price'][back_idx]],
                ))
                i = back_idx + 1
                continue

        # no pattern starting at i
        i += 1

    return rows



# ---------------------------------------------------------------------
# Wrappers to run detectors over chunks
# ---------------------------------------------------------------------

# def _process_chunk(detector_name: str, chunk: List[Tuple[int, int, int]], min_victims: int) -> List[Dict[str, Any]]:
#     out: List[Dict[str, Any]] = []
#     for block, lo, hi in chunk:
#         sub = _GDF.iloc[lo:hi]
#         if detector_name == 'JIT':
#             out.extend(detect_jit_in_block(sub, _GSCHEMA))
#         else:
#             out.extend(detect_sandwich_in_block(sub, _GSCHEMA, min_victims))
#     return out

def _process_chunk(detector_name: str, chunk: List[Tuple[int, int, int]], min_victims: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block, lo, hi in chunk:
        sub = _GDF.iloc[lo:hi]
        if detector_name == 'JIT':
            out.extend(detect_jit_in_block(sub, _GSCHEMA))
        elif detector_name == 'RBACKRUN':
            out.extend(detect_reverse_backrun_in_block(sub, _GSCHEMA))
        else:
            out.extend(detect_sandwich_in_block(sub, _GSCHEMA, min_victims))
    return out



def run_detector_mp(df: pd.DataFrame, schema: Schema, name: str, n_jobs: int, chunk_size: int, min_victims: int, quiet: bool) -> pd.DataFrame:
    slices = _build_block_slices(df, schema)
    chunks = _chunkify(slices, chunk_size)
    total = len(chunks)

    if not quiet:
        if name == 'JIT':
            label = 'JIT collector ⚡'
        elif name == 'RBACKRUN':
            label = 'Reverse Arbitrage collector 🔄'
        else:
            label = 'Sandwich collector 🥪'
        print(f"🧪 {label}: {len(slices):,} blocks → {total:,} chunks (size={chunk_size})")

    if n_jobs in (None, 0) or n_jobs < -1:
        n_jobs = -1
    if n_jobs == -1:
        n_jobs = max(1, mp.cpu_count())

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_pool_init, initargs=(df, schema)) as pool, tqdm(total=total, disable=quiet, desc=f"{name} progress", unit="chunk", mininterval=0.2, smoothing=0.1) as pbar:
        futures = [pool.submit(_process_chunk, name, chunk, min_victims) for chunk in chunks]
        for fut in as_completed(futures):
            rows.extend(fut.result())
            pbar.update(1)

    out = pd.DataFrame(rows)
    if not quiet:
        if name == 'JIT':
            print(f"✅ JIT found: {len(out):,} events.")
        elif name == 'RBACKRUN':
            print(f"✅ Reverse Arbitrage found: {len(out):,} events.")
        else:
            print(f"✅ Sandwich found: {len(out):,} events.")
    return out


# ---------------------------------------------------------------------
# Augment with Section 3 metrics (DIRECTION-AWARE)
# ---------------------------------------------------------------------

# def _safe_div(n: np.ndarray, d: np.ndarray) -> np.ndarray:
#     out = np.full_like(n, np.nan, dtype=float)
#     mask = (d != 0) & np.isfinite(n) & np.isfinite(d)
#     out[mask] = n[mask] / d[mask]
#     return out

def _safe_div(n, d) -> np.ndarray:
    """
    Elementwise safe division that accepts scalars or arrays and broadcasts shapes.
    Returns NaN where d==0 or either side is non-finite.
    """
    n = np.asarray(n, dtype=float)
    d = np.asarray(d, dtype=float)
    shape = np.broadcast(n, d).shape
    out = np.full(shape, np.nan, dtype=float)
    mask = (d != 0) & np.isfinite(n) & np.isfinite(d)
    # np.divide handles broadcasting; 'where' keeps NaNs elsewhere in 'out'
    np.divide(n, d, out=out, where=mask)
    return out


def augment_reverse_backruns(df: pd.DataFrame, fee_bps: Optional[float]) -> pd.DataFrame:
    """
    Augment reverse back-run pairs with Section 3.2 theory, observed PnL (valued at P0),
    and diagnostics that explain why many pairs are not viable MEV back-runs.

    Inputs & required columns in `df` (one row per victim/back-run pair, single pool):
      - x0, y0:     virtual reserves immediately BEFORE the victim swap.
      - victim_dir: 'x2y' or 'y2x' (direction of the victim).
      - S_net_token0, S_net_token1: victim pool deltas (net of fee). By convention, pool delta >0
        means the pool’s balance of that token increased.
      - back_a0, back_a1:           back-run pool deltas (net), same sign convention.
      - P_victim_pre, P_victim_post: price before and after the victim swap.
      - P_back_post (optional but recommended): price after the back-run swap.

    Notation:
      - f  = fee_bps / 1e4, r = 1 - f.
      - P0 = y0 / x0 (price BEFORE victim).
      - Gross victim size: divide net pool deltas by r.
      - Normalized size σ:  σ0 = S0_gross / x0  (victim x→y native side),
                            σ1 = S1_gross / y0  (victim y→x native side).
        We select the “native” σ according to victim_dir.

    Theory computed (Section 3.2):
      - I_theory(σ): price impact for x→y is 1/(1+rσ)^2 - 1; mirror for y→x is (1+rσ)^2 - 1.
      - σ_min_backrun: viability threshold for a profitable optimal back-run.
      - br_pi_star_token0: optimal back-run profit Π*_br(σ), expressed in token0 units
        (convert from native base using P0 when victim_dir=='y2x').
      - br_pi_star_per_x0: Π*_br normalized by x0 for comparability across pools/snapshots.

    Observed PnL (valued at P0 as in the derivation):
      - profit_obs_token0 = -back_a0 - back_a1 / P0
        (trader flow = − pool delta; convert token1 to token0 via P0).
      - profit_obs_per_x0 = profit_obs_token0 / x0.

    Diagnostics added to help explain negative observed PnL:
      - I_measured = P_victim_post / P_victim_pre - 1   (sign check vs victim_dir).
      - viable_sigma:  σ_native_gross ≥ σ_min_backrun.
      - reverted:      price moved back toward P0 after back-run:
                         victim x→y → expect P_back_post > P_victim_post
                         victim y→x → expect P_back_post < P_victim_post
        (uses P_back_post if available; otherwise False/NaN-safe.)
      - near_target:   |P_back_post - r*P0| ≤ 0.5% * P0 (rough Eq. 14 target proximity).
      - reverse_label: categorical reason:
                         'subthreshold' (σ below σ_min),
                         'wrong_sign'   (victim impact sign inconsistent),
                         'no_reversion' (back-run didn’t revert toward P0),
                         'candidate'    (passes all checks).

    Returns:
      The input DataFrame with all theory/observed columns and diagnostics appended.
      This function does NOT drop rows; you can filter by `reverse_label` downstream.

    Notes:
      - If `fee_bps` is None, we default to 5 bps (0.05%) and print a warning.
      - All divisions are safe/broadcasting via `_safe_div`.
    """
    if df.empty:
        return df

    # --- Fee & retention factor
    f = (fee_bps or 5.0) / 10_000.0
    if fee_bps is None:
        print("⚠️  --fee-bps not provided; defaulting to 5 bps (0.05%).")
    r = 1.0 - f

    # --- Pre-victim state
    x0 = df['x0'].astype(float).to_numpy()
    y0 = df['y0'].astype(float).to_numpy()
    P0 = _safe_div(y0, x0)

    # --- Victim direction
    dir_x2y = (df['victim_dir'].astype(str).values == 'x2y')

    # --- Victim sizes (pool deltas, net) and gross equivalents
    S0_net = df['S_net_token0'].astype(float).to_numpy()
    S1_net = df['S_net_token1'].astype(float).to_numpy()
    S0_gross = _safe_div(S0_net, r)
    S1_gross = _safe_div(S1_net, r)

    # --- Normalized sizes σ (both sides), then pick native according to direction
    sigma0_net   = _safe_div(S0_net,   x0)
    sigma0_gross = _safe_div(S0_gross, x0)
    sigma1_net   = _safe_div(S1_net,   y0)
    sigma1_gross = _safe_div(S1_gross, y0)

    sigma_native_net   = np.where(dir_x2y, sigma0_net,   sigma1_net)
    sigma_native_gross = np.where(dir_x2y, sigma0_gross, sigma1_gross)

    # --- Theory: price impact & viability threshold
    I_theory = np.where(
        dir_x2y,
        price_impact_x2y(r, sigma_native_gross),
        price_impact_y2x(r, sigma_native_gross),
    )
    sigma_min_br = backrun_sigma_min(r)

    # --- Optimal back-run profit in native base, then convert to token0
    base = np.where(dir_x2y, x0, y0)
    pi_star_native = np.array(
        [
            backrun_opt_profit_base(b, r, s) if (np.isfinite(b) and np.isfinite(s)) else np.nan
            for b, s in zip(base, sigma_native_gross)
        ],
        dtype=float,
    )
    pi_star_token0 = np.where(dir_x2y, pi_star_native, _safe_div(pi_star_native, P0))
    pi_star_per_x0 = _safe_div(pi_star_token0, x0)

    # --- Observed back-run PnL in token0 (value at P0)
    back_a0 = df['back_a0'].astype(float).to_numpy()
    back_a1 = df['back_a1'].astype(float).to_numpy()
    # infer back-run direction from signs (robust)
    back_x2y = (back_a0 > 0) & (back_a1 < 0)   # token0 in, token1 out
    back_y2x = ~back_x2y                        # token1 in, token0 out
    # trader-side flows (fee-correct)
    x_cost = np.where(back_x2y,  back_a0 / r, 0.0)   # token0 input
    x_get  = np.where(back_y2x, -back_a0,     0.0)   # token0 received
    y_cost = np.where(back_y2x,  back_a1 / r, 0.0)   # token1 input
    y_get  = np.where(back_x2y, -back_a1,     0.0)   # token1 received
    profit_obs_token0 = (x_get - x_cost) + _safe_div((y_get - y_cost), P0)
    profit_obs_per_x0 = _safe_div(profit_obs_token0, x0)


    # --- Measured price impact on victim
    P_pre  = df['P_victim_pre'].astype(float).to_numpy()
    P_post = df['P_victim_post'].astype(float).to_numpy()
    I_measured = _safe_div(P_post, P_pre) - 1.0

    # --- Reversion diagnostics (requires P_back_post; if missing, returns False)
    if 'P_back_post' in df.columns:
        P_back = df['P_back_post'].astype(float).to_numpy()
    else:
        P_back = np.full_like(P0, np.nan, dtype=float)

    # 1) Viability by σ threshold
    viable_sigma = sigma_native_gross >= sigma_min_br

    # 2) Victim moved the price in the direction implied by its trade
    dir_ok = np.where(dir_x2y, I_measured < 0, I_measured > 0)

    # 3) Back-run actually reverts toward P0 (coarse)
    reverted = np.where(
        dir_x2y,                       # victim x→y: P should go down on victim, then back up
        P_back > P_post,               # moved back toward P0
        P_back < P_post,               # victim y→x: back down toward P0
    )
    # If P_back is NaN, mark as False (not verifiable)
    reverted = np.where(np.isfinite(P_back) & np.isfinite(P_post), reverted, False)

    # 4) Near theoretical target r*P0 (fine)
    TOL_NEAR_TARGET = 5e-3  # 0.5%
    near_target = (
        np.isfinite(P0) & np.isfinite(P_back) &
        (np.abs(P_back - r * P0) <= TOL_NEAR_TARGET * np.abs(P0))
    )

    # Summary label
    reverse_label = np.where(
        ~viable_sigma, 'subthreshold',
        np.where(
            ~dir_ok, 'wrong_sign',
            np.where(~reverted, 'no_reversion', 'candidate')
        )
    )

    # --- JIT bribe ceiling for same victim, in token0 units (for comparisons)
    mmax_JIT_token0 = np.where(
        dir_x2y,
        f * S0_gross,                   # already token0 units
        f * _safe_div(S1_gross, P0),    # convert token1 -> token0 via P0
    )

    return df.assign(
        fee_fraction=f, r=r,
        sigma_net=sigma_native_net,
        sigma_gross=sigma_native_gross,
        sigma0_net=sigma0_net, sigma0_gross=sigma0_gross,
        sigma1_net=sigma1_net, sigma1_gross=sigma1_gross,
        I_theory=I_theory, I_measured=I_measured,
        sigma_min_backrun=sigma_min_br,
        br_pi_star_token0=pi_star_token0,
        br_pi_star_per_x0=pi_star_per_x0,
        profit_obs_token0=profit_obs_token0,
        profit_obs_per_x0=profit_obs_per_x0,
        # diagnostics
        viable_sigma=viable_sigma,
        reverted=reverted,
        near_target=near_target,
        reverse_label=reverse_label,
        # extra for cross-strategy comparisons
        mmax_JIT_token0=mmax_JIT_token0,
    )



def augment_jit(df: pd.DataFrame, fee_bps: Optional[float]) -> pd.DataFrame:
    if df.empty:
        return df

    f = (fee_bps or 5.0) / 10_000.0
    if fee_bps is None:
        print("⚠️  --fee-bps not provided; defaulting to 5 bps (0.05%).")
    r = 1.0 - f

    x0 = df['x0'].astype(float).to_numpy()
    y0 = df['y0'].astype(float).to_numpy()
    P0 = _safe_div(y0, x0)  # price (token1 per token0)

    # Victim net sizes (pool deltas) and gross (pre-fee) equivalents
    S0_net = df['S_net_token0'].astype(float).to_numpy()
    S1_net = df['S_net_token1'].astype(float).to_numpy()
    S0_gross = S0_net / r
    S1_gross = S1_net / r

    # Direction of the victim flow (native base = token0 for x2y, token1 for y2x)
    dir_x2y = S0_net >= 0
    victim_dir = np.where(dir_x2y, 'x2y', 'y2x')

    # σ normalized in native base
    sigma0_net   = _safe_div(S0_net,   x0)
    sigma0_gross = _safe_div(S0_gross, x0)
    sigma1_net   = _safe_div(S1_net,   y0)
    sigma1_gross = _safe_div(S1_gross, y0)

    sigma_native_net   = np.where(dir_x2y, sigma0_net,   sigma1_net)
    sigma_native_gross = np.where(dir_x2y, sigma0_gross, sigma1_gross)

    # Theoretical price impact (direction-aware)
    I_theory = np.where(
        dir_x2y,
        price_impact_x2y(r, sigma_native_gross),
        price_impact_y2x(r, sigma_native_gross),
    )

    # ---- JIT bribe ceilings ----
    # 1) Normalized per-x0 (dimensionless, comparable across pools)
    # base_over_x0 = _safe_div(y0, x0)
    # mmax_JIT_per_x0 = f * sigma_native_gross * np.where(dir_x2y, 1.0, base_over_x0)
    mmax_JIT_per_x0 = f * sigma_native_gross

    # 2) Absolute ceiling in token0 (X) units (for plotting with profits in X)
    #    X→Y: f * S0_gross (already token0)
    #    Y→X: f * S1_gross / P0 (convert token1 -> token0)
    mmax_JIT_token0 = np.where(
        dir_x2y,
        f * S0_gross,
        f * _safe_div(S1_gross, P0)
    )

    df = df.assign(
        fee_fraction=f,
        r=r,
        victim_dir=victim_dir,

        # Direction-aware normalizations
        sigma_net=sigma_native_net,
        sigma_gross=sigma_native_gross,
        sigma0_net=sigma0_net,
        sigma0_gross=sigma0_gross,
        sigma1_net=sigma1_net,
        sigma1_gross=sigma1_gross,

        # Theory impact and ceilings
        I_theory=I_theory,
        mmax_JIT_per_x0=mmax_JIT_per_x0,   # unitless, per-x0 normalization
        mmax_JIT_token0=mmax_JIT_token0,   # absolute amount in token0 units
    )
    return df

def augment_sandwich(df: pd.DataFrame, fee_bps: Optional[float], gamma: float, grid_n: int) -> pd.DataFrame:
    if df.empty:
        return df
    f = (fee_bps or 5.0) / 10_000.0
    if fee_bps is None:
        print("⚠️  --fee-bps not provided; defaulting to 5 bps (0.05%).")
    r = 1.0 - f

    x0 = df['x0'].astype(float).to_numpy()
    y0 = df['y0'].astype(float).to_numpy()
    P0 = _safe_div(y0, x0)  # price (token1 per token0) at pre-front state

    # Victim/front direction (x2y means token0 is the native base)
    if 'front_dir' in df.columns:
        dir_vals = df['front_dir'].astype(str).to_numpy()
        victim_dir = np.where(dir_vals == 'swap_x2y', 'x2y', 'y2x')
    else:
        S0_net_tmp = df['S_net_token0'].astype(float).to_numpy()
        victim_dir = np.where(S0_net_tmp >= 0, 'x2y', 'y2x')
    dir_x2y = (victim_dir == 'x2y')

    # Pool deltas recorded for the attacker's two legs
    front_a0 = df['front_a0'].astype(float).fillna(0.0).to_numpy()
    back_a0  = df['back_a0'].astype(float).fillna(0.0).to_numpy()
    front_a1 = df['front_a1'].astype(float).fillna(0.0).to_numpy()
    back_a1  = df['back_a1'].astype(float).fillna(0.0).to_numpy()

    # ---- Realized attacker profit in token0 (fee-correct, direction-aware) ----
    # Convert pool deltas -> trader flows.
    # Token0 spent (input) occurs on the X→Y leg; token0 received occurs on the Y→X leg.
    x_cost = np.where(dir_x2y,  front_a0 / r,              back_a0 / r)   # token0 input by trader
    x_get  = np.where(dir_x2y, -back_a0,                  -front_a0)      # token0 received by trader

    # Token1 flows (used only to value any residual; cancels to ~0 for self-funded cycles)
    y_cost = np.where(dir_x2y,  back_a1 / r,               front_a1 / r)  # token1 input by trader
    y_get  = np.where(dir_x2y, -front_a1,                 -back_a1)       # token1 received by trader

    profit_token0 = (x_get - x_cost) + _safe_div((y_get - y_cost), P0)
    profit_per_x0 = _safe_div(profit_token0, x0)

    # ---- Native front-run size ε (direction-aware) ----
    eps_net_native = np.where(dir_x2y, _safe_div(np.abs(front_a0), x0),
                                         _safe_div(np.abs(front_a1), y0))
    eps_gross_native = eps_net_native / r

    # ---- Victim size σ (direction-aware) ----
    S0_net = df['S_net_token0'].astype(float).to_numpy()
    S1_net = df['S_net_token1'].astype(float).to_numpy()
    S0_gross = S0_net / r
    S1_gross = S1_net / r

    sigma0_net   = _safe_div(S0_net, x0)
    sigma0_gross = _safe_div(S0_gross, x0)
    sigma1_net   = _safe_div(S1_net, y0)
    sigma1_gross = _safe_div(S1_gross, y0)

    sigma_native_net   = np.where(dir_x2y, sigma0_net,   sigma1_net)
    sigma_native_gross = np.where(dir_x2y, sigma0_gross, sigma1_gross)

    # ---- Theory (direction-aware) ----
    I_theory = np.where(dir_x2y,
                        price_impact_x2y(r, sigma_native_gross),
                        price_impact_y2x(r, sigma_native_gross))
    sigma_min_br = backrun_sigma_min(r)

    # ---- Back-run optimum (native base) -> token0 ----
    base = np.where(dir_x2y, x0, y0)
    br_pi_star_native_units = np.array(
        [backrun_opt_profit_base(b, r, s) if (np.isfinite(b) and np.isfinite(s)) else np.nan
         for b, s in zip(base, sigma_native_gross)],
        dtype=float,
    )
    br_pi_star_token0 = np.where(dir_x2y, br_pi_star_native_units,
                                 _safe_div(br_pi_star_native_units, P0))
    br_pi_star_per_x0 = _safe_div(br_pi_star_token0, x0)

    # ---- Sandwich π (native base) -> token0 ----
    sand_pi_obs_native = [sandwich_profit_normalized(e, s, r)
                          for e, s in zip(eps_gross_native, sigma_native_gross)]
    eps_star_list, sand_pi_star_native = [], []
    for s in sigma_native_gross:
        e_star, pi_star = sandwich_profit_star(s, r, gamma, grid_n)
        eps_star_list.append(e_star)
        sand_pi_star_native.append(pi_star)

    sand_obs_native_units  = base * np.array(sand_pi_obs_native,  dtype=float)
    sand_star_native_units = base * np.array(sand_pi_star_native, dtype=float)

    sand_pi_obs_token0  = np.where(dir_x2y, sand_obs_native_units,
                                   _safe_div(sand_obs_native_units,  P0))
    sand_pi_star_token0 = np.where(dir_x2y, sand_star_native_units,
                                   _safe_div(sand_star_native_units, P0))

    sand_pi_obs_per_x0  = _safe_div(sand_pi_obs_token0,  x0)
    sand_pi_star_per_x0 = _safe_div(sand_pi_star_token0, x0)

    # ---- Measured impact around the victim span ----
    P_pre  = df['P_victim_pre'].astype(float).to_numpy()
    P_post = df['P_victim_post'].astype(float).to_numpy()
    I_measured = _safe_div(P_post, P_pre) - 1.0

    out = df.assign(
        fee_fraction=f,
        r=r,
        victim_dir=victim_dir,

        # Realized PnL (token0) and normalized
        profit_token0=profit_token0,
        profit_per_x0=profit_per_x0,

        # Sizes
        eps_net=eps_net_native,
        eps_gross=eps_gross_native,
        sigma_net=sigma_native_net,
        sigma_gross=sigma_native_gross,
        sigma0_net=sigma0_net,
        sigma0_gross=sigma0_gross,
        sigma1_net=sigma1_net,
        sigma1_gross=sigma1_gross,

        # Impacts / thresholds
        I_theory=I_theory,
        I_measured=I_measured,
        sigma_min_backrun=sigma_min_br,

        # THEORETICAL PROFITS — token0 units
        br_pi_star_token0=br_pi_star_token0,
        sand_pi_obs_token0=sand_pi_obs_token0,
        sand_pi_star_token0=sand_pi_star_token0,

        # Optional normalized series for plots
        br_pi_star_per_x0=br_pi_star_per_x0,
        sand_pi_obs_per_x0=sand_pi_obs_per_x0,
        sand_pi_star=sand_pi_star_per_x0,

        # Slippage cap and crossover σ*
        eps_max=evaluate_eps_max_series(pd.Series(sigma_native_gross), gamma, r),
        gamma_used=gamma,
        sigma_star_br_vs_jit_phi0=sigma_star_backrun_vs_jit(f, phi=0.0, r=r),
    )
    return out

def evaluate_eps_max_series(sigmas: pd.Series, gamma: float, r: float) -> pd.Series:
    return pd.Series([eps_max_under_slippage(float(s), gamma, r) for s in sigmas], index=sigmas.index)


# ---------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect JIT cycles and Sandwich attacks + Section 3 metrics (no column normalization).")
    p.add_argument("--in", dest="in_path", required=True, help="Input CSV or Parquet.")
    p.add_argument("--outdir", default="./mev_out", help="Output directory.")
    p.add_argument("--n-jobs", type=int, default=-1, help="Worker processes (-1 = all cores).")
    p.add_argument("--chunk-size", type=int, default=64, help="Blocks per task chunk to reduce IPC overhead.")
    p.add_argument("--min-victims", type=int, default=1, help="Min victim swaps for a sandwich (default: 1).")
    p.add_argument("--fee-bps", type=float, default=None, help="Pool fee tier in basis points (e.g., 5, 30, 100). If omitted, default 5 bps.")
    p.add_argument("--gamma", type=float, default=0.01, help="Victim slippage tolerance used in theory constraints (default: 0.01 = 1%).")
    p.add_argument("--grid-npoints", type=int, default=128, help="Grid size to maximize π(ε) under γ (default: 128).")
    p.add_argument("--quiet", action="store_true", help="Less verbose printing.")
    p.add_argument("--recompute", action="store_true", help="Recompute all detectors even if output files exist.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load (no normalization!)
    print("📥 Loading dataset…")
    ext = os.path.splitext(args.in_path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(args.in_path)
    elif ext in (".csv", ".gz"):
        df = pd.read_csv(args.in_path, low_memory=False)
    elif ext in (".pkl", ".pickle"):
        df = pd.read_pickle(args.in_path)
    else:
        raise ValueError("Unsupported input file. Use CSV or Parquet.")
    print(f"✅ Loaded {len(df):,} rows from {args.in_path}")

    # Resolve schema and sort by (block, logIndex)
    schema = resolve_schema(df)
    df = df.sort_values([schema.col_block, schema.col_log_index], kind='mergesort').reset_index(drop=True)

    # Decide worker processes
    n_jobs = args.n_jobs if args.n_jobs != -1 else max(1, mp.cpu_count())

    # Run detectors
    print("🚀 Starting collectors…")
    if os.path.exists("ABM_Uni_v3/mev_out/jit_cycles_tidy.csv") and not args.recompute:
        print("JIT cycles already detected, skipping...")
        jit_df = pd.read_csv("ABM_Uni_v3/mev_out/jit_cycles_tidy.csv")
    else:
        jit_df = run_detector_mp(df, schema, name='JIT', n_jobs=n_jobs, chunk_size=args.chunk_size, min_victims=args.min_victims, quiet=args.quiet)
    if os.path.exists("ABM_Uni_v3/mev_out/sandwich_attacks_tidy.csv") and not args.recompute:
        print("Sandwich attacks already detected, skipping...")
        sand_df = pd.read_csv("ABM_Uni_v3/mev_out/sandwich_attacks_tidy.csv")
    else:
        sand_df = run_detector_mp(df, schema, name='SANDWICH', n_jobs=n_jobs, chunk_size=args.chunk_size, min_victims=args.min_victims, quiet=args.quiet)
    if os.path.exists("ABM_Uni_v3/mev_out/reverse_backruns_tidy.csv") and not args.recompute:
        print("Reverse back-runs already detected, skipping...")
        rback_df = pd.read_csv("ABM_Uni_v3/mev_out/reverse_backruns_tidy.csv")
    else:
        rback_df = run_detector_mp(df, schema, name='RBACKRUN', n_jobs=n_jobs, chunk_size=args.chunk_size, min_victims=args.min_victims, quiet=args.quiet)

    # Exclude any pair whose back-run TX is part of a detected sandwich ending (strict de-dup)
    if not sand_df.empty and not rback_df.empty:
        sand_back = set(sand_df['back_tx'].dropna().astype(str).tolist())
        before = len(rback_df)
        rback_df = rback_df[~rback_df['back_tx'].astype(str).isin(sand_back)].reset_index(drop=True)
        removed = before - len(rback_df)
        if not args.quiet:
            print(f"🧹 Removed {removed:,} reverse pairs that were sandwich back-runs.")

    # Augment with Section 3 metrics
    print("🧮 Computing Section 3 metrics…")
    jit_df  = augment_jit(jit_df, fee_bps=args.fee_bps)
    sand_df = augment_sandwich(sand_df, fee_bps=args.fee_bps, gamma=args.gamma, grid_n=args.grid_npoints)
    rback_df = augment_reverse_backruns(rback_df, fee_bps=args.fee_bps)

    # Save
    def save_csv(df_out: pd.DataFrame, name: str):
        path = os.path.join(args.outdir, f"{name}.csv")
        df_out.to_csv(path, index=False)
        print(f"💾 Saved {len(df_out):,} rows → {path}")

    if not rback_df.empty:
        path = os.path.join(args.outdir, "reverse_backruns_tidy.csv")
        rback_df.to_csv(path, index=False)
        print(f"💾 Saved {len(rback_df):,} rows → {path}")
    else:
        print("ℹ️ No reverse back-run arbitrages detected.")

    if not jit_df.empty:
        save_csv(jit_df, "jit_cycles_tidy")
    else:
        print("ℹ️ No JIT cycles detected.")

    if not sand_df.empty:
        # add a compact hash bundle for convenience
        def compact_hashes(row: pd.Series) -> List[str]:
            out = []
            if isinstance(row.get('front_tx'), str): out.append(row['front_tx'])
            if isinstance(row.get('mint_tx'), str): out.append(row['mint_tx'])
            if isinstance(row.get('victim_txs'), list): out.extend(row['victim_txs'])
            if isinstance(row.get('burn_tx'), str): out.append(row['burn_tx'])
            if isinstance(row.get('back_tx'), str): out.append(row['back_tx'])
            return out
        sand_df = sand_df.assign(tx_sequence=sand_df.apply(compact_hashes, axis=1))
        save_csv(sand_df, "sandwich_attacks_tidy")
    else:
        print("ℹ️ No sandwich attacks detected.")

    print("🏁 Done. 🎉")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        mp.set_start_method("spawn", force=True)
    main()
