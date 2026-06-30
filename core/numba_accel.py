"""
Numba-accelerated helper functions for Uniswap V3 ABM simulation.

Provides JIT-compiled versions of hot-path math functions with
graceful fallback to pure Python if Numba is not available.
"""
from typing import Tuple

# Try to import numba; if not available, create a no-op decorator
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # No-op decorator that just returns the function unchanged
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        # Handle both @njit and @njit() syntax
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit(cache=True)
def _current_amounts_impl(L: float, sa: float, sb: float, S: float) -> Tuple[float, float]:
    """
    Core Uniswap V3 current amounts computation.
    
    Given liquidity L and sqrt-price range [sa, sb), computes (token0, token1)
    at current sqrt-price S.
    
    This is the hot-path function called ~1.5M times per simulation.
    """
    if S <= sa:
        return (L * (1.0 / sa - 1.0 / sb), 0.0)
    if S >= sb:
        return (0.0, L * (sb - sa))
    return (L * (1.0 / S - 1.0 / sb), L * (S - sa))


def current_amounts_fast(L: float, sa: float, sb: float, S: float) -> Tuple[float, float]:
    """
    Wrapper for _current_amounts_impl with the same interface.
    Exists for explicit calling when you have L, sa, sb, S directly.
    """
    return _current_amounts_impl(L, sa, sb, S)


import numpy as np


@njit(cache=True)
def _broadcast_accrue_numba(
    M_new: float,
    x_prev: np.ndarray,
    cum_R: np.ndarray,
    last_M: np.ndarray,
    initialized: np.ndarray,
) -> None:
    """
    Vectorized rebalancer accrual across all LPs.

    Replaces the Python loop ``for lp in LPs: _accrue_price_move(lp, M_new)``
    with a single Numba-compiled pass over parallel arrays, eliminating ~2.3M
    Python function calls + abs() overhead at T=10k.
    """
    n = len(x_prev)
    for i in range(n):
        if not initialized[i]:
            last_M[i] = M_new
            continue
        delta = M_new - last_M[i]
        if delta != 0.0:
            cum_R[i] += x_prev[i] * delta
            last_M[i] = M_new


# Warm up Numba JIT on import (compile the function)
if NUMBA_AVAILABLE:
    try:
        # Trigger compilation with dummy values
        _ = _current_amounts_impl(1.0, 1.0, 2.0, 1.5)
        _tmp = np.zeros(1)
        _broadcast_accrue_numba(1.0, _tmp, _tmp.copy(), _tmp.copy(), np.zeros(1, dtype=np.bool_))
    except Exception:
        pass  # Compilation will happen on first real call
