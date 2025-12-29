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


# Warm up Numba JIT on import (compile the function)
if NUMBA_AVAILABLE:
    try:
        # Trigger compilation with dummy values
        _ = _current_amounts_impl(1.0, 1.0, 2.0, 1.5)
    except Exception:
        pass  # Compilation will happen on first real call
