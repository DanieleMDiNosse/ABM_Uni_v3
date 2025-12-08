'''How to run tests:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/test_pool_and_simulation.py'''

import copy
from typing import Tuple

import pytest

from utils import build_empty_pool
from uniswapv3_pool import V3Pool, BoundaryIndex
from run import simulate


def _prepare_pool() -> Tuple[V3Pool, BoundaryIndex]:
    """
    Build a pool with symmetric liquidity around the active tick and a matching boundary index.
    """
    pool, _ = build_empty_pool()
    # add symmetric liquidity around the active tick so quotes have cross-tick depth
    L = 50_000.0
    lower = pool.tick - pool.tick_spacing * 2
    upper = pool.tick + pool.tick_spacing * 2
    pool.add_liquidity_range(lower, upper, L)
    pool.recompute_active_L()
    bidx = BoundaryIndex(pool.liquidity_net)
    return pool, bidx


def test_quote_x_to_y_matches_swap_result():
    """
    The read-only X→Y quote should match an actual swap when the trade is executed on an identical pool copy with identical liquidity.
    """
    pool, bidx = _prepare_pool()
    dx_in = 1.0

    quoted = pool.quote_x_to_y(dx_in, bidx)

    pool_for_swap = copy.deepcopy(pool)
    used_dx, dy_out, _ = pool_for_swap.swap_x_to_y(dx_in, fee_cb=None)
    assert used_dx == pytest.approx(dx_in)
    assert dy_out == pytest.approx(quoted, rel=1e-9, abs=1e-9)


def test_quote_y_to_x_matches_swap_result():
    """
    The read-only Y→X quote should reproduce the exact output of a swap consuming the same input on a cloned pool.
    """
    pool, bidx = _prepare_pool()
    dy_in = 1.0

    quoted = pool.quote_y_to_x(dy_in, bidx)

    pool_for_swap = copy.deepcopy(pool)
    used_dy, dx_out, _ = pool_for_swap.swap_y_to_x(dy_in, fee_cb=None)
    assert used_dy == pytest.approx(dy_in)
    assert dx_out == pytest.approx(quoted, rel=1e-9, abs=1e-9)


def test_quote_returns_zero_without_liquidity():
    """
    Quoting should short-circuit to zero when the pool has no active liquidity so callers do not rely on stale or undefined depth.
    """
    pool, _ = build_empty_pool()
    bidx = BoundaryIndex(pool.liquidity_net)
    assert pool.quote_x_to_y(10.0, bidx) == 0.0
    assert pool.quote_y_to_x(10.0, bidx) == 0.0


def test_swap_inactive_pool_returns_zero():
    """
    Swaps on an empty pool should yield zero flow and leave the price unchanged, guaranteeing graceful handling of missing liquidity.
    """
    pool, _ = build_empty_pool()
    start_price = pool.price
    used_dx, dy_out, fee = pool.swap_x_to_y(5.0, fee_cb=None)
    assert used_dx == 0.0
    assert dy_out == 0.0
    assert fee == 0.0
    assert pool.price == start_price


def test_boundary_index_tracks_liquidity_across_ticks():
    """
    BoundaryIndex must expose accurate prefix sums so that consumers can locate active liquidity before and after mint/burn events.
    """
    pool, _ = build_empty_pool()
    spacing = pool.tick_spacing
    lower1 = pool.tick - spacing
    upper1 = pool.tick + spacing
    pool.add_liquidity_range(lower1, upper1, 1_000.0)
    lower2 = upper1
    upper2 = lower2 + spacing
    pool.add_liquidity_range(lower2, upper2, 2_000.0)
    bidx = BoundaryIndex(pool.liquidity_net)

    assert bidx.active_liquidity_at_tick(pool.tick) == pytest.approx(1_000.0)
    assert bidx.active_liquidity_at_tick(lower2) == pytest.approx(2_000.0)
    assert bidx.next_up(lower1) == lower2
    assert bidx.prev_down(lower2) == lower1


def test_simulate_outputs_consistent_lengths():
    """
    Simulation series describing trader and smart-router flows must have consistent lengths so plotting and analysis can zip the arrays safely.
    """
    out = simulate(T=5, block_time=1, visualize=False, seed=1)

    assert len(out['smart_router_pnl_steps']) == len(out['smart_router_notional_y'])
    assert len(out['noise_trader_pnl_steps']) == len(out['noise_trader_notional_y'])
    assert len(out['smart_router_exec_count']) == len(out['smart_router_pnl_steps'])
    assert len(out['noise_trader_exec_count']) == len(out['noise_trader_pnl_steps'])
    assert len(out['smart_router_pnl_steps']) == len(out['smart_router_pnl_cum'])
    assert len(out['noise_trader_pnl_steps']) == len(out['noise_trader_pnl_cum'])
    assert out['fee_mode'] == 'volatility'


def test_simulate_invalid_fee_mode():
    """
    simulate() should defensively reject unsupported fee controller modes before any random state is touched.
    """
    with pytest.raises(ValueError, match="Invalid fee_mode"):
        simulate(T=1, block_time=1, visualize=False, fee_mode="nonsense")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k_out_min": 0, "k_out_max": 1},
        {"k_out_min": 5, "k_out_max": 3},
    ],
)
def test_simulate_validates_k_out_bounds(kwargs):
    """
    The k_out bounds govern price-impact sampling and must be positive and ordered; invalid tuples should raise cleanly.
    """
    with pytest.raises(ValueError):
        simulate(T=1, block_time=1, visualize=False, **kwargs)


def test_simulate_seed_determinism():
    """
    Re-running the simulation with the same seed should reproduce every series exactly, while a different seed should alter at least one path.
    """
    out_a = simulate(T=3, block_time=1, visualize=False, seed=42)
    out_b = simulate(T=3, block_time=1, visualize=False, seed=42)
    out_c = simulate(T=3, block_time=1, visualize=False, seed=999)

    dex_a = tuple(out_a["DEX_price"])
    dex_b = tuple(out_b["DEX_price"])
    dex_c = tuple(out_c["DEX_price"])
    sr_pnl_a = tuple(out_a["smart_router_pnl_steps"])
    sr_pnl_b = tuple(out_b["smart_router_pnl_steps"])

    assert dex_a == dex_b
    assert sr_pnl_a == sr_pnl_b
    assert any(abs(a - b) > 1e-12 for a, b in zip(dex_a, dex_c))


def test_simulate_heston_mode_requires_parameters():
    """
    Heston volatility mode must fail fast when required cex_heston_* parameters are missing.
    """
    with pytest.raises(ValueError, match="cex_sigma_mode='heston' requires parameters"):
        simulate(T=1, block_time=1, visualize=False, cex_sigma_mode="heston")


def test_simulate_heston_mode_runs_and_emits_sigma_series():
    """
    With valid Heston parameters, simulate() should run and produce a non-negative cex_sigma_series.
    """
    out = simulate(
        T=3,
        block_time=1,
        visualize=False,
        cex_sigma_mode="heston",
        cex_sigma=0.001,  # used as sqrt(v0) when cex_heston_v0 is omitted
        cex_heston_kappa=1.0,
        cex_heston_theta=1e-6,
        cex_heston_sigma_v=0.1,
        cex_heston_rho=-0.5,
    )
    sigma_series = out["cex_sigma_series"]
    cex_series = out["CEX_price"]
    assert len(sigma_series) == len(cex_series)
    assert all(s >= 0.0 for s in sigma_series)
