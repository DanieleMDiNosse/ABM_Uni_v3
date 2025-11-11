import copy

import pytest

from utils import build_empty_pool
from uniswapv3_pool import V3Pool, BoundaryIndex
from run import simulate


def _prepare_pool() -> tuple[V3Pool, BoundaryIndex]:
    pool, _ = build_empty_pool()
    # add symmetric liquidity around the active tick so quotes have depth
    L = 50_000.0
    lower = pool.tick - pool.tick_spacing * 2
    upper = pool.tick + pool.tick_spacing * 2
    pool.add_liquidity_range(lower, upper, L)
    pool.recompute_active_L()
    bidx = BoundaryIndex(pool.liquidity_net)
    return pool, bidx


def test_quote_x_to_y_matches_swap_result():
    pool, bidx = _prepare_pool()
    dx_in = 1.0

    quoted = pool.quote_x_to_y(dx_in, bidx)

    pool_for_swap = copy.deepcopy(pool)
    used_dx, dy_out, _ = pool_for_swap.swap_x_to_y(dx_in, fee_cb=None)
    assert used_dx == pytest.approx(dx_in)
    assert dy_out == pytest.approx(quoted, rel=1e-9, abs=1e-9)


def test_quote_y_to_x_matches_swap_result():
    pool, bidx = _prepare_pool()
    dy_in = 1.0

    quoted = pool.quote_y_to_x(dy_in, bidx)

    pool_for_swap = copy.deepcopy(pool)
    used_dy, dx_out, _ = pool_for_swap.swap_y_to_x(dy_in, fee_cb=None)
    assert used_dy == pytest.approx(dy_in)
    assert dx_out == pytest.approx(quoted, rel=1e-9, abs=1e-9)


def test_simulate_outputs_consistent_lengths():
    out = simulate(T=5, block_size=1, visualize=False, seed=1)

    assert len(out['smart_router_pnl_steps']) == len(out['smart_router_notional_y'])
    assert len(out['noise_trader_pnl_steps']) == len(out['noise_trader_notional_y'])
    assert len(out['smart_router_exec_count']) == len(out['smart_router_pnl_steps'])
    assert len(out['noise_trader_exec_count']) == len(out['noise_trader_pnl_steps'])

