"""
AMM Operation Tests for Uniswap v3 Pool.

Tests cover:
- Swap mechanics (X→Y, Y→X, single tick, multi-tick)
- Fee deduction (fee-on-input model)
- Tick boundary crossing
- Desert bridging
- Liquidity management (mint/burn)
- BoundaryIndex functionality
- Fee allocation

Run with:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/test_amm_operations.py -v
"""

import copy
import math
from typing import Dict, Tuple

import numpy as np
import pytest

from core.utils import build_empty_pool, minted_amounts_at_S, EPS_LIQ, EPS_BOUNDARY
from core.uniswapv3_pool import V3Pool, BoundaryIndex
from core.agents import Position


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def empty_pool() -> V3Pool:
    """Build an empty pool with no initial liquidity."""
    pool, _ = build_empty_pool()
    return pool


@pytest.fixture
def pool_with_liquidity() -> V3Pool:
    """Build a pool with symmetric liquidity around the active tick."""
    pool, _ = build_empty_pool()
    L = 50_000.0
    lower = pool.tick - pool.tick_spacing * 2
    upper = pool.tick + pool.tick_spacing * 2
    pool.add_liquidity_range(lower, upper, L)
    pool.recompute_active_L()
    return pool


@pytest.fixture
def pool_with_multi_range_liquidity() -> V3Pool:
    """Build a pool with liquidity in multiple disjoint ranges."""
    pool, _ = build_empty_pool()
    spacing = pool.tick_spacing
    # Range 1: around active tick
    pool.add_liquidity_range(pool.tick - spacing, pool.tick + spacing, 30_000.0)
    # Range 2: above
    pool.add_liquidity_range(pool.tick + spacing, pool.tick + 3 * spacing, 20_000.0)
    # Range 3: below
    pool.add_liquidity_range(pool.tick - 3 * spacing, pool.tick - spacing, 10_000.0)
    pool.recompute_active_L()
    return pool


# =============================================================================
# 1.1 Swap Mechanics Tests
# =============================================================================

class TestSwapMechanics:
    """Tests for swap operations within and across tick boundaries."""

    def test_swap_x_to_y_single_tick_reserves_update(self, pool_with_liquidity: V3Pool):
        """
        Verify that after an X→Y swap that stays within one tick,
        reserves and price update correctly according to v3 formula.
        
        For X→Y: S decreases (price P = S² decreases)
        dx_eff = dx_in * r
        S_new = 1 / (1/S + dx_eff / L)
        dy_out = L * (S - S_new)  (negative change in S times L)
        """
        pool = pool_with_liquidity
        S_before = pool.S
        L_before = pool.L_active
        tick_before = pool.tick
        
        # Calculate a swap small enough to stay within the tick
        # Max dx that doesn't cross down: L * (1/S_lo - 1/S) / r
        S_lo = pool.s_lower()
        dx_max = L_before * (1.0 / S_lo - 1.0 / S_before) / pool.r
        dx_in = dx_max * 0.1  # Use only 10% to stay well within tick
        dx_eff = dx_in * pool.r
        
        used_dx, dy_out, fee_x = pool.swap_x_to_y(dx_in, fee_cb=None)
        
        # Verify input was fully consumed
        assert used_dx == pytest.approx(dx_in)
        
        # Verify fee calculation (fee-on-input model)
        assert fee_x == pytest.approx(dx_in * pool.f)
        
        # Verify price decreased (X→Y pushes price down)
        assert pool.S < S_before
        
        # Verify the v3 formula: S_new = 1 / (1/S + dx_eff / L)
        expected_S = 1.0 / (1.0 / S_before + dx_eff / L_before)
        assert pool.S == pytest.approx(expected_S, rel=1e-9)
        
        # Verify output amount: dy = L * (S_new - S_old), output is positive
        expected_dy = L_before * (S_before - expected_S)
        assert dy_out == pytest.approx(expected_dy, rel=1e-9)
        
        # Tick should not have changed for small swap
        assert pool.tick == tick_before

    def test_swap_y_to_x_single_tick_reserves_update(self, pool_with_liquidity: V3Pool):
        """
        Verify that after a Y→X swap that stays within one tick,
        reserves and price update correctly.
        
        For Y→X: S increases (price P = S² increases)
        dy_eff = dy_in * r
        S_new = S + dy_eff / L
        dx_out = L * (1/S - 1/S_new)
        """
        pool = pool_with_liquidity
        S_before = pool.S
        L_before = pool.L_active
        tick_before = pool.tick
        
        # Small swap that won't cross tick boundary
        dy_in = 100.0
        dy_eff = dy_in * pool.r
        
        used_dy, dx_out, fee_y = pool.swap_y_to_x(dy_in, fee_cb=None)
        
        # Verify input was fully consumed
        assert used_dy == pytest.approx(dy_in)
        
        # Verify fee calculation
        assert fee_y == pytest.approx(dy_in * pool.f)
        
        # Verify price increased (Y→X pushes price up)
        assert pool.S > S_before
        
        # Verify the v3 formula: S_new = S + dy_eff / L
        expected_S = S_before + dy_eff / L_before
        assert pool.S == pytest.approx(expected_S, rel=1e-9)
        
        # Verify output amount: dx = L * (1/S_old - 1/S_new)
        expected_dx = L_before * (1.0 / S_before - 1.0 / expected_S)
        assert dx_out == pytest.approx(expected_dx, rel=1e-9)
        
        # Tick should not have changed for small swap
        assert pool.tick == tick_before

    def test_swap_crosses_tick_boundary_updates_tick_and_L_active(self):
        """
        Verify that a swap that crosses exactly one tick boundary:
        - advances tick by exactly tick_spacing
        - updates L_active according to liquidity_net at the crossed boundary
        - reports the full pre-fee input as used input
        """
        pool, _ = build_empty_pool()
        spacing = pool.tick_spacing

        # Construct a two-band setup with known post-crossing liquidity:
        # - Band 0 (active): liquidity = L0
        # - Band 1 (next up): liquidity = L1
        L0 = 30_000.0
        L1 = 10_000.0
        pool.add_liquidity_range(pool.tick - spacing, pool.tick + spacing, L0)
        pool.add_liquidity_range(pool.tick + spacing, pool.tick + 2 * spacing, L1)
        pool.recompute_active_L()

        tick_before = pool.tick
        S_before = pool.S
        assert pool.L_active == pytest.approx(L0)

        # Post-fee dy needed to reach the next upper boundary.
        S_hi = pool.s_upper()
        dy_to_boundary = L0 * (S_hi - S_before)

        # Add a small extra to enter the next band, but not enough to cross again.
        S_hi_next = pool.s_upper(tick_before + spacing)
        dy_to_next_boundary = L1 * (S_hi_next - S_hi)
        dy_eff_total = dy_to_boundary + min(0.5 * dy_to_next_boundary, 1.0)
        dy_in = dy_eff_total / pool.r

        used_dy, _dx_out, fee_y = pool.swap_y_to_x(dy_in, fee_cb=None)

        assert pool.tick == tick_before + spacing
        assert used_dy == pytest.approx(dy_in)
        assert fee_y == pytest.approx(dy_in * pool.f, rel=1e-12, abs=1e-12)
        assert pool.L_active == pytest.approx(L1)
        # After the extra consumption inside the next band, S is strictly above S_hi.
        assert pool.S > S_hi

    def test_swap_fee_deduction_correct(self, pool_with_liquidity: V3Pool):
        """
        Confirm that the fee is correctly deducted from the input amount.
        Fee-on-input model: dx_eff = dx_in * (1 - f), fee = dx_in * f
        
        Note: This holds exactly when the full input is consumed within a single tick.
        For multi-tick swaps, fee accumulates per segment.
        """
        pool = pool_with_liquidity
        f = pool.f
        r = pool.r
        
        # Calculate a swap small enough to stay within the tick
        S_lo = pool.s_lower()
        S = pool.S
        L = pool.L_active
        dx_max = L * (1.0 / S_lo - 1.0 / S) / r
        dx_in = dx_max * 0.1  # Small swap stays in tick
        
        used_dx, dy_out, fee_x = pool.swap_x_to_y(dx_in, fee_cb=None)
        
        # Fee should be exactly f * input when swap stays in one tick
        assert fee_x == pytest.approx(dx_in * f, rel=1e-12)
        
        # Effective amount used for price movement
        dx_eff = dx_in * r
        assert dx_eff == pytest.approx(dx_in - fee_x, rel=1e-12)

    def test_swap_roundtrip_conservation(self, pool_with_liquidity: V3Pool):
        """
        Swap X→Y and then Y→X. Verify no tokens are created; 
        final amount ≤ initial (accounting for fees).
        """
        pool = pool_with_liquidity
        
        initial_dx = 1.0
        
        # Swap X→Y
        used_dx, dy_out, fee1 = pool.swap_x_to_y(initial_dx, fee_cb=None)
        
        # Swap Y→X with the output
        used_dy, dx_back, fee2 = pool.swap_y_to_x(dy_out, fee_cb=None)
        
        # Due to fees on both legs, we should get back less than we started with
        assert dx_back < initial_dx
        
        # The difference should be approximately the fee impact
        # Each leg loses ~f fraction, so expect roughly (1-f)^2 of initial
        expected_ratio = (pool.r) ** 2
        assert dx_back / initial_dx == pytest.approx(expected_ratio, rel=0.01)

    def test_swap_does_not_bridge_desert_when_L_active_is_zero(self):
        """
        If the active band has zero liquidity, swaps should return zeros and leave
        the pool state unchanged even if liquidity exists in adjacent bands.
        """
        pool, _ = build_empty_pool()
        spacing = pool.tick_spacing
        
        # Add liquidity only above the active tick (creating a "desert" at active tick)
        upper_lower = pool.tick + spacing
        upper_upper = pool.tick + 2 * spacing
        pool.add_liquidity_range(upper_lower, upper_upper, 50_000.0)
        pool.recompute_active_L()
        
        # Active tick has no liquidity
        assert pool.L_active == pytest.approx(0.0, abs=EPS_LIQ)
        
        tick_before = pool.tick
        S_before = pool.S
        
        # Try a Y→X swap (pushes price up toward the liquidity)
        dy_in = 1000.0
        used_dy, dx_out, fee_y = pool.swap_y_to_x(dy_in, fee_cb=None)

        assert used_dy == 0.0
        assert dx_out == 0.0
        assert fee_y == 0.0
        assert pool.tick == tick_before
        assert pool.S == pytest.approx(S_before, rel=0.0, abs=EPS_BOUNDARY)

    def test_swap_returns_zero_on_empty_pool(self, empty_pool: V3Pool):
        """
        Swaps on a pool with no liquidity return (0, 0, 0) and leave price unchanged.
        """
        pool = empty_pool
        start_price = pool.price
        start_S = pool.S
        
        # Try X→Y swap
        used_dx, dy_out, fee_x = pool.swap_x_to_y(5.0, fee_cb=None)
        assert used_dx == 0.0
        assert dy_out == 0.0
        assert fee_x == 0.0
        assert pool.price == start_price
        assert pool.S == start_S
        
        # Try Y→X swap
        used_dy, dx_out, fee_y = pool.swap_y_to_x(1000.0, fee_cb=None)
        assert used_dy == 0.0
        assert dx_out == 0.0
        assert fee_y == 0.0
        assert pool.price == start_price
        assert pool.S == start_S


# =============================================================================
# 1.2 Liquidity (Mint/Burn) Tests
# =============================================================================

class TestLiquidityManagement:
    """Tests for adding and removing liquidity."""

    def test_add_liquidity_updates_liquidity_net(self, empty_pool: V3Pool):
        """
        Verify add_liquidity_range correctly updates liquidity_net:
        liquidity_net[lower] += L
        liquidity_net[upper] -= L
        """
        pool = empty_pool
        lower = pool.tick - pool.tick_spacing
        upper = pool.tick + pool.tick_spacing
        L = 10_000.0
        
        # Record initial values
        net_lower_before = pool.liquidity_net.get(lower, 0.0)
        net_upper_before = pool.liquidity_net.get(upper, 0.0)
        
        pool.add_liquidity_range(lower, upper, L)
        
        # Verify liquidity_net updated correctly
        assert pool.liquidity_net[lower] == pytest.approx(net_lower_before + L)
        assert pool.liquidity_net[upper] == pytest.approx(net_upper_before - L)

    def test_add_liquidity_updates_L_active(self, empty_pool: V3Pool):
        """
        If the minted range includes the active tick, L_active should increase by exactly L.
        """
        pool = empty_pool
        assert pool.L_active == 0.0
        
        # Range that includes the active tick
        lower = pool.tick - pool.tick_spacing
        upper = pool.tick + pool.tick_spacing
        L = 25_000.0
        
        pool.add_liquidity_range(lower, upper, L)
        
        # L_active should increase by L
        assert pool.L_active == pytest.approx(L)

    def test_add_liquidity_no_L_active_change_out_of_range(self, empty_pool: V3Pool):
        """
        If the minted range does NOT include the active tick, L_active should NOT change.
        """
        pool = empty_pool
        L_before = pool.L_active
        
        # Range entirely above the active tick
        lower = pool.tick + pool.tick_spacing
        upper = pool.tick + 2 * pool.tick_spacing
        L = 15_000.0
        
        pool.add_liquidity_range(lower, upper, L)
        
        # L_active should be unchanged
        assert pool.L_active == pytest.approx(L_before)

    def test_remove_liquidity_reverts_add(self, empty_pool: V3Pool):
        """
        Adding then removing the same liquidity should restore liquidity_net
        and L_active to original values.
        """
        pool = empty_pool
        lower = pool.tick - pool.tick_spacing
        upper = pool.tick + pool.tick_spacing
        L = 30_000.0
        
        # Record initial state
        L_active_before = pool.L_active
        net_lower_before = pool.liquidity_net.get(lower, 0.0)
        net_upper_before = pool.liquidity_net.get(upper, 0.0)
        
        # Add liquidity
        pool.add_liquidity_range(lower, upper, L)
        
        # Verify it was added
        assert pool.L_active == pytest.approx(L_active_before + L)
        
        # Remove same liquidity (add negative L)
        pool.add_liquidity_range(lower, upper, -L)
        
        # Should be back to original state
        assert pool.L_active == pytest.approx(L_active_before)
        assert pool.liquidity_net.get(lower, 0.0) == pytest.approx(net_lower_before)
        assert pool.liquidity_net.get(upper, 0.0) == pytest.approx(net_upper_before)

    def test_position_current_amounts_below_range(self):
        """
        When S < s_a (below range), position holds all token0, no token1.
        x = L * (1/s_a - 1/s_b), y = 0
        """
        L = 10_000.0
        sa = 50.0
        sb = 60.0
        S_below = 40.0  # Below sa
        
        pos = Position(
            owner=1, lower=-100, upper=100, L=L, sa=sa, sb=sb,
            amt0_init=0, amt1_init=0
        )
        
        amt0, amt1 = pos.current_amounts(S_below)
        
        expected_x = L * (1.0 / sa - 1.0 / sb)
        assert amt0 == pytest.approx(expected_x, rel=1e-9)
        assert amt1 == pytest.approx(0.0, abs=1e-12)

    def test_position_current_amounts_above_range(self):
        """
        When S > s_b (above range), position holds all token1, no token0.
        x = 0, y = L * (s_b - s_a)
        """
        L = 10_000.0
        sa = 50.0
        sb = 60.0
        S_above = 70.0  # Above sb
        
        pos = Position(
            owner=1, lower=-100, upper=100, L=L, sa=sa, sb=sb,
            amt0_init=0, amt1_init=0
        )
        
        amt0, amt1 = pos.current_amounts(S_above)
        
        expected_y = L * (sb - sa)
        assert amt0 == pytest.approx(0.0, abs=1e-12)
        assert amt1 == pytest.approx(expected_y, rel=1e-9)

    def test_position_current_amounts_in_range(self):
        """
        When s_a < S < s_b (in range), position holds both tokens.
        x = L * (1/S - 1/s_b), y = L * (S - s_a)
        """
        L = 10_000.0
        sa = 50.0
        sb = 60.0
        S_in = 55.0  # In range
        
        pos = Position(
            owner=1, lower=-100, upper=100, L=L, sa=sa, sb=sb,
            amt0_init=0, amt1_init=0
        )
        
        amt0, amt1 = pos.current_amounts(S_in)
        
        expected_x = L * (1.0 / S_in - 1.0 / sb)
        expected_y = L * (S_in - sa)
        assert amt0 == pytest.approx(expected_x, rel=1e-9)
        assert amt1 == pytest.approx(expected_y, rel=1e-9)

    def test_minted_amounts_match_position_algebra(self, empty_pool: V3Pool):
        """
        Verify minted_amounts_at_S() returns values consistent with Position.current_amounts().
        """
        pool = empty_pool
        L = 20_000.0
        lower = pool.tick - pool.tick_spacing
        upper = pool.tick + pool.tick_spacing
        sa = pool.s_lower(lower)
        sb = pool.s_upper(upper)
        S = pool.S
        
        # Get amounts from utility function
        amt0_util, amt1_util = minted_amounts_at_S(L, sa, sb, S)
        
        # Get amounts from Position
        pos = Position(
            owner=1, lower=lower, upper=upper, L=L, sa=sa, sb=sb,
            amt0_init=0, amt1_init=0
        )
        amt0_pos, amt1_pos = pos.current_amounts(S)
        
        assert amt0_util == pytest.approx(amt0_pos, rel=1e-12)
        assert amt1_util == pytest.approx(amt1_pos, rel=1e-12)


# =============================================================================
# 1.3 BoundaryIndex Tests
# =============================================================================

class TestBoundaryIndex:
    """Tests for the BoundaryIndex sparse index structure."""

    def test_boundary_index_prefix_sum(self, pool_with_multi_range_liquidity: V3Pool):
        """
        Verify active_liquidity_at_tick() returns the correct prefix sum of liquidity_net.
        """
        pool = pool_with_multi_range_liquidity
        bidx = BoundaryIndex(pool.liquidity_net)
        
        # Calculate expected L at various ticks manually
        sorted_ticks = sorted(pool.liquidity_net.keys())
        
        for test_tick in [pool.tick - 2 * pool.tick_spacing, pool.tick, pool.tick + pool.tick_spacing]:
            expected_L = sum(
                dL for t, dL in pool.liquidity_net.items() if t <= test_tick
            )
            actual_L = bidx.active_liquidity_at_tick(test_tick)
            assert actual_L == pytest.approx(expected_L, rel=1e-9)

    def test_boundary_index_next_up(self, pool_with_multi_range_liquidity: V3Pool):
        """
        Verify next_up() correctly locates the next initialized tick boundary upward.
        """
        pool = pool_with_multi_range_liquidity
        bidx = BoundaryIndex(pool.liquidity_net)
        
        # Get sorted initialized ticks
        sorted_ticks = sorted([k for k, v in pool.liquidity_net.items() if abs(v) > EPS_LIQ])
        
        if len(sorted_ticks) >= 2:
            # next_up from first tick should give second tick
            first_tick = sorted_ticks[0]
            second_tick = sorted_ticks[1]
            assert bidx.next_up(first_tick) == second_tick
        
        # next_up from below all ticks should give first tick
        if sorted_ticks:
            min_tick = min(sorted_ticks)
            assert bidx.next_up(min_tick - pool.tick_spacing) == min_tick

    def test_boundary_index_prev_down(self, pool_with_multi_range_liquidity: V3Pool):
        """
        Verify prev_down() correctly locates the previous initialized tick boundary downward.
        """
        pool = pool_with_multi_range_liquidity
        bidx = BoundaryIndex(pool.liquidity_net)
        
        # Get sorted initialized ticks
        sorted_ticks = sorted([k for k, v in pool.liquidity_net.items() if abs(v) > EPS_LIQ])
        
        if len(sorted_ticks) >= 2:
            # prev_down from last tick should give second-to-last tick
            last_tick = sorted_ticks[-1]
            second_last_tick = sorted_ticks[-2]
            assert bidx.prev_down(last_tick) == second_last_tick
        
        # prev_down from above all ticks should give last tick
        if sorted_ticks:
            max_tick = max(sorted_ticks)
            assert bidx.prev_down(max_tick + pool.tick_spacing) == max_tick

    def test_boundary_index_dirty_flag(self, empty_pool: V3Pool):
        """
        After add_liquidity_range, the index should be marked dirty
        and auto-refresh on next query.
        """
        pool = empty_pool
        lower = pool.tick
        upper = pool.tick + pool.tick_spacing
        
        # Initial state
        assert pool.bidx.active_liquidity_at_tick(pool.tick) == 0.0
        
        # Add liquidity
        pool.add_liquidity_range(lower, upper, 10_000.0)
        
        # BoundaryIndex should be dirty
        assert pool.bidx.dirty is True
        
        # Query should auto-refresh
        L = pool.bidx.active_liquidity_at_tick(pool.tick)
        assert L == pytest.approx(10_000.0)
        assert pool.bidx.dirty is False


# =============================================================================
# 1.4 Fee Allocation Tests
# =============================================================================

class TestFeeAllocation:
    """Tests for fee allocation to liquidity providers."""

    def test_fee_callback_records_segment_fee_single_tick(self, pool_with_liquidity: V3Pool):
        """
        For a single-tick swap, fee_cb should be called exactly once with the
        segment fee matching the returned fee and the correct tick/L snapshot.
        """
        pool = pool_with_liquidity
        tick_before = pool.tick
        L_before = pool.L_active

        calls = []

        def fee_cb(token: str, fee_amt: float, tick_snapshot: int, L_snapshot: float) -> None:
            calls.append((token, fee_amt, tick_snapshot, L_snapshot))

        # Calculate a swap small enough to stay within the tick.
        S_lo = pool.s_lower()
        dx_max = L_before * (1.0 / S_lo - 1.0 / pool.S) / pool.r
        dx_in = dx_max * 0.1

        used_dx, _dy_out, fee_x = pool.swap_x_to_y(dx_in, fee_cb=fee_cb)

        assert used_dx == pytest.approx(dx_in)
        assert fee_x == pytest.approx(dx_in * pool.f, rel=1e-12, abs=1e-12)
        assert len(calls) == 1

        token, fee_amt, tick_snap, L_snap = calls[0]
        assert token == "x"
        assert tick_snap == tick_before
        assert L_snap == pytest.approx(L_before, rel=1e-12, abs=1e-12)
        assert fee_amt == pytest.approx(fee_x, rel=1e-12, abs=1e-12)

    def test_fee_callback_segments_sum_to_returned_fee_across_tick_crossing(self):
        """
        Across a swap that crosses exactly one boundary, fee_cb should be called
        once per segment with the expected tick/L snapshots, and the sum of
        segment fees should match the returned fee.
        """
        pool, _ = build_empty_pool()
        spacing = pool.tick_spacing

        L0 = 30_000.0
        L1 = 10_000.0
        pool.add_liquidity_range(pool.tick - spacing, pool.tick + spacing, L0)
        pool.add_liquidity_range(pool.tick + spacing, pool.tick + 2 * spacing, L1)
        pool.recompute_active_L()

        tick_before = pool.tick
        S_before = pool.S
        S_hi = pool.s_upper()
        dy_to_boundary = L0 * (S_hi - S_before)

        # Cross one boundary, then move slightly in the next band.
        S_hi_next = pool.s_upper(tick_before + spacing)
        dy_to_next_boundary = L1 * (S_hi_next - S_hi)
        dy_eff_total = dy_to_boundary + min(0.5 * dy_to_next_boundary, 1.0)
        dy_in = dy_eff_total / pool.r

        calls = []

        def fee_cb(token: str, fee_amt: float, tick_snapshot: int, L_snapshot: float) -> None:
            calls.append((token, fee_amt, tick_snapshot, L_snapshot))

        used_dy, _dx_out, fee_y = pool.swap_y_to_x(dy_in, fee_cb=fee_cb)

        assert used_dy == pytest.approx(dy_in)
        assert pool.tick == tick_before + spacing
        assert len(calls) == 2
        assert [c[0] for c in calls] == ["y", "y"]
        assert [c[2] for c in calls] == [tick_before, tick_before + spacing]
        assert calls[0][3] == pytest.approx(L0, rel=1e-12, abs=1e-12)
        assert calls[1][3] == pytest.approx(L1, rel=1e-12, abs=1e-12)

        fee_sum = sum(c[1] for c in calls)
        assert fee_sum == pytest.approx(fee_y, rel=1e-12, abs=1e-12)
        assert fee_y == pytest.approx(dy_in * pool.f, rel=1e-12, abs=1e-12)

    def test_fees_accumulate_on_position(self):
        """
        Verify that pos.fees0 and pos.fees1 accumulate correctly after swaps
        and are reflected in pos.fees_value_y(m).
        """
        # Create a position
        pos = Position(
            owner=1, lower=-10, upper=10, L=10_000.0, sa=40.0, sb=50.0,
            amt0_init=100.0, amt1_init=4000.0
        )
        
        # Accumulate fees
        pos.fees0 += 0.5
        pos.fees1 += 10.0
        pos.fees0 += 0.3
        pos.fees1 += 5.0
        
        assert pos.fees0 == pytest.approx(0.8)
        assert pos.fees1 == pytest.approx(15.0)
        
        # Check fees_value_y at a given price m
        m = 2000.0
        expected_value = pos.fees0 * m + pos.fees1
        assert pos.fees_value_y(m) == pytest.approx(expected_value)


# =============================================================================
# Quote Consistency Tests
# =============================================================================

class TestQuoteConsistency:
    """Tests that quotes match actual swap results."""

    def test_quote_does_not_mutate_pool_state(self, pool_with_liquidity: V3Pool):
        """
        quote_* methods should be read-only with respect to economic pool state:
        (tick, S, L_active) must not change.
        """
        pool = pool_with_liquidity
        tick_before = pool.tick
        S_before = pool.S
        L_before = pool.L_active

        _ = pool.quote_x_to_y(1.0)
        assert pool.tick == tick_before
        assert pool.S == pytest.approx(S_before, rel=0.0, abs=EPS_BOUNDARY)
        assert pool.L_active == pytest.approx(L_before, rel=1e-12, abs=1e-12)

        _ = pool.quote_y_to_x(1.0)
        assert pool.tick == tick_before
        assert pool.S == pytest.approx(S_before, rel=0.0, abs=EPS_BOUNDARY)
        assert pool.L_active == pytest.approx(L_before, rel=1e-12, abs=1e-12)

    def test_quote_x_to_y_matches_swap_result(self, pool_with_liquidity: V3Pool):
        """
        The read-only X→Y quote should match an actual swap on a cloned pool.
        """
        pool = pool_with_liquidity
        dx_in = 1.0
        
        quoted = pool.quote_x_to_y(dx_in)
        
        pool_for_swap = copy.deepcopy(pool)
        used_dx, dy_out, _ = pool_for_swap.swap_x_to_y(dx_in, fee_cb=None)
        
        assert used_dx == pytest.approx(dx_in)
        assert dy_out == pytest.approx(quoted, rel=1e-9, abs=1e-9)

    def test_quote_y_to_x_matches_swap_result(self, pool_with_liquidity: V3Pool):
        """
        The read-only Y→X quote should match an actual swap on a cloned pool.
        """
        pool = pool_with_liquidity
        dy_in = 1000.0
        
        quoted = pool.quote_y_to_x(dy_in)
        
        pool_for_swap = copy.deepcopy(pool)
        used_dy, dx_out, _ = pool_for_swap.swap_y_to_x(dy_in, fee_cb=None)
        
        assert used_dy == pytest.approx(dy_in)
        assert dx_out == pytest.approx(quoted, rel=1e-9, abs=1e-9)

    def test_quote_returns_zero_without_liquidity(self, empty_pool: V3Pool):
        """
        Quoting should return zero when the pool has no active liquidity.
        """
        pool = empty_pool
        assert pool.quote_x_to_y(10.0) == 0.0
        assert pool.quote_y_to_x(10.0) == 0.0

    def test_quote_multi_tick_consistency(self, pool_with_multi_range_liquidity: V3Pool):
        """
        Quote should match swap even when crossing multiple tick boundaries.
        """
        pool = pool_with_multi_range_liquidity
        
        # Large swap that crosses ticks
        dy_in = 50_000.0
        
        quoted = pool.quote_y_to_x(dy_in)
        
        pool_for_swap = copy.deepcopy(pool)
        used_dy, dx_out, _ = pool_for_swap.swap_y_to_x(dy_in, fee_cb=None)
        
        # Quote should match actual output (within tolerance)
        assert dx_out == pytest.approx(quoted, rel=1e-6, abs=1e-9)
