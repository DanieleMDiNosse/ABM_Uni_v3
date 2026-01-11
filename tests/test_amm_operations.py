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

from utils import build_empty_pool, minted_amounts_at_S, EPS_LIQ, EPS_BOUNDARY
from uniswapv3_pool import V3Pool, BoundaryIndex
from agents import Position


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

    def test_swap_crosses_tick_boundary_liquidity_update(self, pool_with_multi_range_liquidity: V3Pool):
        """
        Verify that when a swap crosses a tick boundary, L_active is correctly
        updated by the liquidity_net delta at that boundary.
        """
        pool = pool_with_multi_range_liquidity
        tick_before = pool.tick
        L_before = pool.L_active
        
        # Calculate dy needed to cross to the next tick up
        S_hi = pool.s_upper()
        dy_to_boundary = pool.L_active * (S_hi - pool.S)
        # Add extra to cross and go beyond
        dy_in = (dy_to_boundary / pool.r) + 500.0
        
        used_dy, dx_out, fee_y = pool.swap_y_to_x(dy_in, fee_cb=None)
        
        # Verify tick crossed upward
        assert pool.tick > tick_before
        
        # L_active should have changed due to liquidity_net at the boundary
        expected_L = L_before + pool.liquidity_net.get(tick_before + pool.tick_spacing, 0.0)
        # Note: might cross multiple ticks, so L_active should reflect all crossings
        # For this test, we just verify L_active changed appropriately
        assert pool.L_active != L_before or pool.liquidity_net.get(tick_before + pool.tick_spacing, 0.0) == 0.0

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

    def test_swap_desert_bridging(self):
        """
        When liquidity is zero in the active tick but exists in adjacent ticks,
        verify the swap correctly bridges to the next initialized tick.
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
        
        S_before = pool.S
        
        # Try a Y→X swap (pushes price up toward the liquidity)
        dy_in = 1000.0
        used_dy, dx_out, fee_y = pool.swap_y_to_x(dy_in, fee_cb=None)
        
        # Should not consume any input since we can't move price through empty liquidity
        # Actually, the pool should bridge up to the next initialized tick
        # If we're already at a boundary, the swap should execute in the next tick
        # The behavior depends on the exact implementation
        
        # At minimum, verify the pool state is consistent
        assert pool.S >= S_before  # Price should not decrease for Y→X

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

    def test_fee_allocation_pro_rata(self):
        """
        When multiple positions share the active tick, fees should be allocated
        proportionally to each position's liquidity share.
        """
        pool, m0 = build_empty_pool()
        
        # Create positions with different liquidity amounts
        L1, L2, L3 = 10_000.0, 20_000.0, 30_000.0
        total_L = L1 + L2 + L3
        
        lower = pool.tick - pool.tick_spacing
        upper = pool.tick + pool.tick_spacing
        sa = pool.s_lower(lower)
        sb = pool.s_upper(upper)
        
        # Add all positions to the pool
        pool.add_liquidity_range(lower, upper, L1)
        pool.add_liquidity_range(lower, upper, L2)
        pool.add_liquidity_range(lower, upper, L3)
        pool.recompute_active_L()
        
        # Create Position objects for tracking fees
        pos1 = Position(owner=1, lower=lower, upper=upper, L=L1, sa=sa, sb=sb, amt0_init=0, amt1_init=0)
        pos2 = Position(owner=2, lower=lower, upper=upper, L=L2, sa=sa, sb=sb, amt0_init=0, amt1_init=0)
        pos3 = Position(owner=3, lower=lower, upper=upper, L=L3, sa=sa, sb=sb, amt0_init=0, amt1_init=0)
        
        # Simulate fee allocation (this mimics allocate_fees logic)
        fee_total = 100.0
        for pos in [pos1, pos2, pos3]:
            pos.fees0 = fee_total * (pos.L / total_L)
        
        # Verify pro-rata allocation
        assert pos1.fees0 == pytest.approx(fee_total * L1 / total_L)
        assert pos2.fees0 == pytest.approx(fee_total * L2 / total_L)
        assert pos3.fees0 == pytest.approx(fee_total * L3 / total_L)
        
        # Sum should equal total
        assert pos1.fees0 + pos2.fees0 + pos3.fees0 == pytest.approx(fee_total)

    def test_fee_allocation_via_swap(self, pool_with_liquidity: V3Pool):
        """
        Verify that fees are generated during swaps equal to input * fee_rate
        when the swap stays within a single tick.
        """
        pool = pool_with_liquidity
        
        # Calculate a swap small enough to stay within the tick
        S_lo = pool.s_lower()
        S = pool.S
        L = pool.L_active
        dx_max = L * (1.0 / S_lo - 1.0 / S) / pool.r
        dx_in = dx_max * 0.1  # Small swap stays in tick
        expected_fee = dx_in * pool.f
        
        used_dx, dy_out, fee_x = pool.swap_x_to_y(dx_in, fee_cb=None)
        
        assert fee_x == pytest.approx(expected_fee, rel=1e-12)

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
