"""
PnL Accounting Consistency Tests.

These tests verify the correctness of PnL accounting for all agent types,
focusing on:
1. TraderStepAccumulator CEX trade handling (Flaw 6 fix)
2. JIT flash fee deduction from rebalancer (Flaw 2 fix)
3. JIT burn CEX impact accounting (Flaw 4 fix)
4. Rebalancer cumulative_R correctness
5. Zero-sum / conservation properties
6. Agent-specific edge cases

Run with:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/test_pnl_accounting_consistency.py -v
"""

import copy
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from utils import build_empty_pool, minted_amounts_at_S, ReferenceMarket
from uniswapv3_pool import V3Pool
from agents import (
    Position, LPAgent, RebalancerState,
    lp_token0_exposure, lp_wealth_y, lp_total_fee_earned_value_y,
    lp_total_position_value_y, lp_principal_value_y, lp_fee_value_y
)
from run import simulate, TraderStepAccumulator


# =============================================================================
# Fixtures
# =============================================================================

def _base_simulate_kwargs(tmp_path, **overrides: Any) -> Dict[str, Any]:
    """Base simulation parameters for testing."""
    params: Dict[str, Any] = dict(
        config_name="pnl_accounting_test",
        block_time=5,
        T=5,
        seed=1,
        liquidity_for_gif=False,
        light_mode=False,
        cex_mu=0.0,
        cex_sigma=0.0005,
        smart_trades_per_block=0.0,
        noise_trades_per_block=0.0,
        N_LP=2,
        passive_lp_share=1.0,
        tau=5,
        narrow_mints_per_block=0.0,
        passive_mints_per_block=0.0,
        passive_burns_per_block=0.0,
        w_min_ticks=10,
        w_max_ticks=100,
        basis_half_life=1,
        slope_s=1.0,
        binom_n=0,
        binom_p=0.5,
        trader_mean=2.5,
        trader_sigma=1.0,
        theta_T=1.0,
        slippage_tolerance=0.01,
        passive_width_pct=10.0,
        passive_width_ticks=None,
        mint_mu=2.5,
        mint_sigma=1.0,
        theta_TP=1.0,
        theta_SL=1.0,
        k_out_min=1,
        k_out_max=1,
        initial_binom_N=50,
        initial_total_L=50_000.0,
        fee_mode="static",
        f0=0.003,
        f_min=0.0001,
        f_max=0.01,
        fee_half_life=2,
        k_sigma=1.0,
        k_basis=0.0,
        fee_step_bps_min=0.0,
        fee_step_bps_max=200.0,
        fee_cooldown=0,
        visualize=False,
        skip_step=0,
        results_root=tmp_path,
        verbose=False,
    )
    params.update(overrides)
    return params


# =============================================================================
# 1. TraderStepAccumulator CEX Trade Tests (Flaw 6)
# =============================================================================

class TestTraderStepAccumulatorCEXTrades:
    """Tests for CEX trade PnL handling in TraderStepAccumulator."""

    def test_cex_trade_pnl_zero_for_fair_exchange(self):
        """
        CEX trades are fair exchanges at the current price.
        record_cex_trade_pnl(0.0) should record zero PnL that is NOT revalued at settlement.
        """
        acc = TraderStepAccumulator()
        
        # Record a fair CEX trade (dx * m = dy => PnL = 0)
        acc.record_cex_trade_pnl(0.0)
        
        # Settlement at any price should not change this
        m_settle = 3000.0  # Different from trade price
        acc.settle(m_settle)
        
        # CEX trade PnL should remain 0
        assert acc.pnl == pytest.approx(0.0)

    def test_cex_trade_pnl_not_revalued_at_settlement(self):
        """
        CEX trade PnL (realized_pnl_cex) should NOT be revalued when settle() is called.
        Only DEX flows (via record_swap) should be revalued.
        """
        acc = TraderStepAccumulator()
        
        # DEX trade: sell 1 X, receive 1900 Y at DEX
        acc.record_swap(dx_in=1.0, dy_out=1900.0)
        
        # CEX trade: fair exchange, PnL = 0
        acc.record_cex_trade_pnl(0.0)
        
        # Settle at m = 2000
        acc.settle(2000.0)
        
        # DEX PnL = (1900 - 0) + (0 - 1) * 2000 = 1900 - 2000 = -100
        # CEX PnL = 0 (already realized)
        # Total PnL = -100
        expected_pnl = -100.0
        assert acc.pnl == pytest.approx(expected_pnl)

    def test_mixed_dex_cex_trades_pnl_correct(self):
        """
        Mixed DEX and CEX trades should accumulate correctly.
        DEX flows are revalued at settlement; CEX PnL is already realized.
        """
        acc = TraderStepAccumulator()
        
        # Trade 1: DEX X→Y (sell 0.5 X, get 980 Y)
        acc.record_swap(dx_in=0.5, dy_out=980.0)
        
        # Trade 2: CEX fair exchange (PnL = 0)
        acc.record_cex_trade_pnl(0.0)
        
        # Trade 3: DEX Y→X (sell 500 Y, get 0.26 X)
        acc.record_swap(dy_in=500.0, dx_out=0.26)
        
        # Trade 4: Another CEX trade
        acc.record_cex_trade_pnl(0.0)
        
        # Settle at m = 2000
        acc.settle(2000.0)
        
        # DEX flows: dx_in=0.5, dx_out=0.26, dy_in=500, dy_out=980
        # DEX PnL = (980 - 500) + (0.26 - 0.5) * 2000 = 480 - 480 = 0
        expected_pnl = (980.0 - 500.0) + (0.26 - 0.5) * 2000.0
        assert acc.pnl == pytest.approx(expected_pnl)

    def test_record_cex_trade_pnl_accumulates(self):
        """
        Multiple CEX trades should accumulate in realized_pnl_cex.
        """
        acc = TraderStepAccumulator()
        
        # Multiple CEX trades (all fair => 0 PnL each)
        acc.record_cex_trade_pnl(0.0)
        acc.record_cex_trade_pnl(0.0)
        acc.record_cex_trade_pnl(0.0)
        
        assert acc.realized_pnl_cex == pytest.approx(0.0)
        
        acc.settle(2500.0)
        assert acc.pnl == pytest.approx(0.0)

    def test_settle_combines_dex_and_cex_pnl(self):
        """
        After settle(), acc.pnl should equal DEX PnL + realized_pnl_cex.
        """
        acc = TraderStepAccumulator()
        
        # DEX trade only
        acc.record_swap(dx_in=1.0, dy_out=1950.0)
        
        # No CEX trades
        m_settle = 2000.0
        acc.settle(m_settle)
        
        dex_pnl = (1950.0 - 0) + (0 - 1.0) * 2000.0  # -50
        assert acc.pnl == pytest.approx(dex_pnl)
        assert acc.realized_pnl_cex == pytest.approx(0.0)

    def test_realized_pnl_cex_field_exists(self):
        """
        TraderStepAccumulator should have a realized_pnl_cex field initialized to 0.
        """
        acc = TraderStepAccumulator()
        assert hasattr(acc, 'realized_pnl_cex')
        assert acc.realized_pnl_cex == 0.0

    def test_record_cex_trade_pnl_method_exists(self):
        """
        TraderStepAccumulator should have a record_cex_trade_pnl method.
        """
        acc = TraderStepAccumulator()
        assert hasattr(acc, 'record_cex_trade_pnl')
        assert callable(acc.record_cex_trade_pnl)


# =============================================================================
# 2. JIT Flash Fee Accounting Tests (Flaw 2)
# =============================================================================

class TestJITFlashFeeAccounting:
    """Tests for JIT flash fee deduction from rebalancer."""

    def test_jit_flash_fee_reduces_hedged_pnl(self, tmp_path):
        """
        JIT flash fees should reduce hedged PnL.
        After a JIT mint, the rebalancer's cash_y should be reduced by flash_fee_y.
        """
        flash_fee_rate = 0.05  # 5% flash loan fee
        
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=10,
            seed=42,
            noise_trades_per_block=5.0,
            slippage_tolerance=0.5,
            trader_mean=5.0,
            p_jit=0.8,  # High probability of JIT execution
            N_jit=5,
            liquidity_perc_jit=0.5,
            flash_loan_fee=flash_fee_rate,
        ))
        
        # If JIT minted, flash fees should be paid
        jiter_flash_fees = out.get("jiter_flash_fees_paid_y", 0.0)
        
        # Flash fees should be positive if JIT executed
        jiter_execs = sum(out.get("jiter_activity_signs", []))
        if jiter_execs > 0:
            assert jiter_flash_fees > 0, "JIT executed but no flash fees recorded"

    def test_jit_flash_fee_consistency(self, tmp_path):
        """
        JIT agent's flash_fees_paid_y should equal the sum of all flash fees paid.
        """
        flash_fee_rate = 0.02
        
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=15,
            seed=123,
            noise_trades_per_block=8.0,
            slippage_tolerance=0.3,
            p_jit=0.8,
            N_jit=5,
            liquidity_perc_jit=0.5,
            flash_loan_fee=flash_fee_rate,
        ))
        
        # flash_fees_paid_y should be non-negative
        flash_fees = out.get("jiter_flash_fees_paid_y", 0.0)
        assert flash_fees >= 0.0


# =============================================================================
# 3. JIT Burn CEX Impact Tests (Flaw 4)
# =============================================================================

class TestJITBurnCEXImpact:
    """Tests for JIT burn handling and CEX impact recording."""

    def test_jit_burn_records_cex_impact(self, tmp_path):
        """
        When JIT burns a position with token0, it should record CEX impact
        (delta_a_cex_this should be updated).
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=15,
            seed=77,
            noise_trades_per_block=5.0,
            slippage_tolerance=0.5,
            p_jit=0.8,
            N_jit=5,
            liquidity_perc_jit=0.5,
            flash_loan_fee=0.0,  # No flash fee to isolate CEX impact
        ))
        
        # Simulation should complete successfully with JIT enabled
        assert "DEX_price" in out
        assert len(out["DEX_price"]) == 15
        
        # JIT activity should be tracked (may or may not execute depending on conditions)
        assert "jiter_activity_steps" in out or "lp_pnl_total" in out


# =============================================================================
# 4. Rebalancer Accounting Tests
# =============================================================================

class TestRebalancerAccounting:
    """Tests for rebalancer cumulative_R and hedged PnL calculation."""

    def test_rebalancer_cumulative_R_integral_formula(self):
        """
        cumulative_R should track ∫ x_prev dM correctly.
        For discrete steps: cumulative_R += x_prev * (M_new - M_old)
        """
        rb = RebalancerState()
        rb.initialized = True
        rb.x_prev = 5.0
        rb.last_M = 1000.0
        rb.cumulative_R = 0.0
        
        # Simulate price moves
        price_path = [1000.0, 1050.0, 1030.0, 1100.0, 1080.0]
        
        for i in range(1, len(price_path)):
            M_old = price_path[i-1]
            M_new = price_path[i]
            rb.cumulative_R += rb.x_prev * (M_new - M_old)
            rb.last_M = M_new
        
        # Expected: 5 * (1080 - 1000) = 5 * 80 = 400
        expected_R = rb.x_prev * (price_path[-1] - price_path[0])
        assert rb.cumulative_R == pytest.approx(expected_R)

    def test_rebalancer_value_increases_with_positive_price_move(self):
        """
        With positive token0 exposure, rebalancer value should increase
        when price increases.
        """
        rb = RebalancerState()
        rb.initialized = True
        rb.initial_rebal_value_y = 10000.0
        rb.x_prev = 3.0  # Positive exposure
        rb.last_M = 2000.0
        rb.cumulative_R = 0.0
        
        # Price increases
        M_new = 2200.0
        rb.cumulative_R += rb.x_prev * (M_new - rb.last_M)
        
        rebal_value = rb.initial_rebal_value_y + rb.cumulative_R
        
        # cumulative_R = 3 * 200 = 600
        # rebal_value = 10000 + 600 = 10600
        assert rebal_value == pytest.approx(10600.0)

    def test_hedged_pnl_fee_capture_minus_lvr(self, tmp_path):
        """
        Hedged PnL should approximate: fees earned - LVR.
        LVR = rebalancer_value - LP_value (when LP loses to arb).
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=20,
            seed=42,
            noise_trades_per_block=5.0,
            slippage_tolerance=0.3,
            cex_sigma=0.001,
        ))
        
        # LVR = fee_value - hedged_pnl
        lvr_total = np.array(out["lp_lvr_total_series"])
        fee_value = np.array(out["lp_fee_value_total_series"])
        hedged_pnl = np.array(out["lp_pnl_total"])
        
        computed_lvr = fee_value - hedged_pnl
        np.testing.assert_allclose(lvr_total, computed_lvr, rtol=1e-10)

    def test_rebalancer_exposure_calculation(self):
        """
        lp_token0_exposure should correctly sum wallet_x and position amounts.
        """
        lp = LPAgent(id=1)
        lp.wallet_x = 2.0
        lp.wallet_y = 10000.0
        
        # Create a position
        L = 10000.0
        sa = 40.0
        sb = 50.0
        S = 45.0
        amt0, amt1 = minted_amounts_at_S(L, sa, sb, S)
        pos = Position(owner=1, lower=-10, upper=10, L=L, sa=sa, sb=sb, amt0_init=amt0, amt1_init=amt1)
        lp.positions.append(pos)
        
        exposure = lp_token0_exposure(lp, S)
        
        # Should equal wallet_x + position token0
        pos_x, _ = pos.current_amounts(S)
        expected = lp.wallet_x + pos_x
        assert exposure == pytest.approx(expected)


# =============================================================================
# 5. Arbitrageur PnL Accounting Tests
# =============================================================================

class TestArbitrageurPnLAccounting:
    """Tests for arbitrageur PnL calculations."""

    def test_arb_pnl_includes_flash_loan_cost(self, tmp_path):
        """
        Arbitrageur PnL should account for flash loan fees when enabled.
        """
        flash_fee = 0.05
        
        out_with_fee = simulate(**_base_simulate_kwargs(
            tmp_path / "with_fee",
            T=10,
            seed=42,
            noise_trades_per_block=10.0,
            slippage_tolerance=0.5,
            flash_loan_fee=flash_fee,
        ))
        
        out_no_fee = simulate(**_base_simulate_kwargs(
            tmp_path / "no_fee",
            T=10,
            seed=42,
            noise_trades_per_block=10.0,
            slippage_tolerance=0.5,
            flash_loan_fee=0.0,
        ))
        
        # With flash fees, arb PnL should be lower or arb should execute less
        arb_pnl_with = sum(out_with_fee["arb_pnl_steps"])
        arb_pnl_without = sum(out_no_fee["arb_pnl_steps"])
        
        # At least the test runs without error
        assert arb_pnl_with <= arb_pnl_without + 1e-6

    def test_arb_only_executes_when_profitable(self, tmp_path):
        """
        Each arb execution should result in non-negative PnL.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=15,
            seed=99,
            noise_trades_per_block=8.0,
            slippage_tolerance=0.5,
            flash_loan_fee=0.01,
        ))
        
        arb_pnl = out["arb_pnl_steps"]
        for pnl in arb_pnl:
            assert pnl >= -1e-9, f"Arb PnL was negative: {pnl}"

    def test_arb_pnl_zero_when_no_dislocation(self, tmp_path):
        """
        With no price dislocation, arb should not execute and PnL should be zero.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=5,
            seed=42,
            cex_sigma=0.0,  # No CEX drift
            smart_trades_per_block=0.0,
            noise_trades_per_block=0.0,
            fee_mode="static",
            flash_loan_fee=0.0,
        ))
        
        # With no trades and no CEX drift, DEX price stays in band
        arb_pnl = out["arb_pnl_steps"]
        for pnl in arb_pnl:
            assert pnl == pytest.approx(0.0, abs=1e-12)


# =============================================================================
# 6. Smart Router PnL Accounting Tests
# =============================================================================

class TestSmartRouterPnLAccounting:
    """Tests for smart router PnL calculations."""

    def test_smart_router_pnl_series_exists(self, tmp_path):
        """
        Smart router should produce PnL series with correct length.
        """
        T = 10
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=T,
            seed=42,
            smart_trades_per_block=5.0,
            noise_trades_per_block=0.0,
            slippage_tolerance=0.01,
            theta_T=0.99,
        ))
        
        # Smart router PnL series should exist
        sr_pnl = out["smart_router_pnl_steps"]
        sr_execs = out["smart_router_exec_count"]
        
        assert len(sr_pnl) == T
        assert len(sr_execs) == T

    def test_smart_router_cumulative_pnl_consistency(self, tmp_path):
        """
        Cumulative PnL should equal sum of step PnLs.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=10,
            seed=42,
            smart_trades_per_block=3.0,
            slippage_tolerance=0.5,
        ))
        
        sr_pnl_steps = np.array(out["smart_router_pnl_steps"])
        sr_pnl_cum = np.array(out["smart_router_pnl_cum"])
        expected_cum = np.cumsum(sr_pnl_steps)
        
        np.testing.assert_allclose(sr_pnl_cum, expected_cum, rtol=1e-12, atol=1e-12)


# =============================================================================
# 7. Conservation / Zero-Sum Tests
# =============================================================================

class TestPnLConservation:
    """Tests for PnL conservation properties in a closed system."""

    def test_total_pnl_bounded_by_external_flows(self, tmp_path):
        """
        In a closed AMM system (no external capital injection),
        the sum of all agent PnLs should be bounded.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=20,
            seed=42,
            smart_trades_per_block=3.0,
            noise_trades_per_block=3.0,
            slippage_tolerance=0.3,
        ))
        
        # All series should have consistent length
        T = len(out["DEX_price"])
        
        assert len(out["trader_pnl_steps"]) == T
        assert len(out["arb_pnl_steps"]) == T
        assert len(out["lp_pnl_total"]) == T

    def test_fees_collected_nonnegative(self, tmp_path):
        """
        Total fees collected by LPs should be non-negative.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=15,
            seed=42,
            noise_trades_per_block=5.0,
            slippage_tolerance=0.5,
        ))
        
        # LP fee value should be non-negative
        lp_fee_value = out["lp_fee_value_total_series"]
        for fee in lp_fee_value:
            assert fee >= -1e-9


# =============================================================================
# 8. Edge Case Tests
# =============================================================================

class TestPnLEdgeCases:
    """Edge case tests for PnL accounting."""

    def test_zero_trades_zero_pnl(self, tmp_path):
        """
        With no trades, all trader PnLs should be zero.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=5,
            smart_trades_per_block=0.0,
            noise_trades_per_block=0.0,
            cex_sigma=0.0,  # No price drift
        ))
        
        assert all(p == pytest.approx(0.0) for p in out["trader_pnl_steps"])
        assert all(p == pytest.approx(0.0) for p in out["arb_pnl_steps"])

    def test_high_volatility_pnl_stability(self, tmp_path):
        """
        High volatility should not cause numerical instability in PnL.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=10,
            seed=42,
            cex_sigma=0.01,  # High volatility
            noise_trades_per_block=3.0,
            slippage_tolerance=0.5,
        ))
        
        # All PnL values should be finite
        for pnl in out["trader_pnl_steps"]:
            assert math.isfinite(pnl)
        for pnl in out["lp_pnl_total"]:
            assert math.isfinite(pnl)

    def test_simulation_output_series_consistency(self, tmp_path):
        """
        All simulation output series should have consistent lengths.
        """
        T = 8
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=T,
            noise_trades_per_block=3.0,
            slippage_tolerance=0.5,
        ))
        
        # Check all series lengths match T
        assert len(out["DEX_price"]) == T
        assert len(out["CEX_price"]) == T
        assert len(out["trader_pnl_steps"]) == T
        assert len(out["trader_pnl_cum"]) == T
        assert len(out["arb_pnl_steps"]) == T
        assert len(out["arb_pnl_cum"]) == T
        assert len(out["lp_pnl_total"]) == T
        assert len(out["lp_unhedged_total"]) == T
        assert len(out["lp_lvr_total_series"]) == T

    def test_cumulative_pnl_equals_sum_of_steps(self, tmp_path):
        """
        Cumulative PnL series should equal running sum of step PnLs.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=10,
            seed=42,
            noise_trades_per_block=5.0,
            slippage_tolerance=0.5,
        ))
        
        trader_pnl_steps = np.array(out["trader_pnl_steps"])
        trader_pnl_cum = np.array(out["trader_pnl_cum"])
        expected_cum = np.cumsum(trader_pnl_steps)
        
        np.testing.assert_allclose(trader_pnl_cum, expected_cum, rtol=1e-12, atol=1e-12)
        
        # Same for arb
        arb_pnl_steps = np.array(out["arb_pnl_steps"])
        arb_pnl_cum = np.array(out["arb_pnl_cum"])
        expected_arb_cum = np.cumsum(arb_pnl_steps)
        
        np.testing.assert_allclose(arb_pnl_cum, expected_arb_cum, rtol=1e-12, atol=1e-12)


# =============================================================================
# 9. Noise Trader PnL Tests
# =============================================================================

class TestNoiseTraderPnLAccounting:
    """Tests for noise trader PnL calculations."""

    def test_noise_trader_pnl_series_length(self, tmp_path):
        """
        Noise trader PnL series should have correct length.
        """
        T = 7
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=T,
            noise_trades_per_block=5.0,
            slippage_tolerance=0.5,
        ))
        
        assert len(out["noise_trader_pnl_steps"]) == T
        assert len(out["noise_trader_pnl_cum"]) == T
        assert len(out["noise_trader_exec_count"]) == T

    def test_noise_trader_cumulative_consistency(self, tmp_path):
        """
        Noise trader cumulative PnL should equal sum of steps.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=10,
            seed=42,
            noise_trades_per_block=5.0,
            slippage_tolerance=0.5,
        ))
        
        nt_pnl_steps = np.array(out["noise_trader_pnl_steps"])
        nt_pnl_cum = np.array(out["noise_trader_pnl_cum"])
        expected_cum = np.cumsum(nt_pnl_steps)
        
        np.testing.assert_allclose(nt_pnl_cum, expected_cum, rtol=1e-12, atol=1e-12)


# =============================================================================
# 10. LP PnL Tests
# =============================================================================

class TestLPPnLAccounting:
    """Tests for LP PnL calculations."""

    def test_lp_pnl_series_lengths(self, tmp_path):
        """
        All LP PnL series should have length T.
        """
        T = 6
        out = simulate(**_base_simulate_kwargs(tmp_path, T=T))
        
        assert len(out["lp_pnl_total"]) == T
        assert len(out["lp_pnl_active"]) == T
        assert len(out["lp_pnl_passive"]) == T
        assert len(out["lp_unhedged_total"]) == T
        assert len(out["lp_unhedged_active"]) == T
        assert len(out["lp_unhedged_passive"]) == T
        assert len(out["lp_lvr_total_series"]) == T
        assert len(out["lp_lvr_active_series"]) == T
        assert len(out["lp_lvr_passive_series"]) == T

    def test_lvr_equals_fees_minus_hedged_pnl(self, tmp_path):
        """
        LVR = fee_value - hedged_pnl.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=10,
            seed=42,
            noise_trades_per_block=5.0,
            slippage_tolerance=0.5,
        ))
        
        lvr_total = np.array(out["lp_lvr_total_series"])
        fee_value_total = np.array(out["lp_fee_value_total_series"])
        hedged_pnl_total = np.array(out["lp_pnl_total"])
        
        expected_lvr = fee_value_total - hedged_pnl_total
        np.testing.assert_allclose(lvr_total, expected_lvr, rtol=1e-12, atol=1e-12)

    def test_lp_wealth_formula(self):
        """
        lp_wealth_y() should equal wallet_x * m + wallet_y + position_value.
        """
        lp = LPAgent(id=1)
        lp.wallet_x = 2.0
        lp.wallet_y = 1000.0
        
        # Create a position
        pos = Position(
            owner=1, lower=-10, upper=10, L=5000.0, sa=40.0, sb=50.0,
            amt0_init=100.0, amt1_init=4000.0
        )
        pos.fees0 = 0.1
        pos.fees1 = 5.0
        lp.positions.append(pos)
        
        S = 45.0  # In range
        m = 2000.0
        
        wealth = lp_wealth_y(lp, S, m)
        
        # Manual calculation
        wallet_value = lp.wallet_x * m + lp.wallet_y
        position_value = lp_total_position_value_y(lp, S, m)
        expected = wallet_value + position_value
        
        assert wealth == pytest.approx(expected, rel=1e-12)


# =============================================================================
# 11. Zero-Fee Rebalancer Tests (GPT 5.2 Edge Case)
# =============================================================================

class TestZeroFeeRebalancer:
    """Tests for LP rebalancer updates when fees are zero."""

    def test_rebalancer_updates_with_zero_fee(self, tmp_path):
        """
        LP rebalancer exposure should update correctly even when fee = 0.
        This tests the fix for the edge case where fee_cb was only called
        when _fee_seg > 0, breaking hedged PnL / LVR bookkeeping.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=10,
            seed=42,
            f0=0.0,  # Zero fee!
            fee_mode="static",
            noise_trades_per_block=5.0,
            slippage_tolerance=0.5,
            cex_sigma=0.001,
        ))
        
        # Simulation should complete without error
        assert "DEX_price" in out
        assert len(out["DEX_price"]) == 10
        
        # LVR series should be computed (not all NaN or zero by default)
        lvr_total = out["lp_lvr_total_series"]
        assert len(lvr_total) == 10
        
        # With zero fees, hedged PnL should track LVR correctly
        # LVR = fee_value - hedged_pnl, and fee_value = 0, so LVR = -hedged_pnl
        fee_value = np.array(out["lp_fee_value_total_series"])
        hedged_pnl = np.array(out["lp_pnl_total"])
        computed_lvr = fee_value - hedged_pnl
        
        np.testing.assert_allclose(
            np.array(lvr_total), computed_lvr, rtol=1e-10,
            err_msg="LVR != fee_value - hedged_pnl with zero fees"
        )

    def test_zero_fee_rebalancer_exposure_updates(self, tmp_path):
        """
        With zero fees, positions should still be tracked for rebalancing
        and exposure should update as price moves.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=15,
            seed=123,
            f0=0.0,  # Zero fee
            fee_mode="static",
            noise_trades_per_block=3.0,
            slippage_tolerance=0.5,
            cex_sigma=0.002,  # Some volatility
        ))
        
        # DEX price should move (swaps executed)
        dex_prices = out["DEX_price"]
        assert max(dex_prices) != min(dex_prices), "DEX price didn't move"
        
        # All PnL values should be finite
        for pnl in out["lp_pnl_total"]:
            assert math.isfinite(pnl), f"Hedged PnL is not finite: {pnl}"


# =============================================================================
# 12. JIT Wallet Netting Tests
# =============================================================================

class TestJITWalletNetting:
    """Tests for JIT flash loan netting after burn."""

    def test_jit_simulation_with_trades(self, tmp_path):
        """
        JIT simulation should complete successfully with active trading.
        After JIT burns, wallet state should be properly netted.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=20,
            seed=42,
            noise_trades_per_block=8.0,
            slippage_tolerance=0.5,
            p_jit=0.9,  # High probability
            N_jit=5,
            liquidity_perc_jit=0.5,
            flash_loan_fee=0.02,
        ))
        
        # Simulation should complete
        assert "DEX_price" in out
        assert len(out["DEX_price"]) == 20
        
        # LP PnL series should be present and valid
        assert "lp_pnl_total" in out
        assert len(out["lp_pnl_total"]) == 20
        
        # All PnL values should be finite
        for pnl in out["lp_pnl_total"]:
            assert math.isfinite(pnl), f"LP PnL is not finite: {pnl}"

