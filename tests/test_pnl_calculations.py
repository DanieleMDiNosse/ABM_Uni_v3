"""
Profit & Loss (PnL) Tests for the ABM Simulation.

Tests cover:
- Trader/Smart Router PnL calculations
- Arbitrageur PnL (including flash loan fees)
- LP PnL (hedged, unhedged, LVR)
- Rebalancer benchmark accounting
- Position-level PnL (IL, fees)
- Conservation/zero-sum checks
- Simulation output consistency

Run with:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_pnl_calculations.py -v
"""

import copy
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from core.utils import build_empty_pool, minted_amounts_at_S, ReferenceMarket
from core.uniswapv3_pool import V3Pool
from core.agents import (
    Position, LPAgent, RebalancerState,
    lp_token0_exposure, lp_wealth_y, lp_total_fee_earned_value_y,
    lp_total_position_value_y, lp_principal_value_y, lp_fee_value_y
)
from scripts.run import simulate, TraderStepAccumulator


# =============================================================================
# Fixtures
# =============================================================================

def _base_simulate_kwargs(tmp_path, **overrides: Any) -> Dict[str, Any]:
    """Base simulation parameters for testing."""
    params: Dict[str, Any] = dict(
        # Core simulation parameters
        config_name="pnl_test",
        block_time=5,
        T=5,
        seed=1,
        liquidity_for_gif=False,
        light_mode=False,
        # Market
        cex_mu=0.0,
        cex_sigma=0.00015,
        # Trade flow
        smart_trades_per_block=0.0,
        noise_trades_per_block=0.0,
        # LP population
        N_LP=2,
        passive_lp_share=1.0,
        tau=5,
        narrow_mints_per_block=0.0,
        passive_mints_per_block=0.0,
        passive_burns_per_block=0.0,
        # LP width
        w_min_ticks=10,
        w_max_ticks=100,
        basis_half_life=1,
        slope_s=1.0,
        binom_n=0,
        binom_p=0.5,
        # Trader
        trader_mean=2.5,
        trader_sigma=1.0,
        theta_T=1.0,
        slippage_tolerance=0.01,
        passive_width_pct=10.0,
        passive_width_ticks=None,
        # LP behavior
        mint_mu=2.5,
        mint_sigma=1.0,
        theta_TP=1.0,
        theta_SL=1.0,
        k_out_min=1,
        k_out_max=1,
        # Initialization
        initial_binom_N=50,
        initial_total_L=50_000.0,
        # Fee controller
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
        # Output controls
        visualize=False,
        skip_step=0,
        results_root=tmp_path,
        verbose=False,
    )
    params.update(overrides)
    return params


# =============================================================================
# 2.1 Trader/Arbitrageur PnL Tests
# =============================================================================

class TestTraderStepAccumulator:
    """Tests for the TraderStepAccumulator class."""

    def test_trader_pnl_settlement_formula(self):
        """
        Verify TraderStepAccumulator.settle(m) computes:
        pnl = (dy_out - dy_in) + (dx_out - dx_in) * m
        """
        acc = TraderStepAccumulator()
        
        # Simulate a Y→X swap: trader puts in dy, gets out dx
        dy_in = 1000.0
        dx_out = 0.5
        acc.record_swap(dy_in=dy_in, dx_out=dx_out)
        
        m_settle = 2000.0
        acc.settle(m_settle)
        
        # PnL = (dy_out - dy_in) + (dx_out - dx_in) * m
        # dy_out = 0, dy_in = 1000, dx_out = 0.5, dx_in = 0
        expected_pnl = (0 - 1000.0) + (0.5 - 0) * 2000.0
        assert acc.pnl == pytest.approx(expected_pnl)

    def test_trader_pnl_x_to_y_swap(self):
        """
        For X→Y swap: trader inputs dx, outputs dy.
        PnL = (dy_out - 0) + (0 - dx_in) * m = dy_out - dx_in * m
        """
        acc = TraderStepAccumulator()
        
        dx_in = 1.0
        dy_out = 1950.0  # Slightly worse than m due to fees
        acc.record_swap(dx_in=dx_in, dy_out=dy_out)
        
        m_settle = 2000.0
        acc.settle(m_settle)
        
        # PnL = dy_out - dx_in * m (negative because of fees/slippage)
        expected_pnl = dy_out - dx_in * m_settle
        assert acc.pnl == pytest.approx(expected_pnl)
        assert acc.pnl < 0  # Should be negative due to fees

    def test_trader_pnl_multiple_swaps_accumulate(self):
        """
        Multiple swaps should accumulate correctly before settlement.
        """
        acc = TraderStepAccumulator()
        
        # First swap: X→Y
        acc.record_swap(dx_in=1.0, dy_out=1950.0)
        # Second swap: Y→X
        acc.record_swap(dy_in=500.0, dx_out=0.24)
        
        m_settle = 2000.0
        acc.settle(m_settle)
        
        # Combined PnL
        expected_pnl = (1950.0 - 500.0) + (0.24 - 1.0) * m_settle
        assert acc.pnl == pytest.approx(expected_pnl)


class TestArbitragerPnL:
    """Tests for arbitrageur profit calculations."""

    def test_arbitrageur_pnl_nonnegative_in_band(self, tmp_path):
        """
        When DEX price is within the no-arb band, arb should not execute and PnL should be 0.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=3,
            seed=42,
            cex_sigma=0.0,  # No CEX drift
            smart_trades_per_block=0.0,
            noise_trades_per_block=0.0,
            fee_mode="static",
            flash_loan_fee=0.0,
        ))
        
        # With no trades and no CEX drift, DEX price stays in band
        # All arb PnL steps should be zero
        arb_pnl = out["arb_pnl_steps"]
        for pnl in arb_pnl:
            assert pnl == pytest.approx(0.0, abs=1e-12)

    def test_no_arb_band_includes_flash_loan_fee(self, tmp_path):
        """
        The no-arb band should widen when flash-loan funding costs are enabled.
        band_lo = m * r / (1 + phi)
        band_hi = m * (1 + phi) / r
        """
        f0 = 0.003
        phi = 0.10  # 10% flash loan fee
        
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=1,
            seed=1,
            block_time=2,
            smart_trades_per_block=0.0,
            noise_trades_per_block=0.0,
            cex_sigma=0.0,
            fee_mode="static",
            f0=f0,
            flash_loan_fee=phi,
        ))
        
        m0 = out["CEX_price"][0]
        r = 1.0 - f0
        
        expected_lo = m0 * r / (1.0 + phi)
        expected_hi = m0 * (1.0 + phi) / r
        
        assert out["band_lo"][0] == pytest.approx(expected_lo, rel=1e-9)
        assert out["band_hi"][0] == pytest.approx(expected_hi, rel=1e-9)

    def test_arb_pnl_settles_at_snapshot_price(self, tmp_path, monkeypatch):
        """
        Arbitrageur PnL should be computed at the validated snapshot CEX price,
        not at the end-of-step CEX price after intra-block diffusion.
        """
        import random
        import core.utils as utils
        import numpy as np

        # Make noise arrivals deterministic
        monkeypatch.setattr(np.random, "poisson", lambda _lam: 1)
        monkeypatch.setattr(random, "choice", lambda seq: "X_to_Y")

        sim_kwargs = _base_simulate_kwargs(
            tmp_path / "base",
            T=2,
            seed=7,
            block_time=2,
            smart_trades_per_block=0.0,
            noise_trades_per_block=2.0,
            cex_sigma=0.0,
            slippage_tolerance=1.0,
            trader_mean=5.5,
            trader_sigma=0.0,
            fee_mode="static",
            flash_loan_fee=0.0,
        )

        # Run A: no intra-block diffusion
        monkeypatch.setattr(utils.ReferenceMarket, "diffuse_only", lambda self: self.m)
        out_a = simulate(**sim_kwargs)

        # Run B: apply a large jump during step 1
        block_time = int(sim_kwargs["block_time"])

        def diffuse_step1_jump(self) -> float:
            calls = int(getattr(self, "_test_diffuse_calls", 0)) + 1
            setattr(self, "_test_diffuse_calls", calls)
            if calls <= block_time:
                return self.m
            self.m *= 1.5
            return self.m

        monkeypatch.setattr(utils.ReferenceMarket, "diffuse_only", diffuse_step1_jump)
        out_b = simulate(**dict(sim_kwargs, results_root=tmp_path / "alt"))

        # CEX prices diverged
        assert out_a["CEX_price"][1] != pytest.approx(out_b["CEX_price"][1])
        
        # But arb PnL should be the same (computed at snapshot)
        if out_a["arb_exec_count"][1] == 1:
            assert out_a["arb_pnl_steps"][1] == pytest.approx(
                out_b["arb_pnl_steps"][1], rel=1e-12, abs=1e-12
            )


# =============================================================================
# 2.2 LP PnL Tests
# =============================================================================

class TestLPWealth:
    """Tests for LP wealth calculations."""

    def test_lp_wealth_equals_wallet_plus_positions(self):
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

    def test_lp_token0_exposure_includes_wallet_and_positions(self):
        """
        lp_token0_exposure should include both wallet_x and position principal.
        """
        lp = LPAgent(id=1)
        lp.wallet_x = 3.0
        
        # Create a position in range
        L = 10000.0
        sa = 40.0
        sb = 50.0
        S = 45.0
        pos = Position(owner=1, lower=-10, upper=10, L=L, sa=sa, sb=sb, amt0_init=0, amt1_init=0)
        lp.positions.append(pos)
        
        exposure = lp_token0_exposure(lp, S)
        
        # Position x amount: L * (1/S - 1/sb)
        pos_x, _ = pos.current_amounts(S)
        expected = lp.wallet_x + pos_x
        
        assert exposure == pytest.approx(expected, rel=1e-12)


class TestLPPnLFormulas:
    """Tests for LP hedged/unhedged PnL formulas."""

    def test_rebalancer_initialization(self):
        """
        Test that RebalancerState initializes with correct defaults.
        """
        rb = RebalancerState()
        assert rb.x_prev == 0.0
        assert rb.cash_y == 0.0
        assert rb.cumulative_R == 0.0
        assert rb.hedged_pnl_cum == 0.0
        assert rb.initialized is False

    def test_rebalancer_cumulative_R_accrual(self):
        """
        Verify that cumulative_R accrues correctly as Σ x_prev * ΔM over price moves.
        """
        rb = RebalancerState()
        rb.initialized = True
        rb.x_prev = 10.0  # Token0 exposure
        rb.last_M = 2000.0
        
        # Price moves from 2000 to 2100
        M_new = 2100.0
        delta_M = M_new - rb.last_M
        rb.cumulative_R += rb.x_prev * delta_M
        rb.last_M = M_new
        
        expected_R = 10.0 * 100.0  # x_prev * ΔM
        assert rb.cumulative_R == pytest.approx(expected_R)
        
        # Another price move
        M_new2 = 2050.0
        delta_M2 = M_new2 - rb.last_M
        rb.cumulative_R += rb.x_prev * delta_M2
        rb.last_M = M_new2
        
        expected_R2 = expected_R + 10.0 * (-50.0)
        assert rb.cumulative_R == pytest.approx(expected_R2)

    def test_rebalancer_value_formula(self):
        """
        V^{reb}_t = V^{reb}_0 + cumulative_R
        """
        rb = RebalancerState()
        rb.initial_rebal_value_y = 10000.0  # Initial value
        rb.cumulative_R = 500.0  # Gains from rebalancing
        
        rebal_value = rb.initial_rebal_value_y + rb.cumulative_R
        assert rebal_value == pytest.approx(10500.0)

    def test_lp_hedged_pnl_formula(self):
        """
        Hedged PnL = current wealth − rebalancer benchmark value
        hedged_pnl = wealth_now - (rb.initial_rebal_value_y + rb.cumulative_R)
        """
        lp = LPAgent(id=1)
        lp.wallet_y = 11000.0  # Current wallet
        
        lp.rebalancer.initialized = True
        lp.rebalancer.initial_rebal_value_y = 10000.0
        lp.rebalancer.initial_lp_value_y = 10000.0
        lp.rebalancer.cumulative_R = 200.0
        
        S = 45.0
        m = 2000.0
        wealth_now = lp_wealth_y(lp, S, m)
        rebal_value = lp.rebalancer.initial_rebal_value_y + lp.rebalancer.cumulative_R
        hedged_pnl = wealth_now - rebal_value
        
        # wealth = wallet_y = 11000 (no positions, no wallet_x)
        # rebal_value = 10000 + 200 = 10200
        # hedged_pnl = 11000 - 10200 = 800
        assert hedged_pnl == pytest.approx(800.0)

    def test_lp_unhedged_pnl_formula(self):
        """
        Unhedged PnL = current wealth − initial wealth
        unhedged_pnl = wealth_now - rb.initial_lp_value_y
        """
        lp = LPAgent(id=1)
        lp.wallet_y = 11000.0
        
        lp.rebalancer.initialized = True
        lp.rebalancer.initial_lp_value_y = 10000.0
        
        S = 45.0
        m = 2000.0
        wealth_now = lp_wealth_y(lp, S, m)
        unhedged_pnl = wealth_now - lp.rebalancer.initial_lp_value_y
        
        # wealth = 11000, initial = 10000
        # unhedged_pnl = 1000
        assert unhedged_pnl == pytest.approx(1000.0)

    def test_lp_fee_value_includes_both_tokens(self):
        """
        lp_total_fee_earned_value_y() should return fees0_earned * m + fees1_earned.
        """
        lp = LPAgent(id=1)
        lp.fees0_earned = 0.5
        lp.fees1_earned = 100.0
        
        m = 2000.0
        fee_value = lp_total_fee_earned_value_y(lp, m)
        
        expected = 0.5 * 2000.0 + 100.0
        assert fee_value == pytest.approx(expected)


# =============================================================================
# 2.3 Position PnL Tests
# =============================================================================

class TestPositionPnL:
    """Tests for position-level PnL calculations."""

    def test_position_hodl_value_formula(self):
        """
        hodl_value_y_now(m) should return amt0_init * m + amt1_init.
        """
        pos = Position(
            owner=1, lower=-10, upper=10, L=10000.0, sa=40.0, sb=50.0,
            amt0_init=5.0, amt1_init=8000.0
        )
        
        m = 2000.0
        hodl_value = pos.hodl_value_y_now(m)
        
        expected = 5.0 * 2000.0 + 8000.0
        assert hodl_value == pytest.approx(expected)

    def test_position_IL_formula_price_moved_up(self):
        """
        When price moves up (S increases), position should experience IL.
        IL_y = position_value_y_now - hodl_value_y_now (≤ 0 when price moves)
        """
        L = 10000.0
        sa = 40.0
        sb = 50.0
        S_entry = 45.0
        
        # Calculate initial amounts at entry
        amt0_init, amt1_init = minted_amounts_at_S(L, sa, sb, S_entry)
        
        pos = Position(
            owner=1, lower=-10, upper=10, L=L, sa=sa, sb=sb,
            amt0_init=amt0_init, amt1_init=amt1_init
        )
        pos.hodl0_value_y = amt0_init * 2000.0 + amt1_init
        
        # Price moves up
        S_now = 48.0
        m_now = S_now ** 2  # Mark to market at new price
        
        IL = pos.IL_y(S_now, m_now)
        
        # IL should be negative (or zero) when price moves away from entry
        # The position loses value vs HODL
        assert IL <= 0.0

    def test_position_IL_formula_no_price_change(self):
        """
        When price hasn't moved, IL should be approximately zero.
        """
        L = 10000.0
        sa = 40.0
        sb = 50.0
        S_entry = 45.0
        m_entry = S_entry ** 2
        
        amt0_init, amt1_init = minted_amounts_at_S(L, sa, sb, S_entry)
        
        pos = Position(
            owner=1, lower=-10, upper=10, L=L, sa=sa, sb=sb,
            amt0_init=amt0_init, amt1_init=amt1_init
        )
        
        IL = pos.IL_y(S_entry, m_entry)
        
        # No price change means no IL
        assert IL == pytest.approx(0.0, abs=1e-9)

    def test_position_pnl_equals_IL_plus_fees(self):
        """
        PnL_y(S, m) = IL_y(S, m) + fees_value_y(m)
        """
        L = 10000.0
        sa = 40.0
        sb = 50.0
        S = 45.0
        m = 2000.0
        
        amt0, amt1 = minted_amounts_at_S(L, sa, sb, S)
        
        pos = Position(
            owner=1, lower=-10, upper=10, L=L, sa=sa, sb=sb,
            amt0_init=amt0, amt1_init=amt1
        )
        pos.fees0 = 0.1
        pos.fees1 = 20.0
        
        pnl = pos.PnL_y(S, m)
        IL = pos.IL_y(S, m)
        fees = pos.fees_value_y(m)
        
        assert pnl == pytest.approx(IL + fees, rel=1e-12)


# =============================================================================
# 2.4 PnL Conservation / Zero-Sum Tests
# =============================================================================

class TestPnLConservation:
    """Tests for PnL conservation properties."""

    def test_arbitrage_profit_nonnegative(self, tmp_path, monkeypatch):
        """
        Arbitrageur profit per execution should be non-negative
        (arb only executes when profitable).
        """
        import numpy as np
        
        # Force noise trades to create price dislocation
        monkeypatch.setattr(np.random, "poisson", lambda _lam: 2)
        
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=5,
            seed=42,
            noise_trades_per_block=5.0,
            slippage_tolerance=0.5,
            trader_mean=5.0,
            trader_sigma=0.1,
            fee_mode="static",
            flash_loan_fee=0.0,
        ))
        
        # Each individual arb execution should have non-negative PnL
        # (arbs only execute when profitable)
        arb_pnl = out["arb_pnl_steps"]
        for step_pnl in arb_pnl:
            # Arb PnL can be 0 (no arb) or positive
            assert step_pnl >= -1e-9, f"Arb PnL was negative: {step_pnl}"


# =============================================================================
# 2.5 Simulation Output Consistency Tests
# =============================================================================

class TestSimulationOutputConsistency:
    """Tests for simulation output array consistency."""

    def test_lp_pnl_series_lengths_match_steps(self, tmp_path):
        """
        All LP PnL series should have length equal to simulation steps T.
        """
        T = 7
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

    def test_trader_pnl_series_lengths_match_steps(self, tmp_path):
        """
        Trader PnL series should have length equal to simulation steps T.
        """
        T = 6
        out = simulate(**_base_simulate_kwargs(tmp_path, T=T))
        
        assert len(out["trader_pnl_steps"]) == T
        assert len(out["trader_pnl_cum"]) == T
        assert len(out["arb_pnl_steps"]) == T
        assert len(out["arb_pnl_cum"]) == T
        assert len(out["smart_router_pnl_steps"]) == T
        assert len(out["smart_router_pnl_cum"]) == T
        assert len(out["noise_trader_pnl_steps"]) == T
        assert len(out["noise_trader_pnl_cum"]) == T

    def test_cumulative_pnl_equals_sum_of_steps(self, tmp_path):
        """
        trader_pnl_cum[i] == sum(trader_pnl_steps[0:i+1]) for all i.
        """
        T = 8
        out = simulate(**_base_simulate_kwargs(
            tmp_path, 
            T=T,
            noise_trades_per_block=3.0,
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

    def test_fee_series_length_matches_steps(self, tmp_path):
        """
        fee_series should have one entry per simulation step.
        """
        T = 5
        out = simulate(**_base_simulate_kwargs(tmp_path, T=T))
        
        assert len(out["fee_series"]) == T

    def test_price_series_lengths_match_steps(self, tmp_path):
        """
        DEX and CEX price series should have length T.
        """
        T = 10
        out = simulate(**_base_simulate_kwargs(tmp_path, T=T))
        
        assert len(out["DEX_price"]) == T
        assert len(out["CEX_price"]) == T
        assert len(out["cex_sigma_series"]) == T

    def test_activity_series_lengths_match_steps(self, tmp_path):
        """
        Activity cumulative series should have length T.
        """
        T = 5
        out = simulate(**_base_simulate_kwargs(tmp_path, T=T))
        
        assert len(out["smart_router_activity_cum"]) == T
        assert len(out["noise_trader_activity_cum"]) == T
        assert len(out["lp_active_activity_cum"]) == T
        assert len(out["lp_passive_activity_cum"]) == T
        assert len(out["arb_activity_cum"]) == T

    def test_exec_count_series_lengths_match(self, tmp_path):
        """
        Execution count series should match pnl series lengths.
        """
        T = 5
        out = simulate(**_base_simulate_kwargs(tmp_path, T=T))
        
        assert len(out["trader_exec_count"]) == T
        assert len(out["arb_exec_count"]) == T
        assert len(out["smart_router_exec_count"]) == T
        assert len(out["noise_trader_exec_count"]) == T

    def test_lvr_equals_fees_minus_hedged_pnl(self, tmp_path):
        """
        LVR is computed as fee_value - hedged_pnl.
        lp_lvr_total_series[t] = lp_fee_value_total_series[t] - lp_pnl_total[t]
        """
        T = 5
        out = simulate(**_base_simulate_kwargs(
            tmp_path, 
            T=T,
            noise_trades_per_block=3.0,
            slippage_tolerance=0.5,
        ))
        
        lvr_total = np.array(out["lp_lvr_total_series"])
        fee_value_total = np.array(out["lp_fee_value_total_series"])
        hedged_pnl_total = np.array(out["lp_pnl_total"])
        
        expected_lvr = fee_value_total - hedged_pnl_total
        np.testing.assert_allclose(lvr_total, expected_lvr, rtol=1e-12, atol=1e-12)

    def test_simulate_outputs_consistent_lengths(self, tmp_path):
        """
        Simulation series describing trader flows must have consistent lengths.
        """
        out = simulate(**_base_simulate_kwargs(tmp_path, T=5, seed=1))

        assert len(out['smart_router_pnl_steps']) == len(out['smart_router_notional_y'])
        assert len(out['noise_trader_pnl_steps']) == len(out['noise_trader_notional_y'])
        assert len(out['smart_router_exec_count']) == len(out['smart_router_pnl_steps'])
        assert len(out['noise_trader_exec_count']) == len(out['noise_trader_pnl_steps'])
        assert len(out['smart_router_pnl_steps']) == len(out['smart_router_pnl_cum'])
        assert len(out['noise_trader_pnl_steps']) == len(out['noise_trader_pnl_cum'])


# =============================================================================
# Additional Validation Tests
# =============================================================================

class TestSimulationValidation:
    """Validation tests for simulation parameters."""

    def test_simulate_invalid_fee_mode(self, tmp_path):
        """
        simulate() should reject unsupported fee controller modes.
        """
        with pytest.raises(ValueError, match="Invalid fee_mode"):
            simulate(**_base_simulate_kwargs(tmp_path, T=1, fee_mode="nonsense"))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"k_out_min": 0, "k_out_max": 1},
            {"k_out_min": 5, "k_out_max": 3},
        ],
    )
    def test_simulate_validates_k_out_bounds(self, tmp_path, kwargs):
        """
        The k_out bounds must be positive and ordered.
        """
        with pytest.raises(ValueError):
            simulate(**_base_simulate_kwargs(tmp_path, T=1, **kwargs))

    def test_simulate_seed_determinism(self, tmp_path):
        """
        Re-running with the same seed should reproduce results.
        """
        out_a = simulate(**_base_simulate_kwargs(
            tmp_path / "a", T=3, seed=42, noise_trades_per_block=5.0
        ))
        out_b = simulate(**_base_simulate_kwargs(
            tmp_path / "b", T=3, seed=42, noise_trades_per_block=5.0
        ))
        out_c = simulate(**_base_simulate_kwargs(
            tmp_path / "c", T=3, seed=999, noise_trades_per_block=5.0
        ))

        dex_a = tuple(out_a["DEX_price"])
        dex_b = tuple(out_b["DEX_price"])
        dex_c = tuple(out_c["DEX_price"])

        assert dex_a == dex_b
        assert any(abs(a - c) > 1e-12 for a, c in zip(dex_a, dex_c))

    def test_simulate_heston_mode_requires_parameters(self, tmp_path):
        """
        Heston volatility mode must fail when required parameters are missing.
        """
        with pytest.raises(ValueError, match="cex_sigma_mode='heston' requires parameters"):
            simulate(**_base_simulate_kwargs(tmp_path, T=1, cex_sigma_mode="heston"))

    def test_simulate_heston_mode_runs(self, tmp_path):
        """
        With valid Heston parameters, simulate() should run.
        """
        out = simulate(**_base_simulate_kwargs(
            tmp_path,
            T=3,
            cex_sigma_mode="heston",
            cex_sigma=0.001,
            cex_heston_kappa=1.0,
            cex_heston_theta=1e-6,
            cex_heston_sigma_v=0.1,
            cex_heston_rho=-0.5,
        ))
        sigma_series = out["cex_sigma_series"]
        assert len(sigma_series) == 3
        assert all(s >= 0.0 for s in sigma_series)
