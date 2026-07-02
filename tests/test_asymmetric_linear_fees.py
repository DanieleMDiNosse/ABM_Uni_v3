"""Tests for Leandro-style asymmetric linear dynamic fees."""

import math
from typing import Any, Dict

from scripts.run import _linear_asymmetric_fee_targets, simulate


def _base_simulate_kwargs(tmp_path, **overrides: Any) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(
        config_name="asym_fee_test",
        block_time=2,
        T=1,
        seed=1,
        liquidity_for_gif=False,
        light_mode=False,
        cex_mu=0.0,
        cex_sigma=0.0,
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
        trader_mean=-5.0,
        trader_sigma=0.0,
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
        fee_mode="linear_asymmetric",
        f0=0.003,
        f_min=0.0001,
        f_max=0.01,
        fee_half_life=2,
        k_sigma=1.0,
        k_basis=0.0,
        fee_step_bps_min=0.0,
        fee_step_bps_max=200.0,
        asymmetric_fee_slope=0.5,
        visualize=False,
        skip_step=0,
        results_root=tmp_path,
        verbose=False,
    )
    params.update(overrides)
    return params


def test_linear_asymmetric_fee_targets_raise_x_to_y_fee_when_dex_price_is_high():
    sell_x_fee, sell_y_fee, signal = _linear_asymmetric_fee_targets(
        base_fee=0.003,
        slope=0.5,
        dex_price=110.0,
        oracle_price=100.0,
        f_min=0.0001,
        f_max=0.01,
    )

    assert signal == math.log(110.0) - math.log(100.0)
    assert sell_x_fee > 0.003
    assert sell_y_fee < 0.003
    assert sell_x_fee > sell_y_fee


def test_linear_asymmetric_fee_targets_raise_y_to_x_fee_when_dex_price_is_low():
    sell_x_fee, sell_y_fee, signal = _linear_asymmetric_fee_targets(
        base_fee=0.003,
        slope=0.5,
        dex_price=90.0,
        oracle_price=100.0,
        f_min=0.0001,
        f_max=0.01,
    )

    assert signal == math.log(90.0) - math.log(100.0)
    assert sell_y_fee > 0.003
    assert sell_x_fee < 0.003
    assert sell_y_fee > sell_x_fee


def test_linear_asymmetric_fee_targets_are_symmetric_at_zero_mispricing():
    sell_x_fee, sell_y_fee, signal = _linear_asymmetric_fee_targets(
        base_fee=0.003,
        slope=0.5,
        dex_price=100.0,
        oracle_price=100.0,
        f_min=0.0001,
        f_max=0.01,
    )

    assert signal == 0.0
    assert sell_x_fee == 0.003
    assert sell_y_fee == 0.003


def test_simulate_records_asymmetric_fee_diagnostics(tmp_path):
    out = simulate(**_base_simulate_kwargs(
        tmp_path,
        T=2,
        cex_mu=0.01,
        fee_step_bps_min=0.0,
        fee_step_bps_max=200.0,
    ))

    assert out["fee_mode"] == "linear_asymmetric"
    assert out["fee_use_ewma"] is False
    assert len(out["fee_x_to_y_series"]) == 2
    assert len(out["fee_y_to_x_series"]) == 2
    assert len(out["fee_signal_series"]) == 2
    assert out["fee_x_to_y_series"][0] == out["fee_y_to_x_series"][0]
    assert out["fee_x_to_y_series"][1] < out["fee_y_to_x_series"][1]


def test_linear_asymmetric_can_ewma_smooth_signed_gap(tmp_path):
    raw = simulate(**_base_simulate_kwargs(
        tmp_path / "raw",
        T=2,
        cex_mu=0.01,
        fee_step_bps_min=0.0,
        fee_step_bps_max=200.0,
        fee_use_ewma=False,
    ))
    smoothed = simulate(**_base_simulate_kwargs(
        tmp_path / "smoothed",
        T=2,
        cex_mu=0.01,
        fee_step_bps_min=0.0,
        fee_step_bps_max=200.0,
        fee_half_life=2,
        fee_use_ewma=True,
    ))

    assert raw["fee_use_ewma"] is False
    assert smoothed["fee_use_ewma"] is True
    assert abs(smoothed["fee_signal_series"][1]) < abs(raw["fee_signal_series"][1])
    assert smoothed["fee_y_to_x_series"][1] < raw["fee_y_to_x_series"][1]
