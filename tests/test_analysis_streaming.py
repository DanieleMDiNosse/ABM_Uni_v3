from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from scripts.analysis.common import _write_volatility_artifact
from scripts.analysis.volatility_conditioned import (
    volatility_binned_analysis,
    volatility_binned_analysis_from_artifacts,
)
from scripts.run import simulate


def _base_simulate_kwargs(results_root: Path, **overrides: Any) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(
        config_name="analysis_streaming_test",
        block_time=5,
        T=8,
        seed=7,
        liquidity_for_gif=False,
        light_mode=False,
        cex_mu=0.0,
        cex_sigma=0.00015,
        smart_trades_per_block=1.5,
        noise_trades_per_block=1.0,
        N_LP=4,
        passive_lp_share=0.5,
        tau=5,
        narrow_mints_per_block=0.0,
        passive_mints_per_block=0.0,
        passive_burns_per_block=0.0,
        w_min_ticks=10,
        w_max_ticks=100,
        basis_half_life=2,
        slope_s=1.0,
        binom_n=0,
        binom_p=0.5,
        trader_mean=2.5,
        trader_sigma=1.0,
        theta_T=1.0,
        slippage_tolerance=0.05,
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
        fee_mode="volatility_cex",
        f0=0.003,
        f_min=0.0001,
        f_max=0.01,
        fee_half_life=2,
        k_sigma=1.0,
        k_basis=0.0,
        fee_step_bps_min=0.0,
        fee_step_bps_max=200.0,
        visualize=False,
        skip_step=0,
        results_root=results_root,
        verbose=False,
    )
    params.update(overrides)
    return params


def test_simulate_record_keys_matches_full_outputs(tmp_path: Path) -> None:
    full = simulate(**_base_simulate_kwargs(tmp_path / "full"))
    reduced = simulate(**_base_simulate_kwargs(
        tmp_path / "reduced",
        record_keys=[
            "CEX_price",
            "DEX_price",
            "lp_pnl_active_final",
            "lp_pnl_passive_final",
            "jiter_pnl_final",
            "noise_trader_pnl_cum_final",
            "smart_router_pnl_cum_final",
            "lp_fee_value_active_final",
            "lp_fee_value_passive_final",
            "lp_fee_value_total_final",
            "lp_fees0_earned_active_series",
            "lp_fees1_earned_active_series",
            "lp_fees0_earned_passive_series",
            "lp_fees1_earned_passive_series",
            "fee_mean",
            "smart_router_dex_share_mean",
            "cex_sigma_series",
            "jiter_activity_cum",
            "jiter_fee_value_series",
            "jiter_fees0_earned_series",
            "jiter_fees1_earned_series",
            "jiter_flash_fee_paid_series",
            "lp_fee_value_active_series",
            "lp_fee_value_passive_series",
            "lp_lvr_active_series",
            "lp_lvr_passive_series",
        ],
    ))

    np.testing.assert_allclose(reduced["CEX_price"], full["CEX_price"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reduced["DEX_price"], full["DEX_price"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reduced["cex_sigma_series"], full["cex_sigma_series"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        reduced["lp_fees0_earned_active_series"],
        full["lp_fees0_earned_active_series"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reduced["lp_fees1_earned_active_series"],
        full["lp_fees1_earned_active_series"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reduced["lp_fees0_earned_passive_series"],
        full["lp_fees0_earned_passive_series"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reduced["lp_fees1_earned_passive_series"],
        full["lp_fees1_earned_passive_series"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reduced["jiter_activity_cum"],
        full["jiter_activity_cum"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reduced["jiter_fee_value_series"],
        full["jiter_fee_value_series"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reduced["jiter_fees0_earned_series"],
        full["jiter_fees0_earned_series"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reduced["jiter_fees1_earned_series"],
        full["jiter_fees1_earned_series"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reduced["jiter_flash_fee_paid_series"],
        full["jiter_flash_fee_paid_series"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reduced["lp_fee_value_active_series"],
        full["lp_fee_value_active_series"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reduced["lp_fee_value_passive_series"],
        full["lp_fee_value_passive_series"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(reduced["lp_lvr_active_series"], full["lp_lvr_active_series"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reduced["lp_lvr_passive_series"], full["lp_lvr_passive_series"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reduced["lp_pnl_active_final"], full["lp_pnl_active"][-1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reduced["lp_pnl_passive_final"], full["lp_pnl_passive"][-1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(reduced["jiter_pnl_final"], full["jiter_pnl_series"][-1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        reduced["noise_trader_pnl_cum_final"],
        full["noise_trader_pnl_cum"][-1],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        reduced["smart_router_pnl_cum_final"],
        full["smart_router_pnl_cum"][-1],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        reduced["lp_fee_value_active_final"],
        full["lp_fee_value_active_series"][-1],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reduced["lp_fee_value_passive_final"],
        full["lp_fee_value_passive_series"][-1],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reduced["lp_fee_value_total_final"],
        full["lp_fee_value_total_series"][-1],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(reduced["fee_mean"], np.mean(full["fee_series"]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        reduced["smart_router_dex_share_mean"],
        full["smart_router_dex_share_mean"],
        rtol=0.0,
        atol=0.0,
    )


def test_volatility_artifact_analysis_matches_in_memory(tmp_path: Path) -> None:
    results = [
        {
            "cex_sigma_series": [0.1, 0.2, 0.3, 0.4, 0.5],
            "lp_fee_value_passive_series": [0.0, 1.0, 3.0, 6.0, 10.0],
            "lp_lvr_passive_series": [0.0, 0.4, 1.2, 2.4, 4.0],
            "lp_fee_value_active_series": [0.0, 0.8, 1.6, 2.8, 4.4],
            "lp_lvr_active_series": [0.0, 0.3, 0.9, 1.8, 3.0],
        },
        {
            "cex_sigma_series": [0.15, 0.25, 0.35, 0.45, 0.55],
            "lp_fee_value_passive_series": [0.0, 0.9, 2.1, 3.6, 5.4],
            "lp_lvr_passive_series": [0.0, 0.45, 1.05, 1.8, 2.7],
            "lp_fee_value_active_series": [0.0, 0.7, 1.5, 2.6, 4.0],
            "lp_lvr_active_series": [0.0, 0.28, 0.84, 1.56, 2.4],
        },
    ]

    artifact_paths = []
    for idx, result in enumerate(results):
        artifact_path = tmp_path / f"seed_{idx}.npz"
        _write_volatility_artifact(result, artifact_path=artifact_path, skip=1)
        artifact_paths.append(artifact_path)

    figure_memory = volatility_binned_analysis(results, n_bins=3, skip=1, cohort="passive")
    figure_disk = volatility_binned_analysis_from_artifacts(artifact_paths, n_bins=3, cohort="passive")

    np.testing.assert_allclose(figure_memory.data[0].y, figure_disk.data[0].y, rtol=0.0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        figure_memory.data[0].error_y.array,
        figure_disk.data[0].error_y.array,
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    assert list(figure_memory.data[0].x) == list(figure_disk.data[0].x)
