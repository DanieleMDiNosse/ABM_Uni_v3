from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from core.utils import load_simulation_parameters
from scripts.LVR_vs_blocksize import (
    _extract_filtered_metric_from_payload,
    _required_record_keys_for_lvr,
    _run_blocksize_sweep_for_fee_mode,
    _store_run_payload_inplace,
)


def test_store_run_payload_inplace_merges_results_by_grid_slot() -> None:
    block_idx = {2: 0, 4: 1}
    seed_idx = {11: 0, 13: 1}
    cohort_names = ["active", "jiter"]
    expected_delta_len = 3

    delta_arrays = {
        name: np.full((2, 2, expected_delta_len), np.nan, dtype=float) for name in cohort_names
    }
    ratio_arrays = {
        name: np.full((2, 2, expected_delta_len), np.nan, dtype=float) for name in cohort_names
    }
    jit_success = np.zeros((2, 2, expected_delta_len), dtype=np.uint8)

    payload_late = {
        "block_time": 4,
        "seed": 13,
        "dLVR_active": [1.0, 2.0, 3.0, 4.0],
        "dLVR_over_dFees_active": [5.0, 6.0],
        "dLVR_jiter": [7.0],
        "jit_success_mask": [True, False, True, True],
    }
    payload_early = {
        "block_time": 2,
        "seed": 11,
        "dLVR_active": [10.0, 20.0],
        "dLVR_over_dFees_active": [30.0, 40.0, 50.0, 60.0],
        "dLVR_jiter": [70.0, 80.0, 90.0],
        "dLVR_over_dFees_jiter": [0.5],
        "jit_success_mask": [False, True],
    }

    # Completion order should not matter because each payload owns one grid slot.
    _store_run_payload_inplace(
        payload_late,
        block_idx=block_idx,
        seed_idx=seed_idx,
        cohort_names=cohort_names,
        expected_delta_len=expected_delta_len,
        delta_arrays=delta_arrays,
        ratio_arrays=ratio_arrays,
        jit_success_arrays=jit_success,
    )
    _store_run_payload_inplace(
        payload_early,
        block_idx=block_idx,
        seed_idx=seed_idx,
        cohort_names=cohort_names,
        expected_delta_len=expected_delta_len,
        delta_arrays=delta_arrays,
        ratio_arrays=ratio_arrays,
        jit_success_arrays=jit_success,
    )

    np.testing.assert_allclose(delta_arrays["active"][1, 1, :], np.array([1.0, 2.0, 3.0]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        ratio_arrays["active"][1, 1, :],
        np.array([5.0, 6.0, np.nan]),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        delta_arrays["jiter"][1, 1, :],
        np.array([7.0, np.nan, np.nan]),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_allclose(jit_success[1, 1, :], np.array([1, 0, 1], dtype=np.uint8), rtol=0.0, atol=0.0)

    np.testing.assert_allclose(
        delta_arrays["active"][0, 0, :],
        np.array([10.0, 20.0, np.nan]),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        ratio_arrays["active"][0, 0, :],
        np.array([30.0, 40.0, 50.0]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(delta_arrays["jiter"][0, 0, :], np.array([70.0, 80.0, 90.0]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        ratio_arrays["jiter"][0, 0, :],
        np.array([0.5, np.nan, np.nan]),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_allclose(jit_success[0, 0, :], np.array([0, 1, 0], dtype=np.uint8), rtol=0.0, atol=0.0)


def test_store_run_payload_inplace_rejects_unknown_grid_slot() -> None:
    delta_arrays = {"active": np.full((1, 1, 2), np.nan, dtype=float)}
    ratio_arrays = {"active": np.full((1, 1, 2), np.nan, dtype=float)}

    with pytest.raises(KeyError, match="unexpected"):
        _store_run_payload_inplace(
            {
                "block_time": 99,
                "seed": 7,
                "dLVR_active": [1.0, 2.0],
            },
            block_idx={2: 0},
            seed_idx={3: 0},
            cohort_names=["active"],
            expected_delta_len=2,
            delta_arrays=delta_arrays,
            ratio_arrays=ratio_arrays,
            jit_success_arrays=None,
        )


def test_required_record_keys_for_lvr_matches_fee_definition() -> None:
    flow_keys = set(
        _required_record_keys_for_lvr(
            cohort_names=["active", "passive", "jiter"],
            fee_definition="flow",
        )
    )
    assert "CEX_price" in flow_keys
    assert "lp_fees0_earned_active_series" in flow_keys
    assert "lp_fees1_earned_passive_series" in flow_keys
    assert "jiter_fees0_earned_series" in flow_keys
    assert "jiter_flash_fee_paid_series" in flow_keys
    assert "lp_lvr_active_series" not in flow_keys

    mtm_keys = set(
        _required_record_keys_for_lvr(
            cohort_names=["active", "passive", "jiter"],
            fee_definition="mtm",
        )
    )
    assert "CEX_price" not in mtm_keys
    assert "lp_lvr_active_series" in mtm_keys
    assert "lp_fee_value_passive_series" in mtm_keys
    assert "jiter_fee_value_series" in mtm_keys
    assert "jiter_fees0_earned_series" not in mtm_keys


def test_extract_filtered_metric_from_payload_masks_jiter_series() -> None:
    payload = {
        "dLVR_jiter": [1.0, np.nan, 2.0, 3.0],
        "jit_success_mask": [False, True, True, False],
    }

    filtered = _extract_filtered_metric_from_payload(
        payload,
        cohort="jiter",
        metric_key="dLVR_jiter",
    )

    np.testing.assert_allclose(filtered, np.array([2.0]), rtol=0.0, atol=0.0)


def _write_lvr_test_config(config_path: Path) -> None:
    config = {
        "fee_mode": "static",
        "simulate": {
            "config_name": "lvr_summary_only_test",
            "block_time": 5,
            "T": 8,
            "seed": 7,
            "liquidity_for_gif": False,
            "light_mode": False,
            "cex_mu": 0.0,
            "cex_sigma": 0.00015,
            "N_LP": 4,
            "passive_lp_share": 0.5,
            "w_min_ticks": 10,
            "w_max_ticks": 100,
            "basis_half_life": 2,
            "slope_s": 1.0,
            "binom_n": 0,
            "binom_p": 0.5,
            "trader_mean": 2.5,
            "trader_sigma": 1.0,
            "theta_T": 1.0,
            "slippage_tolerance": 0.05,
            "passive_width_pct": 10.0,
            "passive_width_ticks": None,
            "mint_mu": 2.5,
            "mint_sigma": 1.0,
            "theta_TP": 1.0,
            "theta_SL": 1.0,
            "k_out_min": 1,
            "k_out_max": 1,
            "initial_binom_N": 50,
            "initial_total_L": 50_000.0,
            "fee_mode": "static",
            "f0": 0.003,
            "f_min": 0.0001,
            "f_max": 0.01,
            "fee_half_life": 2,
            "k_sigma": 1.0,
            "k_basis": 0.0,
            "fee_step_bps_min": 0.0,
            "fee_step_bps_max": 200.0,
            "fee_cooldown": 0,
            "smart_trades_per_block": 1.5,
            "noise_trades_per_block": 1.0,
            "narrow_mints_per_block": 0.0,
            "passive_mints_per_block": 0.0,
            "passive_burns_per_block": 0.0,
            "tau": 5,
            "flash_loan_fee": 0.0,
            "jit_flash_loan_fee": 0.0,
            "p_jit": 1.0,
            "N_jit": 1,
            "liquidity_perc_jit": 0.5,
            "cex_sigma_mode": "static",
            "visualize": False,
            "skip_step": 0,
            "n_block_SR_ratio": 1,
            "verbose": False,
            "results_root": str(config_path.parent / "results"),
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_summary_only_matches_full_archive_summaries(tmp_path: Path) -> None:
    config_path = tmp_path / "lvr_summary_only_test.yml"
    _write_lvr_test_config(config_path)
    _, base_params = load_simulation_parameters(config_path)
    base_params = dict(base_params)

    common_args = dict(
        seed_base=None,
        block_min=2,
        block_max=3,
        seed_step=1,
        runs=2,
        max_workers=1,
        keep_run_artifacts=False,
        skip_png=True,
    )
    args_full = argparse.Namespace(summary_only=False, **common_args)
    args_summary = argparse.Namespace(summary_only=True, **common_args)

    full_out = _run_blocksize_sweep_for_fee_mode(
        args=args_full,
        config_path=config_path,
        base_params=base_params,
        fee_mode_label="static",
        fee_def_label="flow",
        yaxis_type="linear",
        yaxis_suffix="",
        plot_violin=False,
        plot_medians=True,
        plot_means=False,
        plot_95_interval=False,
    )
    summary_out = _run_blocksize_sweep_for_fee_mode(
        args=args_summary,
        config_path=config_path,
        base_params=base_params,
        fee_mode_label="static",
        fee_def_label="flow",
        yaxis_type="linear",
        yaxis_suffix="",
        plot_violin=False,
        plot_medians=True,
        plot_means=False,
        plot_95_interval=False,
    )

    full_summary = pd.read_csv(full_out / "dLVR_summary.csv").sort_values(["cohort", "block_time"]).reset_index(drop=True)
    summary_only_summary = pd.read_csv(summary_out / "dLVR_summary.csv").sort_values(["cohort", "block_time"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(full_summary, summary_only_summary, check_dtype=False)

    full_ratio = pd.read_csv(full_out / "dLVR_over_dFees_summary.csv").sort_values(["cohort", "block_time"]).reset_index(drop=True)
    summary_only_ratio = pd.read_csv(summary_out / "dLVR_over_dFees_summary.csv").sort_values(["cohort", "block_time"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(full_ratio, summary_only_ratio, check_dtype=False)

    assert any(full_out.glob("dLVR_arrays_*.npz"))
    assert not any(summary_out.glob("dLVR_arrays_*.npz"))
