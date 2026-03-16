from __future__ import annotations

import numpy as np
import pytest

from scripts.LVR_vs_blocksize import _store_run_payload_inplace


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
