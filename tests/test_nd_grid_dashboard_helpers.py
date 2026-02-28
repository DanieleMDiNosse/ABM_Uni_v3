"""Tests for ND grid runner/builder helper utilities."""

from __future__ import annotations

import shutil
from pathlib import Path

from scripts.build_parameter_surface_nd_pnl_fee_dashboard import _build_meta_summary
from scripts.run_parameter_surface_nd_pnl_fee_dashboard import (
    CACHE_SCHEMA_VERSION,
    SCRIPT_VERSION,
    _canon_fingerprint_payload,
    _effective_config_content_hash,
    _make_worker_run_root,
)


def test_effective_config_hash_is_stable_to_mapping_order() -> None:
    """Config-content hash should ignore dict insertion order."""
    params_a = {
        "seed": 7,
        "T": 100,
        "fee_mode": "static",
        "results_root": Path("abm_results/tmp"),
        "nested": {"b": 2, "a": 1},
    }
    params_b = {
        "results_root": Path("abm_results/tmp"),
        "nested": {"a": 1, "b": 2},
        "fee_mode": "static",
        "T": 100,
        "seed": 7,
    }

    hash_a = _effective_config_content_hash(scenario_label="static", base_params=params_a)
    hash_b = _effective_config_content_hash(scenario_label="static", base_params=params_b)

    assert hash_a == hash_b


def test_effective_config_hash_changes_with_content() -> None:
    """Changing any effective config value should change the hash."""
    params_base = {"seed": 7, "T": 100, "fee_mode": "static"}
    params_changed = {"seed": 8, "T": 100, "fee_mode": "static"}

    hash_base = _effective_config_content_hash(scenario_label="static", base_params=params_base)
    hash_changed = _effective_config_content_hash(scenario_label="static", base_params=params_changed)

    assert hash_base != hash_changed


def test_fingerprint_changes_when_config_hash_changes() -> None:
    """Fingerprint must invalidate when effective config hash changes."""
    common_kwargs = {
        "sweeps": {"k_sigma": [0.0, 1.0], "mint_sigma": [1.0, 2.0]},
        "runs_per_point": 2,
        "seed_base": 1,
        "common_seeds": False,
        "fee_hist_bins": 8,
        "smart_router_dex_share_hist_bins": 8,
        "pnl_summary": "step_rate_mean_diff",
    }

    fingerprint_a, payload_a = _canon_fingerprint_payload(
        **common_kwargs,
        config_content_hash="aaaa1111bbbb2222",
    )
    fingerprint_b, payload_b = _canon_fingerprint_payload(
        **common_kwargs,
        config_content_hash="cccc3333dddd4444",
    )

    assert fingerprint_a != fingerprint_b
    assert payload_a["config_content_hash"] == "aaaa1111bbbb2222"
    assert payload_b["config_content_hash"] == "cccc3333dddd4444"
    assert payload_a["script_version"] == SCRIPT_VERSION
    assert payload_a["cache_schema_version"] == CACHE_SCHEMA_VERSION


def test_make_worker_run_root_creates_unique_dirs(tmp_path: Path) -> None:
    """Worker run roots should be unique and nested under the worker temp root."""
    worker_root = tmp_path / "worker_tmp"

    run_a = _make_worker_run_root(
        worker_temp_root=worker_root,
        point_index=3,
        run_index=0,
        seed_value=11,
    )
    run_b = _make_worker_run_root(
        worker_temp_root=worker_root,
        point_index=3,
        run_index=0,
        seed_value=11,
    )

    try:
        assert run_a.exists()
        assert run_b.exists()
        assert run_a != run_b
        assert str(run_a).startswith(str(worker_root))
        assert str(run_b).startswith(str(worker_root))
    finally:
        shutil.rmtree(run_a, ignore_errors=True)
        shutil.rmtree(run_b, ignore_errors=True)
        shutil.rmtree(worker_root, ignore_errors=True)


def test_build_meta_summary_is_backward_compatible() -> None:
    """Metadata summary should work with both new and old metadata shapes."""
    new_meta = {
        "cache_schema_version": 2,
        "script_version": "nd_grid_runner_v2",
        "config_content_hash": "abcd1234ef567890",
    }
    summary = _build_meta_summary(new_meta)

    assert "Cache schema: 2" in summary
    assert "Runner: nd_grid_runner_v2" in summary
    assert "Config hash: abcd1234ef567890" in summary

    assert _build_meta_summary(None) == ""
    assert _build_meta_summary({}) == ""

