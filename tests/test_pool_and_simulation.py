"""Simulation arrival-process tests.

Run with:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_pool_and_simulation.py
"""

from typing import Any, Dict

import pytest

from scripts.run import simulate


def _base_simulate_kwargs(tmp_path, **overrides: Any) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(
        # Core simulation parameters
        config_name="unit_test",
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
        fee_mode="volatility_cex",
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


def test_poisson_noise_arrivals_can_exceed_block_time(tmp_path, monkeypatch):
    """
    In block mode, noise_trades_per_block is a Poisson rate, so multiple noise intents
    can arrive in the same micro-step (i.e., realized arrivals per block can exceed block_time).
    """
    import numpy as np

    def fixed_poisson(_lam):
        return 3

    monkeypatch.setattr(np.random, "poisson", fixed_poisson)

    out = simulate(**_base_simulate_kwargs(
        tmp_path,
        T=1,
        seed=123,
        block_time=5,
        smart_trades_per_block=0.0,
        noise_trades_per_block=100.0,
        slippage_tolerance=1.0,
        trader_mean=-5.0,
        trader_sigma=0.0,
    ))

    assert int(out["noise_trader_exec_count"][0]) > 5


def test_poisson_narrow_lp_mints_can_exceed_lp_count(tmp_path, monkeypatch):
    """
    In block mode, narrow_mints_per_block is a Poisson target count per block, so realized
    mint intents can exceed the number of narrow LPs (multiple mints per LP in the same block).
    """
    import numpy as np

    def fixed_poisson(_lam):
        return 7

    monkeypatch.setattr(np.random, "poisson", fixed_poisson)

    out = simulate(**_base_simulate_kwargs(
        tmp_path,
        T=1,
        seed=123,
        block_time=5,
        N_LP=1,
        passive_lp_share=0.0,  # one narrow LP
        tau=1,                # always due
        narrow_mints_per_block=100.0,
        passive_mints_per_block=0.0,
        passive_burns_per_block=0.0,
        smart_trades_per_block=0.0,
        noise_trades_per_block=0.0,
        mint_mu=-20.0,
        mint_sigma=0.0,
    ))

    assert len(out["mint_steps"]) == 7


def test_trade_arrival_lambda_legacy_per_block_is_divided_by_block_time(tmp_path, monkeypatch):
    """Legacy: *_trades_per_block is expected per block, so lambda_micro = lambda_block / block_time."""
    import numpy as np

    lams = []

    def recording_poisson(lam):
        lams.append(float(lam))
        return 0

    monkeypatch.setattr(np.random, "poisson", recording_poisson)

    block_time = 5
    lam_block = 10.0
    _ = simulate(**_base_simulate_kwargs(
        tmp_path,
        T=1,
        seed=123,
        block_time=block_time,
        smart_trades_per_block=lam_block,
        noise_trades_per_block=0.0,
        narrow_mints_per_block=0.0,
        passive_mints_per_block=0.0,
        passive_burns_per_block=0.0,
    ))

    assert len(lams) == block_time
    assert all(abs(v - (lam_block / block_time)) < 1e-12 for v in lams)


def test_trade_arrival_lambda_per_second_override(tmp_path, monkeypatch):
    """Per-second override: *_trades_per_second is the per-micro-step (1s) Poisson intensity."""
    import numpy as np

    lams = []

    def recording_poisson(lam):
        lams.append(float(lam))
        return 0

    monkeypatch.setattr(np.random, "poisson", recording_poisson)

    block_time = 5
    lam_second = 0.7
    _ = simulate(**_base_simulate_kwargs(
        tmp_path,
        T=1,
        seed=123,
        block_time=block_time,
        smart_trades_per_block=123.0,  # should be ignored when *_per_second is set
        smart_trades_per_second=lam_second,
        noise_trades_per_block=0.0,
        narrow_mints_per_block=0.0,
        passive_mints_per_block=0.0,
        passive_burns_per_block=0.0,
    ))

    assert len(lams) == block_time
    assert all(abs(v - lam_second) < 1e-12 for v in lams)


def test_narrow_mints_per_second_scales_targets_per_block(tmp_path, monkeypatch):
    """LP Poisson targets: if *_per_second is set, lambda_block = block_time * lambda_second."""
    import numpy as np

    lams = []

    def recording_poisson(lam):
        lams.append(float(lam))
        return 0

    monkeypatch.setattr(np.random, "poisson", recording_poisson)

    block_time = 5
    lam_second = 0.25
    _ = simulate(**_base_simulate_kwargs(
        tmp_path,
        T=1,
        seed=123,
        block_time=block_time,
        # Disable traders so only the LP target Poisson draw is exercised.
        smart_trades_per_block=0.0,
        noise_trades_per_block=0.0,
        # Ensure a narrow LP is due so the scheduler draws the Poisson target.
        N_LP=1,
        passive_lp_share=0.0,
        tau=1,
        narrow_mints_per_block=999.0,  # should be ignored when *_per_second is set
        narrow_mints_per_second=lam_second,
        passive_mints_per_block=0.0,
        passive_burns_per_block=0.0,
        mint_mu=-20.0,
        mint_sigma=0.0,
    ))

    # Exactly one Poisson draw for the narrow mint target (per block).
    assert len(lams) == 1
    assert abs(lams[0] - (lam_second * block_time)) < 1e-12
