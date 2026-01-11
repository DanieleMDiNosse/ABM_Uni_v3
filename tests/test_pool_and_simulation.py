"""How to run tests:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. pytest tests/test_pool_and_simulation.py
"""

import copy
from typing import Any, Dict

import pytest

from utils import build_empty_pool
from uniswapv3_pool import V3Pool, BoundaryIndex
from run import simulate


def _prepare_pool() -> V3Pool:
    """
    Build a pool with symmetric liquidity around the active tick.
    """
    pool, _ = build_empty_pool()
    # add symmetric liquidity around the active tick so quotes have cross-tick depth
    L = 50_000.0
    lower = pool.tick - pool.tick_spacing * 2
    upper = pool.tick + pool.tick_spacing * 2
    pool.add_liquidity_range(lower, upper, L)
    pool.recompute_active_L()
    return pool


def _base_simulate_kwargs(tmp_path, **overrides: Any) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(
        # Core simulation parameters
        config_name="unit_test",
        block_time=5,
        T=5,
        seed=1,
        liquidity_for_gif=False,
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
        fee_mode="volatility",
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
        light_mode=False,
    )
    params.update(overrides)
    return params


def test_quote_x_to_y_matches_swap_result():
    """
    The read-only X→Y quote should match an actual swap when the trade is executed on an identical pool copy with identical liquidity.
    """
    pool = _prepare_pool()
    dx_in = 1.0

    quoted = pool.quote_x_to_y(dx_in)

    pool_for_swap = copy.deepcopy(pool)
    used_dx, dy_out, _ = pool_for_swap.swap_x_to_y(dx_in, fee_cb=None)
    assert used_dx == pytest.approx(dx_in)
    assert dy_out == pytest.approx(quoted, rel=1e-9, abs=1e-9)


def test_quote_y_to_x_matches_swap_result():
    """
    The read-only Y→X quote should reproduce the exact output of a swap consuming the same input on a cloned pool.
    """
    pool = _prepare_pool()
    dy_in = 1.0

    quoted = pool.quote_y_to_x(dy_in)

    pool_for_swap = copy.deepcopy(pool)
    used_dy, dx_out, _ = pool_for_swap.swap_y_to_x(dy_in, fee_cb=None)
    assert used_dy == pytest.approx(dy_in)
    assert dx_out == pytest.approx(quoted, rel=1e-9, abs=1e-9)


def test_swap_y_to_x_tracks_used_input_across_tick_crossing():
    """
    swap_y_to_x() should report the full input amount even when the swap crosses one or more tick boundaries.
    """
    pool = _prepare_pool()
    tick_before = pool.tick
    S_before = pool.S
    S_hi = pool.s_upper()
    dy_to = pool.L_active * (S_hi - S_before)  # post-fee amount needed to reach the next boundary
    dy_in = dy_to / pool.r + 1.0  # cross one boundary, then consume a tiny remainder

    used_dy, _dx_out, fee_y = pool.swap_y_to_x(dy_in, fee_cb=None)

    assert pool.tick == tick_before + pool.tick_spacing
    assert used_dy == pytest.approx(dy_in)
    assert fee_y == pytest.approx(dy_in * pool.f)


def test_quote_returns_zero_without_liquidity():
    """
    Quoting should short-circuit to zero when the pool has no active liquidity so callers do not rely on stale or undefined depth.
    """
    pool, _ = build_empty_pool()
    assert pool.quote_x_to_y(10.0) == 0.0
    assert pool.quote_y_to_x(10.0) == 0.0


def test_swap_inactive_pool_returns_zero():
    """
    Swaps on an empty pool should yield zero flow and leave the price unchanged, guaranteeing graceful handling of missing liquidity.
    """
    pool, _ = build_empty_pool()
    start_price = pool.price
    used_dx, dy_out, fee = pool.swap_x_to_y(5.0, fee_cb=None)
    assert used_dx == 0.0
    assert dy_out == 0.0
    assert fee == 0.0
    assert pool.price == start_price


def test_boundary_index_tracks_liquidity_across_ticks():
    """
    BoundaryIndex must expose accurate prefix sums so that consumers can locate active liquidity before and after mint/burn events.
    """
    pool, _ = build_empty_pool()
    spacing = pool.tick_spacing
    lower1 = pool.tick - spacing
    upper1 = pool.tick + spacing
    pool.add_liquidity_range(lower1, upper1, 1_000.0)
    lower2 = upper1
    upper2 = lower2 + spacing
    pool.add_liquidity_range(lower2, upper2, 2_000.0)
    bidx = BoundaryIndex(pool.liquidity_net)

    assert bidx.active_liquidity_at_tick(pool.tick) == pytest.approx(1_000.0)
    assert bidx.active_liquidity_at_tick(lower2) == pytest.approx(2_000.0)
    assert bidx.next_up(lower1) == lower2
    assert bidx.prev_down(lower2) == lower1


def test_simulate_outputs_consistent_lengths(tmp_path):
    """
    Simulation series describing trader and smart-router flows must have consistent lengths so plotting and analysis can zip the arrays safely.
    """
    out = simulate(**_base_simulate_kwargs(tmp_path, T=5, seed=1))

    assert len(out['smart_router_pnl_steps']) == len(out['smart_router_notional_y'])
    assert len(out['noise_trader_pnl_steps']) == len(out['noise_trader_notional_y'])
    assert len(out['smart_router_exec_count']) == len(out['smart_router_pnl_steps'])
    assert len(out['noise_trader_exec_count']) == len(out['noise_trader_pnl_steps'])
    assert len(out['smart_router_pnl_steps']) == len(out['smart_router_pnl_cum'])
    assert len(out['noise_trader_pnl_steps']) == len(out['noise_trader_pnl_cum'])
    assert out['fee_mode'] == 'volatility'


def test_simulate_invalid_fee_mode(tmp_path):
    """
    simulate() should defensively reject unsupported fee controller modes before any random state is touched.
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
def test_simulate_validates_k_out_bounds(tmp_path, kwargs):
    """
    The k_out bounds govern price-impact sampling and must be positive and ordered; invalid tuples should raise cleanly.
    """
    with pytest.raises(ValueError):
        simulate(**_base_simulate_kwargs(tmp_path, T=1, **kwargs))


def test_simulate_seed_determinism(tmp_path):
    """
    Re-running the simulation with the same seed should reproduce every series exactly, while a different seed should alter at least one path.
    """
    out_a = simulate(**_base_simulate_kwargs(tmp_path / "a", T=3, seed=42, noise_trades_per_block=5.0))
    out_b = simulate(**_base_simulate_kwargs(tmp_path / "b", T=3, seed=42, noise_trades_per_block=5.0))
    out_c = simulate(**_base_simulate_kwargs(tmp_path / "c", T=3, seed=999, noise_trades_per_block=5.0))

    dex_a = tuple(out_a["DEX_price"])
    dex_b = tuple(out_b["DEX_price"])
    dex_c = tuple(out_c["DEX_price"])
    sr_pnl_a = tuple(out_a["smart_router_pnl_steps"])
    sr_pnl_b = tuple(out_b["smart_router_pnl_steps"])

    assert dex_a == dex_b
    assert sr_pnl_a == sr_pnl_b
    assert any(abs(a - b) > 1e-12 for a, b in zip(dex_a, dex_c))


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


def test_simulate_heston_mode_requires_parameters(tmp_path):
    """
    Heston volatility mode must fail fast when required cex_heston_* parameters are missing.
    """
    with pytest.raises(ValueError, match="cex_sigma_mode='heston' requires parameters"):
        simulate(**_base_simulate_kwargs(tmp_path, T=1, cex_sigma_mode="heston"))


def test_simulate_heston_mode_runs_and_emits_sigma_series(tmp_path):
    """
    With valid Heston parameters, simulate() should run and produce a non-negative cex_sigma_series.
    """
    out = simulate(**_base_simulate_kwargs(
        tmp_path,
        T=3,
        cex_sigma_mode="heston",
        cex_sigma=0.001,  # used as sqrt(v0) when cex_heston_v0 is omitted
        cex_heston_kappa=1.0,
        cex_heston_theta=1e-6,
        cex_heston_sigma_v=0.1,
        cex_heston_rho=-0.5,
    ))
    sigma_series = out["cex_sigma_series"]
    cex_series = out["CEX_price"]
    assert len(sigma_series) == len(cex_series)
    assert all(s >= 0.0 for s in sigma_series)


def test_no_arb_band_includes_flash_loan_fee(tmp_path):
    """
    The reported no-arb band should widen when flash-loan funding costs are enabled.
    """
    f0 = 0.003
    phi = 0.10
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
    assert out["band_lo"][0] == pytest.approx(m0 * r / (1.0 + phi))
    assert out["band_hi"][0] == pytest.approx(m0 * (1.0 + phi) / r)


def test_arb_pnl_settles_on_snapshot_not_end_of_step(tmp_path, monkeypatch):
    """
    Arbitrageur PnL should be computed at the validated snapshot CEX price used for
    the arb decision/unwind, not at the end-of-step CEX price after intra-block diffusion.
    """
    import random
    import utils
    import numpy as np

    # Make noise arrivals deterministic: exactly one intent per micro-step.
    monkeypatch.setattr(np.random, "poisson", lambda _lam: 1)
    # Force all noise trades to push DEX price in one direction.
    monkeypatch.setattr(random, "choice", lambda seq: "X_to_Y")

    sim_kwargs = _base_simulate_kwargs(
        tmp_path / "base",
        T=2,
        seed=7,
        block_time=2,
        smart_trades_per_block=0.0,
        noise_trades_per_block=2.0,
        cex_sigma=0.0,
        slippage_tolerance=1.0,  # never reject on slippage
        trader_mean=5.5,         # deterministic trades (sigma=0 below)
        trader_sigma=0.0,
        fee_mode="static",
        flash_loan_fee=0.0,
    )

    # Run A: no intra-block diffusion at all.
    monkeypatch.setattr(utils.ReferenceMarket, "diffuse_only", lambda self: self.m)
    out_a = simulate(**sim_kwargs)

    # Run B: keep step-0 diffusion identical, then apply a large jump during step 1.
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

    # Sanity: we really changed the end-of-step CEX price in step 1.
    assert out_a["CEX_price"][1] != pytest.approx(out_b["CEX_price"][1])
    # Ensure an arb actually executed in step 1 (otherwise this test is vacuous).
    assert out_a["arb_exec_count"][1] == 1
    assert abs(out_a["arb_pnl_steps"][1]) > 1e-12
    # Core assertion: arb PnL is invariant to intra-block CEX diffusion when
    # it is computed at the validated snapshot price.
    assert out_a["arb_pnl_steps"][1] == pytest.approx(out_b["arb_pnl_steps"][1], rel=1e-12, abs=1e-12)
