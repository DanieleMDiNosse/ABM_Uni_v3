"""Regression tests for scripts.visualize_distributions."""

import math
import sys

import numpy as np
from numpy.testing import assert_allclose

import scripts.visualize_distributions as visualize_distributions
from scripts.visualize_distributions import (
    _effective_jit_attempt_probability,
    _effective_review_interval_and_unit,
    DistributionParams,
    build_distribution_params,
    plot_arrival_distributions,
    plot_distribution_suite,
    sample_lp_widths_from_reference_prices,
    sample_post_burn_cooldown,
)


def test_build_distribution_params_reads_width_rule_controls() -> None:
    """Scenario parsing should thread width-rule controls into the visualization params."""
    params = build_distribution_params(
        {
            "seed": 11,
            "w_min_ticks": 30,
            "w_max_ticks": 250,
            "basis_half_life": 7,
            "slope_s": 2.5,
        },
        n_steps=25,
        n_samples=50,
    )

    assert isinstance(params, DistributionParams)
    assert params.seed == 11
    assert params.w_min_ticks == 30
    assert params.w_max_ticks == 250
    assert params.basis_half_life == 7
    assert params.slope_s == 2.5


def test_sample_post_burn_cooldown_matches_scheduler_units(monkeypatch) -> None:
    """Cooldown draws should stay in blocks unless tau_seconds is set, then scale by block_time."""

    def fake_randint(low: int, high: int, size: int) -> np.ndarray:
        assert low == 3
        assert high == 9
        assert size == 4
        return np.asarray([3, 4, 7, 8], dtype=int)

    monkeypatch.setattr(np.random, "randint", fake_randint)

    in_blocks = sample_post_burn_cooldown(block_time=5, tau_seconds=None, n_samples=4)
    in_seconds = sample_post_burn_cooldown(block_time=5, tau_seconds=25.0, n_samples=4)

    assert_allclose(in_blocks, np.asarray([3.0, 4.0, 7.0, 8.0]))
    assert_allclose(in_seconds, np.asarray([15.0, 20.0, 35.0, 40.0]))


def test_sample_lp_widths_from_reference_prices_matches_run_formula_without_noise() -> None:
    """Width sampling should match the clipped snap-to-grid rule used in scripts.run."""
    prices = np.asarray([2000.0, 2010.0, 2025.0], dtype=float)
    widths = sample_lp_widths_from_reference_prices(
        prices,
        basis_half_life=1,
        w_min_ticks=10,
        w_max_ticks=1000,
        slope_s=1.0,
        binom_n=0,
        binom_p=0.5,
        tick_spacing=10,
    )

    lam = math.exp(-math.log(2.0) / 1.0)
    vol_obs_1 = abs(math.log(prices[1]) - math.log(prices[0]))
    vol_obs_2 = abs(math.log(prices[2]) - math.log(prices[1]))
    vol_hat_1 = (1.0 - lam) * vol_obs_1
    vol_hat_2 = lam * vol_hat_1 + (1.0 - lam) * vol_obs_2
    expected_1 = int(round((10.0 + vol_hat_1 / math.log(1.0001)) / 10.0)) * 10
    expected_2 = int(round((10.0 + vol_hat_2 / math.log(1.0001)) / 10.0)) * 10

    assert_allclose(widths, np.asarray([float(expected_1), float(expected_2)]))


def test_effective_review_interval_matches_run_semantics() -> None:
    """Visualization review intervals should use the same effective mean as the simulator."""
    mean_tau, tau_unit = _effective_review_interval_and_unit(DistributionParams(tau=7, tau_seconds=None))
    mean_seconds, seconds_unit = _effective_review_interval_and_unit(
        DistributionParams(tau=7, tau_seconds=0.25)
    )

    assert mean_tau == 7.0
    assert tau_unit == "blocks"
    assert mean_seconds == 1.0
    assert seconds_unit == "s"


def test_effective_jit_attempt_probability_respects_enable_knobs() -> None:
    """JIT attempts should disappear from the plot when the searcher is disabled in run.py terms."""
    enabled = DistributionParams(p_jit=0.6, N_jit=1, liquidity_perc_jit=0.9)
    disabled_by_targets = DistributionParams(p_jit=0.6, N_jit=0, liquidity_perc_jit=0.9)
    disabled_by_liquidity = DistributionParams(p_jit=0.6, N_jit=1, liquidity_perc_jit=0.0)

    assert _effective_jit_attempt_probability(enabled) == 0.6
    assert _effective_jit_attempt_probability(disabled_by_targets) == 0.0
    assert _effective_jit_attempt_probability(disabled_by_liquidity) == 0.0


def test_distribution_figures_do_not_draw_vertical_marker_lines() -> None:
    """Core and arrival figures should not include the old mean-marker vlines."""
    params = DistributionParams(
        seed=3,
        n_steps=8,
        n_samples=64,
        cex_sigma_mode="static",
        cex_sigma=1e-4,
        binom_n=0,
        p_jit=0.0,
    )

    core_fig = plot_distribution_suite(params, reference_prices=[2000.0] * (params.n_steps + 1))
    arrival_fig = plot_arrival_distributions(params)

    assert len(core_fig.layout.shapes or ()) == 0
    assert len(arrival_fig.layout.shapes or ()) == 0


def test_main_skips_reference_market_export(tmp_path, monkeypatch) -> None:
    """The script should only export the core and arrival figures."""
    config_path = tmp_path / "mini.yml"
    config_path.write_text(
        "\n".join(
            [
                "fee_mode: static",
                "simulate:",
                "  seed: 1",
                "  block_time: 5",
                "  cex_mu: 0.0",
                "  cex_sigma: 0.0001",
                "  cex_sigma_mode: static",
                "  trader_mean: 0.0",
                "  trader_sigma: 0.1",
                "  mint_mu: 0.0",
                "  mint_sigma: 0.1",
                "  tau: 5",
                "  k_out_min: 1",
                "  k_out_max: 2",
                "  smart_trades_per_block: 0.0",
                "  noise_trades_per_block: 0.0",
                "  narrow_mints_per_block: 0.0",
                "  passive_mints_per_block: 0.0",
                "  passive_burns_per_block: 0.0",
                "  p_jit: 0.0",
                "  N_jit: 1",
                "  liquidity_perc_jit: 0.9",
                "  initial_binom_N: 4",
                "  initial_total_L: 100.0",
                "  w_min_ticks: 10",
                "  w_max_ticks: 100",
                "  basis_half_life: 1",
                "  slope_s: 1.0",
                "  binom_n: 0",
                "  binom_p: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    saved_html_names = []

    class DummyManifest:
        def to_dict(self):
            return {}

    def fake_save_plotly_figure(fig, png_path, html_path, source="plot", **kwargs):
        saved_html_names.append(str(html_path.name))

    def fake_make_unique_dir(path):
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(visualize_distributions, "save_plotly_figure", fake_save_plotly_figure)
    monkeypatch.setattr(visualize_distributions, "scenario_output_root", lambda _config: tmp_path / "scenario")
    monkeypatch.setattr(visualize_distributions, "make_unique_dir", fake_make_unique_dir)
    monkeypatch.setattr(visualize_distributions, "snapshot_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(visualize_distributions, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(visualize_distributions, "write_csv_rows", lambda *args, **kwargs: None)
    monkeypatch.setattr(visualize_distributions, "build_run_manifest", lambda **kwargs: DummyManifest())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_distributions.py",
            "--config",
            str(config_path),
            "--n-steps",
            "5",
            "--n-samples",
            "20",
        ],
    )

    visualize_distributions.main()

    assert len(saved_html_names) == 2
    assert all("reference_market" not in name for name in saved_html_names)
