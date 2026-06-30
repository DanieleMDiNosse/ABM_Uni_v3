from pathlib import Path

import pandas as pd

from scripts.analysis import plot_model2_lvr_ratio_combined as lvr_ratio
from scripts.analysis.paper_figure_style import load_figure_style
from scripts.analysis.regenerate_paper_figures import _acf_no_correlation_band


def test_load_figure_style_merges_global_and_per_figure_override(tmp_path: Path) -> None:
    style_path = tmp_path / "figure_style.yml"
    style_path.write_text(
        """
        paper_figure_style:
          width: 111
          font:
            base_size: 10
            tick_size: 8
        figures:
          example_figure:
            width: 222
            font:
              axis_title_size: 12
        """,
        encoding="utf-8",
    )

    style = load_figure_style(
        style_path,
        {"width": 1, "height": 2, "font": {"base_size": 1, "axis_title_size": 1}},
        figure_key="example_figure",
    )

    assert style["width"] == 222
    assert style["height"] == 2
    assert style["font"]["base_size"] == 10
    assert style["font"]["tick_size"] == 8
    assert style["font"]["axis_title_size"] == 12


def test_blocksize_ratio_uses_per_figure_schedule_color_override(tmp_path: Path) -> None:
    style_path = tmp_path / "figure_style.yml"
    style_path.write_text(
        """
        paper_figure_style:
          width: 700
          height: 400
          font:
            base_size: 12
            axis_title_size: 13
            tick_size: 11
            legend_size: 10
            subplot_title_size: 14
          margins:
            l: 20
            r: 20
            t: 20
            b: 20
        figures:
          model2_blocksize_ratio_combined:
            legend:
              show: false
            fee_schedules:
              static:
                color: "#123456"
        """,
        encoding="utf-8",
    )
    rows = []
    for cohort in lvr_ratio.COHORTS:
        for schedule in lvr_ratio.SCHEDULES:
            rows.append(
                {
                    "block_time": 2,
                    "fee_schedule": schedule,
                    "cohort": cohort,
                    "median_R": 1.0,
                }
            )
    df = pd.DataFrame(rows)

    style = lvr_ratio.load_paper_style(style_path)
    fig = lvr_ratio.build_figure(df, style)

    assert fig.data[0].line.color == "#123456"
    assert fig.data[0].showlegend is False
    assert fig.layout.width == 700
    assert fig.layout.height == 400


def test_acf_no_correlation_band_is_analytic_large_sample_formula() -> None:
    # Under the white-noise null, rho_hat(k) is asymptotically N(0, 1/N).
    # For N=9999 returns and a two-sided 95% band this is z_0.975/sqrt(N),
    # without bootstrap/permutation randomness.
    band = _acf_no_correlation_band(9999, confidence_level=0.95)

    assert abs(band - 0.019601) < 1e-6
