from __future__ import annotations

from abm_webapp.app import _build_fee_figure


def test_linear_asymmetric_fee_figure_plots_both_directional_fee_paths() -> None:
    rows = [
        {
            "t": 0,
            "fee": 0.003,
            "fee_x_to_y": 0.003,
            "fee_y_to_x": 0.003,
            "fee_signal": 0.0,
            "fee_sigma": 0.0,
            "fee_basis_ticks": 0.0,
        },
        {
            "t": 1,
            "fee": 0.003,
            "fee_x_to_y": 0.004,
            "fee_y_to_x": 0.002,
            "fee_signal": 0.01,
            "fee_sigma": 0.0,
            "fee_basis_ticks": 0.0,
        },
        {
            "t": 2,
            "fee": 0.003,
            "fee_x_to_y": 0.0025,
            "fee_y_to_x": 0.0035,
            "fee_signal": -0.005,
            "fee_sigma": 0.0,
            "fee_basis_ticks": 0.0,
        },
    ]

    fig = _build_fee_figure(rows, fee_mode="linear_asymmetric")
    trace_names = {trace.name for trace in fig.data}

    assert "X→Y fee" in trace_names
    assert "Y→X fee" in trace_names
    assert "log(DEX/CEX)" in trace_names
