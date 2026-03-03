"""scripts/analysis/fee_lvr_decomposition.py — Fee vs LVR 4-panel figure.

Shows the *mechanism* behind hedged PnL by decomposing it into:
  (a) cumulative fees  F_t,
  (b) cumulative LVR,
  (c) hedged PnL = F_t − LVR_t,
  (d) fee level  f_t  over time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.analysis.common import (
    COLORS, FONT, GRID_STYLE, LEGEND_STYLE, PLOTLY_TEMPLATE,
    align_multi_run_series, hex_to_rgba, save_figure, slice_series,
)

_COHORT_MAP = {
    "passive": ("lp_fee_value_passive_series", "lp_lvr_passive_series", "lp_pnl_passive"),
    "active":  ("lp_fee_value_active_series",  "lp_lvr_active_series",  "lp_pnl_active"),
    "total":   ("lp_fee_value_total_series",   "lp_lvr_total_series",   "lp_pnl_total"),
}


def _add_band(fig, *, x, mean, std, name, color, row, col,
              dash=None, band_alpha=0.18):
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fill = f"rgba({r},{g},{b},{band_alpha})"
    fig.add_trace(go.Scatter(x=x, y=mean - std, mode="lines",
                             line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), row=row, col=col)
    fig.add_trace(go.Scatter(x=x, y=mean + std, mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor=fill, showlegend=False,
                             hoverinfo="skip"), row=row, col=col)
    line = dict(width=2, color=color)
    if dash:
        line["dash"] = dash
    fig.add_trace(go.Scatter(x=x, y=mean, mode="lines", name=name,
                             line=line), row=row, col=col)


def fee_lvr_panels(
    results: List[Dict[str, Any]],
    *,
    cohort: str = "passive",
    skip: int = 0,
    band_mult: float = 2.0,
) -> go.Figure:
    """Build a 4-panel decomposition figure (mean ± band across seeds).

    Parameters
    ----------
    results : list[dict]
        Per-seed output dicts from ``simulate()``.
    cohort : str
        ``"passive"``, ``"active"``, or ``"total"``.
    skip : int
        Burn-in blocks to skip.
    band_mult : float
        Band = mean ± band_mult × std.

    Returns
    -------
    go.Figure
    """
    fee_key, lvr_key, pnl_key = _COHORT_MAP[cohort]

    fees_mat = align_multi_run_series(results, fee_key, skip)
    lvr_mat  = align_multi_run_series(results, lvr_key, skip)
    pnl_mat  = align_multi_run_series(results, pnl_key, skip)
    fee_level_mat = align_multi_run_series(results, "fee_series", skip)

    T = fees_mat.shape[1] if fees_mat.ndim == 2 and fees_mat.shape[1] > 0 else 0
    if T == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False, x=0.5, y=0.5)
        return fig

    x = np.arange(skip, skip + T)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=(
            f"Cumulative fees ({cohort})",
            f"Cumulative LVR ({cohort})",
            f"Hedged PnL = Fees − LVR ({cohort})",
            "Fee level f_t",
        ),
    )

    for mat, name, color, row in [
        (fees_mat, "Fees", COLORS["fees"], 1),
        (lvr_mat,  "LVR",  COLORS["lvr"],  2),
        (pnl_mat,  "Hedged PnL", COLORS["passive_lp"] if cohort == "passive"
                    else COLORS["active_lp"], 3),
    ]:
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0) * band_mult
        _add_band(fig, x=x, mean=mean, std=std, name=name, color=color,
                  row=row, col=1)

    # Fee level panel
    if fee_level_mat.shape[1] > 0:
        mean_fee = np.mean(fee_level_mat, axis=0)
        std_fee = np.std(fee_level_mat, axis=0) * band_mult
        _add_band(fig, x=x, mean=mean_fee, std=std_fee,
                  name="Fee level", color="#333333", row=4, col=1)

    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"), row=3, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=1100,
        title=f"Fee / LVR decomposition — {cohort} LPs",
        font=FONT,
        legend=LEGEND_STYLE,
        showlegend=True,
    )
    fig.update_yaxes(title_text="Token-1", row=1, col=1)
    fig.update_yaxes(title_text="Token-1", row=2, col=1)
    fig.update_yaxes(title_text="Token-1", row=3, col=1)
    fig.update_yaxes(title_text="Fee rate", row=4, col=1)
    fig.update_xaxes(title_text="Block", row=4, col=1)

    return fig
