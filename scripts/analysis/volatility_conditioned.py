"""scripts/analysis/volatility_conditioned.py — Volatility-binned analysis.

Bins simulation blocks by the CEX instantaneous volatility σ_t (from
``cex_sigma_series``) and computes, for each bin, the aggregate
ΔLVR / ΔFees ratio.

This directly tests the paper's claim that dynamic fees protect LPs
during high-volatility regimes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from scripts.analysis.common import (
    COLORS, FONT, PLOTLY_TEMPLATE, diff_cumulative, slice_series,
)

_COHORT_MAP = {
    "passive": ("lp_fee_value_passive_series", "lp_lvr_passive_series"),
    "active":  ("lp_fee_value_active_series",  "lp_lvr_active_series"),
    "total":   ("lp_fee_value_total_series",   "lp_lvr_total_series"),
}


def volatility_binned_analysis(
    results: List[Dict[str, Any]],
    *,
    n_bins: int = 5,
    skip: int = 0,
    cohort: str = "total",
    bin_labels: Optional[List[str]] = None,
) -> go.Figure:
    """Bin blocks by σ_t quantile and show per-bin aggregate ΔLVR / ΔFees ratio.

    Parameters
    ----------
    results : list[dict]
        Per-seed output dicts.
    n_bins : int
        Number of volatility quantile bins (default 5).
    skip : int
        Burn-in blocks to discard.
    cohort : str
        ``"passive"``, ``"active"``, or ``"total"``.

    Returns
    -------
    go.Figure  (single bar chart: per-bin LVR/fee ratio)
    """
    fee_key, lvr_key = _COHORT_MAP[cohort]

    # Pool per-block increments across all seeds.
    all_sigma: List[np.ndarray] = []
    all_d_fee: List[np.ndarray] = []
    all_d_lvr: List[np.ndarray] = []

    for r in results:
        sigma = slice_series(r.get("cex_sigma_series", []), skip)
        fees_cum = slice_series(r.get(fee_key, []), skip)
        lvr_cum = slice_series(r.get(lvr_key, []), skip)

        n = min(sigma.size, fees_cum.size, lvr_cum.size)
        if n < 2:
            continue

        sigma = sigma[:n]
        d_fee = diff_cumulative(fees_cum[:n])
        d_lvr = diff_cumulative(lvr_cum[:n])

        all_sigma.append(sigma)
        all_d_fee.append(d_fee)
        all_d_lvr.append(d_lvr)

    if not all_sigma:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False, x=0.5, y=0.5)
        return fig

    sigma_cat = np.concatenate(all_sigma)
    d_fee_cat = np.concatenate(all_d_fee)
    d_lvr_cat = np.concatenate(all_d_lvr)

    # Compute quantile edges.
    edges = np.quantile(sigma_cat, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.digitize(sigma_cat, edges[1:-1])  # 0 .. n_bins-1

    if bin_labels is None:
        bin_labels = [f"Q{i+1}" for i in range(n_bins)]

    mean_ratio = []
    se_ratio = []
    sigma_range_labels = []

    for b in range(n_bins):
        mask = bin_idx == b
        count = mask.sum()
        if count == 0:
            mean_ratio.append(float("nan")); se_ratio.append(0.0)
            sigma_range_labels.append(bin_labels[b])
            continue
        df = d_fee_cat[mask]
        dl = d_lvr_cat[mask]

        total_fee = float(np.sum(df))
        total_lvr = float(np.sum(dl))
        ratio = total_lvr / total_fee if abs(total_fee) > 1e-12 else float("nan")
        mean_ratio.append(ratio)
        per_block_ratio = np.where(np.abs(df) > 1e-12, dl / df, np.nan)
        valid = per_block_ratio[np.isfinite(per_block_ratio)]
        se_ratio.append(float(np.std(valid, ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else 0.0)

        lo = float(edges[b])
        hi = float(edges[b + 1])
        sigma_range_labels.append(f"{bin_labels[b]}<br>[{lo:.2e}, {hi:.2e}]")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=sigma_range_labels, y=mean_ratio, marker_color="#e377c2",
        error_y=dict(type="data", array=se_ratio, visible=True),
        name="ΔLVR/ΔFees",
    ))

    fig.add_hline(y=1, line=dict(color="gray", dash="dot", width=1))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Aggregate ΔLVR / ΔFees by σ_t quantile — {cohort} LPs ({n_bins} bins)",
        font=FONT,
        showlegend=False,
        height=450,
    )
    fig.update_xaxes(title_text="σ_t quantile")
    fig.update_yaxes(title_text="Ratio")

    return fig
