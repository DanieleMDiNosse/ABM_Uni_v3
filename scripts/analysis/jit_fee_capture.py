"""scripts/analysis/jit_fee_capture.py — JIT fee-sniping analysis.

Produces two figures:
  1. Per-block fee share captured by JIT vs other LPs, as a function of the
     fee level in that block.
  2. Distribution of fee level at JIT-entry blocks vs non-entry blocks
     (boxplot / overlaid histograms).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.analysis.common import (
    COLORS, FONT, PLOTLY_TEMPLATE, diff_cumulative, slice_series,
)


def _extract_jit_block_data(
    results: List[Dict[str, Any]],
    skip: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pooled (fee_level, jit_fee_increment, lp_fee_increment, jit_active)."""
    all_fee_lvl: List[np.ndarray] = []
    all_jit_dfee: List[np.ndarray] = []
    all_lp_dfee: List[np.ndarray] = []
    all_jit_active: List[np.ndarray] = []

    for r in results:
        fee_lvl = slice_series(r.get("fee_series", []), skip)
        jit_fee_cum = slice_series(r.get("jiter_fee_value_series", []), skip)
        lp_fee_cum = slice_series(r.get("lp_fee_value_total_series", []), skip)
        jit_act_cum = slice_series(r.get("jiter_activity_cum", []), skip)

        n = min(fee_lvl.size, jit_fee_cum.size, lp_fee_cum.size, jit_act_cum.size)
        if n < 2:
            continue

        fee_lvl = fee_lvl[:n]
        d_jit = diff_cumulative(jit_fee_cum[:n])
        d_lp = diff_cumulative(lp_fee_cum[:n])
        d_act = diff_cumulative(jit_act_cum[:n])

        all_fee_lvl.append(fee_lvl)
        all_jit_dfee.append(d_jit)
        all_lp_dfee.append(d_lp)
        all_jit_active.append(d_act)

    if not all_fee_lvl:
        return (np.array([]), np.array([]), np.array([]), np.array([]))

    return (
        np.concatenate(all_fee_lvl),
        np.concatenate(all_jit_dfee),
        np.concatenate(all_lp_dfee),
        np.concatenate(all_jit_active),
    )


def jit_fee_share(
    results: List[Dict[str, Any]],
    *,
    skip: int = 0,
    n_fee_bins: int = 10,
) -> go.Figure:
    """Fee share of JIT vs other LPs, binned by fee level.

    Parameters
    ----------
    results : list[dict]
        Per-seed output dicts (must have JIT enabled).
    skip : int
        Burn-in blocks to discard.
    n_fee_bins : int
        Number of fee-level bins.

    Returns
    -------
    go.Figure  (stacked bar: JIT share vs LP share by fee-level bin)
    """
    fee_lvl, d_jit, d_lp, d_act = _extract_jit_block_data(results, skip)
    if fee_lvl.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="No JIT data", showarrow=False, x=0.5, y=0.5)
        return fig

    # Only consider blocks where some fees were earned
    total_fee = d_jit + d_lp
    mask = total_fee > 1e-12
    fee_lvl = fee_lvl[mask]
    d_jit = d_jit[mask]
    total_fee_m = total_fee[mask]

    if fee_lvl.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="No blocks with positive fees", showarrow=False,
                           x=0.5, y=0.5)
        return fig

    edges = np.quantile(fee_lvl, np.linspace(0, 1, n_fee_bins + 1))
    bin_idx = np.clip(np.digitize(fee_lvl, edges[1:-1]), 0, n_fee_bins - 1)

    jit_shares = []
    lp_shares = []
    labels = []
    for b in range(n_fee_bins):
        m = bin_idx == b
        if m.sum() == 0:
            jit_shares.append(0)
            lp_shares.append(0)
            labels.append(f"{edges[b]:.4f}–{edges[b+1]:.4f}")
            continue
        jit_total = float(np.sum(d_jit[m]))
        all_total = float(np.sum(total_fee_m[m]))
        js = jit_total / all_total if all_total > 0 else 0
        jit_shares.append(js)
        lp_shares.append(1 - js)
        labels.append(f"{edges[b]:.4f}–{edges[b+1]:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=jit_shares, name="JIT share",
                         marker_color=COLORS["jiter"]))
    fig.add_trace(go.Bar(x=labels, y=lp_shares, name="Other LPs share",
                         marker_color=COLORS["passive_lp"]))
    fig.update_layout(
        barmode="stack",
        template=PLOTLY_TEMPLATE,
        title="Fee share: JIT vs other LPs (by fee-level bin)",
        xaxis_title="Fee level bin",
        yaxis_title="Share of block fees",
        font=FONT,
        height=500,
    )
    return fig


def jit_entry_fee_distribution(
    results: List[Dict[str, Any]],
    *,
    skip: int = 0,
) -> go.Figure:
    """Compare fee level distributions at JIT-entry vs non-entry blocks.

    Returns
    -------
    go.Figure  (overlaid histograms)
    """
    fee_lvl, _, _, d_act = _extract_jit_block_data(results, skip)
    if fee_lvl.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="No JIT data", showarrow=False, x=0.5, y=0.5)
        return fig

    jit_mask = d_act > 0.5  # JIT was active in this block
    fee_jit = fee_lvl[jit_mask]
    fee_no_jit = fee_lvl[~jit_mask]

    fig = go.Figure()
    if fee_no_jit.size > 0:
        fig.add_trace(go.Histogram(
            x=fee_no_jit, name=f"Non-JIT blocks (n={fee_no_jit.size})",
            marker_color=COLORS["passive_lp"], opacity=0.6,
            histnorm="probability density",
        ))
    if fee_jit.size > 0:
        fig.add_trace(go.Histogram(
            x=fee_jit, name=f"JIT-entry blocks (n={fee_jit.size})",
            marker_color=COLORS["jiter"], opacity=0.6,
            histnorm="probability density",
        ))
    fig.update_layout(
        barmode="overlay",
        template=PLOTLY_TEMPLATE,
        title="Fee level distribution: JIT-entry blocks vs non-entry blocks",
        xaxis_title="Fee level",
        yaxis_title="Density",
        font=FONT,
        height=450,
    )
    return fig
