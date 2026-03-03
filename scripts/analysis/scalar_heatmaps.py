"""scripts/analysis/scalar_heatmaps.py — DEX-share and fee-value heatmaps.

Same visual style as ``pnl_heatmap.py`` but for scalar / per-scenario metrics
(one row per scenario, no cohort axis).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go

from scripts.analysis.common import (
    FONT, PLOTLY_TEMPLATE, final_values, save_figure,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _scalar_values(
    runs: List[Dict[str, Any]], key: str,
) -> np.ndarray:
    """Extract a scalar value from each run dict (float or last element)."""
    vals = []
    for r in runs:
        v = r.get(key)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, (list, np.ndarray)) and len(v) > 0:
            vals.append(float(v[-1]))
    return np.array(vals, dtype=float)


def _mean_series(
    runs: List[Dict[str, Any]], key: str,
) -> np.ndarray:
    """Compute the time-averaged mean of a series for each run."""
    vals = []
    for r in runs:
        s = r.get(key)
        if s is None:
            continue
        arr = np.asarray(s, dtype=float)
        if arr.size > 0:
            vals.append(float(np.mean(arr)))
    return np.array(vals, dtype=float)


# ── DEX share heatmap ─────────────────────────────────────────────────────

def dex_share_heatmap(
    scenario_results: Dict[str, List[Dict[str, Any]]],
) -> go.Figure:
    """Single-row heatmap of mean DEX share across scenarios.

    Uses ``smart_router_dex_share_mean`` (scalar per run).
    """
    labels = list(scenario_results.keys())
    z_row: List[float] = []
    ann_row: List[str] = []

    for label in labels:
        runs = scenario_results[label]
        vals = _scalar_values(runs, "smart_router_dex_share_mean")
        if vals.size == 0:
            z_row.append(float("nan"))
            ann_row.append("")
        else:
            mu = float(np.mean(vals))
            sigma = float(np.std(vals))
            z_row.append(mu)
            ann_row.append(f"{mu:.2%} ± {sigma:.2%}")

    z_arr = np.array([z_row], dtype=float)

    fig = go.Figure(data=go.Heatmap(
        z=z_arr,
        x=labels,
        y=["DEX share"],
        colorscale="Blues",
        zmin=0,
        zmax=1,
        text=[ann_row],
        texttemplate="%{text}",
        textfont=dict(size=16),
        hovertemplate="Scenario: %{x}<br>DEX share: %{z:.2%}<extra></extra>",
        colorbar=dict(title="DEX share"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="DEX routing share (mean ± σ across seeds)",
        xaxis=dict(title="Scenario", tickangle=-35, tickfont=dict(size=20)),
        yaxis=dict(tickfont=dict(size=20)),
        font=FONT,
        height=350,
    )
    return fig


# ── Fee value heatmap ─────────────────────────────────────────────────────

_FEE_COHORT_KEYS = {
    "Passive LP": "lp_fee_value_passive_series",
    "Active LP": "lp_fee_value_active_series",
    "Total": "lp_fee_value_total_series",
}

_INACTIVE: Dict[str, set] = {
    "Model0": {"Active LP"},
    "Model1": set(),
    "Model2": set(),
}


def _is_inactive(cohort: str, scenario_label: str) -> bool:
    for model, inactive_set in _INACTIVE.items():
        if model in scenario_label and cohort in inactive_set:
            return True
    return False


def fee_value_heatmap(
    scenario_results: Dict[str, List[Dict[str, Any]]],
) -> go.Figure:
    """Annotated heatmap of final cumulative fee value (mean ± σ) per cohort."""
    cohorts = list(_FEE_COHORT_KEYS.keys())
    labels = list(scenario_results.keys())
    z_vals: List[List[float]] = []
    annotations: List[List[str]] = []

    for cohort in cohorts:
        row_z: List[float] = []
        row_ann: List[str] = []
        key = _FEE_COHORT_KEYS[cohort]
        for label in labels:
            if _is_inactive(cohort, label):
                row_z.append(float("nan"))
                row_ann.append("")
                continue
            runs = scenario_results[label]
            vals = final_values(runs, key)
            if vals.size == 0:
                row_z.append(float("nan"))
                row_ann.append("")
            else:
                mu = float(np.mean(vals))
                sigma = float(np.std(vals))
                row_z.append(mu)
                row_ann.append(f"{mu:.1f} ± {sigma:.1f}")
        z_vals.append(row_z)
        annotations.append(row_ann)

    z_arr = np.array(z_vals, dtype=float)
    vmax = float(np.nanmax(z_arr)) if z_arr.size and np.any(np.isfinite(z_arr)) else 1.0
    vmax = max(vmax, 1e-6)

    fig = go.Figure(data=go.Heatmap(
        z=z_arr,
        x=labels,
        y=cohorts,
        colorscale="YlGn",
        zmin=0,
        zmax=vmax,
        text=annotations,
        texttemplate="%{text}",
        textfont=dict(size=16),
        hovertemplate="Scenario: %{x}<br>Cohort: %{y}<br>Fees: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Cumulative fees<br>(token-1)"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Cumulative fee value (mean ± σ across seeds)",
        xaxis=dict(title="Scenario", tickangle=-35, tickfont=dict(size=20)),
        yaxis=dict(title="LP Cohort", tickfont=dict(size=20)),
        font=FONT,
        height=350 + 60 * len(cohorts),
    )
    return fig
