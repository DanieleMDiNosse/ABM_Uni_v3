"""scripts/analysis/scalar_heatmaps.py — DEX-share and fee-value bar plots.

Bar charts with error bars (mean ± σ) for DEX routing share and
cumulative fee values across scenarios.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go

from scripts.analysis.common import (
    COLORS, FONT, PLOTLY_TEMPLATE, final_values, save_figure,
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


# ── DEX share bar plot ────────────────────────────────────────────────────

def dex_share_barplot(
    scenario_results: Dict[str, List[Dict[str, Any]]],
) -> go.Figure:
    """Bar plot of mean DEX share (± σ) across scenarios."""
    labels = list(scenario_results.keys())
    means: List[float] = []
    sds: List[float] = []

    for label in labels:
        runs = scenario_results[label]
        vals = _scalar_values(runs, "smart_router_dex_share_mean")
        if vals.size == 0:
            means.append(0.0)
            sds.append(0.0)
        else:
            means.append(float(np.mean(vals)))
            sds.append(float(np.std(vals)))

    fig = go.Figure(data=go.Bar(
        x=labels,
        y=means,
        error_y=dict(type="data", array=sds, visible=True),
        marker_color="#1f77b4",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        # title="DEX routing share (mean ± σ across seeds)",
        xaxis=dict(title="Scenario", tickangle=-35, tickfont=dict(size=14)),
        yaxis=dict(title="DEX share", tickformat=".0%"),
        font=FONT,
        height=450,
    )
    return fig


# ── Fee value bar plot ────────────────────────────────────────────────────

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

_COHORT_COLORS = {
    "Passive LP": COLORS["passive_lp"],
    "Active LP": COLORS["active_lp"],
    "Total": "#2ca02c",
}


def _is_inactive(cohort: str, scenario_label: str) -> bool:
    for model, inactive_set in _INACTIVE.items():
        if model in scenario_label and cohort in inactive_set:
            return True
    return False


def fee_value_barplot(
    scenario_results: Dict[str, List[Dict[str, Any]]],
) -> go.Figure:
    """Grouped bar plot of final cumulative fee value (mean ± σ) per cohort."""
    cohorts = list(_FEE_COHORT_KEYS.keys())
    labels = list(scenario_results.keys())

    fig = go.Figure()
    for cohort in cohorts:
        key = _FEE_COHORT_KEYS[cohort]
        means: List[float] = []
        sds: List[float] = []
        for label in labels:
            if _is_inactive(cohort, label):
                means.append(0.0)
                sds.append(0.0)
                continue
            runs = scenario_results[label]
            vals = final_values(runs, key)
            if vals.size == 0:
                means.append(0.0)
                sds.append(0.0)
            else:
                means.append(float(np.mean(vals)))
                sds.append(float(np.std(vals)))
        fig.add_trace(go.Bar(
            name=cohort,
            x=labels,
            y=means,
            error_y=dict(type="data", array=sds, visible=True),
            marker_color=_COHORT_COLORS[cohort],
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="group",
        # title="Cumulative fee value (mean ± σ across seeds)",
        xaxis=dict(title="Scenario", tickangle=-35, tickfont=dict(size=14)),
        yaxis=dict(title="Cumulative fees (token-1)"),
        font=FONT,
        height=500,
    )
    return fig


# ── Mean fee level bar plot ───────────────────────────────────────────────

def mean_fee_barplot(
    scenario_results: Dict[str, List[Dict[str, Any]]],
) -> go.Figure:
    """Bar plot of time-averaged fee level (mean ± σ across seeds).

    Scenarios whose label contains ``"static"`` are excluded automatically.
    """
    labels = [l for l in scenario_results if "static" not in l.lower()]
    means: List[float] = []
    sds: List[float] = []

    for label in labels:
        runs = scenario_results[label]
        per_run_means: List[float] = []
        for r in runs:
            s = r.get("fee_series")
            if s is None:
                continue
            arr = np.asarray(s, dtype=float)
            if arr.size > 0:
                per_run_means.append(float(np.mean(arr)))
        vals = np.array(per_run_means, dtype=float)
        means.append(float(np.mean(vals)) if vals.size else 0.0)
        sds.append(float(np.std(vals)) if vals.size else 0.0)

    fig = go.Figure(data=go.Bar(
        x=labels,
        y=means,
        error_y=dict(type="data", array=sds, visible=True),
        marker_color="#ff7f0e",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        # title="Mean liquidity-taker fee (mean ± σ across seeds, static excluded)",
        xaxis=dict(title="Scenario", tickangle=-35, tickfont=dict(size=14)),
        yaxis=dict(title="Fee rate"),
        font=FONT,
        height=450,
    )
    return fig
