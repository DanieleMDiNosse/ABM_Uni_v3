"""scripts/analysis/pnl_heatmap.py — Cross-scenario PnL comparison table.

Produces a single heatmap / annotated table showing final hedged PnL
(mean ± std across seeds) for every (model, fee_mode, LP cohort) combination.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from scripts.analysis.common import (
    COLORS, FONT, PLOTLY_TEMPLATE, final_values, save_figure,
)

# LP cohort keys in the simulate() output dict.
_COHORT_KEYS = {
    "Passive LP": "lp_pnl_passive",
    "Active LP": "lp_pnl_active",
    "JIT": "jiter_pnl_series",
}

# Cohorts that are inactive for each model variant.
_INACTIVE: Dict[str, set] = {
    "Model0": {"Active LP", "JIT"},
    "Model1": {"JIT"},
    "Model2": set(),
}


def _is_inactive(cohort: str, scenario_label: str) -> bool:
    """Return True if *cohort* is not active in the scenario."""
    for model, inactive_set in _INACTIVE.items():
        if model in scenario_label and cohort in inactive_set:
            return True
    return False


def pnl_summary_table(
    scenario_results: Dict[str, List[Dict[str, Any]]],
    *,
    cohorts: Optional[List[str]] = None,
) -> go.Figure:
    """Build an annotated heatmap of final hedged PnL across scenarios.

    Parameters
    ----------
    scenario_results : dict
        Maps a scenario label (e.g. ``"Model0 — static"``) to the list of
        per-seed output dicts returned by ``common.run_multi_seed``.
    cohorts : list[str] or None
        Subset of ``{"Passive LP", "Active LP", "JIT"}``.  ``None`` → all.

    Returns
    -------
    go.Figure
    """
    if cohorts is None:
        cohorts = list(_COHORT_KEYS.keys())

    scenario_labels = list(scenario_results.keys())
    z_vals: List[List[float]] = []
    annotations: List[List[str]] = []

    for cohort in cohorts:
        row_z: List[float] = []
        row_ann: List[str] = []
        key = _COHORT_KEYS[cohort]
        for label in scenario_labels:
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
                row_ann.append(f"{mu:+.1f} ± {sigma:.1f}")
        z_vals.append(row_z)
        annotations.append(row_ann)

    z_arr = np.array(z_vals, dtype=float)
    # Symmetric color scale centered at 0
    abs_max = float(np.nanmax(np.abs(z_arr))) if z_arr.size else 1.0
    abs_max = max(abs_max, 1e-6)

    fig = go.Figure(data=go.Heatmap(
        z=z_arr,
        x=scenario_labels,
        y=cohorts,
        colorscale="RdYlGn",
        zmin=-abs_max,
        zmax=abs_max,
        text=annotations,
        texttemplate="%{text}",
        textfont=dict(size=13),
        hovertemplate="Scenario: %{x}<br>Cohort: %{y}<br>Mean PnL: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Mean hedged PnL<br>(token-1)"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Hedged PnL summary (mean ± σ across seeds)",
        xaxis=dict(title="Scenario", tickangle=-35, tickfont=dict(size=14)),
        yaxis=dict(title="LP Cohort", tickfont=dict(size=16)),
        font=FONT,
        height=350 + 60 * len(cohorts),
    )
    return fig
