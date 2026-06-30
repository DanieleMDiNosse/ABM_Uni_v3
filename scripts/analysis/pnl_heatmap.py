"""scripts/analysis/pnl_heatmap.py — Cross-scenario PnL comparison table.

Produces a single heatmap / annotated table showing final hedged PnL
(mean ± SEM across seeds) for every (model, fee_mode, LP cohort) combination.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from scripts.analysis.common import (
    COLORS, FONT, PLOTLY_TEMPLATE, final_values, save_figure,
)

# Cohort/agent keys in the simulate() output dict.
_COHORT_KEYS = {
    "Passive LP": "lp_pnl_passive",
    "Active LP": "lp_pnl_active",
    "JIT": "jiter_pnl_series",
    "Noise Trader": "noise_trader_pnl_cum",
    "Smart Router": "smart_router_pnl_cum",
}

# Cohorts that are inactive for each model variant.
_INACTIVE: Dict[str, set] = {
    "Model0": {"Active LP", "JIT"},
    "Model1": {"JIT"},
    "Model2": set(),
}

_FEE_LABELS = {
    "static": "static",
    "toxicity": "toxicity",
    "volatility_dex": "vol DEX",
    "volatility_cex": "vol CEX",
}


def _compact_scenario_label(scenario_label: str) -> str:
    """Return a compact, two-line tick label for a paper-width heatmap."""
    if " — " not in scenario_label:
        return scenario_label
    model, fee_mode = scenario_label.split(" — ", 1)
    model = model.replace("Model", "M")
    fee_mode = _FEE_LABELS.get(fee_mode, fee_mode.replace("_", " "))
    return f"{model}<br>{fee_mode}"


def _annotation_text(mu: float, sem: float) -> str:
    """Format mean and standard error compactly while keeping both visible."""
    return f"{mu:+.1f}<br>±{sem:.1f}"


def _annotation_color(value: float, abs_max: float) -> str:
    """Use white text on saturated cells and black text near the neutral center."""
    if not np.isfinite(value) or abs_max <= 0:
        return "black"
    return "white" if abs(value) / abs_max >= 0.58 else "black"


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
        Subset of ``{"Passive LP", "Active LP", "JIT", "Noise Trader",
        "Smart Router"}``.  ``None`` → all.

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
                sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                row_z.append(mu)
                row_ann.append(_annotation_text(mu, sem))
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
      # remove heatmap text rendering (we’ll replace it with annotations)
      text=None,
      texttemplate=None,
      hovertemplate="Scenario: %{x}<br>Cohort: %{y}<br>Mean PnL: %{z:.2f}<extra></extra>",
      colorbar=dict(
          title=dict(text="Mean hedged PnL<br>(token-1)", font=dict(size=20)),
          tickfont=dict(size=18),
      ),
  ))

    for i, cohort in enumerate(cohorts):
        for j, scen in enumerate(scenario_labels):
            fig.add_annotation(
                x=scen, y=cohort,
                xref="x", yref="y",
                text=str(annotations[i][j]),
                showarrow=False,
                textangle=0,
                font=dict(size=20, color=_annotation_color(z_arr[i, j], abs_max)),
                xanchor="center", yanchor="middle",
                align="center",
            )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=dict(size=22, color="black"),
        margin=dict(l=145, r=145, t=30, b=120),
    )
    fig.update_xaxes(
        tickangle=0,
        tickfont=dict(size=18, color="black"),
        tickvals=scenario_labels,
        ticktext=[_compact_scenario_label(label) for label in scenario_labels],
        automargin=True,
    )
    fig.update_yaxes(tickfont=dict(size=22, color="black"), automargin=True)
    return fig
