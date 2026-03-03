"""scripts/analysis/welfare_decomposition.py — Agent surplus breakdown.

Produces a grouped / stacked bar chart showing the final PnL of every agent
type for each scenario, making the welfare transfer visible in a single glance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from scripts.analysis.common import (
    COLORS, FONT, PLOTLY_TEMPLATE, final_values,
)

# (display name, simulate() key, color)
_AGENT_SPECS = [
    ("Arbitrageur",   "arb_pnl_cum",           COLORS["arb"]),
    ("Smart router",  "smart_router_pnl_cum",   COLORS["smart_router"]),
    ("Noise trader",  "noise_trader_pnl_cum",   COLORS["noise_trader"]),
    ("Passive LP",    "lp_pnl_passive",         COLORS["passive_lp"]),
    ("Active LP",     "lp_pnl_active",          COLORS["active_lp"]),
    ("JIT",           "jiter_pnl_series",        COLORS["jiter"]),
]


def welfare_breakdown(
    scenario_results: Dict[str, List[Dict[str, Any]]],
    *,
    agents: Optional[List[str]] = None,
) -> go.Figure:
    """Grouped bar chart of mean final PnL by agent, per scenario.

    Parameters
    ----------
    scenario_results : dict
        Maps scenario label → list of per-seed output dicts.
    agents : list[str] or None
        Agent display names to include.  ``None`` → auto-detect (skip agents
        whose PnL is all-NaN or whose data is absent).

    Returns
    -------
    go.Figure
    """
    scenario_labels = list(scenario_results.keys())

    fig = go.Figure()

    for agent_name, key, color in _AGENT_SPECS:
        if agents is not None and agent_name not in agents:
            continue

        means: List[float] = []
        errors: List[float] = []
        has_data = False

        for label in scenario_labels:
            vals = final_values(scenario_results[label], key)
            if vals.size > 0:
                has_data = True
                means.append(float(np.mean(vals)))
                errors.append(float(np.std(vals)))
            else:
                means.append(float("nan"))
                errors.append(0.0)

        if not has_data:
            continue

        fig.add_trace(go.Bar(
            x=scenario_labels,
            y=means,
            name=agent_name,
            marker_color=color,
            error_y=dict(type="data", array=errors, visible=True),
        ))

    fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1))

    fig.update_layout(
        barmode="group",
        template=PLOTLY_TEMPLATE,
        title="Welfare decomposition — mean final PnL by agent (± σ)",
        xaxis_title="Scenario",
        yaxis_title="Final PnL (token-1)",
        font=FONT,
        legend=dict(font=dict(size=13)),
        height=550,
    )
    return fig
