"""scripts/analysis/sensitivity_sweep.py — 1-D parameter sweep.

Sweeps a single controller parameter (e.g. ``k_basis`` for toxicity gain
or ``k_sigma`` for volatility gain) and plots mean final hedged PnL
for passive and active LPs as a function of that parameter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from scripts.analysis.common import (
    COLORS, FONT, PLOTLY_TEMPLATE, final_values, run_multi_seed,
)


def sensitivity_sweep(
    base_params: Dict[str, Any],
    param_name: str,
    param_values: List[float],
    *,
    n_seeds: int = 10,
    seed_base: int = 1,
    max_workers: int = 4,
    cohorts: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """Sweep *param_name* over *param_values* and plot mean hedged PnL.

    Parameters
    ----------
    base_params : dict
        Baseline ``simulate()`` kwargs.
    param_name : str
        The parameter to sweep (e.g. ``"k_basis"``, ``"k_sigma"``).
    param_values : list[float]
        Values to test.
    n_seeds : int
        Seeds per point.
    cohorts : dict or None
        Maps display name → simulate output key.
        Default: ``{"Passive LP": "lp_pnl_passive", "Active LP": "lp_pnl_active"}``.

    Returns
    -------
    go.Figure
    """
    if cohorts is None:
        cohorts = {
            "Passive LP": "lp_pnl_passive",
            "Active LP": "lp_pnl_active",
        }

    # Storage: {cohort_name: {"mean": [...], "std": [...]}}
    data: Dict[str, Dict[str, List[float]]] = {
        name: {"mean": [], "std": []} for name in cohorts
    }

    for val in param_values:
        params = dict(base_params)
        params[param_name] = val
        tmp = Path(f"/tmp/abm_sweep_{param_name}_{val}_{os.getpid()}")
        results = run_multi_seed(
            params, n_seeds,
            seed_base=seed_base, max_workers=max_workers, tmp_root=tmp,
        )
        for name, key in cohorts.items():
            finals = final_values(results, key)
            data[name]["mean"].append(float(np.mean(finals)) if finals.size else float("nan"))
            data[name]["std"].append(float(np.std(finals)) if finals.size else 0.0)

    x = [float(v) for v in param_values]
    colors = list(COLORS.values())
    fig = go.Figure()

    for i, (name, d) in enumerate(data.items()):
        mu = np.array(d["mean"])
        sigma = np.array(d["std"])
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=x, y=mu + sigma, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=mu - sigma, mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=mu, mode="lines+markers", name=name,
            line=dict(color=color, width=2),
        ))

    fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Sensitivity: mean hedged PnL vs {param_name} ({n_seeds} seeds/point)",
        xaxis_title=param_name,
        yaxis_title="Final hedged PnL (token-1)",
        font=FONT,
        height=550,
    )
    return fig
