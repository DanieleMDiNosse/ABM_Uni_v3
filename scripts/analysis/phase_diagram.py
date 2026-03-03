"""scripts/analysis/phase_diagram.py — 2-D positive-PnL phase diagram.

Sweeps two parameters (e.g. block time *B* and fee-controller gain *k*)
and produces a heatmap showing where mean hedged PnL is positive / negative.
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


def phase_diagram(
    base_params: Dict[str, Any],
    param_x: str,
    values_x: List[float],
    param_y: str,
    values_y: List[float],
    *,
    pnl_key: str = "lp_pnl_passive",
    n_seeds: int = 5,
    seed_base: int = 1,
    max_workers: int = 4,
) -> go.Figure:
    """2-D heatmap of mean hedged PnL in (*param_x*, *param_y*) space.

    Parameters
    ----------
    base_params : dict
        Baseline ``simulate()`` kwargs.
    param_x, param_y : str
        Parameters to sweep (columns / rows of the heatmap).
    values_x, values_y : list[float]
        Grid values for each axis.
    pnl_key : str
        Output key for hedged PnL to evaluate.
    n_seeds : int
        Seeds per grid point.

    Returns
    -------
    go.Figure
    """
    nx, ny = len(values_x), len(values_y)
    z = np.full((ny, nx), float("nan"))

    for iy, vy in enumerate(values_y):
        for ix, vx in enumerate(values_x):
            params = dict(base_params)
            params[param_x] = vx
            params[param_y] = vy
            tmp = Path(f"/tmp/abm_phase_{param_x}{vx}_{param_y}{vy}_{os.getpid()}")
            results = run_multi_seed(
                params, n_seeds,
                seed_base=seed_base, max_workers=max_workers, tmp_root=tmp,
            )
            finals = final_values(results, pnl_key)
            if finals.size > 0:
                z[iy, ix] = float(np.mean(finals))

    abs_max = float(np.nanmax(np.abs(z))) if np.any(np.isfinite(z)) else 1.0
    abs_max = max(abs_max, 1e-6)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{v:.4g}" for v in values_x],
        y=[f"{v:.4g}" for v in values_y],
        colorscale="RdYlGn",
        zmid=0,
        zmin=-abs_max,
        zmax=abs_max,
        colorbar=dict(title="Mean PnL"),
        hovertemplate=f"{param_x}: %{{x}}<br>{param_y}: %{{y}}<br>PnL: %{{z:.2f}}<extra></extra>",
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Phase diagram: hedged PnL in ({param_x}, {param_y}) space",
        xaxis_title=param_x,
        yaxis_title=param_y,
        font=FONT,
        height=600,
        width=800,
    )
    return fig
