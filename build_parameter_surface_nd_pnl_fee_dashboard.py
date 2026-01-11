#!/usr/bin/env python3
"""
Build a standalone interactive HTML dashboard from a cached ND grid CSV.

Inputs:
  - Cached CSV produced by `run_parameter_surface_nd_pnl_fee_dashboard.py`
  - Optional metadata JSON (written alongside the CSV by the runner)

Output:
  - A single HTML file (Plotly loaded via CDN) with:
      * a 3D surface of a cached PnL summary (select X/Y axes + metric)
      * sliders for non-axis parameters
      * a fee histogram for the selected surface point (click-to-update)
      * a smart-router DEX share histogram for the selected surface point (click-to-update)

This script exists to keep the expensive grid computation separate from the (iterable) dashboard UI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import run as run_module
from utils import load_simulation_parameters


BASE_CONFIG_PATH = Path("abm_results/scenarios/test.yml")


PNL_METRICS: Tuple[Tuple[str, str], ...] = (
    ("lp_pnl_passive", "Passive LP hedged PnL"),
    ("lp_pnl_active", "Active LP hedged PnL"),
    ("arb_pnl_cum", "Arbitrageur PnL"),
    ("noise_trader_pnl_cum", "Noise trader PnL"),
)


INT_PARAMS: set[str] = {
    "narrow_mints_per_block",
    "passive_mints_per_block",
    "noise_trades_per_block",
    "passive_burns_per_block",
}


def _infer_tag_from_cache_path(cache_path: Path) -> str:
    name = cache_path.name
    if name.startswith("grid_") and name.endswith(".csv"):
        return name[len("grid_") : -len(".csv")]
    return cache_path.stem


def _infer_fee_hist_bins(columns: Sequence[str]) -> int:
    indices: List[int] = []
    prefix = "fee_hist_"
    for col in columns:
        if not col.startswith(prefix):
            continue
        tail = col[len(prefix) :]
        if not tail.isdigit():
            continue
        indices.append(int(tail))
    if not indices:
        raise SystemExit("Cache CSV has no fee histogram columns (fee_hist_0, fee_hist_1, ...).")
    return int(max(indices) + 1)


def _infer_smart_router_dex_share_hist_bins(columns: Sequence[str]) -> int:
    indices: List[int] = []
    prefix = "smart_router_dex_share_hist_"
    for col in columns:
        if not col.startswith(prefix):
            continue
        tail = col[len(prefix) :]
        if not tail.isdigit():
            continue
        indices.append(int(tail))
    if not indices:
        raise SystemExit(
            "Cache CSV has no smart-router DEX share histogram columns "
            "(smart_router_dex_share_hist_0, smart_router_dex_share_hist_1, ...)."
        )
    return int(max(indices) + 1)


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.floating):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def _infer_param_order_from_columns(columns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for col in columns:
        if not col.startswith("i__"):
            continue
        name = col[len("i__") :]
        if not name:
            continue
        out.append(name)
    if len(out) < 2:
        raise SystemExit("Need at least 2 swept parameters (expected i__<param> columns) to build a surface.")
    return out


def _infer_sweeps_from_cache(dataframe: pd.DataFrame, *, param_order: Sequence[str]) -> Dict[str, List[float | int]]:
    sweeps: Dict[str, List[float | int]] = {}
    for name in param_order:
        idx_col = f"i__{name}"
        val_col = f"v__{name}"
        if idx_col not in dataframe.columns or val_col not in dataframe.columns:
            raise SystemExit(f"Cache is missing required columns: {idx_col}, {val_col}")
        subset = dataframe[[idx_col, val_col]].dropna()
        subset = subset.drop_duplicates(subset=[idx_col], keep="last").sort_values(by=idx_col, kind="mergesort")
        values: List[float | int] = []
        for _, row in subset.iterrows():
            v = row[val_col]
            if name in INT_PARAMS:
                values.append(int(round(float(v))))
            else:
                values.append(float(v))
        if not values:
            raise SystemExit(f"Could not infer sweep values for '{name}' from cache.")
        sweeps[name] = values
    return sweeps


def _default_selection_indices(
    *,
    base_params: Mapping[str, Any],
    param_order: Sequence[str],
    sweep_values: Mapping[str, Sequence[float | int]],
    int_params: set[str],
) -> Dict[str, int]:
    defaults: Dict[str, int] = {}
    for name in param_order:
        values = list(sweep_values[name])
        base_value = base_params.get(name)
        if base_value is None:
            defaults[name] = 0
            continue
        if name in int_params:
            try:
                target = int(round(float(base_value)))
            except Exception:
                defaults[name] = 0
                continue
            best = 0
            best_dist = None
            for idx, v in enumerate(values):
                vi = int(round(float(v)))
                dist = abs(vi - target)
                if best_dist is None or dist < best_dist:
                    best = idx
                    best_dist = dist
            defaults[name] = int(best)
        else:
            try:
                target_f = float(base_value)
            except Exception:
                defaults[name] = 0
                continue
            best = 0
            best_dist = None
            for idx, v in enumerate(values):
                vf = float(v)
                dist = abs(vf - target_f)
                if best_dist is None or dist < best_dist:
                    best = idx
                    best_dist = dist
            defaults[name] = int(best)
    return defaults


def _write_dashboard_html(
    output_path: Path,
    *,
    title: str,
    scenario_label: str,
    config_path: Path,
    pnl_summary_label: str,
    param_order: Sequence[str],
    sweep_values: Mapping[str, Sequence[float | int]],
    int_params: Sequence[str],
    metrics: Sequence[Mapping[str, str]],
    default_indices: Mapping[str, int],
    fee_bin_edges: Sequence[float],
    smart_router_dex_share_bin_edges: Optional[Sequence[float]],
    records_idx: Sequence[Sequence[int]],
    records_pnl: Sequence[Sequence[float]],
    records_pnl_p10: Optional[Sequence[Sequence[float]]],
    records_pnl_p90: Optional[Sequence[Sequence[float]]],
    records_pnl_p_loss: Optional[Sequence[Sequence[float]]],
    records_fee_mean: Sequence[float],
    records_fee_median: Sequence[float],
    records_fee_hist: Sequence[Sequence[int]],
    records_smart_router_dex_share_mean: Optional[Sequence[float]],
    records_smart_router_dex_share_median: Optional[Sequence[float]],
    records_smart_router_dex_share_hist: Optional[Sequence[Sequence[int]]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "title": title,
        "scenario_label": scenario_label,
        "config_path": str(config_path),
        "pnl_summary": {"label": str(pnl_summary_label)},
        "param_order": list(param_order),
        "param_values": {k: list(v) for k, v in sweep_values.items()},
        "int_params": list(int_params),
        "metrics": list(metrics),
        "defaults": {k: int(v) for k, v in default_indices.items()},
        "fee_bin_edges": list(map(float, fee_bin_edges)),
        "smart_router_dex_share_bin_edges": None
        if smart_router_dex_share_bin_edges is None
        else list(map(float, smart_router_dex_share_bin_edges)),
        "records": {
            "idx": [list(map(int, row)) for row in records_idx],
            "pnl": [list(map(float, row)) for row in records_pnl],
            "pnl_p10": None if records_pnl_p10 is None else [list(map(float, row)) for row in records_pnl_p10],
            "pnl_p90": None if records_pnl_p90 is None else [list(map(float, row)) for row in records_pnl_p90],
            "pnl_p_loss": None if records_pnl_p_loss is None else [list(map(float, row)) for row in records_pnl_p_loss],
            "fee_mean": list(map(float, records_fee_mean)),
            "fee_median": list(map(float, records_fee_median)),
            "fee_hist": [list(map(int, row)) for row in records_fee_hist],
            "smart_router_dex_share_mean": None
            if records_smart_router_dex_share_mean is None
            else list(map(float, records_smart_router_dex_share_mean)),
            "smart_router_dex_share_median": None
            if records_smart_router_dex_share_median is None
            else list(map(float, records_smart_router_dex_share_median)),
            "smart_router_dex_share_hist": None
            if records_smart_router_dex_share_hist is None
            else [list(map(int, row)) for row in records_smart_router_dex_share_hist],
        },
    }

    payload = _sanitize_for_json(payload)
    data_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link
    href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap"
    rel="stylesheet"
  />
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --bg: #0b0f14;
      --bg-elev: #0f172a;
      --panel: #0f1621;
      --panel-border: #1f2937;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-soft: #0ea5e9;
      --bg-grad-1: #101826;
      --bg-grad-2: #0b0f14;
      --bg-grad-3: #070a0d;
    }}
    [data-theme="light"] {{
      --bg: #f6f7fb;
      --bg-elev: #ffffff;
      --panel: #ffffff;
      --panel-border: #e2e8f0;
      --text: #0f172a;
      --muted: #475569;
      --accent: #1d4ed8;
      --accent-soft: #2563eb;
      --bg-grad-1: #f0f4ff;
      --bg-grad-2: #f6f7fb;
      --bg-grad-3: #ffffff;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      font-family: "Space Grotesk", "SF Pro Display", "Segoe UI", system-ui, sans-serif;
      margin: 0;
      padding: 0;
      background:
        radial-gradient(900px circle at 10% 10%, var(--bg-grad-1) 0%, var(--bg-grad-2) 55%, var(--bg-grad-3) 100%);
      color: var(--text);
    }}
    header {{
      padding: 18px 22px;
      background: rgba(10, 14, 20, 0.85);
      border-bottom: 1px solid var(--panel-border);
      backdrop-filter: blur(6px);
    }}
    [data-theme="light"] header {{
      background: rgba(255, 255, 255, 0.85);
    }}
    header h1 {{
      font-size: 18px;
      margin: 0 0 4px 0;
      font-weight: 600;
      letter-spacing: 0.2px;
    }}
    header .meta {{
      font-size: 12px;
      color: var(--muted);
    }}
    header code {{
      color: var(--accent);
    }}
    .container {{
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 16px;
      padding: 16px;
    }}
    .panel {{
      background: linear-gradient(180deg, rgba(15, 22, 33, 0.98), rgba(12, 18, 28, 0.98));
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 10px 30px rgba(2, 6, 12, 0.35);
    }}
    .panel-tall {{
      min-height: 720px;
    }}
    [data-theme="light"] .panel {{
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.98));
      box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
    }}
    .panel h2 {{
      font-size: 13px;
      margin: 0 0 10px 0;
      color: var(--text);
      font-weight: 600;
      letter-spacing: 0.2px;
    }}
    .controls {{
      display: grid;
      gap: 10px;
    }}
    .row {{
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 10px;
      align-items: center;
    }}
    label {{
      font-size: 12px;
      color: var(--muted);
    }}
    select, input[type="range"] {{
      width: 100%;
    }}
    select {{
      appearance: none;
      background: var(--bg-elev);
      color: var(--text);
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 6px 10px;
      font-size: 12px;
    }}
    select:focus {{
      outline: 2px solid rgba(56, 189, 248, 0.35);
      border-color: var(--accent);
    }}
    input[type="range"] {{
      accent-color: var(--accent-soft);
    }}
    .slider-meta {{
      font-size: 12px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }}
    .plot {{
      width: 100%;
      height: 620px;
      border-radius: 12px;
    }}
    .plot-small {{
      width: 100%;
      height: 420px;
      border-radius: 12px;
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: minmax(360px, 1.6fr) minmax(260px, 1fr);
      gap: 14px;
      align-items: start;
    }}
    .stats-box {{
      margin-top: 10px;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--panel-border);
      background: rgba(2, 6, 12, 0.18);
    }}
    [data-theme="light"] .stats-box {{
      background: rgba(226, 232, 240, 0.55);
    }}
    .stats-title {{
      font-size: 12px;
      color: var(--muted);
      margin: 0 0 8px 0;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }}
    .stat {{
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(15, 23, 42, 0.25);
      border: 1px solid rgba(148, 163, 184, 0.16);
    }}
    [data-theme="light"] .stat {{
      background: rgba(255, 255, 255, 0.7);
      border: 1px solid rgba(15, 23, 42, 0.12);
    }}
    .stat .k {{
      font-size: 11px;
      color: var(--muted);
    }}
    .stat .v {{
      margin-top: 2px;
      font-size: 13px;
      font-weight: 600;
      color: var(--text);
    }}
    .hint {{
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
      line-height: 1.35;
    }}
    .warning {{
      margin: 12px 0 10px 0;
      padding: 10px 12px;
      border-radius: 10px;
      background: rgba(56, 189, 248, 0.08);
      border: 1px solid rgba(56, 189, 248, 0.35);
      color: var(--text);
      font-size: 12px;
      line-height: 1.35;
    }}
    [data-theme="light"] .warning {{
      background: rgba(37, 99, 235, 0.08);
      border: 1px solid rgba(37, 99, 235, 0.3);
    }}
    .footer {{
      font-size: 12px;
      color: var(--muted);
      padding: 0 12px 14px 12px;
    }}
    @media (max-width: 1100px) {{
      .container {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 1200px) {{
      .plot-grid {{ grid-template-columns: 1fr; }}
      .stats-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    }}
    @media (max-width: 700px) {{
      .stats-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="meta">Scenario: <b>{scenario_label}</b> • Config: <code>{config_path}</code></div>
  </header>
  <div class="container">
    <div class="panel panel-tall">
      <h2>Controls</h2>
      <div class="controls">
        <div class="row">
          <label for="themeSelect">Theme</label>
          <select id="themeSelect">
            <option value="dark">Dark</option>
            <option value="light">Light</option>
          </select>
        </div>
        <div class="row">
          <label for="xAxis">X axis</label>
          <select id="xAxis"></select>
        </div>
        <div class="row">
          <label for="yAxis">Y axis</label>
          <select id="yAxis"></select>
        </div>
        <div class="row">
          <label for="metric">PnL metric</label>
          <select id="metric"></select>
        </div>
        <div id="sliders"></div>
        <div class="hint">
          - Surface shows <b>{pnl_summary_label}</b> over repeated runs.<br/>
          - Click a point on the surface to update the <b>fee distribution</b> and <b>smart-router DEX share</b> for that exact parameter combo.
        </div>
      </div>
    </div>
    <div class="panel panel-tall">
      <div class="plot-grid">
        <section>
          <h2>PnL Surface</h2>
          <div id="pnlSurface" class="plot"></div>
          <div id="seedWarning" class="warning" style="display: none;">
            <b>Note:</b> with <code>passive_lp_share = 1</code>, narrow mints are inactive. Any
            surface changes you see while varying <code>narrow_mints_per_block</code> come from
            random seeding across runs.
          </div>
        </section>
        <section>
          <h2>Fee Distribution (Selected Point)</h2>
          <div id="feeHist" class="plot-small"></div>
          <div id="feePercentilesBox" class="stats-box">
            <div class="stats-title">Fee percentiles (empirical)</div>
            <div class="stats-grid">
              <div class="stat">
                <div class="k">P25</div>
                <div class="v" id="feeP25">—</div>
              </div>
              <div class="stat">
                <div class="k">P50</div>
                <div class="v" id="feeP50">—</div>
              </div>
              <div class="stat">
                <div class="k">P75</div>
                <div class="v" id="feeP75">—</div>
              </div>
              <div class="stat">
                <div class="k">P95</div>
                <div class="v" id="feeP95">—</div>
              </div>
              <div class="stat">
                <div class="k">P99</div>
                <div class="v" id="feeP99">—</div>
              </div>
            </div>
          </div>
          <div id="srDexShareWrapper" style="margin-top: 14px;">
            <h2>Smart-Router DEX Share (Selected Point)</h2>
            <div id="srDexShareHist" class="plot-small"></div>
          </div>
        </section>
      </div>
    </div>
  </div>
  <div class="footer">
    Generated by <code>build_parameter_surface_nd_pnl_fee_dashboard.py</code>.
  </div>

  <script id="grid-data" type="application/json">{data_json}</script>
  <script>
    const GRID = JSON.parse(document.getElementById("grid-data").textContent);

    const PARAM_ORDER = GRID.param_order;
    const PARAM_VALUES = GRID.param_values;
    const INT_PARAMS = new Set(GRID.int_params || []);
    const METRICS = GRID.metrics;
    const DEFAULTS = GRID.defaults || {{}};
    const FEE_EDGES = GRID.fee_bin_edges;
    const FEE_CENTERS = (() => {{
      const centers = [];
      for (let i = 0; i < FEE_EDGES.length - 1; i++) {{
        centers.push(0.5 * (FEE_EDGES[i] + FEE_EDGES[i + 1]));
      }}
      return centers;
    }})();

	    const REC_IDX = GRID.records.idx;
	    const REC_PNL = GRID.records.pnl;
	    const PNL_SUMMARY_LABEL = GRID?.pnl_summary?.label ? String(GRID.pnl_summary.label) : "median final PnL";
	    const REC_PNL_P10 = GRID.records.pnl_p10;
	    const REC_PNL_P90 = GRID.records.pnl_p90;
	    const REC_PNL_P_LOSS = GRID.records.pnl_p_loss;
	    const HAS_PNL_BANDS =
	      Array.isArray(REC_PNL_P10) && Array.isArray(REC_PNL_P90) && Array.isArray(REC_PNL_P_LOSS);
	    const REC_FEE_MEAN = GRID.records.fee_mean;
	    const REC_FEE_MEDIAN = GRID.records.fee_median;
	    const REC_FEE_HIST = GRID.records.fee_hist;
	    const SR_EDGES = GRID.smart_router_dex_share_bin_edges;
	    const REC_SR_DEX_SHARE_MEAN = GRID.records.smart_router_dex_share_mean;
	    const REC_SR_DEX_SHARE_MEDIAN = GRID.records.smart_router_dex_share_median;
	    const REC_SR_DEX_SHARE_HIST = GRID.records.smart_router_dex_share_hist;
	    const HAS_SR_DEX_SHARE_HIST =
	      Array.isArray(SR_EDGES) &&
	      SR_EDGES.length > 1 &&
	      Array.isArray(REC_SR_DEX_SHARE_HIST) &&
	      Array.isArray(REC_SR_DEX_SHARE_MEAN) &&
	      Array.isArray(REC_SR_DEX_SHARE_MEDIAN);
	    const SR_CENTERS = (() => {{
	      if (!HAS_SR_DEX_SHARE_HIST) return [];
	      const centers = [];
	      for (let i = 0; i < SR_EDGES.length - 1; i++) {{
	        centers.push(0.5 * (SR_EDGES[i] + SR_EDGES[i + 1]));
	      }}
	      return centers;
	    }})();
    const THEMES = {{
      dark: {{
        bg: "#0b0f14",
        grid: "#1f2937",
        text: "#e2e8f0",
        muted: "#94a3b8",
        accent: "#38bdf8",
        accentSoft: "#0ea5e9",
        mean: "#f59e0b",
        median: "#22c55e",
        template: "plotly_dark",
        colorscale: "Cividis",
        axisText: "#e2e8f0",
        legendText: "#94a3b8",
      }},
      light: {{
        bg: "#ffffff",
        grid: "#cbd5e1",
        text: "#0f172a",
        muted: "#475569",
        accent: "#1d4ed8",
        accentSoft: "#2563eb",
        mean: "#b45309",
        median: "#22c55e",
        template: "plotly_white",
        colorscale: "Viridis",
        axisText: "#000000",
        legendText: "#000000",
      }},
    }};
    let currentTheme = THEMES.dark;

    const N_PARAMS = PARAM_ORDER.length;
    const PARAM_LENGTHS = PARAM_ORDER.map(p => (PARAM_VALUES[p] || []).length);
    const STRIDES = Array(N_PARAMS).fill(1);
    let totalSize = 1;
    for (let i = N_PARAMS - 1; i >= 0; i--) {{
      STRIDES[i] = totalSize;
      totalSize *= PARAM_LENGTHS[i];
    }}

    // Map idxRow -> record index. Dense array is fast but can be huge for large grids,
    // so fall back to a sparse Map above a threshold.
    const DENSE_LIMIT = 5_000_000; // ~20MB Int32Array
    const USE_DENSE = Number.isFinite(totalSize) && totalSize > 0 && totalSize <= DENSE_LIMIT;
    let FLAT_TO_REC = null;
    let FLAT_MAP = null;
    if (USE_DENSE) {{
      FLAT_TO_REC = new Int32Array(totalSize);
      FLAT_TO_REC.fill(-1);
      for (let r = 0; r < REC_IDX.length; r++) {{
        const idxRow = REC_IDX[r];
        let flat = 0;
        for (let j = 0; j < N_PARAMS; j++) {{
          flat += idxRow[j] * STRIDES[j];
        }}
        FLAT_TO_REC[flat] = r;
      }}
    }} else {{
      FLAT_MAP = new Map();
      for (let r = 0; r < REC_IDX.length; r++) {{
        const idxRow = REC_IDX[r];
        let flat = 0;
        for (let j = 0; j < N_PARAMS; j++) {{
          flat += idxRow[j] * STRIDES[j];
        }}
        FLAT_MAP.set(flat, r);
      }}
    }}

    function getRecordIndex(idxRow) {{
      // Validate idxRow: finite ints within bounds.
      let flat = 0;
      for (let j = 0; j < N_PARAMS; j++) {{
        const idx = idxRow[j];
        if (!Number.isFinite(idx)) return -1;
        const idxInt = Math.trunc(idx);
        if (idxInt !== idx) return -1;
        if (idxInt < 0 || idxInt >= PARAM_LENGTHS[j]) return -1;
        flat += idxInt * STRIDES[j];
      }}
      if (FLAT_TO_REC) {{
        return FLAT_TO_REC[flat];
      }}
      const v = FLAT_MAP.get(flat);
      return (v === undefined) ? -1 : v;
    }}

    function formatValue(param, value) {{
      if (INT_PARAMS.has(param)) return String(Math.round(value));
      const v = Number(value);
      if (!Number.isFinite(v)) return String(value);
      const abs = Math.abs(v);
      if (abs >= 1000 || (abs > 0 && abs < 1e-3)) return v.toExponential(3);
      return v.toPrecision(6).replace(/0+$/,'').replace(/\\.$/,'');
    }}

    const state = {{
      xParam: PARAM_ORDER[0],
      yParam: PARAM_ORDER.length > 1 ? PARAM_ORDER[1] : PARAM_ORDER[0],
      metricIndex: 0,
      selected: {{}},           // param -> idx (we store for all)
      selectedPoint: null,      // {{ xIdx, yIdx }}
    }};

    for (const p of PARAM_ORDER) {{
      const maxIdx = Math.max(0, (PARAM_VALUES[p] || []).length - 1);
      const d = (DEFAULTS[p] !== undefined) ? DEFAULTS[p] : 0;
      state.selected[p] = Math.max(0, Math.min(maxIdx, Number(d)));
    }}

    const xAxisEl = document.getElementById("xAxis");
    const yAxisEl = document.getElementById("yAxis");
    const metricEl = document.getElementById("metric");
    const themeEl = document.getElementById("themeSelect");
    const slidersEl = document.getElementById("sliders");
    const seedWarningEl = document.getElementById("seedWarning");
    const feePercentilesBoxEl = document.getElementById("feePercentilesBox");
    const feeP25El = document.getElementById("feeP25");
    const feeP50El = document.getElementById("feeP50");
    const feeP75El = document.getElementById("feeP75");
    const feeP95El = document.getElementById("feeP95");
    const feeP99El = document.getElementById("feeP99");
    const srDexShareWrapperEl = document.getElementById("srDexShareWrapper");

    function populateSelect(el, options, selectedValue) {{
      el.innerHTML = "";
      for (const opt of options) {{
        const o = document.createElement("option");
        o.value = opt.value;
        o.textContent = opt.label;
        el.appendChild(o);
      }}
      el.value = selectedValue;
    }}

    let pnlSurfaceInitialized = false;
    let feeHistInitialized = false;
    let srDexShareHistInitialized = false;
    let pendingSurfaceRAF = null;
    let pendingFeeRAF = null;
    let pendingMarkerRAF = null;

    function histogramQuantile(counts, edges, q) {{
      if (!counts || !edges) return NaN;
      const n = counts.length;
      if (edges.length !== n + 1) return NaN;
      const qq = Math.min(1, Math.max(0, Number(q)));
      if (!Number.isFinite(qq)) return NaN;

      let total = 0;
      for (let i = 0; i < n; i++) {{
        const c = Number(counts[i]);
        if (Number.isFinite(c) && c > 0) total += c;
      }}
      if (!(total > 0)) return NaN;

      const target = qq * total;
      let cum = 0;
      for (let i = 0; i < n; i++) {{
        const c = Number(counts[i]);
        if (!Number.isFinite(c) || c <= 0) continue;
        const next = cum + c;
        if (next >= target) {{
          const left = Number(edges[i]);
          const right = Number(edges[i + 1]);
          if (!Number.isFinite(left) || !Number.isFinite(right)) return NaN;
          const frac = (target - cum) / c;
          const t = Math.min(1, Math.max(0, frac));
          return left + t * (right - left);
        }}
        cum = next;
      }}
      const last = Number(edges[edges.length - 1]);
      return Number.isFinite(last) ? last : NaN;
    }}

    function setPercentileValue(el, value) {{
      if (!el) return;
      el.textContent = Number.isFinite(value) ? formatValue("fee", value) : "—";
    }}

    function clearFeePercentiles() {{
      setPercentileValue(feeP25El, NaN);
      setPercentileValue(feeP50El, NaN);
      setPercentileValue(feeP75El, NaN);
      setPercentileValue(feeP95El, NaN);
      setPercentileValue(feeP99El, NaN);
      if (feePercentilesBoxEl) feePercentilesBoxEl.style.display = "none";
    }}

    function updateFeePercentiles(counts, feeMedian) {{
      const p25 = histogramQuantile(counts, FEE_EDGES, 0.25);
      const p75 = histogramQuantile(counts, FEE_EDGES, 0.75);
      const p95 = histogramQuantile(counts, FEE_EDGES, 0.95);
      const p99 = histogramQuantile(counts, FEE_EDGES, 0.99);
      const p50 = Number.isFinite(feeMedian) ? Number(feeMedian) : histogramQuantile(counts, FEE_EDGES, 0.5);

      setPercentileValue(feeP25El, p25);
      setPercentileValue(feeP50El, p50);
      setPercentileValue(feeP75El, p75);
      setPercentileValue(feeP95El, p95);
      setPercentileValue(feeP99El, p99);
      if (feePercentilesBoxEl) feePercentilesBoxEl.style.display = "block";
    }}

    function applyTheme(name) {{
      const themeName = THEMES[name] ? name : "dark";
      currentTheme = THEMES[themeName];
      document.documentElement.dataset.theme = themeName;
      renderSurface();
      if (state.selectedPoint) {{
        renderFeeHist(state.selectedPoint.xIdx, state.selectedPoint.yIdx);
        if (HAS_SR_DEX_SHARE_HIST) {{
          renderSmartRouterDexShareHist(state.selectedPoint.xIdx, state.selectedPoint.yIdx);
        }}
      }}
    }}

    function scheduleSurfaceRender() {{
      if (pendingSurfaceRAF !== null) return;
      pendingSurfaceRAF = requestAnimationFrame(() => {{
        pendingSurfaceRAF = null;
        renderSurface();
      }});
    }}

    function scheduleFeeRender() {{
      if (!state.selectedPoint) return;
      if (pendingFeeRAF !== null) return;
      pendingFeeRAF = requestAnimationFrame(() => {{
        pendingFeeRAF = null;
        renderFeeHist(state.selectedPoint.xIdx, state.selectedPoint.yIdx);
        if (HAS_SR_DEX_SHARE_HIST) {{
          renderSmartRouterDexShareHist(state.selectedPoint.xIdx, state.selectedPoint.yIdx);
        }}
      }});
    }}

    function scheduleMarkerUpdate() {{
      if (!state.selectedPoint) return;
      if (pendingMarkerRAF !== null) return;
      pendingMarkerRAF = requestAnimationFrame(() => {{
        pendingMarkerRAF = null;
        updateSelectedMarker(state.selectedPoint.xIdx, state.selectedPoint.yIdx);
      }});
    }}

    function selectedParamValue(param) {{
      const values = PARAM_VALUES[param] || [];
      if (!values.length) return null;
      if (param === state.xParam || param === state.yParam) {{
        if (!state.selectedPoint) return null;
        const idx = (param === state.xParam) ? state.selectedPoint.xIdx : state.selectedPoint.yIdx;
        if (!Number.isFinite(idx)) return null;
        const idxInt = Math.trunc(idx);
        if (idxInt < 0 || idxInt >= values.length) return null;
        return Number(values[idxInt]);
      }}
      const idx = state.selected[param];
      if (!Number.isFinite(idx)) return null;
      const idxInt = Math.trunc(idx);
      if (idxInt < 0 || idxInt >= values.length) return null;
      return Number(values[idxInt]);
    }}

    function updateSeedWarning() {{
      if (!seedWarningEl) return;
      const v = selectedParamValue("passive_lp_share");
      const show = Number.isFinite(v) && Math.abs(v - 1) < 1e-9;
      seedWarningEl.style.display = show ? "block" : "none";
    }}

    function rebuildSliders() {{
      slidersEl.innerHTML = "";
      for (const p of PARAM_ORDER) {{
        if (p === state.xParam || p === state.yParam) continue;
        const values = PARAM_VALUES[p] || [];
        if (values.length <= 1) continue;

        const wrapper = document.createElement("div");
        wrapper.style.marginTop = "6px";

        const meta = document.createElement("div");
        meta.className = "slider-meta";
        const left = document.createElement("div");
        left.textContent = p;
        const right = document.createElement("div");
        right.id = `label_${{p}}`;
        right.textContent = formatValue(p, values[state.selected[p]]);
        meta.appendChild(left);
        meta.appendChild(right);

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = 0;
        slider.max = values.length - 1;
        slider.step = 1;
        slider.value = state.selected[p];
        slider.addEventListener("input", () => {{
          state.selected[p] = Number(slider.value);
          right.textContent = formatValue(p, values[state.selected[p]]);
          scheduleSurfaceRender();
          scheduleFeeRender();
          updateSeedWarning();
        }});

        wrapper.appendChild(meta);
        wrapper.appendChild(slider);
        slidersEl.appendChild(wrapper);
      }}
      updateSeedWarning();
    }}

    function currentSelectionIdxRow(xIdx, yIdx) {{
      const row = Array(N_PARAMS).fill(0);
      for (let j = 0; j < N_PARAMS; j++) {{
        const p = PARAM_ORDER[j];
        if (p === state.xParam) row[j] = xIdx;
        else if (p === state.yParam) row[j] = yIdx;
        else row[j] = state.selected[p];
      }}
      return row;
    }}

    function metricLabel() {{
      return METRICS[state.metricIndex]?.label || METRICS[0].label;
    }}

    function computePointValue(xIdx, yIdx) {{
      const idxRow = currentSelectionIdxRow(xIdx, yIdx);
      const rec = getRecordIndex(idxRow);
      let val = NaN;
      if (rec >= 0) {{
        val = REC_PNL[rec][state.metricIndex];
      }}
      return {{ idxRow, rec, val }};
    }}

    function wrapTitleLines(parts, maxLen) {{
      const lines = [];
      let current = "";
      for (const part of parts) {{
        if (!current) {{
          current = part;
          continue;
        }}
        const candidate = `${{current}}, ${{part}}`;
        if (candidate.length > maxLen) {{
          lines.push(current);
          current = part;
        }} else {{
          current = candidate;
        }}
      }}
      if (current) lines.push(current);
      return lines;
    }}

    function updateSelectedMarker(xIdx, yIdx) {{
      if (!pnlSurfaceInitialized) return;
      const xVals = PARAM_VALUES[state.xParam] || [];
      const yVals = PARAM_VALUES[state.yParam] || [];
      if (xIdx < 0 || xIdx >= xVals.length || yIdx < 0 || yIdx >= yVals.length) {{
        Plotly.restyle("pnlSurface", {{ visible: false }}, [1]);
        return;
      }}

      const {{ rec, val }} = computePointValue(xIdx, yIdx);
      if (rec < 0 || !Number.isFinite(val)) {{
        Plotly.restyle("pnlSurface", {{ visible: false }}, [1]);
        return;
      }}

      Plotly.restyle(
        "pnlSurface",
        {{
          x: [[xVals[xIdx]]],
          y: [[yVals[yIdx]]],
          z: [[val]],
          visible: true,
        }},
        [1]
      );
    }}

	    function renderSurface() {{
	      const xVals = PARAM_VALUES[state.xParam] || [];
	      const yVals = PARAM_VALUES[state.yParam] || [];

	      const z = [];
	      const customdata = HAS_PNL_BANDS ? [] : null;
	      let zMin = null;
	      let zMax = null;
	      for (let yi = 0; yi < yVals.length; yi++) {{
	        const row = [];
	        const rowCustom = HAS_PNL_BANDS ? [] : null;
        for (let xi = 0; xi < xVals.length; xi++) {{
          const {{ rec, val }} = computePointValue(xi, yi);
          if (rec >= 0 && Number.isFinite(val)) {{
            zMin = (zMin === null) ? val : Math.min(zMin, val);
            zMax = (zMax === null) ? val : Math.max(zMax, val);
          }}
	          if (HAS_PNL_BANDS) {{
	            let p10 = NaN;
	            let p90 = NaN;
	            let pLoss = NaN;
	            if (rec >= 0) {{
	              p10 = REC_PNL_P10[rec][state.metricIndex];
	              p90 = REC_PNL_P90[rec][state.metricIndex];
	              pLoss = REC_PNL_P_LOSS[rec][state.metricIndex];
	            }}
	            rowCustom.push([p10, p90, pLoss]);
	          }}
	          row.push(val);
	        }}
	        z.push(row);
	        if (HAS_PNL_BANDS) {{
	          customdata.push(rowCustom);
	        }}
	      }}
	      if (zMin === null || zMax === null) {{
	        zMin = 0;
	        zMax = 1;
	      }}

	      const hoverTemplateBase =
	        `${{state.xParam}}=%{{x}}<br>${{state.yParam}}=%{{y}}<br>` +
	        `${{metricLabel()}} (${{PNL_SUMMARY_LABEL}})=%{{z}}`;
	      const hoverTemplate = HAS_PNL_BANDS
	        ? (hoverTemplateBase +
	          `<br>P10=%{{customdata[0]}}<br>P90=%{{customdata[1]}}<br>P(rate&lt;0)=%{{customdata[2]:.1%}}<extra></extra>`)
	        : (hoverTemplateBase + `<extra></extra>`);

	      const surfaceTrace = {{
	        type: "surface",
	        x: xVals,
	        y: yVals,
	        z: z,
	        colorscale: currentTheme.colorscale || "Cividis",
	        cmin: zMin,
	        cmax: zMax,
	        colorbar: {{ title: metricLabel(), len: 0.75 }},
	        hovertemplate: hoverTemplate,
	      }};
	      if (HAS_PNL_BANDS) {{
	        surfaceTrace.customdata = customdata;
	      }}

      // Always include a dedicated marker trace; on-click we move it via Plotly.restyle
      // (avoids full Plotly.react on each click, which can freeze on some browsers/GPU stacks).
      const markerTrace = (() => {{
        if (!state.selectedPoint) {{
          return {{
            type: "scatter3d",
            mode: "markers",
            x: [],
            y: [],
            z: [],
            marker: {{ size: 5, color: currentTheme.text }},
            name: "Selected",
            hoverinfo: "skip",
            showlegend: false,
            visible: false,
          }};
        }}
        const xi = state.selectedPoint.xIdx;
        const yi = state.selectedPoint.yIdx;
        const {{ rec, val }} = computePointValue(xi, yi);
        if (rec < 0 || !Number.isFinite(val) || xi < 0 || yi < 0 || xi >= xVals.length || yi >= yVals.length) {{
          return {{
            type: "scatter3d",
            mode: "markers",
            x: [],
            y: [],
            z: [],
            marker: {{ size: 5, color: currentTheme.text }},
            name: "Selected",
            hoverinfo: "skip",
            showlegend: false,
            visible: false,
          }};
        }}
        return {{
          type: "scatter3d",
          mode: "markers",
          x: [xVals[xi]],
          y: [yVals[yi]],
          z: [val],
          marker: {{ size: 5, color: currentTheme.text }},
          name: "Selected",
          hoverinfo: "skip",
          showlegend: false,
          visible: true,
        }};
      }})();

      const layout = {{
        template: currentTheme.template,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
	        font: {{ color: currentTheme.text }},
	        margin: {{ l: 0, r: 0, t: 24, b: 0 }},
	        title: {{ text: `${{metricLabel()}} (${{PNL_SUMMARY_LABEL}})`, x: 0.01, xanchor: "left", font: {{ size: 14 }} }},
	        scene: {{
	          bgcolor: currentTheme.bg,
	          xaxis: {{
            title: state.xParam,
            color: currentTheme.text,
            gridcolor: currentTheme.grid,
            gridwidth: 1.2,
            zerolinecolor: currentTheme.grid,
            backgroundcolor: currentTheme.bg,
            showbackground: true,
          }},
          yaxis: {{
            title: state.yParam,
            color: currentTheme.text,
            gridcolor: currentTheme.grid,
            gridwidth: 1.2,
            zerolinecolor: currentTheme.grid,
            backgroundcolor: currentTheme.bg,
            showbackground: true,
          }},
          zaxis: {{
            title: metricLabel(),
            range: [zMin, zMax],
            color: currentTheme.text,
            gridcolor: currentTheme.grid,
            gridwidth: 1.2,
            zerolinecolor: currentTheme.grid,
            backgroundcolor: currentTheme.bg,
            showbackground: true,
          }},
        }},
      }};

      if (!pnlSurfaceInitialized) {{
        Plotly.newPlot("pnlSurface", [surfaceTrace, markerTrace], layout, {{ responsive: true }});
        pnlSurfaceInitialized = true;
        const div = document.getElementById("pnlSurface");

        function nearestIndex(values, target) {{
          if (!values || values.length === 0 || !Number.isFinite(target)) return null;
          // Exact match fast path.
          const exact = values.indexOf(target);
          if (exact >= 0) return exact;
          let best = 0;
          let bestDist = Math.abs(Number(values[0]) - target);
          for (let i = 1; i < values.length; i++) {{
            const d = Math.abs(Number(values[i]) - target);
            if (d < bestDist) {{
              best = i;
              bestDist = d;
            }}
          }}
          return best;
        }}

        div.on("plotly_click", (ev) => {{
          if (!ev?.points?.length) return;
          const pt = ev.points[0];
          const xValsNow = PARAM_VALUES[state.xParam] || [];
          const yValsNow = PARAM_VALUES[state.yParam] || [];

          // Always use pt.x and pt.y coordinate values for reliability.
          // Plotly's pointNumber/i/j can be inconsistent for 3D surfaces.
          let xi = nearestIndex(xValsNow, Number(pt.x));
          let yi = nearestIndex(yValsNow, Number(pt.y));

          if (!Number.isFinite(xi) || !Number.isFinite(yi)) return;
          xi = Math.trunc(xi);
          yi = Math.trunc(yi);
          if (xi < 0 || yi < 0 || xi >= xValsNow.length || yi >= yValsNow.length) return;

          state.selectedPoint = {{ xIdx: xi, yIdx: yi }};
          scheduleMarkerUpdate();
          scheduleFeeRender();
          updateSeedWarning();
        }});
      }} else {{
        Plotly.react("pnlSurface", [surfaceTrace, markerTrace], layout, {{ responsive: true }});
      }}
    }}

    function renderFeeHist(xIdx, yIdx) {{
      const {{ idxRow, rec }} = computePointValue(xIdx, yIdx);
      if (rec < 0) {{
        clearFeePercentiles();
        return;
      }}

      const counts = REC_FEE_HIST[rec];
      if (!counts || counts.length === 0) {{
        clearFeePercentiles();
        return;
      }}
      const feeMean = REC_FEE_MEAN[rec];
      const feeMedian = REC_FEE_MEDIAN[rec];

      let maxCount = 1;
      for (const c of counts) {{
        if (Number.isFinite(c) && c > maxCount) maxCount = c;
      }}

      const centers = FEE_CENTERS;
      const meanVisible = Number.isFinite(feeMean);
      const medianVisible = Number.isFinite(feeMedian);

      const bar = {{
        type: "bar",
        x: centers,
        y: counts,
        marker: {{ color: currentTheme.accent }},
        name: "Fee distribution",
        opacity: 0.8,
      }};
      const meanTrace = {{
        type: "scatter",
        x: meanVisible ? [feeMean, feeMean] : [0, 0],
        y: meanVisible ? [0, maxCount] : [0, 0],
        mode: "lines",
        line: {{ color: currentTheme.mean, width: 2, dash: "dash" }},
        name: "Mean",
        visible: meanVisible,
      }};
      const medianTrace = {{
        type: "scatter",
        x: medianVisible ? [feeMedian, feeMedian] : [0, 0],
        y: medianVisible ? [0, maxCount] : [0, 0],
        mode: "lines",
        line: {{ color: currentTheme.median, width: 2, dash: "dot" }},
        name: "Median",
        visible: medianVisible,
      }};
      const data = [bar, meanTrace, medianTrace];

      const title = "Fee distribution";
      const topMargin = 40;

      const axisText = currentTheme.axisText || currentTheme.text;
      const legendText = currentTheme.legendText || currentTheme.muted || currentTheme.text;
      const layout = {{
        template: currentTheme.template,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {{ color: currentTheme.text }},
        margin: {{ l: 40, r: 10, t: topMargin, b: 40 }},
        title: {{ text: title, x: 0.01, xanchor: "left", font: {{ size: 12 }} }},
        xaxis: {{
          title: {{ text: "Fee", font: {{ color: axisText }} }},
          tickfont: {{ color: axisText }},
          gridcolor: currentTheme.grid,
          zerolinecolor: currentTheme.grid,
          color: axisText,
        }},
        yaxis: {{
          title: {{ text: "Count", font: {{ color: axisText }} }},
          tickfont: {{ color: axisText }},
          gridcolor: currentTheme.grid,
          zerolinecolor: currentTheme.grid,
          color: axisText,
        }},
        legend: {{
          orientation: "v",
          yanchor: "top",
          y: 1,
          xanchor: "right",
          x: 1,
          font: {{ color: legendText }},
        }},
      }};

      updateFeePercentiles(counts, feeMedian);
      if (!feeHistInitialized) {{
        Plotly.newPlot("feeHist", data, layout, {{ responsive: true }});
        feeHistInitialized = true;
      }} else {{
        Plotly.restyle("feeHist", {{ x: [centers], y: [counts] }}, [0]);
        if (meanVisible) {{
          Plotly.restyle(
            "feeHist",
            {{ x: [[feeMean, feeMean]], y: [[0, maxCount]], visible: true }},
            [1]
          );
        }} else {{
          Plotly.restyle("feeHist", {{ visible: false }}, [1]);
        }}
        if (medianVisible) {{
          Plotly.restyle(
            "feeHist",
            {{ x: [[feeMedian, feeMedian]], y: [[0, maxCount]], visible: true }},
            [2]
          );
        }} else {{
          Plotly.restyle("feeHist", {{ visible: false }}, [2]);
        }}
        Plotly.relayout("feeHist", {{ ...layout, "yaxis.autorange": true }});
      }}
    }}

    function renderSmartRouterDexShareHist(xIdx, yIdx) {{
      if (!HAS_SR_DEX_SHARE_HIST) return;
      const {{ idxRow, rec }} = computePointValue(xIdx, yIdx);
      if (rec < 0) return;

      const counts = REC_SR_DEX_SHARE_HIST[rec];
      if (!counts || counts.length === 0) return;
      const srMean = REC_SR_DEX_SHARE_MEAN[rec];
      const srMedian = REC_SR_DEX_SHARE_MEDIAN[rec];

      let maxCount = 1;
      for (const c of counts) {{
        if (Number.isFinite(c) && c > maxCount) maxCount = c;
      }}

      const centers = SR_CENTERS;
      const meanVisible = Number.isFinite(srMean);
      const medianVisible = Number.isFinite(srMedian);

      const bar = {{
        type: "bar",
        x: centers,
        y: counts,
        marker: {{ color: currentTheme.accentSoft }},
        name: "DEX share distribution",
        opacity: 0.82,
      }};
      const meanTrace = {{
        type: "scatter",
        x: meanVisible ? [srMean, srMean] : [0, 0],
        y: meanVisible ? [0, maxCount] : [0, 0],
        mode: "lines",
        line: {{ color: currentTheme.mean, width: 2, dash: "dash" }},
        name: "Mean",
        visible: meanVisible,
      }};
      const medianTrace = {{
        type: "scatter",
        x: medianVisible ? [srMedian, srMedian] : [0, 0],
        y: medianVisible ? [0, maxCount] : [0, 0],
        mode: "lines",
        line: {{ color: currentTheme.median, width: 2, dash: "dot" }},
        name: "Median",
        visible: medianVisible,
      }};
      const data = [bar, meanTrace, medianTrace];

      const title = "Smart-router DEX share distribution";
      const topMargin = 40;

      const axisText = currentTheme.axisText || currentTheme.text;
      const legendText = currentTheme.legendText || currentTheme.muted || currentTheme.text;
      const layout = {{
        template: currentTheme.template,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {{ color: currentTheme.text }},
        margin: {{ l: 40, r: 10, t: topMargin, b: 40 }},
        title: {{ text: title, x: 0.01, xanchor: "left", font: {{ size: 12 }} }},
        xaxis: {{
          title: {{ text: "DEX share", font: {{ color: axisText }} }},
          tickfont: {{ color: axisText }},
          tickformat: ".0%",
          gridcolor: currentTheme.grid,
          zerolinecolor: currentTheme.grid,
          color: axisText,
        }},
        yaxis: {{
          title: {{ text: "Count", font: {{ color: axisText }} }},
          tickfont: {{ color: axisText }},
          gridcolor: currentTheme.grid,
          zerolinecolor: currentTheme.grid,
          color: axisText,
        }},
        legend: {{
          orientation: "v",
          yanchor: "top",
          y: 1,
          xanchor: "right",
          x: 1,
          font: {{ color: legendText }},
        }},
      }};

      if (!srDexShareHistInitialized) {{
        Plotly.newPlot("srDexShareHist", data, layout, {{ responsive: true }});
        srDexShareHistInitialized = true;
      }} else {{
        Plotly.restyle("srDexShareHist", {{ x: [centers], y: [counts] }}, [0]);
        if (meanVisible) {{
          Plotly.restyle(
            "srDexShareHist",
            {{ x: [[srMean, srMean]], y: [[0, maxCount]], visible: true }},
            [1]
          );
        }} else {{
          Plotly.restyle("srDexShareHist", {{ visible: false }}, [1]);
        }}
        if (medianVisible) {{
          Plotly.restyle(
            "srDexShareHist",
            {{ x: [[srMedian, srMedian]], y: [[0, maxCount]], visible: true }},
            [2]
          );
        }} else {{
          Plotly.restyle("srDexShareHist", {{ visible: false }}, [2]);
        }}
        Plotly.relayout("srDexShareHist", {{ ...layout, "yaxis.autorange": true }});
      }}
    }}

    function setAxes(xParam, yParam) {{
      state.xParam = xParam;
      state.yParam = yParam;
      if (state.xParam === state.yParam) {{
        const idx = PARAM_ORDER.indexOf(state.yParam);
        const alt = PARAM_ORDER[(idx + 1) % PARAM_ORDER.length];
        state.yParam = alt;
      }}
      rebuildSliders();
      const xLen = (PARAM_VALUES[state.xParam] || []).length;
      const yLen = (PARAM_VALUES[state.yParam] || []).length;
      state.selectedPoint = {{
        xIdx: Math.floor(Math.max(0, xLen - 1) / 2),
        yIdx: Math.floor(Math.max(0, yLen - 1) / 2),
      }};
      updateSeedWarning();
      renderSurface();
      renderFeeHist(state.selectedPoint.xIdx, state.selectedPoint.yIdx);
      if (HAS_SR_DEX_SHARE_HIST) {{
        renderSmartRouterDexShareHist(state.selectedPoint.xIdx, state.selectedPoint.yIdx);
      }}
    }}

    function init() {{
      populateSelect(
        xAxisEl,
        PARAM_ORDER.map(p => ({{ value: p, label: p }})),
        state.xParam
      );
      populateSelect(
        yAxisEl,
        PARAM_ORDER.map(p => ({{ value: p, label: p }})),
        state.yParam
      );
      populateSelect(
        metricEl,
        METRICS.map((m, i) => ({{ value: String(i), label: m.label }})),
        String(state.metricIndex)
      );

      xAxisEl.addEventListener("change", () => {{
        setAxes(xAxisEl.value, yAxisEl.value);
        yAxisEl.value = state.yParam;
      }});
      yAxisEl.addEventListener("change", () => {{
        setAxes(xAxisEl.value, yAxisEl.value);
        yAxisEl.value = state.yParam;
      }});
      metricEl.addEventListener("change", () => {{
        state.metricIndex = Number(metricEl.value);
        renderSurface();
        if (state.selectedPoint) {{
          renderFeeHist(state.selectedPoint.xIdx, state.selectedPoint.yIdx);
          if (HAS_SR_DEX_SHARE_HIST) {{
            renderSmartRouterDexShareHist(state.selectedPoint.xIdx, state.selectedPoint.yIdx);
          }}
        }}
      }});
      themeEl.addEventListener("change", () => {{
        applyTheme(themeEl.value);
      }});

      if (!HAS_SR_DEX_SHARE_HIST && srDexShareWrapperEl) {{
        srDexShareWrapperEl.style.display = "none";
      }}

      setAxes(state.xParam, state.yParam);
      applyTheme(themeEl.value);
    }}

    init();
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def _read_meta(meta_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read meta JSON at {meta_path}: {exc}") from exc


def _find_latest_cache(*, data_dir: Path, stem: str) -> Path:
    if not data_dir.exists():
        raise SystemExit(f"Cache directory not found: {data_dir}")
    matches = sorted(data_dir.glob(f"grid_{stem}_*.csv"))
    if not matches:
        raise SystemExit(f"No cached CSV found under {data_dir} for config stem '{stem}'.")
    return max(matches, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML dashboard from an ND grid cache CSV.")
    parser.add_argument("--cache", type=Path, default=None, help="Path to cached grid CSV (grid_<tag>.csv).")
    parser.add_argument("--meta", type=Path, default=None, help="Optional meta JSON (meta_<tag>.json).")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Scenario YAML config (used if meta is missing; also shown in header).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default under abm_results/.../html/).")
    parser.add_argument("--title", type=str, default=None, help="Override HTML title.")
    args = parser.parse_args()

    global_root = Path("abm_results") / "grid_search" / "dashboard_nd"
    data_dir = global_root / "data"

    cache_path = args.cache
    if cache_path is None:
        config_path_guess = args.config or BASE_CONFIG_PATH
        cache_path = _find_latest_cache(data_dir=data_dir, stem=config_path_guess.stem)

    if not cache_path.exists():
        raise SystemExit(f"Cache CSV not found: {cache_path}")

    tag = _infer_tag_from_cache_path(cache_path)

    meta_path: Optional[Path] = args.meta
    if meta_path is None:
        candidate = cache_path.with_name(f"meta_{tag}.json")
        if candidate.exists():
            meta_path = candidate

    meta: Optional[Dict[str, Any]] = _read_meta(meta_path) if meta_path is not None else None

    config_path = args.config
    if config_path is None and meta is not None and isinstance(meta.get("config_path"), str):
        config_path = Path(meta["config_path"])
    if config_path is None:
        config_path = BASE_CONFIG_PATH

    scenario_label = "unknown"
    base_params: Dict[str, Any] = {}
    try:
        scenario_label, base_params = load_simulation_parameters(config_path, simulate_func=run_module.simulate)
    except Exception:
        scenario_label = str(meta.get("scenario_label", "unknown")) if meta is not None else "unknown"
        base_params = {}

    dataframe = pd.read_csv(cache_path)
    if dataframe.empty:
        raise SystemExit(f"Cache CSV is empty: {cache_path}")

    param_order = (
        list(meta["param_order"])
        if meta is not None and isinstance(meta.get("param_order"), list) and len(meta["param_order"]) >= 2
        else _infer_param_order_from_columns(list(dataframe.columns))
    )
    sweeps = (
        {k: list(v) for k, v in meta["sweeps"].items()}
        if meta is not None and isinstance(meta.get("sweeps"), dict)
        else _infer_sweeps_from_cache(dataframe, param_order=param_order)
    )
    int_params = (
        sorted({str(v) for v in meta.get("int_params", [])})
        if meta is not None and isinstance(meta.get("int_params"), list)
        else sorted(INT_PARAMS)
    )

    metrics = (
        list(meta.get("metrics", []))
        if meta is not None and isinstance(meta.get("metrics"), list) and meta.get("metrics")
        else [{"key": k, "label": label} for k, label in PNL_METRICS]
    )

    fee_hist_bins = (
        int(meta["fee_hist_bins"])
        if meta is not None and meta.get("fee_hist_bins") is not None
        else _infer_fee_hist_bins(list(dataframe.columns))
    )
    if fee_hist_bins <= 1:
        raise SystemExit("Invalid fee_hist_bins; expected > 1.")

    if meta is not None and isinstance(meta.get("fee_bin_edges"), list) and len(meta["fee_bin_edges"]) == fee_hist_bins + 1:
        fee_bin_edges = [float(v) for v in meta["fee_bin_edges"]]
    else:
        f_min = float(base_params.get("f_min", 0.0))
        f_max = float(base_params.get("f_max", 0.0))
        if not (np.isfinite(f_min) and np.isfinite(f_max) and f_max > f_min):
            raise SystemExit(
                "Could not determine fee bin edges: meta is missing fee_bin_edges and config has invalid f_min/f_max."
            )
        fee_bin_edges = np.linspace(f_min, f_max, fee_hist_bins + 1, dtype=float).tolist()

    smart_router_dex_share_hist_bins: Optional[int]
    if meta is not None and meta.get("smart_router_dex_share_hist_bins") is not None:
        smart_router_dex_share_hist_bins = int(meta["smart_router_dex_share_hist_bins"])
    else:
        try:
            smart_router_dex_share_hist_bins = _infer_smart_router_dex_share_hist_bins(list(dataframe.columns))
        except SystemExit:
            smart_router_dex_share_hist_bins = None

    smart_router_dex_share_bin_edges: Optional[List[float]] = None
    if smart_router_dex_share_hist_bins is not None:
        if (
            meta is not None
            and isinstance(meta.get("smart_router_dex_share_bin_edges"), list)
            and len(meta["smart_router_dex_share_bin_edges"]) == smart_router_dex_share_hist_bins + 1
        ):
            smart_router_dex_share_bin_edges = [float(v) for v in meta["smart_router_dex_share_bin_edges"]]
        else:
            # DEX share is a ratio in [0, 1]. We use fixed bin edges so the histogram is comparable across points.
            smart_router_dex_share_bin_edges = np.linspace(
                0.0, 1.0, int(smart_router_dex_share_hist_bins) + 1, dtype=float
            ).tolist()

    default_indices = _default_selection_indices(
        base_params=base_params,
        param_order=param_order,
        sweep_values=sweeps,
        int_params=set(int_params),
    )

    # Stable sort by index columns so record order is deterministic.
    dataframe = dataframe.copy()
    sort_cols = [f"i__{name}" for name in param_order]
    if all(col in dataframe.columns for col in sort_cols):
        dataframe = dataframe.drop_duplicates(subset=sort_cols, keep="last")
        dataframe = dataframe.sort_values(by=sort_cols, kind="mergesort")

    metric_keys: List[str] = []
    for metric in metrics:
        key = metric.get("key")
        if not isinstance(key, str) or not key:
            raise SystemExit(f"Invalid metric entry in meta; expected {{'key': str, 'label': str}}, got: {metric}")
        metric_keys.append(key)

    # Prefer the newer, horizon-normalized PnL summaries if present; fall back to the legacy
    # `median_final_*` columns for old caches.
    pnl_summary_label = "median final PnL"
    pnl_cols = [f"median_final_{key}" for key in metric_keys]
    pnl_p10_cols: Optional[List[str]] = None
    pnl_p90_cols: Optional[List[str]] = None
    pnl_p_loss_cols: Optional[List[str]] = None

    candidate_rate_p50 = [f"pnl_rate_p50_{key}" for key in metric_keys]
    if all(col in dataframe.columns for col in candidate_rate_p50):
        pnl_summary_label = "p50 per-step PnL rate"
        pnl_cols = candidate_rate_p50
        candidate_rate_p10 = [f"pnl_rate_p10_{key}" for key in metric_keys]
        candidate_rate_p90 = [f"pnl_rate_p90_{key}" for key in metric_keys]
        candidate_rate_p_loss = [f"pnl_rate_p_loss_{key}" for key in metric_keys]
        if all(col in dataframe.columns for col in candidate_rate_p10):
            pnl_p10_cols = candidate_rate_p10
        if all(col in dataframe.columns for col in candidate_rate_p90):
            pnl_p90_cols = candidate_rate_p90
        if all(col in dataframe.columns for col in candidate_rate_p_loss):
            pnl_p_loss_cols = candidate_rate_p_loss
    hist_cols = [f"fee_hist_{b}" for b in range(fee_hist_bins)]
    required_cols = set(sort_cols) | set(pnl_cols) | {"fee_mean", "fee_median"} | set(hist_cols)
    sr_hist_cols: Optional[List[str]] = None
    if smart_router_dex_share_hist_bins is not None:
        sr_hist_cols = [f"smart_router_dex_share_hist_{b}" for b in range(int(smart_router_dex_share_hist_bins))]
        required_cols |= {
            "smart_router_dex_share_mean",
            "smart_router_dex_share_median",
            *sr_hist_cols,
        }
    missing_cols = sorted(required_cols - set(dataframe.columns))
    if missing_cols:
        raise SystemExit(f"Cache is missing required columns: {missing_cols}")

    records_idx: List[List[int]] = []
    records_pnl: List[List[float]] = []
    records_pnl_p10: Optional[List[List[float]]] = [] if pnl_p10_cols is not None else None
    records_pnl_p90: Optional[List[List[float]]] = [] if pnl_p90_cols is not None else None
    records_pnl_p_loss: Optional[List[List[float]]] = [] if pnl_p_loss_cols is not None else None
    records_fee_mean: List[float] = []
    records_fee_median: List[float] = []
    records_fee_hist: List[List[int]] = []
    records_smart_router_dex_share_mean: Optional[List[float]] = [] if sr_hist_cols is not None else None
    records_smart_router_dex_share_median: Optional[List[float]] = [] if sr_hist_cols is not None else None
    records_smart_router_dex_share_hist: Optional[List[List[int]]] = [] if sr_hist_cols is not None else None

    for _, row in dataframe.iterrows():
        records_idx.append([int(row[f"i__{name}"]) for name in param_order])
        records_pnl.append([float(row[col]) for col in pnl_cols])
        if records_pnl_p10 is not None and pnl_p10_cols is not None:
            records_pnl_p10.append([float(row[col]) for col in pnl_p10_cols])
        if records_pnl_p90 is not None and pnl_p90_cols is not None:
            records_pnl_p90.append([float(row[col]) for col in pnl_p90_cols])
        if records_pnl_p_loss is not None and pnl_p_loss_cols is not None:
            records_pnl_p_loss.append([float(row[col]) for col in pnl_p_loss_cols])
        records_fee_mean.append(float(row["fee_mean"]))
        records_fee_median.append(float(row["fee_median"]))
        records_fee_hist.append([int(row[col]) for col in hist_cols])
        if (
            records_smart_router_dex_share_mean is not None
            and records_smart_router_dex_share_median is not None
            and records_smart_router_dex_share_hist is not None
            and sr_hist_cols is not None
        ):
            records_smart_router_dex_share_mean.append(float(row["smart_router_dex_share_mean"]))
            records_smart_router_dex_share_median.append(float(row["smart_router_dex_share_median"]))
            records_smart_router_dex_share_hist.append([int(row[col]) for col in sr_hist_cols])

    title = args.title
    if title is None:
        fingerprint = str(meta.get("fingerprint", "")) if meta is not None else ""
        if fingerprint:
            title = f"ABM Parameter Dashboard (ND grid) • {config_path.stem} • {fingerprint}"
        else:
            title = f"ABM Parameter Dashboard (ND grid) • {tag}"

    output_path = args.output
    if output_path is None:
        output_path = global_root / "html" / f"dashboard_{tag}.html"

    _write_dashboard_html(
        output_path,
        title=title,
        scenario_label=str(scenario_label),
        config_path=config_path,
        pnl_summary_label=pnl_summary_label,
        param_order=param_order,
        sweep_values=sweeps,
        int_params=int_params,
        metrics=metrics,
        default_indices=default_indices,
        fee_bin_edges=fee_bin_edges,
        smart_router_dex_share_bin_edges=smart_router_dex_share_bin_edges,
        records_idx=records_idx,
        records_pnl=records_pnl,
        records_pnl_p10=records_pnl_p10,
        records_pnl_p90=records_pnl_p90,
        records_pnl_p_loss=records_pnl_p_loss,
        records_fee_mean=records_fee_mean,
        records_fee_median=records_fee_median,
        records_fee_hist=records_fee_hist,
        records_smart_router_dex_share_mean=records_smart_router_dex_share_mean,
        records_smart_router_dex_share_median=records_smart_router_dex_share_median,
        records_smart_router_dex_share_hist=records_smart_router_dex_share_hist,
    )

    print(f"[dashboard_nd] cache: {cache_path}")
    if meta_path is not None:
        print(f"[dashboard_nd] meta:  {meta_path}")
    print(f"[dashboard_nd] html:   {output_path}")


if __name__ == "__main__":
    main()
