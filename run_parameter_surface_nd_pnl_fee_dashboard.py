#!/usr/bin/env python3
"""
Precompute an N-parameter grid and render a standalone interactive HTML dashboard.

Dashboard features (in a single HTML file):
  - Choose X and Y axes (any swept parameter) for a 3D surface of median *final* PnL.
  - Choose the PnL metric: passive LP (hedged), active LP (hedged), arbitrageur, noise trader.
  - For every other swept parameter (not on the axes), use a slider to select a fixed value.
  - Click a point on the surface to show the empirical fee distribution (histogram) for that
    specific parameter combination, aggregated over all runs and all time steps (after skip_step).

Notes:
  - This script runs the full simulation for every parameter combination (potentially expensive).
  - Results are cached to a scenario-scoped CSV to support resume / incremental runs.
  - The HTML is "standalone" in the sense it is a single file; Plotly JS is loaded via CDN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import run as run_module
from utils import load_simulation_parameters


def _silent_tqdm(iterable=None, **kwargs):
    """Silence tqdm inside run.simulate to avoid nested progress bars."""
    if iterable is None:
        total = int(kwargs.get("total", 0))
        return range(total)
    return iterable


run_module.tqdm = _silent_tqdm  # type: ignore[attr-defined]
simulate = run_module.simulate


BASE_CONFIG_PATH = Path("abm_results/scenarios/test.yml")

RUNS_PER_POINT_DEFAULT = 20
SEED_BASE_DEFAULT = 1
FEE_HIST_BINS_DEFAULT = 60


def linspace_int(start: int, stop: int, steps: int) -> List[int]:
    """Inclusive integer linspace with rounding + de-dup, preserving order."""
    if steps <= 1:
        return [int(start)]
    raw = np.linspace(float(start), float(stop), int(steps))
    rounded = [int(round(float(v))) for v in raw.tolist()]
    # Ensure endpoints are included and preserve order.
    if rounded:
        rounded[0] = int(start)
        rounded[-1] = int(stop)
    seen: set[int] = set()
    out: List[int] = []
    for v in rounded:
        if v in seen:
            continue
        out.append(int(v))
        seen.add(int(v))
    return out


# -----------------------------------------------------------------------------
# Sweep configuration (edit here)
# -----------------------------------------------------------------------------
# Provide discrete values for each parameter you want to sweep. The full grid is
# the cartesian product of all values.
DEFAULT_SWEEPS: Dict[str, Sequence[float | int]] = {
    "passive_lp_share": np.linspace(0.0, 1.0, 5).tolist(),
    "narrow_mints_per_block": linspace_int(0, 10, 5),
    "passive_mints_per_block": linspace_int(0, 10, 5),
    "noise_trades_per_block": linspace_int(0, 10, 5),
    "passive_burns_per_block": linspace_int(0, 6, 3),
    "k_sigma": np.linspace(0.01, 15.0, 5).tolist(),
    "p_jit": np.linspace(0.0, 1.0, 5).tolist(),
}

# Parameters treated as integers for UI display + casting.
INT_PARAMS: set[str] = {
    "narrow_mints_per_block",
    "passive_mints_per_block",
    "noise_trades_per_block",
    "passive_burns_per_block",
}
# -----------------------------------------------------------------------------


PNL_METRICS: Tuple[Tuple[str, str], ...] = (
    ("lp_pnl_passive", "Passive LP hedged PnL"),
    ("lp_pnl_active", "Active LP hedged PnL"),
    ("arb_pnl_cum", "Arbitrageur PnL"),
    ("noise_trader_pnl_cum", "Noise trader PnL"),
)


def _slice_series(values: Sequence[float], skip: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    skip_clamped = max(0, min(int(skip), array.size))
    return array[skip_clamped:]


def _canon_fingerprint_payload(
    *,
    sweeps: Mapping[str, Sequence[float | int]],
    runs_per_point: int,
    seed_base: int,
    common_seeds: bool,
    fee_hist_bins: int,
) -> str:
    payload = {
        "sweeps": {k: list(v) for k, v in sorted(sweeps.items())},
        "runs_per_point": int(runs_per_point),
        "seed_base": int(seed_base),
        "common_seeds": bool(common_seeds),
        "fee_hist_bins": int(fee_hist_bins),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


@dataclass(frozen=True)
class GridPoint:
    index: int
    indices: Tuple[int, ...]  # aligned with param_order
    seed_base: int


def _append_rows_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    write_header = not csv_path.exists()
    dataframe.to_csv(csv_path, mode="a", header=write_header, index=False)


def _load_cache(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _existing_status(
    dataframe: pd.DataFrame, *, param_order: Sequence[str]
) -> Tuple[set[Tuple[int, ...]], set[Tuple[int, ...]]]:
    if dataframe.empty:
        return set(), set()
    required = {f"i__{name}" for name in param_order}
    required |= {"fee_mean", "fee_median"}
    if not required.issubset(set(dataframe.columns)):
        return set(), set()

    sort_cols = [f"i__{name}" for name in param_order]
    if all(col in dataframe.columns for col in sort_cols):
        dataframe = dataframe.drop_duplicates(subset=sort_cols, keep="last")

    ok_keys: set[Tuple[int, ...]] = set()
    failed_keys: set[Tuple[int, ...]] = set()
    for _, row in dataframe.iterrows():
        try:
            indices = tuple(int(row[f"i__{name}"]) for name in param_order)
            if np.isfinite(row["fee_mean"]) and np.isfinite(row["fee_median"]):
                ok_keys.add(indices)
            else:
                failed_keys.add(indices)
        except Exception:
            continue
    return ok_keys, failed_keys


def _evaluate_grid_point(
    point: GridPoint,
    *,
    base_params: Dict[str, Any],
    param_order: Sequence[str],
    sweep_values: Mapping[str, Sequence[float | int]],
    fee_bins: np.ndarray,
    runs_per_point: int,
    common_seeds: bool,
    global_seed_base: int,
) -> Dict[str, Any]:
    logging.getLogger("uniswapv3_pool").setLevel(logging.ERROR)
    params = dict(base_params)
    for name, idx in zip(param_order, point.indices):
        value = sweep_values[name][int(idx)]
        if name in INT_PARAMS:
            params[name] = int(round(float(value)))
        else:
            params[name] = float(value)

    params["visualize"] = False
    params["verbose"] = False
    params["light_mode"] = True

    skip_step = max(0, int(params.get("skip_step", 0)))

    pnl_samples_by_key: Dict[str, List[float]] = {key: [] for key, _ in PNL_METRICS}
    fee_values: List[float] = []
    fee_hist = np.zeros(int(fee_bins.size - 1), dtype=np.int64)

    ok = True
    error_type: str = ""
    error_message: str = ""
    error_run_index: Optional[int] = None
    error_seed: Optional[int] = None

    try:
        for run_index in range(int(runs_per_point)):
            if common_seeds:
                seed_value = int(global_seed_base + run_index)
            else:
                seed_value = int(point.seed_base + run_index)
            params["seed"] = seed_value

            try:
                output = simulate(**params)
            except Exception as exc:
                ok = False
                error_type = type(exc).__name__
                error_message = str(exc)
                error_run_index = int(run_index)
                error_seed = int(seed_value)
                break

            for metric_key, _ in PNL_METRICS:
                series = _slice_series(output.get(metric_key, []), skip_step)
                if series.size == 0:
                    raise ValueError(f"Series '{metric_key}' empty after applying skip_step={skip_step}.")
                pnl_samples_by_key[metric_key].append(float(series[-1]))

            fee_series = _slice_series(output.get("fee_series", []), skip_step)
            if fee_series.size == 0:
                raise ValueError("fee_series empty after applying skip_step.")
            fee_values.extend([float(v) for v in fee_series.tolist()])
            fee_hist += np.histogram(fee_series, bins=fee_bins)[0].astype(np.int64)
    except Exception as exc:
        ok = False
        error_type = type(exc).__name__
        error_message = str(exc)

    if ok:
        fee_arr = np.asarray(fee_values, dtype=float)
        fee_mean = float(np.mean(fee_arr)) if fee_arr.size else np.nan
        fee_median = float(np.median(fee_arr)) if fee_arr.size else np.nan

        medians: Dict[str, float] = {}
        for metric_key, _ in PNL_METRICS:
            samples = np.asarray(pnl_samples_by_key[metric_key], dtype=float)
            medians[f"median_final_{metric_key}"] = float(np.median(samples)) if samples.size else np.nan
        fee_hist_out = fee_hist.tolist()
    else:
        fee_mean = np.nan
        fee_median = np.nan
        medians = {f"median_final_{metric_key}": np.nan for metric_key, _ in PNL_METRICS}
        fee_hist_out = [0 for _ in range(int(fee_bins.size - 1))]

    return {
        "grid_index": int(point.index),
        "indices": point.indices,
        "seed_base": int(point.seed_base),
        "ok": bool(ok),
        "error_type": error_type,
        "error_message": error_message,
        "error_run_index": error_run_index,
        "error_seed": error_seed,
        **medians,
        "fee_mean": fee_mean,
        "fee_median": fee_median,
        "fee_hist": fee_hist_out,
    }


def _result_to_row(
    result: Dict[str, Any],
    *,
    param_order: Sequence[str],
    sweep_values: Mapping[str, Sequence[float | int]],
    runs_per_point: int,
    fee_hist_bins: int,
) -> Dict[str, Any]:
    indices = tuple(int(v) for v in result["indices"])
    row: Dict[str, Any] = {
        "runs_per_point": int(runs_per_point),
        "seed_base": int(result["seed_base"]),
        "fee_mean": float(result["fee_mean"]),
        "fee_median": float(result["fee_median"]),
    }
    for metric_key, _ in PNL_METRICS:
        row[f"median_final_{metric_key}"] = float(result[f"median_final_{metric_key}"])

    for name, idx in zip(param_order, indices):
        row[f"i__{name}"] = int(idx)
        value = sweep_values[name][int(idx)]
        if name in INT_PARAMS:
            row[f"v__{name}"] = int(round(float(value)))
        else:
            row[f"v__{name}"] = float(value)

    hist = result["fee_hist"]
    if not isinstance(hist, list) or len(hist) != int(fee_hist_bins):
        raise ValueError("Internal error: fee_hist has unexpected shape.")
    for b in range(int(fee_hist_bins)):
        row[f"fee_hist_{b}"] = int(hist[b])
    return row


def _default_selection_indices(
    *,
    base_params: Mapping[str, Any],
    param_order: Sequence[str],
    sweep_values: Mapping[str, Sequence[float | int]],
) -> Dict[str, int]:
    defaults: Dict[str, int] = {}
    for name in param_order:
        values = list(sweep_values[name])
        base_value = base_params.get(name)
        if base_value is None:
            defaults[name] = 0
            continue
        if name in INT_PARAMS:
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
    param_order: Sequence[str],
    sweep_values: Mapping[str, Sequence[float | int]],
    int_params: Sequence[str],
    default_indices: Mapping[str, int],
    fee_bin_edges: Sequence[float],
    records_idx: Sequence[Sequence[int]],
    records_pnl: Sequence[Sequence[float]],
    records_fee_mean: Sequence[float],
    records_fee_median: Sequence[float],
    records_fee_hist: Sequence[Sequence[int]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "title": title,
        "scenario_label": scenario_label,
        "config_path": str(config_path),
        "param_order": list(param_order),
        "param_values": {k: list(v) for k, v in sweep_values.items()},
        "int_params": list(int_params),
        "metrics": [{"key": k, "label": label} for k, label in PNL_METRICS],
        "defaults": {k: int(v) for k, v in default_indices.items()},
        "fee_bin_edges": list(map(float, fee_bin_edges)),
        "records": {
            "idx": [list(map(int, row)) for row in records_idx],
            "pnl": [list(map(float, row)) for row in records_pnl],
            "fee_mean": list(map(float, records_fee_mean)),
            "fee_median": list(map(float, records_fee_median)),
            "fee_hist": [list(map(int, row)) for row in records_fee_hist],
        },
    }

    data_json = json.dumps(payload, separators=(",", ":"))

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      margin: 0;
      padding: 0;
      background: #fafafa;
      color: #111827;
    }}
    header {{
      padding: 16px 20px;
      background: #ffffff;
      border-bottom: 1px solid #e5e7eb;
    }}
    header h1 {{
      font-size: 16px;
      margin: 0 0 4px 0;
      font-weight: 650;
    }}
    header .meta {{
      font-size: 12px;
      color: #6b7280;
    }}
    .container {{
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 12px;
      padding: 12px;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 12px;
    }}
    .panel h2 {{
      font-size: 13px;
      margin: 0 0 10px 0;
      color: #111827;
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
      color: #374151;
    }}
    select, input[type="range"] {{
      width: 100%;
    }}
    .slider-meta {{
      font-size: 12px;
      color: #374151;
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }}
    .plot {{
      width: 100%;
      height: 520px;
    }}
    .plot-small {{
      width: 100%;
      height: 360px;
    }}
    .hint {{
      font-size: 12px;
      color: #6b7280;
      margin-top: 8px;
      line-height: 1.35;
    }}
    .footer {{
      font-size: 12px;
      color: #6b7280;
      padding: 0 12px 14px 12px;
    }}
    @media (max-width: 1100px) {{
      .container {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="meta">Scenario: <b>{scenario_label}</b> • Config: <code>{config_path}</code></div>
  </header>
  <div class="container">
    <div class="panel">
      <h2>Controls</h2>
      <div class="controls">
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
          - Surface shows the <b>median final PnL</b> over repeated runs.<br/>
          - Click a point on the surface to update the <b>fee distribution</b> for that exact parameter combo.
        </div>
      </div>
    </div>
    <div class="panel">
      <h2>Median Final PnL Surface</h2>
      <div id="pnlSurface" class="plot"></div>
      <h2 style="margin-top: 14px;">Fee Distribution (Selected Point)</h2>
      <div id="feeHist" class="plot-small"></div>
    </div>
  </div>
  <div class="footer">
    Generated by <code>run_parameter_surface_nd_pnl_fee_dashboard.py</code>.
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

    const REC_IDX = GRID.records.idx;
    const REC_PNL = GRID.records.pnl;
    const REC_FEE_MEAN = GRID.records.fee_mean;
    const REC_FEE_MEDIAN = GRID.records.fee_median;
    const REC_FEE_HIST = GRID.records.fee_hist;

    const N_PARAMS = PARAM_ORDER.length;
    const PARAM_LENGTHS = PARAM_ORDER.map(p => (PARAM_VALUES[p] || []).length);
    const STRIDES = Array(N_PARAMS).fill(1);
    let totalSize = 1;
    for (let i = N_PARAMS - 1; i >= 0; i--) {{
      STRIDES[i] = totalSize;
      totalSize *= PARAM_LENGTHS[i];
    }}
    const FLAT_TO_REC = new Int32Array(totalSize);
    FLAT_TO_REC.fill(-1);
    for (let r = 0; r < REC_IDX.length; r++) {{
      const idxRow = REC_IDX[r];
      let flat = 0;
      for (let j = 0; j < N_PARAMS; j++) {{
        flat += idxRow[j] * STRIDES[j];
      }}
      FLAT_TO_REC[flat] = r;
    }}
    function flatIndex(idxRow) {{
      let flat = 0;
      for (let j = 0; j < N_PARAMS; j++) {{
        flat += idxRow[j] * STRIDES[j];
      }}
      return flat;
    }}
    function getRecordIndex(idxRow) {{
      const flat = flatIndex(idxRow);
      if (flat < 0 || flat >= FLAT_TO_REC.length) return -1;
      return FLAT_TO_REC[flat];
    }}
    function formatValue(param, value) {{
      if (INT_PARAMS.has(param)) return String(Math.round(value));
      const v = Number(value);
      // Compact float formatting.
      if (!isFinite(v)) return String(value);
      const abs = Math.abs(v);
      if (abs >= 1000 || (abs > 0 && abs < 1e-3)) return v.toExponential(3);
      return v.toPrecision(6).replace(/0+$/,'').replace(/\\.$/,'');
    }}

    const state = {{
      xParam: PARAM_ORDER[0],
      yParam: PARAM_ORDER.length > 1 ? PARAM_ORDER[1] : PARAM_ORDER[0],
      metricIndex: 0,
      selected: {{}},           // param -> idx (for non-axis sliders, but we store for all)
      selectedPoint: null,      // {{ xIdx, yIdx }}
    }};

    // Initialize selected indices from defaults (nearest to YAML base values if present).
    for (const p of PARAM_ORDER) {{
      const maxIdx = Math.max(0, (PARAM_VALUES[p] || []).length - 1);
      const d = (DEFAULTS[p] !== undefined) ? DEFAULTS[p] : 0;
      state.selected[p] = Math.max(0, Math.min(maxIdx, Number(d)));
    }}

    const xAxisEl = document.getElementById("xAxis");
    const yAxisEl = document.getElementById("yAxis");
    const metricEl = document.getElementById("metric");
    const slidersEl = document.getElementById("sliders");

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
          renderSurface();
          // Update fee plot if we already have a selected point.
          if (state.selectedPoint) renderFeeHist(state.selectedPoint.xIdx, state.selectedPoint.yIdx);
        }});

        wrapper.appendChild(meta);
        wrapper.appendChild(slider);
        slidersEl.appendChild(wrapper);
      }}
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

    function metricKey() {{
      return METRICS[state.metricIndex]?.key || METRICS[0].key;
    }}
    function metricLabel() {{
      return METRICS[state.metricIndex]?.label || METRICS[0].label;
    }}

    let pnlSurfaceInitialized = false;
    let feeHistInitialized = false;

    function renderSurface() {{
      const xVals = PARAM_VALUES[state.xParam] || [];
      const yVals = PARAM_VALUES[state.yParam] || [];

      const metricIdx = state.metricIndex;

      const z = [];
      let zMin = null;
      let zMax = null;
      for (let yi = 0; yi < yVals.length; yi++) {{
        const row = [];
        for (let xi = 0; xi < xVals.length; xi++) {{
          const idxRow = currentSelectionIdxRow(xi, yi);
          const rec = getRecordIndex(idxRow);
          let val = NaN;
          if (rec >= 0) {{
            val = REC_PNL[rec][metricIdx];
            if (isFinite(val)) {{
              zMin = (zMin === null) ? val : Math.min(zMin, val);
              zMax = (zMax === null) ? val : Math.max(zMax, val);
            }}
          }}
          row.push(val);
        }}
        z.push(row);
      }}
      if (zMin === null || zMax === null) {{
        zMin = 0;
        zMax = 1;
      }}

      // Selected marker (if any).
      const markerTrace = (() => {{
        if (!state.selectedPoint) return null;
        const xi = state.selectedPoint.xIdx;
        const yi = state.selectedPoint.yIdx;
        const zz = z[yi]?.[xi];
        if (zz === undefined || !isFinite(zz)) return null;
        return {{
          type: "scatter3d",
          mode: "markers",
          x: [xVals[xi]],
          y: [yVals[yi]],
          z: [zz],
          marker: {{ size: 5, color: "#111827" }},
          name: "Selected",
          hoverinfo: "skip",
          showlegend: false,
        }};
      }})();

      const surfaceTrace = {{
        type: "surface",
        x: xVals,
        y: yVals,
        z: z,
        colorscale: "Viridis",
        cmin: zMin,
        cmax: zMax,
        colorbar: {{ title: metricLabel(), len: 0.75 }},
        hovertemplate:
          `${{state.xParam}}=%{{x}}<br>${{state.yParam}}=%{{y}}<br>${{metricLabel()}}=%{{z}}<extra></extra>`,
      }};

      const data = markerTrace ? [surfaceTrace, markerTrace] : [surfaceTrace];
      const layout = {{
        template: "plotly_white",
        margin: {{ l: 0, r: 0, t: 24, b: 0 }},
        title: {{ text: `${{metricLabel()}} (median final)`, x: 0.01, xanchor: "left", font: {{ size: 14 }} }},
        scene: {{
          xaxis: {{ title: state.xParam }},
          yaxis: {{ title: state.yParam }},
          zaxis: {{ title: metricLabel(), range: [zMin, zMax] }},
        }},
      }};

      if (!pnlSurfaceInitialized) {{
        Plotly.newPlot("pnlSurface", data, layout, {{ responsive: true }});
        pnlSurfaceInitialized = true;
        const div = document.getElementById("pnlSurface");
        div.on("plotly_click", (ev) => {{
          if (!ev?.points?.length) return;
          const pt = ev.points[0];
          const pn = pt.pointNumber;
          if (!pn || pn.length < 2) return;
          const yi = Number(pn[0]);
          const xi = Number(pn[1]);
          state.selectedPoint = {{ xIdx: xi, yIdx: yi }};
          renderSurface(); // refresh marker
          renderFeeHist(xi, yi);
        }});
      }} else {{
        Plotly.react("pnlSurface", data, layout, {{ responsive: true }});
      }}
    }}

    function renderFeeHist(xIdx, yIdx) {{
      const idxRow = currentSelectionIdxRow(xIdx, yIdx);
      const rec = getRecordIndex(idxRow);
      if (rec < 0) return;

      const edges = FEE_EDGES;
      const centers = [];
      for (let i = 0; i < edges.length - 1; i++) {{
        centers.push(0.5 * (edges[i] + edges[i + 1]));
      }}
      const counts = REC_FEE_HIST[rec];
      const feeMean = REC_FEE_MEAN[rec];
      const feeMedian = REC_FEE_MEDIAN[rec];
      const maxCount = Math.max(1, ...counts);

      const bar = {{
        type: "bar",
        x: centers,
        y: counts,
        marker: {{ color: "#1f77b4" }},
        name: "Fee distribution",
        opacity: 0.8,
      }};
      const data = [bar];
      if (isFinite(feeMean)) {{
        data.push({{
          type: "scatter",
          x: [feeMean, feeMean],
          y: [0, maxCount],
          mode: "lines",
          line: {{ color: "firebrick", width: 2, dash: "dash" }},
          name: "Mean (fee)",
        }});
      }}
      if (isFinite(feeMedian)) {{
        data.push({{
          type: "scatter",
          x: [feeMedian, feeMedian],
          y: [0, maxCount],
          mode: "lines",
          line: {{ color: "black", width: 2, dash: "dot" }},
          name: "Median (fee)",
        }});
      }}

      // Build a compact title showing the full parameter selection for this point.
      const parts = [];
      for (let j = 0; j < N_PARAMS; j++) {{
        const p = PARAM_ORDER[j];
        const values = PARAM_VALUES[p];
        const idx = idxRow[j];
        parts.push(`${{p}}=${{formatValue(p, values[idx])}}`);
      }}
      const title = `Fee distribution • ${{parts.join(", ")}}`;

      const layout = {{
        template: "plotly_white",
        margin: {{ l: 40, r: 10, t: 50, b: 40 }},
        title: {{ text: title, x: 0.01, xanchor: "left", font: {{ size: 12 }} }},
        xaxis: {{ title: "Fee" }},
        yaxis: {{ title: "Count" }},
        legend: {{ orientation: "h", yanchor: "bottom", y: 1.02, x: 0 }},
      }};

      if (!feeHistInitialized) {{
        Plotly.newPlot("feeHist", data, layout, {{ responsive: true }});
        feeHistInitialized = true;
      }} else {{
        Plotly.react("feeHist", data, layout, {{ responsive: true }});
      }}
    }}

    function setAxes(xParam, yParam) {{
      state.xParam = xParam;
      state.yParam = yParam;
      if (state.xParam === state.yParam) {{
        // Ensure distinct axes: pick the next parameter if possible.
        const idx = PARAM_ORDER.indexOf(state.yParam);
        const alt = PARAM_ORDER[(idx + 1) % PARAM_ORDER.length];
        state.yParam = alt;
      }}
      rebuildSliders();
      // Reset selected point to the center of the surface (best-effort).
      const xLen = (PARAM_VALUES[state.xParam] || []).length;
      const yLen = (PARAM_VALUES[state.yParam] || []).length;
      state.selectedPoint = {{
        xIdx: Math.floor(Math.max(0, xLen - 1) / 2),
        yIdx: Math.floor(Math.max(0, yLen - 1) / 2),
      }};
      renderSurface();
      renderFeeHist(state.selectedPoint.xIdx, state.selectedPoint.yIdx);
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
        if (state.selectedPoint) renderFeeHist(state.selectedPoint.xIdx, state.selectedPoint.yIdx);
      }});

      setAxes(state.xParam, state.yParam);
    }}

    init();
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="N-parameter grid runner + standalone HTML dashboard (PnL surface + fee histogram)."
    )
    parser.add_argument("--config", type=Path, default=BASE_CONFIG_PATH, help="Base YAML scenario config path.")
    parser.add_argument("--runs-per-point", type=int, default=RUNS_PER_POINT_DEFAULT)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE_DEFAULT)
    parser.add_argument("--common-seeds", action="store_true", help="Use the same seeds for all grid points.")
    parser.add_argument("--fee-hist-bins", type=int, default=FEE_HIST_BINS_DEFAULT, help="Histogram bins for fee plot.")
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument(
        "--index-start",
        type=int,
        default=0,
        help="Grid index start (inclusive) in the cartesian-product enumeration order.",
    )
    parser.add_argument(
        "--index-stop",
        type=int,
        default=None,
        help="Grid index stop (exclusive) in the cartesian-product enumeration order.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run cached grid points with non-finite fee stats (previous failures).",
    )
    parser.add_argument("--recompute", action="store_true", help="Ignore cache and recompute all grid points.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved grid and exit.")
    args = parser.parse_args()

    runs_per_point = int(args.runs_per_point)
    if runs_per_point <= 0:
        raise SystemExit("--runs-per-point must be positive.")
    fee_hist_bins = int(args.fee_hist_bins)
    if fee_hist_bins <= 1:
        raise SystemExit("--fee-hist-bins must be > 1.")

    sweeps = {k: list(v) for k, v in DEFAULT_SWEEPS.items()}
    param_order = list(sweeps.keys())
    if len(param_order) < 2:
        raise SystemExit("Need at least 2 swept parameters to build a surface.")

    scenario_label, base_params = load_simulation_parameters(args.config, simulate_func=simulate)
    base_params["results_root"] = Path("abm_results") / "grid_search"

    f_min = float(base_params.get("f_min", 0.0))
    f_max = float(base_params.get("f_max", 0.0))
    if not (np.isfinite(f_min) and np.isfinite(f_max) and f_max > f_min):
        raise SystemExit(f"Invalid fee bounds from config: f_min={f_min}, f_max={f_max}")
    fee_bin_edges = np.linspace(f_min, f_max, fee_hist_bins + 1, dtype=float)

    fingerprint = _canon_fingerprint_payload(
        sweeps=sweeps,
        runs_per_point=runs_per_point,
        seed_base=int(args.seed_base),
        common_seeds=bool(args.common_seeds),
        fee_hist_bins=fee_hist_bins,
    )

    # --- outputs -------------------------------------------------------------
    global_root = Path("abm_results") / "grid_search" / "dashboard_nd"

    stem = args.config.stem
    tag = f"{stem}_{fingerprint}"

    csv_global = global_root / "data" / f"grid_{tag}.csv"
    html_global = global_root / "html" / f"dashboard_{tag}.html"
    errors_global = global_root / "data" / f"errors_{tag}.csv"

    # --- dry run -------------------------------------------------------------
    grid_sizes = {k: len(v) for k, v in sweeps.items()}
    total_points = int(np.prod([max(1, n) for n in grid_sizes.values()], dtype=np.int64))
    index_start = max(0, int(args.index_start))
    index_stop = int(total_points) if args.index_stop is None else int(args.index_stop)
    index_stop = max(index_start, min(int(total_points), index_stop))
    slice_total = int(index_stop - index_start)
    if args.dry_run:
        print("Resolved ND grid:")
        print(f"  config: {args.config}")
        print(f"  scenario_label (from YAML): {scenario_label}")
        print(f"  runs_per_point: {runs_per_point}")
        print(f"  seed_mode: {'common' if args.common_seeds else 'per_point'} (seed_base={args.seed_base})")
        print(f"  fee histogram bins: {fee_hist_bins} (edges from f_min={f_min} to f_max={f_max})")
        if slice_total != total_points:
            print(f"  index slice: [{index_start}, {index_stop}) => {slice_total} points")
        print("  swept parameters:")
        for name in param_order:
            values = sweeps[name]
            preview = values if len(values) <= 10 else (list(values[:5]) + ["..."] + list(values[-2:]))
            print(f"    - {name}: {len(values)} values: {preview}")
        print(f"  total grid points: {total_points}")
        print(f"  cache (global): {csv_global}")
        print(f"  html (global):  {html_global}")
        return

    # --- build grid ----------------------------------------------------------
    value_lengths = [len(sweeps[name]) for name in param_order]
    if any(n <= 0 for n in value_lengths):
        empty = [name for name in param_order if len(sweeps[name]) <= 0]
        raise SystemExit(f"Empty sweep values for parameters: {empty}")

    cached = _load_cache(csv_global)
    ok_cached, failed_cached = _existing_status(cached, param_order=param_order)
    if args.recompute:
        existing: set[Tuple[int, ...]] = set()
    else:
        existing = set(ok_cached)
        if not args.retry_failed:
            existing |= set(failed_cached)

    print(
        f"[dashboard_nd] config={args.config} | grid={total_points} points | "
        f"slice=[{index_start},{index_stop}) ({slice_total} points) | workers={args.max_workers}"
    )
    if slice_total > 0:
        print(f"[dashboard_nd] cache (global): {csv_global}")

    # --- run simulations -----------------------------------------------------
    pending_rows: List[Dict[str, Any]] = []
    progress_overall: Optional[tqdm] = (
        tqdm(total=int(slice_total), desc="Grid points (slice)", unit="pt") if slice_total > 0 else None
    )

    cached_in_slice = 0
    run_in_slice = 0
    failed_in_slice = 0
    pending_error_rows: List[Dict[str, Any]] = []

    if slice_total > 0:
        max_pending = max(1, int(args.max_workers) * 4)
        indices_iter = product(*[range(n) for n in value_lengths])
        with ProcessPoolExecutor(max_workers=int(args.max_workers)) as executor:
            pending: set = set()
            for idx, combo in enumerate(indices_iter):
                if idx < index_start:
                    continue
                if idx >= index_stop:
                    break
                indices = tuple(int(v) for v in combo)
                if not args.recompute and indices in existing:
                    cached_in_slice += 1
                    if progress_overall is not None:
                        progress_overall.update(1)
                    continue

                seed_base_point = int(args.seed_base + int(idx) * runs_per_point)
                point = GridPoint(index=int(idx), indices=indices, seed_base=seed_base_point)
                future = executor.submit(
                    _evaluate_grid_point,
                    point,
                    base_params=base_params,
                    param_order=param_order,
                    sweep_values=sweeps,
                    fee_bins=fee_bin_edges,
                    runs_per_point=runs_per_point,
                    common_seeds=bool(args.common_seeds),
                    global_seed_base=int(args.seed_base),
                )
                pending.add(future)
                run_in_slice += 1

                # Keep the number of in-flight futures bounded to avoid high memory usage.
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        result = fut.result()
                        if not bool(result.get("ok", True)):
                            failed_in_slice += 1
                            indices = result.get("indices")
                            error_row: Dict[str, Any] = {
                                "grid_index": int(result.get("grid_index", -1)),
                                "seed_base": int(result.get("seed_base", 0)),
                                "error_type": str(result.get("error_type", "")),
                                "error_message": str(result.get("error_message", ""))[:500],
                                "error_run_index": (
                                    "" if result.get("error_run_index") is None else int(result.get("error_run_index"))
                                ),
                                "error_seed": "" if result.get("error_seed") is None else int(result.get("error_seed")),
                            }
                            if isinstance(indices, (list, tuple)) and len(indices) == len(param_order):
                                for name, idx_val in zip(param_order, indices):
                                    idx_int = int(idx_val)
                                    error_row[f"i__{name}"] = idx_int
                                    value = sweeps[name][idx_int]
                                    if name in INT_PARAMS:
                                        error_row[f"v__{name}"] = int(round(float(value)))
                                    else:
                                        error_row[f"v__{name}"] = float(value)
                            pending_error_rows.append(error_row)
                        row = _result_to_row(
                            result,
                            param_order=param_order,
                            sweep_values=sweeps,
                            runs_per_point=runs_per_point,
                            fee_hist_bins=fee_hist_bins,
                        )
                        pending_rows.append(row)
                        if progress_overall is not None:
                            progress_overall.update(1)
                        if len(pending_rows) >= 25:
                            _append_rows_csv(csv_global, pending_rows)
                            pending_rows.clear()
                        if len(pending_error_rows) >= 10:
                            _append_rows_csv(errors_global, pending_error_rows)
                            pending_error_rows.clear()

            # Drain remaining futures.
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    result = fut.result()
                    if not bool(result.get("ok", True)):
                        failed_in_slice += 1
                        indices = result.get("indices")
                        error_row = {
                            "grid_index": int(result.get("grid_index", -1)),
                            "seed_base": int(result.get("seed_base", 0)),
                            "error_type": str(result.get("error_type", "")),
                            "error_message": str(result.get("error_message", ""))[:500],
                            "error_run_index": (
                                "" if result.get("error_run_index") is None else int(result.get("error_run_index"))
                            ),
                            "error_seed": "" if result.get("error_seed") is None else int(result.get("error_seed")),
                        }
                        if isinstance(indices, (list, tuple)) and len(indices) == len(param_order):
                            for name, idx_val in zip(param_order, indices):
                                idx_int = int(idx_val)
                                error_row[f"i__{name}"] = idx_int
                                value = sweeps[name][idx_int]
                                if name in INT_PARAMS:
                                    error_row[f"v__{name}"] = int(round(float(value)))
                                else:
                                    error_row[f"v__{name}"] = float(value)
                        pending_error_rows.append(error_row)
                    row = _result_to_row(
                        result,
                        param_order=param_order,
                        sweep_values=sweeps,
                        runs_per_point=runs_per_point,
                        fee_hist_bins=fee_hist_bins,
                    )
                    pending_rows.append(row)
                    if progress_overall is not None:
                        progress_overall.update(1)
                    if len(pending_rows) >= 25:
                        _append_rows_csv(csv_global, pending_rows)
                        pending_rows.clear()
                    if len(pending_error_rows) >= 10:
                        _append_rows_csv(errors_global, pending_error_rows)
                        pending_error_rows.clear()
        if pending_rows:
            _append_rows_csv(csv_global, pending_rows)
            pending_rows.clear()
        if pending_error_rows:
            _append_rows_csv(errors_global, pending_error_rows)
            pending_error_rows.clear()

    if progress_overall is not None:
        progress_overall.close()
    if slice_total > 0:
        print(
            f"[dashboard_nd] slice summary: ran={run_in_slice}, cached={cached_in_slice}, "
            f"failed={failed_in_slice}, total={slice_total}"
        )
        if failed_in_slice > 0:
            print(f"[dashboard_nd] failures logged to {errors_global}")

    # --- build dashboard from cache -----------------------------------------
    dataframe = _load_cache(csv_global)
    if dataframe.empty:
        raise SystemExit("No cached data found; nothing to plot.")

    # Ensure we have one row per full grid point.
    expected = int(total_points)
    if len(dataframe) < expected:
        print(
            f"[dashboard_nd] Warning: cache has {len(dataframe)}/{expected} rows. "
            "Surface will show gaps until you finish the grid."
        )

    # Extract arrays for JS payload.
    dataframe = dataframe.copy()
    # Stable sort by index columns so that record order is deterministic.
    sort_cols = [f"i__{name}" for name in param_order]
    if all(col in dataframe.columns for col in sort_cols):
        dataframe = dataframe.drop_duplicates(subset=sort_cols, keep="last")
        dataframe = dataframe.sort_values(by=sort_cols, kind="mergesort")

    records_idx: List[List[int]] = []
    records_pnl: List[List[float]] = []
    records_fee_mean: List[float] = []
    records_fee_median: List[float] = []
    records_fee_hist: List[List[int]] = []

    pnl_cols = [f"median_final_{key}" for key, _ in PNL_METRICS]
    hist_cols = [f"fee_hist_{b}" for b in range(fee_hist_bins)]
    required_cols = set(sort_cols) | set(pnl_cols) | {"fee_mean", "fee_median"} | set(hist_cols)
    missing_cols = sorted(required_cols - set(dataframe.columns))
    if missing_cols:
        raise SystemExit(f"Cache is missing required columns: {missing_cols}")

    for _, row in dataframe.iterrows():
        idx_row = [int(row[f"i__{name}"]) for name in param_order]
        pnl_row = [float(row[col]) for col in pnl_cols]
        hist_row = [int(row[col]) for col in hist_cols]
        records_idx.append(idx_row)
        records_pnl.append(pnl_row)
        records_fee_mean.append(float(row["fee_mean"]))
        records_fee_median.append(float(row["fee_median"]))
        records_fee_hist.append(hist_row)

    default_indices = _default_selection_indices(
        base_params=base_params,
        param_order=param_order,
        sweep_values=sweeps,
    )

    title = f"ABM Parameter Dashboard (ND grid) • {stem} • {fingerprint}"
    _write_dashboard_html(
        html_global,
        title=title,
        scenario_label=scenario_label,
        config_path=args.config,
        param_order=param_order,
        sweep_values=sweeps,
        int_params=sorted(INT_PARAMS),
        default_indices=default_indices,
        fee_bin_edges=fee_bin_edges.tolist(),
        records_idx=records_idx,
        records_pnl=records_pnl,
        records_fee_mean=records_fee_mean,
        records_fee_median=records_fee_median,
        records_fee_hist=records_fee_hist,
    )

    print(f"[dashboard_nd] HTML written to {html_global}")


if __name__ == "__main__":
    main()
