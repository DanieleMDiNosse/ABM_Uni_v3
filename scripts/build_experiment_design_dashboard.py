#!/usr/bin/env python3
"""Build an interactive HTML dashboard for a sampled experiment cache CSV.

This dashboard targets *non-grid* designs (LHS/Sobol/Saltelli/adaptive/BO). It
renders:
- a scatter view for any (x, y) parameter pair, colored by a selected metric,
- basic filtering on remaining parameters,
- an optional binned heatmap view for quick "surface-like" inspection.

Inputs
------
- Cached CSV produced by `scripts/run_experiment_design.py`
- Optional meta JSON written alongside the cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _sanitize_for_json(value: Any) -> Any:
    """Convert a nested payload to JSON-safe primitives (no NaN/Inf)."""
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


def _escape_html(text: str) -> str:
    """Escape HTML special characters for safe literal rendering."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _load_meta(meta_path: Optional[Path]) -> Optional[Mapping[str, Any]]:
    if meta_path is None:
        return None
    p = Path(meta_path)
    if not p.exists():
        raise SystemExit(f"Meta JSON not found: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to parse meta JSON: {p} ({exc})")


def _infer_param_names(columns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for col in columns:
        if not col.startswith("p__"):
            continue
        name = col[len("p__") :]
        if not name:
            continue
        out.append(name)
    if not out:
        raise SystemExit("Cache CSV has no parameter columns (expected p__<param> columns).")
    return out


def _infer_metric_columns(columns: Sequence[str]) -> List[str]:
    """Infer scalar metric columns from a cache CSV."""
    preferred = [
        "fee_mean",
        "fee_median",
        "smart_router_dex_share_mean",
        "smart_router_dex_share_median",
    ]
    out: List[str] = [c for c in preferred if c in columns]

    for c in columns:
        if c.startswith("pnl_rate_"):
            out.append(c)
    # De-dup while preserving order.
    seen: set[str] = set()
    uniq: List[str] = []
    for c in out:
        if c in seen:
            continue
        uniq.append(c)
        seen.add(c)
    if not uniq:
        raise SystemExit("Could not infer any metric columns from cache CSV.")
    return uniq


def _metric_label(key: str) -> str:
    if key.startswith("pnl_rate_"):
        return key.replace("_", " ")
    return key.replace("_", " ")


def _space_kind_from_meta(meta: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if meta is None:
        return out
    raw = meta.get("space")
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        kind = entry.get("kind")
        if isinstance(name, str) and isinstance(kind, str):
            out[name] = kind
    return out


def _unique_values(dataframe: pd.DataFrame, col: str) -> List[float]:
    series = dataframe[col]
    vals = series.dropna().unique().tolist()
    out: List[float] = []
    for v in vals:
        try:
            out.append(float(v))
        except Exception:
            continue
    out = sorted(out)
    return out


def _write_dashboard_html(
    output_path: Path,
    *,
    title: str,
    header_lines: Sequence[str],
    payload: Mapping[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_title = _escape_html(title)
    header_html = "\n".join([f"<div class=\"meta\">{_escape_html(line)}</div>" for line in header_lines if line])

    data_json = json.dumps(_sanitize_for_json(dict(payload)), separators=(",", ":"), allow_nan=False)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: rgba(255,255,255,0.06);
      --panel2: rgba(255,255,255,0.04);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.65);
      --border: rgba(255,255,255,0.12);
      --accent: #7dd3fc;
      --shadow: 0 14px 40px rgba(0,0,0,0.38);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }}
    body {{
      margin: 0;
      background: radial-gradient(1200px 800px at 20% 0%, rgba(125,211,252,0.12), transparent 60%),
                  radial-gradient(900px 700px at 80% 20%, rgba(99,102,241,0.10), transparent 55%),
                  var(--bg);
      color: var(--text);
      font-family: var(--sans);
    }}
    header {{
      padding: 18px 18px 10px 18px;
    }}
    header h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }}
    header .meta {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 14px;
      padding: 14px 18px 18px 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .panel .head {{
      padding: 12px 12px 10px 12px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.0));
    }}
    .panel .head h2 {{
      margin: 0;
      font-size: 13px;
      font-weight: 700;
      color: var(--text);
    }}
    .panel .body {{
      padding: 12px;
      background: var(--panel2);
    }}
    label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    select, input[type="number"] {{
      width: 100%;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.18);
      color: var(--text);
      outline: none;
      font-family: var(--sans);
    }}
    .row {{
      margin-bottom: 10px;
    }}
    .hint {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.35;
    }}
    .kpi {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }}
    .stat {{
      padding: 10px 10px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.14);
    }}
    .stat .k {{
      font-size: 11px;
      color: var(--muted);
    }}
    .stat .v {{
      margin-top: 4px;
      font-size: 13px;
      font-weight: 700;
      color: var(--text);
      font-family: var(--mono);
    }}
    .filters {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }}
    .filter {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
      background: rgba(0,0,0,0.10);
    }}
    .filter .title {{
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .grid2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .chk {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
    }}
    .plotWrap {{
      padding: 12px;
    }}
    #plot {{
      width: 100%;
      height: calc(100vh - 140px);
      min-height: 520px;
    }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr; }}
      #plot {{ height: 520px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{safe_title}</h1>
    {header_html}
  </header>
  <div class="layout">
    <div class="panel">
      <div class="head"><h2>Controls</h2></div>
      <div class="body">
        <div class="row">
          <label for="metricSelect">Metric</label>
          <select id="metricSelect"></select>
        </div>
        <div class="grid2">
          <div class="row">
            <label for="xSelect">X axis</label>
            <select id="xSelect"></select>
          </div>
          <div class="row">
            <label for="ySelect">Y axis</label>
            <select id="ySelect"></select>
          </div>
        </div>
        <div class="row chk">
          <input type="checkbox" id="heatmapToggle" />
          <label for="heatmapToggle" style="margin:0">Show binned heatmap</label>
        </div>
        <div class="row chk">
          <input type="checkbox" id="includeFailedToggle" />
          <label for="includeFailedToggle" style="margin:0">Include failed points</label>
        </div>
        <div class="hint">
          Filters apply to all other parameters (not on X/Y). Discrete parameters use value sets;
          continuous parameters use numeric min/max bounds.
        </div>
        <div class="kpi">
          <div class="stat"><div class="k">Points</div><div class="v" id="kpiPoints">-</div></div>
          <div class="stat"><div class="k">Metric range</div><div class="v" id="kpiRange">-</div></div>
        </div>
        <div class="filters" id="filters"></div>
      </div>
    </div>
    <div class="panel">
      <div class="head"><h2>Scatter / Heatmap</h2></div>
      <div class="plotWrap">
        <div id="plot"></div>
      </div>
    </div>
  </div>

  <script id="payload" type="application/json">{data_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById("payload").textContent);
    const paramOrder = payload.param_order;
    const paramKinds = payload.param_kinds || {{}};
    const paramUniques = payload.param_uniques || {{}};
    const metrics = payload.metrics;
    const records = payload.records;

    function el(id) {{ return document.getElementById(id); }}
    function fmt(v) {{
      if (v === null || v === undefined) return "-";
      if (!Number.isFinite(v)) return "-";
      const a = Math.abs(v);
      if (a !== 0 && (a < 1e-4 || a >= 1e6)) return v.toExponential(3);
      return v.toFixed(6).replace(/\\.0+$/,"").replace(/(\\.[0-9]*?)0+$/,"$1");
    }}

    function buildSelect(selectEl, options, defaultValue) {{
      selectEl.innerHTML = "";
      for (const opt of options) {{
        const o = document.createElement("option");
        o.value = opt.value;
        o.textContent = opt.label;
        if (defaultValue !== undefined && opt.value === defaultValue) {{
          o.selected = true;
        }}
        selectEl.appendChild(o);
      }}
    }}

    function currentState() {{
      return {{
        metric: el("metricSelect").value,
        x: el("xSelect").value,
        y: el("ySelect").value,
        heatmap: el("heatmapToggle").checked,
        includeFailed: el("includeFailedToggle").checked,
      }};
    }}

    function makeFilters() {{
      const state = currentState();
      const root = el("filters");
      root.innerHTML = "";

      const filters = {{}};
      for (const p of paramOrder) {{
        if (p === state.x || p === state.y) continue;
        const kind = paramKinds[p] || "continuous";
        const uniques = paramUniques[p] || [];
        const col = records.params[p];

        const card = document.createElement("div");
        card.className = "filter";
        const title = document.createElement("div");
        title.className = "title";
        title.textContent = p;
        card.appendChild(title);

        if (kind === "discrete" || uniques.length <= 20) {{
          const sel = document.createElement("select");
          sel.multiple = true;
          sel.size = Math.min(10, Math.max(3, uniques.length || 3));
          for (const v of uniques) {{
            const o = document.createElement("option");
            o.value = String(v);
            o.textContent = String(v);
            o.selected = true;
            sel.appendChild(o);
          }}
          sel.addEventListener("change", () => updatePlot());
          card.appendChild(sel);
          filters[p] = {{ type: "discrete", control: sel }};
        }} else {{
          let lo = Infinity, hi = -Infinity;
          for (const v of col) {{
            if (!Number.isFinite(v)) continue;
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
          }}
          if (!Number.isFinite(lo) || !Number.isFinite(hi)) {{
            lo = 0; hi = 1;
          }}

          const grid = document.createElement("div");
          grid.className = "grid2";
          const loInput = document.createElement("input");
          loInput.type = "number";
          loInput.step = "any";
          loInput.value = String(lo);
          const hiInput = document.createElement("input");
          hiInput.type = "number";
          hiInput.step = "any";
          hiInput.value = String(hi);
          loInput.addEventListener("change", () => updatePlot());
          hiInput.addEventListener("change", () => updatePlot());
          grid.appendChild(loInput);
          grid.appendChild(hiInput);
          card.appendChild(grid);
          filters[p] = {{ type: "range", loInput, hiInput }};
        }}
        root.appendChild(card);
      }}
      return filters;
    }}

    function passesFilters(i, state, filters) {{
      if (!state.includeFailed && records.ok[i] !== true) return false;
      for (const [p, f] of Object.entries(filters)) {{
        const v = records.params[p][i];
        if (!Number.isFinite(v)) return false;
        if (f.type === "discrete") {{
          const allowed = new Set(Array.from(f.control.selectedOptions).map(o => o.value));
          if (!allowed.has(String(v))) return false;
        }} else {{
          const lo = Number(f.loInput.value);
          const hi = Number(f.hiInput.value);
          if (Number.isFinite(lo) && v < lo) return false;
          if (Number.isFinite(hi) && v > hi) return false;
        }}
      }}
      return true;
    }}

    function binnedHeatmap(x, y, z, bins=40) {{
      let xmin=Infinity, xmax=-Infinity, ymin=Infinity, ymax=-Infinity;
      for (let i=0;i<x.length;i++) {{
        xmin = Math.min(xmin, x[i]); xmax = Math.max(xmax, x[i]);
        ymin = Math.min(ymin, y[i]); ymax = Math.max(ymax, y[i]);
      }}
      if (!(Number.isFinite(xmin)&&Number.isFinite(xmax)&&Number.isFinite(ymin)&&Number.isFinite(ymax))) {{
        return null;
      }}
      if (xmax === xmin) xmax = xmin + 1e-9;
      if (ymax === ymin) ymax = ymin + 1e-9;

      const sum = Array.from({{length: bins}}, () => Array(bins).fill(0));
      const cnt = Array.from({{length: bins}}, () => Array(bins).fill(0));
      for (let i=0;i<x.length;i++) {{
        const xi = Math.min(bins-1, Math.max(0, Math.floor((x[i]-xmin)/(xmax-xmin)*bins)));
        const yi = Math.min(bins-1, Math.max(0, Math.floor((y[i]-ymin)/(ymax-ymin)*bins)));
        const zi = z[i];
        if (!Number.isFinite(zi)) continue;
        sum[yi][xi] += zi;
        cnt[yi][xi] += 1;
      }}
      const zmat = sum.map((row, r) => row.map((v, c) => cnt[r][c] ? v/cnt[r][c] : null));
      const xcent = Array.from({{length: bins}}, (_, i) => xmin + (i+0.5)*(xmax-xmin)/bins);
      const ycent = Array.from({{length: bins}}, (_, i) => ymin + (i+0.5)*(ymax-ymin)/bins);
      return {{ x: xcent, y: ycent, z: zmat }};
    }}

    let filters = null;
    function updatePlot() {{
      const state = currentState();
      if (filters === null) {{
        filters = makeFilters();
      }}
      // Rebuild filters if x/y changed (since those params should be removed from filters).
      // Cheap approach: if any filter references current x/y, rebuild.
      if (filters[state.x] || filters[state.y]) {{
        filters = makeFilters();
      }}

      const xcol = records.params[state.x];
      const ycol = records.params[state.y];
      const mcol = records.metrics[state.metric];

      const xs = [];
      const ys = [];
      const ms = [];
      const hover = [];
      for (let i=0;i<records.point_id.length;i++) {{
        if (!passesFilters(i, state, filters)) continue;
        const xv = xcol[i], yv = ycol[i], mv = mcol[i];
        if (!Number.isFinite(xv) || !Number.isFinite(yv) || !Number.isFinite(mv)) continue;
        xs.push(xv); ys.push(yv); ms.push(mv);
        hover.push("point_id=" + records.point_id[i] + "<br>" +
                   state.x + "=" + fmt(xv) + "<br>" +
                   state.y + "=" + fmt(yv) + "<br>" +
                   state.metric + "=" + fmt(mv));
      }}

      el("kpiPoints").textContent = xs.length + " / " + records.point_id.length;
      if (ms.length) {{
        let lo = Math.min(...ms), hi = Math.max(...ms);
        el("kpiRange").textContent = fmt(lo) + " .. " + fmt(hi);
      }} else {{
        el("kpiRange").textContent = "-";
      }}

      const traces = [];
      if (state.heatmap && xs.length) {{
        const hm = binnedHeatmap(xs, ys, ms, 45);
        if (hm) {{
          traces.push({{
            type: "heatmap",
            x: hm.x,
            y: hm.y,
            z: hm.z,
            colorscale: "Viridis",
            showscale: true,
            opacity: 0.88,
            colorbar: {{ title: state.metric }},
            hoverinfo: "skip",
          }});
        }}
      }}

      traces.push({{
        type: "scattergl",
        mode: "markers",
        x: xs,
        y: ys,
        text: hover,
        hoverinfo: "text",
        marker: {{
          size: 7,
          color: ms,
          colorscale: "Turbo",
          opacity: 0.92,
          colorbar: state.heatmap ? undefined : {{ title: state.metric }},
          line: {{ width: 0 }}
        }}
      }});

      const layout = {{
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(255,255,255,0.02)",
        margin: {{ l: 55, r: 25, t: 10, b: 45 }},
        xaxis: {{ title: state.x, zeroline: false, gridcolor: "rgba(255,255,255,0.06)" }},
        yaxis: {{ title: state.y, zeroline: false, gridcolor: "rgba(255,255,255,0.06)" }},
        font: {{ color: "rgba(255,255,255,0.88)" }},
        showlegend: false,
      }};
      Plotly.react("plot", traces, layout, {{displayModeBar: true, responsive: true}});
    }}

    function init() {{
      buildSelect(
        el("metricSelect"),
        metrics.map(m => ({{value: m.key, label: m.label}})),
        metrics[0].key
      );
      buildSelect(
        el("xSelect"),
        paramOrder.map(p => ({{value: p, label: p}})),
        paramOrder[0]
      );
      buildSelect(
        el("ySelect"),
        paramOrder.map(p => ({{value: p, label: p}})),
        paramOrder.length > 1 ? paramOrder[1] : paramOrder[0]
      );

      el("metricSelect").addEventListener("change", () => updatePlot());
      el("xSelect").addEventListener("change", () => {{ filters = null; updatePlot(); }});
      el("ySelect").addEventListener("change", () => {{ filters = null; updatePlot(); }});
      el("heatmapToggle").addEventListener("change", () => updatePlot());
      el("includeFailedToggle").addEventListener("change", () => updatePlot());
      updatePlot();
    }}
    init();
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    """Entry point for building a sampled-design HTML dashboard.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - This script is cache-only: it does not re-run simulations.
    - It expects a CSV produced by `scripts/run_experiment_design.py`.

    Examples
    --------
    `python -m scripts.build_experiment_design_dashboard --cache abm_results/experiments_runs/<tag>/data/points_<tag>.csv --meta abm_results/experiments_runs/<tag>/data/meta_<tag>.json`
    """
    parser = argparse.ArgumentParser(description="Build an HTML dashboard from an experiment cache CSV.")
    parser.add_argument("--cache", type=Path, required=True, help="Path to cached points CSV (points_<tag>.csv).")
    parser.add_argument("--meta", type=Path, default=None, help="Optional meta JSON (meta_<tag>.json).")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default next to cache).")
    parser.add_argument("--title", type=str, default=None, help="Override HTML title.")
    args = parser.parse_args()

    cache_path = args.cache.expanduser().resolve()
    if not cache_path.exists():
        raise SystemExit(f"Cache CSV not found: {cache_path}")
    dataframe = pd.read_csv(cache_path)
    if dataframe.empty:
        raise SystemExit(f"Cache CSV is empty: {cache_path}")

    meta = _load_meta(args.meta)
    columns = list(map(str, dataframe.columns))

    param_order = None
    if meta is not None:
        space = meta.get("space")
        if isinstance(space, list):
            names = []
            for entry in space:
                if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                    names.append(str(entry["name"]))
            if names:
                param_order = names
    if param_order is None:
        param_order = _infer_param_names(columns)

    param_cols = [f"p__{name}" for name in param_order]
    missing_params = [c for c in param_cols if c not in dataframe.columns]
    if missing_params:
        raise SystemExit(f"Cache is missing required parameter columns: {missing_params}")

    metric_cols = _infer_metric_columns(columns)
    metrics = [{"key": c, "label": _metric_label(c)} for c in metric_cols]

    param_kinds = _space_kind_from_meta(meta)
    param_uniques = {name: _unique_values(dataframe, f"p__{name}") for name in param_order}

    # Records (column-oriented for faster JS filtering)
    records = {
        "point_id": [int(v) for v in dataframe["point_id"].tolist()] if "point_id" in dataframe.columns else list(range(len(dataframe))),
        "ok": [bool(v) for v in dataframe["ok"].tolist()] if "ok" in dataframe.columns else [True for _ in range(len(dataframe))],
        "params": {name: [float(v) if np.isfinite(v) else None for v in dataframe[f"p__{name}"].astype(float).tolist()] for name in param_order},
        "metrics": {c: [float(v) if np.isfinite(v) else None for v in dataframe[c].astype(float).tolist()] for c in metric_cols},
    }

    title = args.title
    if title is None:
        title = f"ABM Experiment Dashboard • {cache_path.stem}"

    header_lines: List[str] = []
    if meta is not None:
        scenario_label = meta.get("scenario_label")
        base_cfg = meta.get("base_config_path") or meta.get("config_path")
        design = meta.get("design", {}).get("type") if isinstance(meta.get("design"), dict) else None
        if scenario_label:
            header_lines.append(f"Scenario: {scenario_label}")
        if base_cfg:
            header_lines.append(f"Base config: {base_cfg}")
        if design:
            header_lines.append(f"Design: {design}")
        if meta.get("git_commit"):
            header_lines.append(f"Git: {meta.get('git_commit')}")
        if meta.get("config_content_hash"):
            header_lines.append(f"Config hash: {meta.get('config_content_hash')}")
        if meta.get("experiment_content_hash"):
            header_lines.append(f"Experiment hash: {meta.get('experiment_content_hash')}")

    payload = {
        "title": title,
        "param_order": list(param_order),
        "param_kinds": dict(param_kinds),
        "param_uniques": dict(param_uniques),
        "metrics": metrics,
        "records": records,
    }

    output_path = args.output
    if output_path is None:
        output_path = cache_path.with_suffix(".dashboard.html")
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        stem = output_path.stem
        suffix = output_path.suffix
        parent = output_path.parent
        k = 1
        while True:
            candidate = parent / f"{stem}_{k}{suffix}"
            if not candidate.exists():
                output_path = candidate
                break
            k += 1

    _write_dashboard_html(output_path, title=title, header_lines=header_lines, payload=payload)
    print(f"[experiment_dashboard] cache: {cache_path}")
    if args.meta is not None:
        print(f"[experiment_dashboard] meta:  {Path(args.meta).expanduser().resolve()}")
    print(f"[experiment_dashboard] html:  {output_path}")


if __name__ == "__main__":
    main()
