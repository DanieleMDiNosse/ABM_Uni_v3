from __future__ import annotations

import argparse
import shutil
import threading
import math
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Event, Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yaml

from abm_webapp.storage import read_metrics, read_status, tail_text_file
from abm_webapp.worker import run_simulation_process

PLOTLY_TEMPLATE = "plotly_dark"


def _list_scenario_files(scenarios_dir: Path) -> List[Path]:
    if not scenarios_dir.exists():
        return []
    return sorted([p for p in scenarios_dir.glob("*.yml") if p.is_file()] + [p for p in scenarios_dir.glob("*.yaml") if p.is_file()])


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _default_run_id() -> str:
    # Keep it filesystem-friendly and sortable.
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _web_runs_root() -> Path:
    return Path("abm_results") / "web_runs"


def _run_root_for(run_id: str) -> Path:
    return _web_runs_root() / run_id


def _safe_yaml_parse(text: str) -> Tuple[bool, str]:
    try:
        yaml.safe_load(text)
    except Exception as exc:
        return False, f"YAML parse error: {exc}"
    return True, ""


def _validate_config_against_simulate(config_yaml: str) -> Tuple[bool, str]:
    """
    Validate YAML config against simulate() signature (fast fail before starting a run).

    Notes
    -----
    This reuses `utils.load_simulation_parameters` for consistency with CLI scenarios.
    """
    ok, err = _safe_yaml_parse(config_yaml)
    if not ok:
        return False, err

    try:
        from tempfile import TemporaryDirectory

        from run import simulate
        from utils import load_simulation_parameters

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "scenario.yml"
            tmp_path.write_text(config_yaml, encoding="utf-8")
            load_simulation_parameters(tmp_path, simulate_func=simulate)
    except Exception as exc:
        return False, f"Config validation failed: {type(exc).__name__}: {exc}"

    return True, ""


def _build_price_figure(rows: List[Dict[str, Any]]) -> go.Figure:
    t = [r["t"] for r in rows]
    dex = [r.get("dex_price") for r in rows]
    cex = [r.get("cex_price") for r in rows]
    band_lo = [r.get("band_lo") for r in rows]
    band_hi = [r.get("band_hi") for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=band_lo, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=t,
            y=band_hi,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(180,180,180,0.35)",
            line=dict(width=0),
            name="No-arb fee band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=t, y=dex, mode="lines", name="DEX price", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=t, y=cex, mode="lines", name="CEX price", line=dict(width=1.6, dash="dash")))
    fig.update_layout(template=PLOTLY_TEMPLATE, title="CEX vs DEX Price (live)")
    fig.update_xaxes(title_text="Block")
    fig.update_yaxes(title_text="Price")
    return fig


def _log_returns(prices: List[Optional[float]]) -> List[float]:
    out: List[float] = []
    prev: Optional[float] = None
    for p in prices:
        if p is None:
            prev = None
            continue
        if p <= 0:
            prev = None
            continue
        if prev is None or prev <= 0:
            prev = p
            continue
        out.append(float(math.log(p) - math.log(prev)))
        prev = p
    return out


def _build_price_distribution_figure(rows: List[Dict[str, Any]]) -> go.Figure:
    """
    Build the "distribution" panel for the price tab.

    Notes
    -----
    The offline runner shows histograms of log-returns (CEX and DEX). We replicate that
    here (as a separate figure shown below the time-series price+band plot).
    """
    cex = [r.get("cex_price") for r in rows]
    dex = [r.get("dex_price") for r in rows]
    cex_rets = _log_returns(cex)
    dex_rets = _log_returns(dex)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("CEX log-returns", "DEX log-returns"))
    fig.add_trace(
        go.Histogram(x=cex_rets, nbinsx=60, marker_color="#1f77b4", opacity=0.85, showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=dex_rets, nbinsx=60, marker_color="#ff7f0e", opacity=0.85, showlegend=False),
        row=1,
        col=2,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, title="Price distributions (log-returns)", bargap=0.05)
    fig.update_xaxes(title_text="Log-return", row=1, col=1)
    fig.update_xaxes(title_text="Log-return", row=1, col=2)
    fig.update_yaxes(title_text="Count", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Count", type="log", row=1, col=2)
    return fig


def _diff_series(vals: List[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    prev: Optional[float] = None
    for v in vals:
        if v is None:
            out.append(None)
            prev = v
            continue
        if prev is None:
            out.append(v)
        else:
            out.append(v - prev)
        prev = v
    return out


def _cumsum(vals: List[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    acc: float = 0.0
    for v in vals:
        if v is None:
            out.append(None)
            continue
        acc += float(v)
        out.append(acc)
    return out


def _build_pnl_per_block_figure(rows: List[Dict[str, Any]]) -> go.Figure:
    t = [r["t"] for r in rows]
    sr = [r.get("sr_pnl_step") for r in rows]
    noise = [r.get("noise_pnl_step") for r in rows]
    arb = [r.get("arb_pnl_step") for r in rows]
    lp_active = _diff_series([r.get("lp_pnl_active") for r in rows])
    lp_passive = _diff_series([r.get("lp_pnl_passive") for r in rows])
    jiter = _diff_series([r.get("jiter_pnl") for r in rows])

    fig = go.Figure()
    fig.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=t, y=sr, mode="lines", name="Smart router"))
    fig.add_trace(go.Scatter(x=t, y=noise, mode="lines", name="Noise trader", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=arb, mode="lines", name="Arbitrageur"))
    fig.add_trace(go.Scatter(x=t, y=lp_active, mode="lines", name="Active LP (Δ)", line=dict(dash="dashdot")))
    fig.add_trace(go.Scatter(x=t, y=lp_passive, mode="lines", name="Passive LP (Δ)", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=jiter, mode="lines", name="Jiter (Δ)", line=dict(width=2)))
    fig.update_layout(template=PLOTLY_TEMPLATE, title="PnL per block (live)")
    fig.update_xaxes(title_text="Block")
    fig.update_yaxes(title_text="PnL (token1)")
    return fig


def _finite(values: List[Optional[float]]) -> List[float]:
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if fv != fv:  # NaN
            continue
        if fv in (float("inf"), float("-inf")):
            continue
        out.append(fv)
    return out


def _build_pnl_per_block_distribution_figure(rows: List[Dict[str, Any]]) -> go.Figure:
    """
    Build the "distribution" panel for per-block PnL.

    Notes
    -----
    - For Smart router / Noise trader / Arbitrageur we already store per-block PnL.
    - For LP and Jiter series we store the *state* (cumulative/level), so we
      compute per-block increments by first differences (Δ series), matching the
      offline dashboard logic.
    """
    series = [
        ("Smart router", _finite([r.get("sr_pnl_step") for r in rows]), "#1f77b4"),
        ("Noise trader", _finite([r.get("noise_pnl_step") for r in rows]), "#ff7f0e"),
        ("Arbitrageur", _finite([r.get("arb_pnl_step") for r in rows]), "#2ca02c"),
        ("Active LP (Δ)", _finite(_diff_series([r.get("lp_pnl_active") for r in rows])), "#9467bd"),
        ("Passive LP (Δ)", _finite(_diff_series([r.get("lp_pnl_passive") for r in rows])), "#8c564b"),
        ("Jiter (Δ)", _finite(_diff_series([r.get("jiter_pnl") for r in rows])), "#d62728"),
    ]

    fig = make_subplots(rows=3, cols=2, subplot_titles=[s[0] for s in series], vertical_spacing=0.12)
    for i, (label, vals, color) in enumerate(series):
        row = (i // 2) + 1
        col = (i % 2) + 1
        fig.add_trace(
            go.Histogram(x=vals, nbinsx=60, marker_color=color, opacity=0.85, showlegend=False),
            row=row,
            col=col,
        )

    fig.update_layout(template=PLOTLY_TEMPLATE, title="PnL distributions per block", bargap=0.05, height=650)
    fig.update_yaxes(title_text="Count", type="log")
    fig.update_xaxes(title_text="PnL")
    return fig


def _build_pnl_cumulative_figure(rows: List[Dict[str, Any]]) -> go.Figure:
    t = [r["t"] for r in rows]
    sr_c = _cumsum([r.get("sr_pnl_step") for r in rows])
    noise_c = _cumsum([r.get("noise_pnl_step") for r in rows])
    arb_c = _cumsum([r.get("arb_pnl_step") for r in rows])
    lp_active = [r.get("lp_pnl_active") for r in rows]
    lp_passive = [r.get("lp_pnl_passive") for r in rows]
    jiter = [r.get("jiter_pnl") for r in rows]

    fig = go.Figure()
    fig.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=t, y=sr_c, mode="lines", name="Smart router"))
    fig.add_trace(go.Scatter(x=t, y=noise_c, mode="lines", name="Noise trader", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=arb_c, mode="lines", name="Arbitrageur"))
    fig.add_trace(go.Scatter(x=t, y=lp_active, mode="lines", name="Active LP", line=dict(dash="dashdot")))
    fig.add_trace(go.Scatter(x=t, y=lp_passive, mode="lines", name="Passive LP", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=jiter, mode="lines", name="Jiter", line=dict(width=2)))
    fig.update_layout(template=PLOTLY_TEMPLATE, title="Cumulative PnL (live)")
    fig.update_xaxes(title_text="Block")
    fig.update_yaxes(title_text="PnL (token1)")
    return fig


def _find_latest_log_in_run_root(run_root: Path) -> Optional[Path]:
    logs_dir = run_root / "logs"
    if not logs_dir.exists():
        return None
    candidates = [p for p in logs_dir.glob("*.txt") if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


_FEE_MODE_CACHE: Dict[str, str] = {}


def _get_fee_mode_for_run(run_root: Path) -> str:
    """
    Infer the fee mode from the run's saved scenario YAML.

    Notes
    -----
    The worker writes `scenario.yml` into the run directory. We keep a tiny cache
    to avoid reparsing the YAML on every polling tick.
    """
    key = str(run_root)
    cached = _FEE_MODE_CACHE.get(key)
    if cached is not None:
        return cached

    scenario_path = run_root / "scenario.yml"
    fee_mode = "static"
    if scenario_path.exists():
        try:
            data = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                fee_mode_raw = data.get("fee_mode")
                if fee_mode_raw is None and isinstance(data.get("simulate"), dict):
                    fee_mode_raw = data["simulate"].get("fee_mode")
                if fee_mode_raw is not None:
                    fee_mode = str(fee_mode_raw)
        except Exception:
            fee_mode = "static"

    _FEE_MODE_CACHE[key] = fee_mode
    return fee_mode


def _build_fee_figure(rows: List[Dict[str, Any]], *, fee_mode: str) -> go.Figure:
    """
    Build the fee+signal panel (time series + fee distribution), matching run.py.
    """
    steps = [r["t"] for r in rows]
    fee_series = [r.get("fee") for r in rows]
    fee_sigma_series = [r.get("fee_sigma") for r in rows]
    fee_basis_ticks_series = [r.get("fee_basis_ticks") for r in rows]
    fee_signal_series = [r.get("fee_signal") for r in rows]

    if fee_mode in ("volatility_cex", "volatility_dex"):
        secondary_vals_full = fee_sigma_series
        secondary_label = "EWMA(σ^2)"
    elif fee_mode == "toxicity":
        secondary_vals_full = fee_basis_ticks_series
        secondary_label = "Basis (ticks)"
    elif fee_mode == "lvr_fee_ewma":
        secondary_vals_full = fee_signal_series
        secondary_label = "EWMA(dLVR - dFees) / notional"
    else:
        secondary_vals_full = fee_signal_series
        secondary_label = "Controller signal"

    # Align: signal at t -> fee applies at t+1. Plot fee_{t+1} at timestamp t.
    if len(steps) > 1:
        steps_fee_plot = steps[:-1]
        fee_plot = fee_series[1:]
        secondary_vals_plot = secondary_vals_full[:-1]
        fee_label = "Fee (applies next step; aligned to signal)"
    else:
        steps_fee_plot = steps
        fee_plot = fee_series
        secondary_vals_plot = secondary_vals_full
        fee_label = "Fee"

    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
    fig.add_trace(
        go.Scatter(x=steps_fee_plot, y=fee_plot, mode="lines", name=fee_label, line=dict(width=1.8)),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=steps_fee_plot,
            y=secondary_vals_plot,
            mode="lines",
            name=secondary_label,
            line=dict(width=1.2, dash="dash"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fee_vals = _finite([float(v) if v is not None else None for v in fee_series])
    fig.add_trace(
        go.Histogram(x=fee_vals, name="Fee distribution", marker_color="#1f77b4", opacity=0.75, showlegend=False),
        row=2,
        col=1,
    )

    if fee_vals:
        fee_arr = np.asarray(fee_vals, dtype=float)
        fee_mean = float(np.mean(fee_arr))
        fee_median = float(np.median(fee_arr))
        percentiles = [(p, float(np.percentile(fee_arr, p))) for p in (5, 25, 75, 95)]

        fig.add_vline(x=fee_mean, row=2, col=1, line=dict(color="firebrick", width=2, dash="dash"))
        fig.add_vline(x=fee_median, row=2, col=1, line=dict(color="white", width=2, dash="dot"))
        for (p, val), style in zip(
            percentiles,
            [
                dict(color="#9ca3af", dash="dot"),
                dict(color="#cbd5e1", dash="dash"),
                dict(color="#cbd5e1", dash="dash"),
                dict(color="#9ca3af", dash="dot"),
            ],
        ):
            fig.add_vline(x=val, row=2, col=1, line=dict(color=style["color"], width=1.6, dash=style["dash"]))

        # Legend handles
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"Mean = {fee_mean:.5f}", line=dict(color="firebrick", width=2, dash="dash")), row=2, col=1)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"Median = {fee_median:.5f}", line=dict(color="white", width=2, dash="dot")), row=2, col=1)
        for p, val in percentiles:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"P{p:02d} = {val:.5f}", line=dict(color="#9ca3af", width=1.6, dash="dot")), row=2, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Fee & Controller Signal (aligned)",
        bargap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=750,
    )
    fig.update_xaxes(title_text="Block", row=1, col=1)
    fig.update_yaxes(title_text="Fee", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text=secondary_label, row=1, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Fee", row=2, col=1)
    fig.update_yaxes(title_text="Count", type="log", row=2, col=1)
    return fig


@dataclass
class _RunController:
    lock: threading.Lock
    run_id: Optional[str] = None
    run_root: Optional[Path] = None
    process: Optional[Process] = None
    stop_event: Optional[Event] = None

    def start(self, *, config_yaml: str, live_every: int, log_flush_every: int) -> Tuple[bool, str]:
        """
        Start a new simulation run (stopping any existing run first).
        """
        with self.lock:
            if self.process is not None and self.process.is_alive():
                return False, "A run is already in progress. Stop it before starting a new one."

            ok, err = _validate_config_against_simulate(config_yaml)
            if not ok:
                return False, err

            run_id = _default_run_id()
            run_root = _run_root_for(run_id)
            run_root.mkdir(parents=True, exist_ok=True)

            stop_event = Event()
            proc = Process(
                target=run_simulation_process,
                kwargs=dict(
                    run_id=run_id,
                    run_root=str(run_root),
                    config_yaml=config_yaml,
                    stop_event=stop_event,
                    live_every=int(live_every),
                    log_flush_every=int(log_flush_every),
                ),
                daemon=True,
            )
            proc.start()

            self.run_id = run_id
            self.run_root = run_root
            self.process = proc
            self.stop_event = stop_event
            return True, f"Started run {run_id}"

    def stop(self) -> Tuple[bool, str]:
        """Request the current run to stop."""
        with self.lock:
            if self.process is None or self.stop_event is None:
                return False, "No active run."
            if not self.process.is_alive():
                return False, "No active run."
            self.stop_event.set()
            return True, "Stop requested."

    def reset_all(self) -> Tuple[bool, str]:
        """
        Stop any running simulation and remove all webapp run outputs.

        Notes
        -----
        This clears only `abm_results/web_runs/` (webapp runs). It does *not* touch
        `abm_results/scenarios/` or other offline outputs.
        """
        with self.lock:
            if self.process is not None and self.stop_event is not None and self.process.is_alive():
                self.stop_event.set()
                self.process.join(timeout=10.0)
                if self.process.is_alive():
                    # Last resort: force terminate to allow cleanup.
                    try:
                        self.process.terminate()
                    except Exception:
                        pass
                    self.process.join(timeout=5.0)

            self.run_id = None
            self.run_root = None
            self.process = None
            self.stop_event = None
            _FEE_MODE_CACHE.clear()

        root = _web_runs_root()
        try:
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return False, f"Reset failed: {type(exc).__name__}: {exc}"
        return True, f"Reset complete: cleared {root}"


_CTRL = _RunController(lock=threading.Lock())


def _build_dash_app():
    try:
        from dash import Dash, Input, Output, State, dcc, html
    except Exception as exc:  # pragma: no cover - dash is optional for non-web workflows
        raise RuntimeError(
            "Dash is required for the webapp. Install it with `pip install dash` (or `conda install -c conda-forge dash`)."
        ) from exc

    scenarios_dir = Path("abm_results") / "scenarios"
    scenario_files = _list_scenario_files(scenarios_dir)
    scenario_options = [{"label": p.name, "value": str(p)} for p in scenario_files]
    default_yaml = _load_text(scenario_files[0]) if scenario_files else ""

    assets_dir = Path(__file__).with_name("assets")
    app = Dash(__name__, assets_folder=str(assets_dir))
    app.layout = html.Div(
        style={
            "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
            "padding": "16px",
            "backgroundColor": "#0b1220",
            "color": "#e5e7eb",
            "minHeight": "100vh",
        },
        children=[
            html.H2("ABM_Uni_v3 — Live Simulation Webapp"),
            html.Div(
                style={"display": "flex", "gap": "16px", "alignItems": "flex-start"},
                children=[
                    html.Div(
                        style={"flex": "1", "minWidth": "420px"},
                        children=[
                            html.H4("Scenario / Parameters"),
                            html.Label("Load scenario file"),
                            dcc.Dropdown(
                                id="scenario-dropdown",
                                options=scenario_options,
                                value=str(scenario_files[0]) if scenario_files else None,
                                placeholder="Select a scenario YAML...",
                            ),
                            html.Div(style={"height": "8px"}),
                            html.Label("Edit YAML config"),
                            dcc.Textarea(
                                id="yaml-editor",
                                value=default_yaml,
                                style={"width": "100%", "height": "340px", "fontFamily": "monospace"},
                            ),
                            html.Div(style={"display": "flex", "gap": "12px", "marginTop": "10px"}),
                            html.Div(
                                style={"display": "flex", "gap": "12px", "alignItems": "center"},
                                children=[
                                    html.Button("Run", id="run-btn", n_clicks=0),
                                    html.Button("Stop", id="stop-btn", n_clicks=0),
                                    dcc.ConfirmDialogProvider(
                                        id="reset-confirm",
                                        message=(
                                            "Reset will stop any running simulation and delete all webapp runs under "
                                            "abm_results/web_runs/. This cannot be undone. Continue?"
                                        ),
                                        children=html.Button("Reset", id="reset-btn", n_clicks=0),
                                    ),
                                    html.Span("live_every"),
                                    dcc.Input(id="live-every", type="number", value=1, min=1, step=1, style={"width": "90px"}),
                                    html.Span("log_flush_every"),
                                    dcc.Input(id="log-flush-every", type="number", value=200, min=1, step=1, style={"width": "110px"}),
                                ],
                            ),
                            html.Div(style={"height": "8px"}),
                            html.Div(id="run-message", style={"whiteSpace": "pre-wrap"}),
                            html.Div(id="run-status", style={"whiteSpace": "pre-wrap", "marginTop": "6px"}),
                            dcc.Store(id="run-root-store"),
                        ],
                    ),
                    html.Div(
                        style={"flex": "2", "minWidth": "520px"},
                        children=[
                            dcc.Tabs(
                                children=[
                                    dcc.Tab(
                                        label="Prices + band",
                                        children=[
                                            dcc.Graph(id="price-graph", style={"height": "520px"}),
                                            dcc.Graph(id="price-dist-graph", style={"height": "360px"}),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="PnL per block",
                                        children=[
                                            dcc.Graph(id="pnl-step-graph", style={"height": "520px"}),
                                            dcc.Graph(id="pnl-step-dist-graph", style={"height": "650px"}),
                                        ],
                                    ),
                                    dcc.Tab(label="Cumulative PnL", children=[dcc.Graph(id="pnl-cum-graph", style={"height": "700px"})]),
                                    dcc.Tab(label="Fees", children=[dcc.Graph(id="fee-graph", style={"height": "760px"})]),
                                    dcc.Tab(
                                        label="Logs",
                                        children=[
                                            html.Pre(
                                                id="logs-pre",
                                                style={
                                                    "height": "650px",
                                                    "overflowY": "scroll",
                                                    "background": "#111827",
                                                    "color": "#E5E7EB",
                                                    "padding": "12px",
                                                },
                                            )
                                        ],
                                    ),
                                ]
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Interval(id="poll-interval", interval=1000, n_intervals=0),
        ],
    )

    @app.callback(Output("yaml-editor", "value"), Input("scenario-dropdown", "value"))
    def _load_scenario(selected_path: Optional[str]) -> str:
        if not selected_path:
            return ""
        p = Path(selected_path)
        if not p.exists():
            return ""
        return _load_text(p)

    @app.callback(
        Output("run-message", "children"),
        Output("run-root-store", "data"),
        Input("run-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        Input("reset-confirm", "submit_n_clicks"),
        State("yaml-editor", "value"),
        State("live-every", "value"),
        State("log-flush-every", "value"),
        prevent_initial_call=True,
    )
    def _handle_run_stop(
        run_clicks: int,
        stop_clicks: int,
        reset_clicks: int,
        yaml_text: str,
        live_every: Any,
        log_flush_every: Any,
    ) -> Tuple[str, Optional[str]]:
        from dash import callback_context

        if not callback_context.triggered:
            return "", None
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "stop-btn":
            ok, msg = _CTRL.stop()
            run_root = str(_CTRL.run_root) if _CTRL.run_root is not None else None
            return msg, run_root
        if trigger == "reset-confirm":
            ok, msg = _CTRL.reset_all()
            return msg, None

        # run-btn
        live_every_i = int(live_every) if live_every is not None else 1
        log_flush_i = int(log_flush_every) if log_flush_every is not None else 200
        ok, msg = _CTRL.start(config_yaml=yaml_text or "", live_every=live_every_i, log_flush_every=log_flush_i)
        run_root = str(_CTRL.run_root) if _CTRL.run_root is not None else None
        return msg, run_root

    @app.callback(
        Output("price-graph", "figure"),
        Output("price-dist-graph", "figure"),
        Output("pnl-step-graph", "figure"),
        Output("pnl-step-dist-graph", "figure"),
        Output("pnl-cum-graph", "figure"),
        Output("fee-graph", "figure"),
        Output("logs-pre", "children"),
        Output("run-status", "children"),
        Input("poll-interval", "n_intervals"),
        State("run-root-store", "data"),
        prevent_initial_call=False,
    )
    def _poll(n: int, run_root_str: Optional[str]):
        empty_fig = go.Figure().update_layout(template=PLOTLY_TEMPLATE)
        if not run_root_str:
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, "", ""

        run_root = Path(run_root_str)
        db_path = run_root / "live.db"
        status = read_status(db_path)
        status_line = ""
        if status is not None:
            status_line = f"run_id={status.run_id} state={status.state} t_last={status.t_last} updated={status.updated_at}\n{status.message}"

        rows = read_metrics(db_path, limit=10_000)
        fig_price = _build_price_figure(rows) if rows else empty_fig
        fig_price_dist = _build_price_distribution_figure(rows) if rows else empty_fig
        fig_step = _build_pnl_per_block_figure(rows) if rows else empty_fig
        fig_step_dist = _build_pnl_per_block_distribution_figure(rows) if rows else empty_fig
        fig_cum = _build_pnl_cumulative_figure(rows) if rows else empty_fig
        fee_mode = _get_fee_mode_for_run(run_root)
        fig_fee = _build_fee_figure(rows, fee_mode=fee_mode) if rows else empty_fig

        log_text = ""
        log_path = Path(status.log_path) if (status and status.log_path) else None
        if log_path is None or not log_path.exists():
            log_path = _find_latest_log_in_run_root(run_root)
        if log_path is not None and log_path.exists():
            log_text = tail_text_file(log_path, max_bytes=80_000)

        return fig_price, fig_price_dist, fig_step, fig_step_dist, fig_cum, fig_fee, log_text, status_line

    return app


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run the Dash webapp.

    Examples
    --------
    `python -m abm_webapp.app --port 8050`
    """
    parser = argparse.ArgumentParser(description="ABM_Uni_v3 live simulation webapp (Dash).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args(argv)

    app = _build_dash_app()
    # Important: disable the reloader; it spawns a second process which confuses multiprocessing state.
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
