from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Event, Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

from abm_webapp.storage import read_metrics, read_status, scan_and_recover
from abm_webapp.worker import run_simulation_process

PLOTLY_TEMPLATE = "plotly_dark"
MAX_TIMESERIES_POINTS = 1800
MAX_DISTRIBUTION_POINTS = 6000
LIVE_TRANSITION_MS = 180
SSE_LOOP_SLEEP_SECONDS = 0.80
SSE_HEARTBEAT_SECONDS = 12.0
DEFAULT_LIVE_EVERY = 1
DEFAULT_LOG_FLUSH_EVERY = 200
LOG_TAIL_MAX_BYTES = 120_000
LOG_DELTA_MAX_BYTES = 65_536
LOG_TEXT_MAX_CHARS = 120_000
MEDIUM_FIG_UPDATE_EVERY = 2
HEAVY_FIG_UPDATE_EVERY = 5
TERMINAL_RUN_STATES = {"finished", "stopped", "error", "abandoned"}
EXPLORATORY_LAB_NOTICE = (
    "Exploratory single-run laboratory: use this webapp to quickly inspect fee schedules and parameter "
    "effects. Use CLI multi-run scripts for paper-grade results, robustness checks, and confirmatory claims."
)
FEE_SCHEDULE_PRESETS = {
    "static": "Static fee",
    "volatility_cex": "CEX volatility fee",
    "volatility_dex": "DEX volatility fee",
    "toxicity": "Toxicity / basis fee",
    "lvr_fee_ewma": "LVR EWMA fee",
    "linear_asymmetric": "Linear asymmetric fee",
}
LINEAR_ASYMMETRIC_DEFAULT_SLOPE = 0.5


def _env_optional_positive_int(name: str, default: Optional[int]) -> Optional[int]:
    """Read an optional positive integer from the environment."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else None


# Keep full live history by default so line plots always retain the run from
# t=0. Users can re-enable trimming explicitly if they need a hard memory cap.
LIVE_METRICS_LIMIT: Optional[int] = _env_optional_positive_int("ABM_WEBAPP_LIVE_METRICS_LIMIT", None)

# Dark-theme palette (high contrast on low-luminance backgrounds).
CLR_DEX = "#2DD4BF"
CLR_CEX = "#E2E8F0"
CLR_BAND_FILL = "rgba(148, 163, 184, 0.28)"
CLR_GUIDE = "#94A3B8"
CLR_SR = "#60A5FA"
CLR_NOISE = "#F59E0B"
CLR_ARB = "#4ADE80"
CLR_LP_ACTIVE = "#C084FC"
CLR_LP_PASSIVE = "#FBBF24"
CLR_JITER = "#FB7185"
CLR_FEE = "#22D3EE"
CLR_SIGNAL = "#A3B3C7"
CLR_MEAN = "#F87171"
CLR_MEDIAN = "#E2E8F0"
CLR_PERCENTILE = "#94A3B8"
CLR_LVR = "#F472B6"
CLR_HIST = "#93C5FD"


def _empty_fig() -> go.Figure:
    """Return a dark-themed empty figure for initial / empty states."""
    return go.Figure().update_layout(template=PLOTLY_TEMPLATE)


@dataclass
class _MetricsCacheEntry:
    rows: List[Dict[str, Any]]
    last_t: int
    data_version: int  # monotonically increasing; bumped whenever new rows arrive


_METRICS_CACHE: Dict[str, _MetricsCacheEntry] = {}
_METRICS_CACHE_LOCK = threading.Lock()

# Per-tier, per-run tracking: maps (run_root_key, tier) → last data_version rendered.
_TIER_LAST_RENDERED: Dict[Tuple[str, str], int] = {}
_TIER_LAST_RENDERED_LOCK = threading.Lock()
# Per-tier, per-run wall-clock of the last render.
_TIER_LAST_RENDER_TIME: Dict[Tuple[str, str], float] = {}


def _clear_tier_tracking(run_root_key: Optional[str] = None) -> None:
    """Clear per-tier render tracking for a specific run or all runs."""
    with _TIER_LAST_RENDERED_LOCK:
        if run_root_key is None:
            _TIER_LAST_RENDERED.clear()
            _TIER_LAST_RENDER_TIME.clear()
        else:
            for tier in ("medium", "heavy"):
                _TIER_LAST_RENDERED.pop((run_root_key, tier), None)
                _TIER_LAST_RENDER_TIME.pop((run_root_key, tier), None)


def _should_render_tier(
    run_root_key: str,
    tier: str,
    current_data_version: int,
    *,
    min_interval_s: float,
    force: bool = False,
) -> bool:
    """
    Decide whether a downstream tier should re-render.

    Returns ``True`` when data has changed since the tier's last render **and**
    enough wall-clock time has elapsed (or ``force`` is set).  This avoids the
    previous race where Dash callback batching could overwrite a
    ``data_changed=True`` store with a subsequent ``data_changed=False`` tick
    before the tier callback had a chance to fire.
    """
    key = (run_root_key, tier)
    with _TIER_LAST_RENDERED_LOCK:
        last_ver = _TIER_LAST_RENDERED.get(key, -1)
        if current_data_version <= last_ver and not force:
            return False  # no new data since last render
        now = time.monotonic()
        last_time = _TIER_LAST_RENDER_TIME.get(key, 0.0)
        if (not force) and (now - last_time < min_interval_s):
            return False  # throttled — will catch up on a later tick
        _TIER_LAST_RENDERED[key] = current_data_version
        _TIER_LAST_RENDER_TIME[key] = now
        return True


def _clear_metrics_cache(run_root_key: Optional[str] = None) -> None:
    """
    Clear in-memory cached metrics used to incrementally hydrate live plots.

    Parameters
    ----------
    run_root_key
        Optional cache key for a single run root. If None, all cache entries are cleared.
    """
    with _METRICS_CACHE_LOCK:
        if run_root_key is None:
            _METRICS_CACHE.clear()
        else:
            _METRICS_CACHE.pop(run_root_key, None)


def _get_cached_metrics(run_root_key: str) -> Optional[_MetricsCacheEntry]:
    """Return cached metrics for a run, if present."""
    with _METRICS_CACHE_LOCK:
        return _METRICS_CACHE.get(run_root_key)


def _set_cached_metrics(run_root_key: str, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Replace cached metrics for a run and enforce the global live row cap.

    Returns
    -------
    Tuple[list, int]
        The (possibly trimmed) rows and the new ``data_version``.
        If *rows* is empty the cache entry is **not** created so that
        ``_get_cached_metrics`` still returns ``None`` and the next callback
        invocation treats it as the initial load (avoiding frozen empty figures).
    """
    if rows:
        trimmed = list(rows[-LIVE_METRICS_LIMIT:]) if LIVE_METRICS_LIMIT is not None else list(rows)
    else:
        trimmed = []
    if not trimmed:
        return [], 0
    last_t = int(trimmed[-1]["t"])
    with _METRICS_CACHE_LOCK:
        prev = _METRICS_CACHE.get(run_root_key)
        new_ver = (prev.data_version + 1) if prev is not None else 1
        _METRICS_CACHE[run_root_key] = _MetricsCacheEntry(rows=trimmed, last_t=last_t, data_version=new_ver)
    return trimmed, new_ver


def _append_cached_metrics(run_root_key: str, new_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Append newly streamed metrics to cache and return the full bounded series.

    Parameters
    ----------
    run_root_key
        Run-root cache key.
    new_rows
        Rows with strictly increasing `t` expected from SQLite polling.

    Returns
    -------
    Tuple[list, int]
        The merged rows and the current ``data_version``.
    """
    if not new_rows:
        cached = _get_cached_metrics(run_root_key)
        if cached is not None:
            return list(cached.rows), cached.data_version
        return [], 0

    with _METRICS_CACHE_LOCK:
        cached = _METRICS_CACHE.get(run_root_key)
        if cached is None:
            merged = list(new_rows)
            prev_ver = 0
        else:
            merged = list(cached.rows)
            prev_ver = cached.data_version
            last_t_seen = cached.last_t
            for row in new_rows:
                t_val = int(row["t"])
                if t_val > last_t_seen:
                    merged.append(row)
                    last_t_seen = t_val
        if LIVE_METRICS_LIMIT is not None:
            merged = merged[-LIVE_METRICS_LIMIT:]
        last_t = int(merged[-1]["t"]) if merged else -1
        new_ver = prev_ver + 1
        _METRICS_CACHE[run_root_key] = _MetricsCacheEntry(rows=merged, last_t=last_t, data_version=new_ver)
        return list(merged), new_ver


_LOG_TEXT_CACHE: Dict[str, str] = {}
_LOG_TEXT_CACHE_LOCK = threading.Lock()
_LOG_OFFSET_CACHE: Dict[str, Tuple[str, int]] = {}
_LOG_OFFSET_CACHE_LOCK = threading.Lock()


def _clear_log_offset_cache(run_root_key: Optional[str] = None) -> None:
    """Clear cached log offsets used for incremental tail updates."""
    with _LOG_OFFSET_CACHE_LOCK:
        if run_root_key is None:
            _LOG_OFFSET_CACHE.clear()
        else:
            _LOG_OFFSET_CACHE.pop(run_root_key, None)


def _get_log_offset_cache(run_root_key: str) -> Tuple[str, int]:
    """Get cached log path and byte offset for a run root."""
    with _LOG_OFFSET_CACHE_LOCK:
        return _LOG_OFFSET_CACHE.get(run_root_key, ("", 0))


def _set_log_offset_cache(run_root_key: str, log_path: str, offset: int) -> None:
    """Persist cached log path and byte offset for a run root."""
    with _LOG_OFFSET_CACHE_LOCK:
        _LOG_OFFSET_CACHE[run_root_key] = (str(log_path), max(0, int(offset)))


def _clear_log_cache(run_root_key: Optional[str] = None) -> None:
    """Clear cached log text used by event-driven UI updates."""
    with _LOG_TEXT_CACHE_LOCK:
        if run_root_key is None:
            _LOG_TEXT_CACHE.clear()
        else:
            _LOG_TEXT_CACHE.pop(run_root_key, None)


def _get_log_cache(run_root_key: str) -> Optional[str]:
    """Get cached log text for a run root, if present."""
    with _LOG_TEXT_CACHE_LOCK:
        return _LOG_TEXT_CACHE.get(run_root_key)


def _set_log_cache(run_root_key: str, text: str) -> str:
    """Set full cached log text for a run root with bounded length."""
    trimmed = str(text or "")[-LOG_TEXT_MAX_CHARS:]
    with _LOG_TEXT_CACHE_LOCK:
        _LOG_TEXT_CACHE[run_root_key] = trimmed
    return trimmed


def _append_log_cache(run_root_key: str, append_text: str) -> str:
    """Append a log chunk to cache and return the bounded full log text."""
    chunk = str(append_text or "")
    with _LOG_TEXT_CACHE_LOCK:
        prev = _LOG_TEXT_CACHE.get(run_root_key, "")
        merged = (prev + chunk)[-LOG_TEXT_MAX_CHARS:]
        _LOG_TEXT_CACHE[run_root_key] = merged
    return merged


def _status_to_dict(status: Any) -> Dict[str, Any]:
    """Convert a RunStatus-like object to a JSON-serializable mapping."""
    if status is None:
        return {}
    try:
        return {
            "run_id": str(getattr(status, "run_id", "")),
            "state": str(getattr(status, "state", "")),
            "t_last": int(getattr(status, "t_last", -1)),
            "message": str(getattr(status, "message", "")),
            "updated_at": str(getattr(status, "updated_at", "")),
            "log_path": str(getattr(status, "log_path", "")),
            "pid": getattr(status, "pid", None),
            "heartbeat_at": getattr(status, "heartbeat_at", None),
            "stop_reason": getattr(status, "stop_reason", None),
        }
    except Exception:
        return {}


def _status_signature(status_dict: Dict[str, Any]) -> Tuple[Any, ...]:
    """Stable tuple signature for status-change detection in SSE emission."""
    return (
        status_dict.get("run_id"),
        status_dict.get("state"),
        status_dict.get("message"),
        status_dict.get("log_path"),
    )


def _resolve_log_path(run_root: Path, status_dict: Dict[str, Any]) -> Optional[Path]:
    """Resolve current log path from status, with fallback scan in run root."""
    status_log = str(status_dict.get("log_path", "") or "")
    if status_log:
        candidate = Path(status_log)
        if candidate.exists():
            return candidate
    return _find_latest_log_in_run_root(run_root)


def _read_text_delta(path: Path, *, offset: int, max_bytes: int) -> Tuple[str, int]:
    """
    Read an incremental UTF-8-safe text chunk from a file.

    Parameters
    ----------
    path
        Text file path to tail.
    offset
        Starting byte offset for the read.
    max_bytes
        Maximum bytes to read for this chunk.

    Returns
    -------
    Tuple[str, int]
        Decoded text chunk and next byte offset.
    """
    if not path.exists():
        return "", max(0, int(offset))

    offset_i = max(0, int(offset))
    try:
        size = int(path.stat().st_size)
    except Exception:
        size = offset_i
    if offset_i > size:
        offset_i = 0

    with path.open("rb") as f:
        f.seek(offset_i)
        data = f.read(max(1, int(max_bytes)))
        next_offset = int(f.tell())

    if not data:
        return "", next_offset
    return data.decode("utf-8", errors="replace"), next_offset


def _format_sse_event(*, event: str, payload: Dict[str, Any], event_id: int) -> str:
    """Encode one SSE frame."""
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    lines = [f"id: {int(event_id)}", f"event: {event}"]
    for part in body.splitlines() or ["{}"]:
        lines.append(f"data: {part}")
    return "\n".join(lines) + "\n\n"


def _format_status_line(status_dict: Dict[str, Any]) -> str:
    """Render run status line from an already-serialized status dictionary."""
    if not status_dict:
        return ""
    run_id = str(status_dict.get("run_id", ""))
    state = str(status_dict.get("state", ""))
    t_last = status_dict.get("t_last", "")
    updated_at = str(status_dict.get("updated_at", ""))
    message = str(status_dict.get("message", ""))
    heartbeat_at = status_dict.get("heartbeat_at")
    stop_reason = status_dict.get("stop_reason")
    line = f"run_id={run_id} state={state} t_last={t_last} updated={updated_at}"
    if heartbeat_at:
        line += f" heartbeat={heartbeat_at}"
    if stop_reason:
        line += f" stop_reason={stop_reason}"
    return f"{line}\n{message}"


def _list_scenario_files(scenarios_dir: Path) -> List[Path]:
    runnable, _rejected = _partition_scenario_files(scenarios_dir)
    return runnable


def _partition_scenario_files(scenarios_dir: Path) -> Tuple[List[Path], Dict[str, str]]:
    """Return runnable scenario YAMLs and a name->reason map for rejected files."""
    if not scenarios_dir.exists():
        return [], {}

    candidates = sorted(
        [p for p in scenarios_dir.glob("*.yml") if p.is_file()]
        + [p for p in scenarios_dir.glob("*.yaml") if p.is_file()]
    )
    runnable: List[Path] = []
    rejected: Dict[str, str] = {}
    for path in candidates:
        try:
            text = _load_text(path)
        except Exception as exc:
            rejected[path.name] = f"Could not read scenario file: {type(exc).__name__}: {exc}"
            continue

        ok, err = _validate_config_against_simulate(text)
        if ok:
            runnable.append(path)
        else:
            rejected[path.name] = err

    return runnable, rejected


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
    from abm_webapp.config import safe_load_yaml
    data, err = safe_load_yaml(text)
    if err:
        return False, err
    return True, ""


def _apply_fee_schedule_preset(config_yaml: str, fee_mode: str) -> Tuple[str, str]:
    """
    Return YAML with top-level and simulate fee modes updated for exploratory webapp runs.

    The webapp remains YAML-first: this helper only changes the fee-schedule selector
    and leaves all other scientific parameters untouched.
    """
    fee_mode_s = str(fee_mode or "").strip()
    if fee_mode_s not in FEE_SCHEDULE_PRESETS:
        return config_yaml, f"Unknown fee schedule '{fee_mode_s}'."
    try:
        data = yaml.safe_load(config_yaml)
    except yaml.YAMLError as exc:
        return config_yaml, f"YAML parse error: {exc}"
    if not isinstance(data, dict):
        return config_yaml, "YAML root must be a mapping."
    simulate_block = data.get("simulate")
    if not isinstance(simulate_block, dict):
        return config_yaml, "Missing required 'simulate' mapping."
    updated = dict(data)
    updated_simulate = dict(simulate_block)
    updated["fee_mode"] = fee_mode_s
    updated_simulate["fee_mode"] = fee_mode_s
    if fee_mode_s == "linear_asymmetric" and "asymmetric_fee_slope" not in updated_simulate:
        updated_simulate["asymmetric_fee_slope"] = LINEAR_ASYMMETRIC_DEFAULT_SLOPE
    updated["simulate"] = updated_simulate
    return yaml.safe_dump(updated, sort_keys=False), ""


def _validate_config_against_simulate(config_yaml: str) -> Tuple[bool, str]:
    """
    Validate YAML config with strict schema checks + simulate() signature check.

    Notes
    -----
    First runs the strict schema validation (bounds, types, unknown keys),
    then validates against simulate() signature for full compatibility.
    """
    from abm_webapp.config import validate_scenario_text

    ok, err = validate_scenario_text(config_yaml)
    if not ok:
        return False, err

    try:
        from tempfile import TemporaryDirectory

        from scripts.run import simulate
        from core.utils import load_simulation_parameters

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "scenario.yml"
            tmp_path.write_text(config_yaml, encoding="utf-8")
            load_simulation_parameters(tmp_path, simulate_func=simulate)
    except Exception as exc:
        return False, f"Config validation failed: {type(exc).__name__}: {exc}"

    return True, ""


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _finite(values: Sequence[Optional[float]]) -> List[float]:
    out: List[float] = []
    for value in values:
        as_float = _to_float(value)
        if as_float is not None:
            out.append(as_float)
    return out


def _optional_sum(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = _finite(values)
    if not finite:
        return None
    return float(np.sum(np.asarray(finite, dtype=float)))


def _diff_series(vals: Sequence[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    prev: Optional[float] = None
    for value in vals:
        as_float = _to_float(value)
        if as_float is None:
            out.append(None)
            prev = None
            continue
        if prev is None:
            out.append(as_float)
        else:
            out.append(as_float - prev)
        prev = as_float
    return out


def _cumsum(vals: Sequence[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    acc = 0.0
    has_value = False
    for value in vals:
        as_float = _to_float(value)
        if as_float is None:
            out.append(None)
            continue
        acc += as_float
        has_value = True
        out.append(acc)
    if not has_value:
        return [None] * len(vals)
    return out


def _sum_pairs(a: Sequence[Optional[float]], b: Sequence[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for left, right in zip(a, b):
        lf = _to_float(left)
        rf = _to_float(right)
        if lf is None and rf is None:
            out.append(None)
        elif lf is None:
            out.append(rf)
        elif rf is None:
            out.append(lf)
        else:
            out.append(lf + rf)
    return out


def _safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    n = _to_float(num)
    d = _to_float(den)
    if n is None or d is None or abs(d) < 1e-18:
        return None
    return n / d


def _rolling_nanmedian(values: Sequence[Optional[float]], window: int) -> List[Optional[float]]:
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    arr = np.asarray([np.nan if _to_float(v) is None else float(v) for v in values], dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if arr.size < window:
        return out.tolist()
    for idx in range(window - 1, arr.size):
        current_window = arr[idx - window + 1 : idx + 1]
        if np.isfinite(current_window).any():
            out[idx] = float(np.nanmedian(current_window))
    return [None if not np.isfinite(v) else float(v) for v in out]


def _format_value(value: Optional[float], *, digits: int = 4, scale: float = 1.0, suffix: str = "") -> str:
    v = _to_float(value)
    if v is None:
        return "n/a"
    return f"{v * scale:,.{digits}f}{suffix}"


def _downsample_indices(size: int, *, max_points: int) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype=int)
    max_points_i = max(2, int(max_points))
    if size <= max_points_i:
        return np.arange(size, dtype=int)

    # Use fixed-stride downsampling so existing points remain stable across most
    # polling ticks; this avoids linspace-induced visual jitter during live redraws.
    step = max(1, int(math.ceil(size / max_points_i)))
    idx = np.arange(0, size, step, dtype=int)
    if idx[-1] != size - 1:
        idx = np.append(idx, size - 1)
    if idx.size > max_points_i:
        idx = idx[-max_points_i:]
    return np.unique(idx)


def _downsample_series(values: Sequence[Any], *, max_points: int) -> List[Any]:
    idx = _downsample_indices(len(values), max_points=max_points)
    return [values[int(i)] for i in idx]


def _downsample_xy(x: Sequence[Any], *ys: Sequence[Any], max_points: int) -> Tuple[List[Any], ...]:
    idx = _downsample_indices(len(x), max_points=max_points)
    out: List[List[Any]] = []
    out.append([x[int(i)] for i in idx])
    for series in ys:
        out.append([series[int(i)] for i in idx])
    return tuple(out)


def _log_returns(prices: Sequence[Optional[float]]) -> List[float]:
    out: List[float] = []
    prev: Optional[float] = None
    for price in prices:
        p = _to_float(price)
        if p is None or p <= 0:
            prev = None
            continue
        if prev is None or prev <= 0:
            prev = p
            continue
        out.append(float(math.log(p) - math.log(prev)))
        prev = p
    return out


def _build_price_figure(rows: List[Dict[str, Any]]) -> go.Figure:
    t = [r["t"] for r in rows]
    dex = [r.get("dex_price") for r in rows]
    cex = [r.get("cex_price") for r in rows]
    band_lo = [r.get("band_lo") for r in rows]
    band_hi = [r.get("band_hi") for r in rows]
    t, dex, cex, band_lo, band_hi = _downsample_xy(
        t, dex, cex, band_lo, band_hi, max_points=MAX_TIMESERIES_POINTS
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=band_lo,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=band_hi,
            mode="lines",
            fill="tonexty",
            fillcolor=CLR_BAND_FILL,
            line=dict(width=0),
            name="No-arb fee band",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=dex,
            mode="lines",
            name="DEX price",
            line=dict(width=2.2, color=CLR_DEX),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=cex,
            mode="lines",
            name="CEX price",
            line=dict(width=1.8, dash="dash", color=CLR_CEX),
        )
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, title="CEX vs DEX Price (live)")
    fig.update_xaxes(title_text="Block")
    fig.update_yaxes(title_text="Price")
    return fig


def _build_price_distribution_figure(rows: List[Dict[str, Any]]) -> go.Figure:
    """
    Build the distribution panel for prices.

    Notes
    -----
    Replicates the offline histogram view over CEX/DEX log-returns.
    """
    cex = [r.get("cex_price") for r in rows]
    dex = [r.get("dex_price") for r in rows]
    cex_rets = _downsample_series(_log_returns(cex), max_points=MAX_DISTRIBUTION_POINTS)
    dex_rets = _downsample_series(_log_returns(dex), max_points=MAX_DISTRIBUTION_POINTS)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("CEX log-returns", "DEX log-returns"))
    fig.add_trace(
        go.Histogram(x=cex_rets, nbinsx=60, marker_color=CLR_HIST, opacity=0.80, showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=dex_rets, nbinsx=60, marker_color=CLR_DEX, opacity=0.80, showlegend=False),
        row=1,
        col=2,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, title="Price distributions (log-returns)", bargap=0.05)
    fig.update_xaxes(title_text="Log-return", row=1, col=1)
    fig.update_xaxes(title_text="Log-return", row=1, col=2)
    fig.update_yaxes(title_text="Count", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Count", type="log", row=1, col=2)
    return fig


def _build_pnl_per_block_figure(rows: List[Dict[str, Any]]) -> go.Figure:
    t = [r["t"] for r in rows]
    sr = [r.get("sr_pnl_step") for r in rows]
    noise = [r.get("noise_pnl_step") for r in rows]
    arb = [r.get("arb_pnl_step") for r in rows]
    lp_active = _diff_series([r.get("lp_pnl_active") for r in rows])
    lp_passive = _diff_series([r.get("lp_pnl_passive") for r in rows])
    jiter = _diff_series([r.get("jiter_pnl") for r in rows])
    t, sr, noise, arb, lp_active, lp_passive, jiter = _downsample_xy(
        t,
        sr,
        noise,
        arb,
        lp_active,
        lp_passive,
        jiter,
        max_points=MAX_TIMESERIES_POINTS,
    )

    fig = go.Figure()
    fig.add_hline(y=0.0, line=dict(color=CLR_GUIDE, width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=t, y=sr, mode="lines", name="Smart router", line=dict(color=CLR_SR)))
    fig.add_trace(
        go.Scatter(
            x=t,
            y=noise,
            mode="lines",
            name="Noise trader",
            line=dict(dash="dash", color=CLR_NOISE),
        )
    )
    fig.add_trace(go.Scatter(x=t, y=arb, mode="lines", name="Arbitrageur", line=dict(color=CLR_ARB)))
    fig.add_trace(
        go.Scatter(x=t, y=lp_active, mode="lines", name="Active LP (Δ)", line=dict(dash="dashdot", color=CLR_LP_ACTIVE))
    )
    fig.add_trace(
        go.Scatter(x=t, y=lp_passive, mode="lines", name="Passive LP (Δ)", line=dict(dash="dot", color=CLR_LP_PASSIVE))
    )
    fig.add_trace(go.Scatter(x=t, y=jiter, mode="lines", name="Jiter (Δ)", line=dict(width=2, color=CLR_JITER)))
    fig.update_layout(template=PLOTLY_TEMPLATE, title="PnL per block (live)")
    fig.update_xaxes(title_text="Block")
    fig.update_yaxes(title_text="PnL (token1)")
    return fig


def _build_pnl_per_block_distribution_figure(rows: List[Dict[str, Any]]) -> go.Figure:
    """
    Build the distribution panel for per-block PnL.

    Notes
    -----
    LP and Jiter level series are converted to per-block deltas by first differences,
    matching offline analysis semantics.
    """
    series = [
        (
            "Smart router",
            _downsample_series(_finite([r.get("sr_pnl_step") for r in rows]), max_points=MAX_DISTRIBUTION_POINTS),
            CLR_SR,
        ),
        (
            "Noise trader",
            _downsample_series(_finite([r.get("noise_pnl_step") for r in rows]), max_points=MAX_DISTRIBUTION_POINTS),
            CLR_NOISE,
        ),
        (
            "Arbitrageur",
            _downsample_series(_finite([r.get("arb_pnl_step") for r in rows]), max_points=MAX_DISTRIBUTION_POINTS),
            CLR_ARB,
        ),
        (
            "Active LP (Δ)",
            _downsample_series(
                _finite(_diff_series([r.get("lp_pnl_active") for r in rows])),
                max_points=MAX_DISTRIBUTION_POINTS,
            ),
            CLR_LP_ACTIVE,
        ),
        (
            "Passive LP (Δ)",
            _downsample_series(
                _finite(_diff_series([r.get("lp_pnl_passive") for r in rows])),
                max_points=MAX_DISTRIBUTION_POINTS,
            ),
            CLR_LP_PASSIVE,
        ),
        (
            "Jiter (Δ)",
            _downsample_series(
                _finite(_diff_series([r.get("jiter_pnl") for r in rows])),
                max_points=MAX_DISTRIBUTION_POINTS,
            ),
            CLR_JITER,
        ),
    ]

    fig = make_subplots(rows=3, cols=2, subplot_titles=[s[0] for s in series], vertical_spacing=0.18)
    for idx, (_, vals, color) in enumerate(series):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        fig.add_trace(
            go.Histogram(x=vals, nbinsx=60, marker_color=color, opacity=0.85, showlegend=False),
            row=row,
            col=col,
        )

    fig.update_layout(template=PLOTLY_TEMPLATE, title="PnL distributions per block", bargap=0.05, height=780)
    fig.update_yaxes(title_text="Count", type="log")
    fig.update_xaxes(title_text="PnL", title_standoff=18, automargin=True)
    return fig


def _build_pnl_cumulative_figure(rows: List[Dict[str, Any]]) -> go.Figure:
    t = [r["t"] for r in rows]
    sr_c = _cumsum([r.get("sr_pnl_step") for r in rows])
    noise_c = _cumsum([r.get("noise_pnl_step") for r in rows])
    arb_c = _cumsum([r.get("arb_pnl_step") for r in rows])
    lp_active = [r.get("lp_pnl_active") for r in rows]
    lp_passive = [r.get("lp_pnl_passive") for r in rows]
    jiter = [r.get("jiter_pnl") for r in rows]
    t, sr_c, noise_c, arb_c, lp_active, lp_passive, jiter = _downsample_xy(
        t,
        sr_c,
        noise_c,
        arb_c,
        lp_active,
        lp_passive,
        jiter,
        max_points=MAX_TIMESERIES_POINTS,
    )

    fig = go.Figure()
    fig.add_hline(y=0.0, line=dict(color=CLR_GUIDE, width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=t, y=sr_c, mode="lines", name="Smart router", line=dict(color=CLR_SR)))
    fig.add_trace(
        go.Scatter(
            x=t,
            y=noise_c,
            mode="lines",
            name="Noise trader",
            line=dict(dash="dash", color=CLR_NOISE),
        )
    )
    fig.add_trace(go.Scatter(x=t, y=arb_c, mode="lines", name="Arbitrageur", line=dict(color=CLR_ARB)))
    fig.add_trace(
        go.Scatter(x=t, y=lp_active, mode="lines", name="Active LP", line=dict(dash="dashdot", color=CLR_LP_ACTIVE))
    )
    fig.add_trace(
        go.Scatter(x=t, y=lp_passive, mode="lines", name="Passive LP", line=dict(dash="dot", color=CLR_LP_PASSIVE))
    )
    fig.add_trace(go.Scatter(x=t, y=jiter, mode="lines", name="Jiter", line=dict(width=2, color=CLR_JITER)))
    fig.update_layout(template=PLOTLY_TEMPLATE, title="Cumulative PnL (live)")
    fig.update_xaxes(title_text="Block")
    fig.update_yaxes(title_text="PnL (token1)")
    return fig


def _build_fee_figure(rows: List[Dict[str, Any]], *, fee_mode: str) -> go.Figure:
    """
    Build the fee and controller signal panel.

    Notes
    -----
    Aligns fee(t+1) with signal(t), as in scripts/run.py.
    """
    steps = [r["t"] for r in rows]
    fee_series = [r.get("fee") for r in rows]
    fee_x_to_y_series = [r.get("fee_x_to_y", r.get("fee")) for r in rows]
    fee_y_to_x_series = [r.get("fee_y_to_x", r.get("fee")) for r in rows]
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
    elif fee_mode == "linear_asymmetric":
        secondary_vals_full = fee_signal_series
        secondary_label = "log(DEX/CEX)"
    else:
        secondary_vals_full = fee_signal_series
        secondary_label = "Controller signal"

    if len(steps) > 1:
        steps_fee_plot = steps[:-1]
        fee_plot = fee_series[1:]
        fee_x_to_y_plot = fee_x_to_y_series[1:]
        fee_y_to_x_plot = fee_y_to_x_series[1:]
        secondary_vals_plot = secondary_vals_full[:-1]
        fee_label = "Fee (applies next step; aligned to signal)"
    else:
        steps_fee_plot = steps
        fee_plot = fee_series
        fee_x_to_y_plot = fee_x_to_y_series
        fee_y_to_x_plot = fee_y_to_x_series
        secondary_vals_plot = secondary_vals_full
        fee_label = "Fee"
    steps_fee_plot, fee_plot, fee_x_to_y_plot, fee_y_to_x_plot, secondary_vals_plot = _downsample_xy(
        steps_fee_plot,
        fee_plot,
        fee_x_to_y_plot,
        fee_y_to_x_plot,
        secondary_vals_plot,
        max_points=MAX_TIMESERIES_POINTS,
    )

    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
    if fee_mode == "linear_asymmetric":
        fig.add_trace(
            go.Scatter(x=steps_fee_plot, y=fee_x_to_y_plot, mode="lines", name="X→Y fee", line=dict(width=1.9, color=CLR_FEE)),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=steps_fee_plot, y=fee_y_to_x_plot, mode="lines", name="Y→X fee", line=dict(width=1.9, color=CLR_LVR)),
            row=1,
            col=1,
            secondary_y=False,
        )
    else:
        fig.add_trace(
            go.Scatter(x=steps_fee_plot, y=fee_plot, mode="lines", name=fee_label, line=dict(width=1.9, color=CLR_FEE)),
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
            line=dict(width=1.2, dash="dash", color=CLR_SIGNAL),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fee_distribution_source = fee_x_to_y_series + fee_y_to_x_series if fee_mode == "linear_asymmetric" else fee_series
    fee_vals = _downsample_series(
        _finite([_to_float(v) for v in fee_distribution_source]),
        max_points=MAX_DISTRIBUTION_POINTS,
    )
    fig.add_trace(
        go.Histogram(x=fee_vals, name="Fee distribution", marker_color=CLR_FEE, opacity=0.70, showlegend=False),
        row=2,
        col=1,
    )

    if fee_vals:
        fee_arr = np.asarray(fee_vals, dtype=float)
        fee_mean = float(np.mean(fee_arr))
        fee_median = float(np.median(fee_arr))
        percentiles = [(p, float(np.percentile(fee_arr, p))) for p in (5, 25, 75, 95)]

        fig.add_vline(x=fee_mean, row=2, col=1, line=dict(color=CLR_MEAN, width=2, dash="dash"))
        fig.add_vline(x=fee_median, row=2, col=1, line=dict(color=CLR_MEDIAN, width=2, dash="dot"))
        for p, val in percentiles:
            dash = "dash" if p in (25, 75) else "dot"
            fig.add_vline(x=val, row=2, col=1, line=dict(color=CLR_PERCENTILE, width=1.5, dash=dash))

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"Mean = {fee_mean:.5f}",
                line=dict(color=CLR_MEAN, width=2, dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"Median = {fee_median:.5f}",
                line=dict(color=CLR_MEDIAN, width=2, dash="dot"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Fee & Controller Signal",
        bargap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=760,
    )
    fig.update_xaxes(title_text="Block", row=1, col=1)
    fig.update_yaxes(title_text="Fee", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text=secondary_label, row=1, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Fee", row=2, col=1)
    fig.update_yaxes(title_text="Count", type="log", row=2, col=1)
    return fig


def _build_lp_decomposition_figure(rows: List[Dict[str, Any]]) -> go.Figure:
    """
    Build LP decomposition panel (hedged/unhedged/fees/LVR).

    Notes
    -----
    This mirrors the richer LP accounting outputs recently added to scripts/run.py.
    """
    t = [r["t"] for r in rows]
    lp_active_hedged = [r.get("lp_pnl_active") for r in rows]
    lp_passive_hedged = [r.get("lp_pnl_passive") for r in rows]
    lp_total_hedged = _sum_pairs(lp_active_hedged, lp_passive_hedged)

    lp_active_unhedged = [r.get("lp_unhedged_active") for r in rows]
    lp_passive_unhedged = [r.get("lp_unhedged_passive") for r in rows]
    lp_total_unhedged = _sum_pairs(lp_active_unhedged, lp_passive_unhedged)

    lp_fee_value_total = [r.get("lp_fee_value_total") for r in rows]
    lp_lvr_total = [r.get("lp_lvr_total") for r in rows]
    (
        t,
        lp_total_hedged,
        lp_total_unhedged,
        lp_active_hedged,
        lp_passive_hedged,
        lp_fee_value_total,
        lp_lvr_total,
    ) = _downsample_xy(
        t,
        lp_total_hedged,
        lp_total_unhedged,
        lp_active_hedged,
        lp_passive_hedged,
        lp_fee_value_total,
        lp_lvr_total,
        max_points=MAX_TIMESERIES_POINTS,
    )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
    fig.add_hline(y=0.0, row=1, col=1, line=dict(color=CLR_GUIDE, width=1, dash="dot"))
    fig.add_trace(
        go.Scatter(x=t, y=lp_total_hedged, mode="lines", name="LP total hedged", line=dict(color=CLR_LP_ACTIVE, width=2.1)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=lp_total_unhedged, mode="lines", name="LP total unhedged", line=dict(color=CLR_NOISE, width=1.9, dash="dash")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=lp_active_hedged, mode="lines", name="Active hedged", line=dict(color=CLR_LP_ACTIVE, dash="dot")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=lp_passive_hedged, mode="lines", name="Passive hedged", line=dict(color=CLR_LP_PASSIVE, dash="dot")),
        row=1,
        col=1,
    )

    fig.add_hline(y=0.0, row=2, col=1, line=dict(color=CLR_GUIDE, width=1, dash="dot"))
    fig.add_trace(
        go.Scatter(x=t, y=lp_fee_value_total, mode="lines", name="LP fee value total", line=dict(color=CLR_FEE, width=1.9)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=lp_lvr_total, mode="lines", name="LP LVR total", line=dict(color=CLR_LVR, width=1.9)),
        row=2,
        col=1,
    )

    fig.update_layout(template=PLOTLY_TEMPLATE, title="LP decomposition: hedged vs unhedged, fees, LVR", height=760)
    fig.update_xaxes(title_text="Block", row=2, col=1)
    fig.update_yaxes(title_text="PnL (token1)", row=1, col=1)
    fig.update_yaxes(title_text="Value (token1)", row=2, col=1)
    return fig


def _build_normalized_lvr_figure(rows: List[Dict[str, Any]], *, smooth_blocks: int = 50) -> go.Figure:
    """
    Build normalized LVR diagnostics from streamed per-step deltas.

    Parameters
    ----------
    rows
        Live rows from SQLite metrics.
    smooth_blocks
        Trailing window length for rolling median.

    Returns
    -------
    plotly.graph_objects.Figure
        Two-row diagnostics chart with rolling medians and distributions.

    Notes
    -----
    Uses:
    - d_lvr_total / dex_notional_y (in bps)
    - d_lvr_total / d_fee_value_total (ratio)
    """
    steps = [r["t"] for r in rows]
    d_lvr = [_to_float(r.get("d_lvr_total")) for r in rows]
    d_fee = [_to_float(r.get("d_fee_value_total")) for r in rows]
    dex_notional = [_to_float(r.get("dex_notional_y")) for r in rows]

    lvr_per_notional_bps: List[Optional[float]] = []
    lvr_over_fee_value: List[Optional[float]] = []
    for dlvr, notional, dfee in zip(d_lvr, dex_notional, d_fee):
        ratio_notional = _safe_ratio(dlvr, notional)
        lvr_per_notional_bps.append(None if ratio_notional is None else 1e4 * ratio_notional)
        lvr_over_fee_value.append(_safe_ratio(dlvr, dfee))

    lvr_per_notional_med = _rolling_nanmedian(lvr_per_notional_bps, smooth_blocks)
    lvr_over_fee_med = _rolling_nanmedian(lvr_over_fee_value, smooth_blocks)
    steps_plot, lvr_per_notional_med_plot, lvr_over_fee_med_plot = _downsample_xy(
        steps,
        lvr_per_notional_med,
        lvr_over_fee_med,
        max_points=MAX_TIMESERIES_POINTS,
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
        subplot_titles=(
            f"ΔLVR / DEX notional (bps) - {smooth_blocks}-block rolling median",
            f"ΔLVR / Δfee value - {smooth_blocks}-block rolling median",
            "Distribution: rolling median bps",
            "Distribution: rolling median ratio",
        ),
    )
    fig.add_hline(y=0.0, line=dict(color=CLR_GUIDE, width=1, dash="dot"), row=1, col=1)
    fig.add_hline(y=0.0, line=dict(color=CLR_GUIDE, width=1, dash="dot"), row=1, col=2)
    fig.add_hline(y=1.0, line=dict(color=CLR_GUIDE, width=1, dash="dash"), row=1, col=2)

    fig.add_trace(
        go.Scatter(
            x=steps_plot,
            y=lvr_per_notional_med_plot,
            mode="lines",
            line=dict(width=2.0, color=CLR_CEX),
            showlegend=False,
            hovertemplate=f"t=%{{x}}<br>{smooth_blocks}-block median bps=%{{y:.4g}}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=steps_plot,
            y=lvr_over_fee_med_plot,
            mode="lines",
            line=dict(width=2.0, color=CLR_CEX),
            showlegend=False,
            hovertemplate=f"t=%{{x}}<br>{smooth_blocks}-block median ratio=%{{y:.4g}}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Histogram(
            x=_downsample_series(_finite(lvr_per_notional_med), max_points=MAX_DISTRIBUTION_POINTS),
            nbinsx=80,
            marker_color=CLR_HIST,
            opacity=0.82,
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=_downsample_series(_finite(lvr_over_fee_med), max_points=MAX_DISTRIBUTION_POINTS),
            nbinsx=80,
            marker_color=CLR_HIST,
            opacity=0.82,
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(template=PLOTLY_TEMPLATE, title="Normalized LVR diagnostics", bargap=0.05, height=760)
    fig.update_xaxes(title_text="Block", row=1, col=1)
    fig.update_xaxes(title_text="Block", row=1, col=2)
    fig.update_xaxes(title_text="bps", row=2, col=1)
    fig.update_xaxes(title_text="ratio", row=2, col=2)
    fig.update_yaxes(title_text="bps", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=1, col=2)
    fig.update_yaxes(title_text="Count", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Count", type="log", row=2, col=2)
    return fig


def _build_activity_figure(rows: List[Dict[str, Any]]) -> go.Figure:
    """
    Build execution activity panel from per-step counts.

    Notes
    -----
    Top row: per-block counts.
    Bottom row: cumulative routing and execution counts.
    """
    t = [r["t"] for r in rows]

    sr_exec = np.asarray([_to_float(r.get("sr_exec_count")) or 0.0 for r in rows], dtype=float)
    noise_exec = np.asarray([_to_float(r.get("noise_exec_count")) or 0.0 for r in rows], dtype=float)
    arb_exec = np.asarray([_to_float(r.get("arb_exec_count")) or 0.0 for r in rows], dtype=float)
    trader_exec = np.asarray([_to_float(r.get("trader_exec_count")) or 0.0 for r in rows], dtype=float)
    sr_cex_exec = np.asarray([_to_float(r.get("sr_cex_exec_count")) or 0.0 for r in rows], dtype=float)
    sr_dex_exec = np.asarray([_to_float(r.get("sr_dex_exec_count")) or 0.0 for r in rows], dtype=float)
    t, sr_exec, noise_exec, arb_exec, sr_cex_cum, sr_dex_cum, trader_cum, arb_cum = _downsample_xy(
        t,
        sr_exec.tolist(),
        noise_exec.tolist(),
        arb_exec.tolist(),
        np.cumsum(sr_cex_exec).tolist(),
        np.cumsum(sr_dex_exec).tolist(),
        np.cumsum(trader_exec).tolist(),
        np.cumsum(arb_exec).tolist(),
        max_points=MAX_TIMESERIES_POINTS,
    )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
    fig.add_trace(
        go.Scatter(x=t, y=sr_exec, mode="lines", name="Smart router execs", line=dict(color=CLR_SR)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=noise_exec, mode="lines", name="Noise trader execs", line=dict(color=CLR_NOISE, dash="dash")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=arb_exec, mode="lines", name="Arb execs", line=dict(color=CLR_ARB)),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=t, y=sr_cex_cum, mode="lines", name="SR routed to CEX (cum)", line=dict(color=CLR_CEX)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=sr_dex_cum, mode="lines", name="SR routed to DEX (cum)", line=dict(color=CLR_DEX)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=trader_cum, mode="lines", name="Trader execs (cum)", line=dict(color=CLR_LP_ACTIVE, dash="dot")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=arb_cum, mode="lines", name="Arb execs (cum)", line=dict(color=CLR_JITER, dash="dot")),
        row=2,
        col=1,
    )

    fig.update_layout(template=PLOTLY_TEMPLATE, title="Execution activity and cumulative routes", height=760)
    fig.update_xaxes(title_text="Block", row=2, col=1)
    fig.update_yaxes(title_text="Exec count / block", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative count", row=2, col=1)
    return fig


def _build_routing_figure(rows: List[Dict[str, Any]]) -> go.Figure:
    """
    Build smart-router routing panel (DEX share over time + distribution).
    """
    t = [r["t"] for r in rows]
    sr_cex_exec = np.asarray([_to_float(r.get("sr_cex_exec_count")) or 0.0 for r in rows], dtype=float)
    sr_dex_exec = np.asarray([_to_float(r.get("sr_dex_exec_count")) or 0.0 for r in rows], dtype=float)
    totals = sr_cex_exec + sr_dex_exec

    per_block_share: List[Optional[float]] = []
    for num, den in zip(sr_dex_exec, totals):
        if den <= 0:
            per_block_share.append(None)
        else:
            per_block_share.append(float(num / den))

    cex_cum = np.cumsum(sr_cex_exec)
    dex_cum = np.cumsum(sr_dex_exec)
    cumulative_share: List[Optional[float]] = []
    for num, den in zip(dex_cum, cex_cum + dex_cum):
        if den <= 0:
            cumulative_share.append(None)
        else:
            cumulative_share.append(float(num / den))
    t, per_block_share, cumulative_share = _downsample_xy(
        t,
        per_block_share,
        cumulative_share,
        max_points=MAX_TIMESERIES_POINTS,
    )

    fig = make_subplots(rows=1, cols=2, column_widths=[0.70, 0.30], horizontal_spacing=0.08)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=cumulative_share,
            mode="lines",
            name="DEX share (cumulative)",
            line=dict(color=CLR_CEX, dash="dash"),
        ),
        row=1,
        col=1,
    )

    share_vals = _downsample_series(_finite(per_block_share), max_points=MAX_DISTRIBUTION_POINTS)
    fig.add_trace(
        go.Histogram(x=share_vals, nbinsx=40, marker_color=CLR_HIST, opacity=0.80, showlegend=False),
        row=1,
        col=2,
    )
    if share_vals:
        share_mean = float(np.mean(np.asarray(share_vals, dtype=float)))
        fig.add_vline(x=share_mean, row=1, col=2, line=dict(color=CLR_MEAN, width=2, dash="dash"))

    fig.update_layout(template=PLOTLY_TEMPLATE, title="Smart-router DEX routing share", height=520)
    fig.update_xaxes(title_text="Block", row=1, col=1)
    fig.update_yaxes(title_text="DEX share", range=[0.0, 1.0], row=1, col=1)
    fig.update_xaxes(title_text="DEX share", row=1, col=2)
    fig.update_yaxes(title_text="Count", type="log", row=1, col=2)
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
    Infer fee mode from the run's scenario YAML.

    Notes
    -----
    Cached to avoid reparsing on every polling tick.
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


@dataclass
class _RunController:
    lock: threading.Lock
    run_id: Optional[str] = None
    run_root: Optional[Path] = None
    process: Optional[Process] = None
    stop_event: Optional[Event] = None

    def start(self, *, config_yaml: str, live_every: int, log_flush_every: int) -> Tuple[bool, str]:
        """
        Start a new simulation run (stopping existing run is required first).
        """
        with self.lock:
            if self.process is not None and self.process.is_alive():
                return False, "A run is already in progress. Stop it before starting a new one."

            ok, err = _validate_config_against_simulate(config_yaml)
            if not ok:
                return False, err

            # Build canonical validated config
            from abm_webapp.config import validate_scenario, canonical_yaml
            validated, verr = validate_scenario(config_yaml)
            if validated is not None:
                config_yaml = canonical_yaml(validated)

            # Collect run metadata for reproducibility
            meta_json_str: Optional[str] = None
            try:
                from abm_webapp import __version__
                from abm_webapp.run_meta import collect_run_meta
                import yaml as _yaml
                parsed = _yaml.safe_load(config_yaml)
                seed = None
                if isinstance(parsed, dict):
                    sim_block = parsed.get("simulate", {})
                    if isinstance(sim_block, dict):
                        seed = sim_block.get("seed")
                meta = collect_run_meta(
                    app_version=__version__,
                    seed=seed,
                    config_yaml=config_yaml,
                )
                import json as _json
                meta_json_str = _json.dumps(meta, default=str)
            except Exception:
                pass

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
                    meta_json=meta_json_str,
                ),
                daemon=True,
            )
            proc.start()

            self.run_id = run_id
            self.run_root = run_root
            self.process = proc
            self.stop_event = stop_event
            _clear_metrics_cache()
            _clear_log_cache()
            _clear_log_offset_cache()
            _clear_tier_tracking()
            return True, f"Started run {run_id}"

    def stop(self) -> Tuple[bool, str]:
        """Request current run to stop (idempotent)."""
        with self.lock:
            if self.process is None or self.stop_event is None:
                return False, "No active run."
            if not self.process.is_alive():
                return True, "Run already finished."
            self.stop_event.set()
            return True, "Stop requested."

    def reset_all(self) -> Tuple[bool, str]:
        """
        Stop any running simulation and remove all webapp run outputs.

        Notes
        -----
        This clears only `abm_results/web_runs/` (webapp runs).
        """
        with self.lock:
            if self.process is not None and self.stop_event is not None and self.process.is_alive():
                self.stop_event.set()
                self.process.join(timeout=10.0)
                if self.process.is_alive():
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
            _clear_metrics_cache()
            _clear_log_cache()
            _clear_log_offset_cache()
            _clear_tier_tracking()

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
        from dash import Dash, Input, Output, State, dcc, html, no_update
    except Exception as exc:  # pragma: no cover - dash is optional for non-web workflows
        raise RuntimeError(
            "Dash is required for the webapp. Install it with `pip install dash` (or `conda install -c conda-forge dash`)."
        ) from exc

    # ── Crash recovery: scan existing runs on startup ──
    web_root = _web_runs_root()
    web_root.mkdir(parents=True, exist_ok=True)
    try:
        recovered = scan_and_recover(web_root)
        if recovered:
            print(f"[webapp] Crash recovery: marked {len(recovered)} abandoned run(s): {recovered}")
    except Exception as exc:
        print(f"[webapp] Warning: crash recovery scan failed: {exc}")

    scenarios_dir = Path("configs") / "scenarios"
    scenario_files, rejected_scenarios = _partition_scenario_files(scenarios_dir)
    for scenario_name, reason in rejected_scenarios.items():
        first_line = reason.splitlines()[0] if reason else "Unknown validation error."
        print(f"[webapp] Ignoring non-runnable scenario file {scenario_name}: {first_line}")
    scenario_options = [{"label": p.name, "value": str(p)} for p in scenario_files]
    default_scenario = next((p for p in scenario_files if p.name == "section4_microstructure_model0_static.yml"), None)
    if default_scenario is None and scenario_files:
        default_scenario = scenario_files[0]
    default_yaml = _load_text(default_scenario) if default_scenario is not None else ""

    assets_dir = Path(__file__).with_name("assets")
    app = Dash(__name__, assets_folder=str(assets_dir), title="ABM Live Lab")

    @app.server.route("/stream/run/<run_id>")
    def _stream_run(run_id: str):
        """
        Server-Sent Events endpoint for live run updates.

        Notes
        -----
        This keeps browser traffic event-driven while SQLite remains the durable source
        of truth written by the simulation process.
        """
        from flask import Response, stream_with_context

        run_id_s = str(run_id or "").strip()
        if not run_id_s:
            return Response("missing run_id", status=400, mimetype="text/plain")

        run_root = _run_root_for(run_id_s)
        db_path = run_root / "live.db"

        def _event_stream():
            event_id = 0
            last_t = -1
            last_status_sig: Optional[Tuple[Any, ...]] = None
            last_emit = time.monotonic()
            terminal_emitted = False

            try:
                # Initial hydrate snapshot.
                rows_init = read_metrics(db_path, limit=LIVE_METRICS_LIMIT)
                if rows_init:
                    last_t = int(rows_init[-1]["t"])

                status_init = read_status(db_path)
                status_init_dict = _status_to_dict(status_init)
                last_status_sig = _status_signature(status_init_dict)

                event_id += 1
                yield _format_sse_event(
                    event="snapshot",
                    payload=dict(
                        run_id=run_id_s,
                        row_count=len(rows_init),
                        status=status_init_dict,
                    ),
                    event_id=event_id,
                )
                last_emit = time.monotonic()

                while True:
                    emitted_this_loop = False

                    status = read_status(db_path)
                    status_dict = _status_to_dict(status)
                    status_sig = _status_signature(status_dict)
                    if status_sig != last_status_sig:
                        last_status_sig = status_sig
                        event_id += 1
                        yield _format_sse_event(
                            event="status_change",
                            payload=dict(run_id=run_id_s, status=status_dict),
                            event_id=event_id,
                        )
                        emitted_this_loop = True
                        last_emit = time.monotonic()

                    new_rows = read_metrics(db_path, since_t=last_t)
                    if new_rows:
                        last_t = int(new_rows[-1]["t"])
                        event_id += 1
                        yield _format_sse_event(
                            event="metrics_delta",
                            payload=dict(run_id=run_id_s, row_count=len(new_rows), last_t=last_t),
                            event_id=event_id,
                        )
                        emitted_this_loop = True
                        last_emit = time.monotonic()

                    run_state = str(status_dict.get("state", "")).lower()
                    if run_state in TERMINAL_RUN_STATES and not new_rows:
                        if not terminal_emitted:
                            event_id += 1
                            yield _format_sse_event(
                                event="end",
                                payload=dict(run_id=run_id_s, state=run_state),
                                event_id=event_id,
                            )
                            terminal_emitted = True
                        break

                    now = time.monotonic()
                    if (not emitted_this_loop) and ((now - last_emit) >= SSE_HEARTBEAT_SECONDS):
                        event_id += 1
                        yield _format_sse_event(
                            event="heartbeat",
                            payload=dict(run_id=run_id_s, ts=datetime.now().isoformat()),
                            event_id=event_id,
                        )
                        last_emit = now

                    time.sleep(SSE_LOOP_SLEEP_SECONDS)
            except GeneratorExit:
                # Client disconnected; exit cleanly without resource leaks.
                return

        response = Response(stream_with_context(_event_stream()), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    app.layout = html.Div(
        className="app-shell",
        children=[
            html.Div(
                className="hero",
                children=[
                    html.H1("ABM Live Lab", className="hero-title"),
                    html.P(
                        (
                            "Fast exploratory control room for live ABM runs with fee-schedule, "
                            "LP, routing, and LVR diagnostics."
                        ),
                        className="hero-subtitle",
                    ),
                    html.P(EXPLORATORY_LAB_NOTICE, className="hero-note"),
                ],
            ),
            html.Div(id="summary-cards", className="metrics-grid", children=[]),
            html.Div(
                className="workspace-grid",
                children=[
                    html.Div(
                        className="panel control-panel",
                        children=[
                            html.H3("Scenario & Controls", className="panel-title"),
                            html.Label("Load scenario file", className="field-label"),
                            dcc.Dropdown(
                                id="scenario-dropdown",
                                options=scenario_options,
                                value=str(default_scenario) if default_scenario is not None else None,
                                placeholder="Select a scenario YAML...",
                            ),
                            html.Label("Fee schedule preset", className="field-label"),
                            dcc.Dropdown(
                                id="fee-preset-dropdown",
                                options=[
                                    {"label": label, "value": value}
                                    for value, label in FEE_SCHEDULE_PRESETS.items()
                                ],
                                value="static",
                                clearable=False,
                            ),
                            html.Div(
                                (
                                    "Preset changes update only fee_mode; edit YAML below for all "
                                    "other exploratory parameters."
                                ),
                                className="field-help",
                            ),
                            html.Label("Edit YAML config", className="field-label"),
                            dcc.Textarea(id="yaml-editor", value=default_yaml, className="yaml-editor"),
                            html.Div(
                                className="controls-row",
                                children=[
                                    html.Button("Run", id="run-btn", n_clicks=0, className="btn btn-run"),
                                    html.Button("Stop", id="stop-btn", n_clicks=0, className="btn btn-stop"),
                                    dcc.ConfirmDialogProvider(
                                        id="reset-confirm",
                                        message=(
                                            "Reset will stop any running simulation and delete all webapp runs under "
                                            "abm_results/web_runs/. This cannot be undone. Continue?"
                                        ),
                                        children=html.Button("Reset", id="reset-btn", n_clicks=0, className="btn btn-reset"),
                                    ),
                                ],
                            ),
                            html.Div(id="run-message", className="run-message"),
                            html.Div(id="run-status", className="run-status"),
                            dcc.Store(id="run-root-store"),
                            dcc.Store(id="data-seq-store"),
                            dcc.Input(id="stream-run-id", type="text", value="", style={"display": "none"}),
                            dcc.Input(id="stream-event-seq", type="number", value=0, style={"display": "none"}),
                        ],
                    ),
                    html.Div(
                        className="panel view-panel",
                        children=[
                            dcc.Tabs(
                                className="main-tabs",
                                children=[
                                    dcc.Tab(
                                        label="Price",
                                        children=[
                                            dcc.Graph(id="price-graph", className="graph-large"),
                                            dcc.Graph(id="price-dist-graph", className="graph-medium"),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="PnL",
                                        children=[
                                            dcc.Graph(id="pnl-step-graph", className="graph-large"),
                                            dcc.Graph(id="pnl-step-dist-graph", className="graph-large"),
                                            dcc.Graph(id="pnl-cum-graph", className="graph-large"),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Fees",
                                        children=[
                                            dcc.Graph(id="fee-graph", className="graph-large"),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="LP & LVR",
                                        children=[
                                            dcc.Graph(id="lp-graph", className="graph-large"),
                                            dcc.Graph(id="lvr-graph", className="graph-large"),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Activity & Routing",
                                        children=[
                                            dcc.Graph(id="activity-graph", className="graph-large"),
                                            dcc.Graph(id="routing-graph", className="graph-medium"),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )

    def _build_summary_cards(rows: List[Dict[str, Any]], status: Any, fee_mode: str) -> List[Any]:
        if not rows:
            return [
                html.Div(
                    className="metric-card",
                    children=[
                        html.Div("Run snapshot", className="metric-label"),
                        html.Div("No data yet", className="metric-value"),
                        html.Div("Start a run to populate live diagnostics.", className="metric-note"),
                    ],
                )
            ]

        latest = rows[-1]
        sr_pnl_cum = _optional_sum([_to_float(r.get("sr_pnl_step")) for r in rows])
        noise_pnl_cum = _optional_sum([_to_float(r.get("noise_pnl_step")) for r in rows])
        arb_pnl_cum = _optional_sum([_to_float(r.get("arb_pnl_step")) for r in rows])

        lp_total_hedged = _sum_pairs([latest.get("lp_pnl_active")], [latest.get("lp_pnl_passive")])[0]
        lp_total_unhedged = _sum_pairs([latest.get("lp_unhedged_active")], [latest.get("lp_unhedged_passive")])[0]
        lp_lvr_total = _to_float(latest.get("lp_lvr_total"))

        dex_price = _to_float(latest.get("dex_price"))
        cex_price = _to_float(latest.get("cex_price"))
        if dex_price is not None and cex_price is not None and dex_price > 0 and cex_price > 0:
            basis_bps = 1e4 * abs(math.log(dex_price) - math.log(cex_price))
        else:
            basis_bps = None

        sr_cex_total = float(
            np.sum(np.asarray([_to_float(r.get("sr_cex_exec_count")) or 0.0 for r in rows], dtype=float))
        )
        sr_dex_total = float(
            np.sum(np.asarray([_to_float(r.get("sr_dex_exec_count")) or 0.0 for r in rows], dtype=float))
        )
        sr_total = sr_cex_total + sr_dex_total
        sr_dex_share = None if sr_total <= 0 else (sr_dex_total / sr_total)

        latest_d_lvr = _to_float(latest.get("d_lvr_total"))
        latest_notional = _to_float(latest.get("dex_notional_y"))
        latest_lvr_bps = _safe_ratio(latest_d_lvr, latest_notional)
        latest_lvr_bps = None if latest_lvr_bps is None else 1e4 * latest_lvr_bps

        run_state = str(getattr(status, "state", "idle")) if status is not None else "idle"
        cards = [
            ("Run state", run_state, f"fee_mode={fee_mode}"),
            ("Blocks", f"{len(rows):,}", f"latest t={int(latest.get('t', len(rows) - 1))}"),
            ("Latest fee", _format_value(_to_float(latest.get("fee")), digits=5), "fraction"),
            ("Price basis", _format_value(basis_bps, digits=2, suffix=" bps"), "|ln(DEX/CEX)|"),
            ("SR cumulative PnL", _format_value(sr_pnl_cum, digits=3), "token1"),
            ("Noise cumulative PnL", _format_value(noise_pnl_cum, digits=3), "token1"),
            ("Arb cumulative PnL", _format_value(arb_pnl_cum, digits=3), "token1"),
            ("LP hedged total", _format_value(lp_total_hedged, digits=3), "token1"),
            ("LP unhedged total", _format_value(lp_total_unhedged, digits=3), "token1"),
            ("LP LVR total", _format_value(lp_lvr_total, digits=3), "token1"),
            (
                "SR DEX share",
                _format_value(sr_dex_share, digits=2, scale=100.0, suffix="%"),
                f"DEX={int(sr_dex_total):,} / total={int(sr_total):,}",
            ),
            ("Latest dLVR/notional", _format_value(latest_lvr_bps, digits=2, suffix=" bps"), "per block"),
        ]

        return [
            html.Div(
                className="metric-card",
                children=[
                    html.Div(label, className="metric-label"),
                    html.Div(value, className="metric-value"),
                    html.Div(note, className="metric-note"),
                ],
            )
            for label, value, note in cards
        ]

    @app.callback(
        Output("yaml-editor", "value"),
        Input("scenario-dropdown", "value"),
        Input("fee-preset-dropdown", "value"),
        State("yaml-editor", "value"),
    )
    def _load_or_update_scenario(selected_path: Optional[str], fee_preset: Optional[str], current_yaml: str) -> str:
        from dash import callback_context

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        if trigger == "fee-preset-dropdown":
            updated, _err = _apply_fee_schedule_preset(current_yaml or "", fee_preset or "static")
            return updated
        if not selected_path:
            return ""
        path = Path(selected_path)
        if not path.exists():
            return ""
        loaded = _load_text(path)
        return loaded

    @app.callback(
        Output("run-message", "children"),
        Output("run-root-store", "data"),
        Output("stream-run-id", "value"),
        Input("run-btn", "n_clicks"),
        Input("stop-btn", "n_clicks"),
        Input("reset-confirm", "submit_n_clicks"),
        State("yaml-editor", "value"),
        prevent_initial_call=True,
    )
    def _handle_run_stop(
        run_clicks: int,
        stop_clicks: int,
        reset_clicks: int,
        yaml_text: str,
    ) -> Tuple[str, Optional[str], str]:
        from dash import callback_context

        if not callback_context.triggered:
            return "", None, ""

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger == "stop-btn":
            _ok, message = _CTRL.stop()
            run_root = str(_CTRL.run_root) if _CTRL.run_root is not None else None
            run_id = str(_CTRL.run_id) if _CTRL.run_id is not None else ""
            return message, run_root, run_id

        if trigger == "reset-confirm":
            _ok, message = _CTRL.reset_all()
            return message, None, ""

        _ok, message = _CTRL.start(
            config_yaml=yaml_text or "",
            live_every=DEFAULT_LIVE_EVERY,
            log_flush_every=DEFAULT_LOG_FLUSH_EVERY,
        )
        run_root = str(_CTRL.run_root) if _CTRL.run_root is not None else None
        run_id = str(_CTRL.run_id) if _CTRL.run_id is not None else ""
        return message, run_root, run_id

    # ── Data-loading + always-tier figures (fires every SSE tick) ─────
    @app.callback(
        Output("price-graph", "figure"),
        Output("pnl-step-graph", "figure"),
        Output("pnl-cum-graph", "figure"),
        Output("fee-graph", "figure"),
        Output("run-status", "children"),
        Output("summary-cards", "children"),
        Output("data-seq-store", "data"),
        Input("stream-event-seq", "value"),
        Input("stream-run-id", "value"),
        prevent_initial_call=False,
    )
    def _on_stream_data_and_core(_seq: Any, stream_run_id: Optional[str]):
        empty = _empty_fig()
        empty_cards = _build_summary_cards([], None, "static")
        empty_store: Dict[str, Any] = {
            "seq": int(_seq or 0),
            "run_id": "",
            "run_root_key": "",
            "data_version": 0,
            "has_rows": False,
            "is_terminal": False,
            "fee_mode": "static",
            "status_dict": {},
        }

        run_id = str(stream_run_id or "").strip()
        if not run_id:
            return empty, empty, empty, empty, "", empty_cards, empty_store

        run_root = _run_root_for(run_id)
        run_root_key = str(run_root)
        db_path = run_root / "live.db"

        cached = _get_cached_metrics(run_root_key)
        is_initial_load = cached is None
        data_changed = False
        data_version = 0

        if cached is None:
            rows, data_version = _set_cached_metrics(
                run_root_key, read_metrics(db_path, limit=LIVE_METRICS_LIMIT)
            )
            data_changed = bool(rows)
        else:
            new_rows = read_metrics(db_path, since_t=int(cached.last_t))
            data_changed = bool(new_rows)
            rows, data_version = _append_cached_metrics(run_root_key, new_rows)

        status = read_status(db_path)
        status_dict = _status_to_dict(status)
        status_line = _format_status_line(status_dict)

        fee_mode = _get_fee_mode_for_run(run_root)
        run_state = str(status_dict.get("state", "")).lower()
        is_terminal = run_state in TERMINAL_RUN_STATES

        # Freeze core-tier figures only when genuinely nothing changed
        # AND we already rendered at least once with actual data.
        freeze = (not data_changed) and (not is_initial_load) and bool(rows)
        if freeze:
            fig_price = no_update
            fig_step = no_update
            fig_cum = no_update
            fig_fee = no_update
        else:
            fig_price = _build_price_figure(rows) if rows else empty
            fig_step = _build_pnl_per_block_figure(rows) if rows else empty
            fig_cum = _build_pnl_cumulative_figure(rows) if rows else empty
            fig_fee = _build_fee_figure(rows, fee_mode=fee_mode) if rows else empty
            for fig in [fig_price, fig_step, fig_cum, fig_fee]:
                fig.update_layout(transition=dict(duration=LIVE_TRANSITION_MS, easing="linear"))

        cards = _build_summary_cards(rows, status, fee_mode)

        # The store carries a monotonic data_version so downstream callbacks
        # can reliably detect whether new data appeared, even if Dash batches
        # multiple rapid Input changes and only delivers the last value.
        store_data: Dict[str, Any] = {
            "seq": int(_seq or 0),
            "run_id": run_id,
            "run_root_key": run_root_key,
            "data_version": data_version,
            "has_rows": bool(rows),
            "is_terminal": is_terminal,
            "fee_mode": fee_mode,
            "status_dict": status_dict,
        }

        return fig_price, fig_step, fig_cum, fig_fee, status_line, cards, store_data

    # ── Medium-tier figures (LP, activity, routing) ────────────────
    @app.callback(
        Output("lp-graph", "figure"),
        Output("activity-graph", "figure"),
        Output("routing-graph", "figure"),
        Input("data-seq-store", "data"),
        prevent_initial_call=True,
    )
    def _on_stream_medium(store_data: Optional[Dict[str, Any]]):
        empty = _empty_fig()
        if not store_data or not store_data.get("run_id"):
            return empty, empty, empty

        if not store_data.get("has_rows", False):
            return no_update, no_update, no_update

        run_root_key = store_data["run_root_key"]
        data_version = int(store_data.get("data_version", 0))
        is_terminal = store_data.get("is_terminal", False)

        # Render when new data has appeared since this tier's last render
        # and enough wall-clock time has elapsed (throttle).  Force render
        # on terminal state so the final frame is always complete.
        if not _should_render_tier(
            run_root_key, "medium", data_version,
            min_interval_s=MEDIUM_FIG_UPDATE_EVERY * SSE_LOOP_SLEEP_SECONDS,
            force=is_terminal,
        ):
            return no_update, no_update, no_update

        cached = _get_cached_metrics(run_root_key)
        rows = list(cached.rows) if cached else []
        if not rows:
            return empty, empty, empty

        fig_lp = _build_lp_decomposition_figure(rows)
        fig_activity = _build_activity_figure(rows)
        fig_routing = _build_routing_figure(rows)
        for fig in [fig_lp, fig_activity, fig_routing]:
            fig.update_layout(transition=dict(duration=LIVE_TRANSITION_MS, easing="linear"))
        return fig_lp, fig_activity, fig_routing

    # ── Heavy-tier figures (distributions, LVR) ───────────────────
    @app.callback(
        Output("price-dist-graph", "figure"),
        Output("pnl-step-dist-graph", "figure"),
        Output("lvr-graph", "figure"),
        Input("data-seq-store", "data"),
        prevent_initial_call=True,
    )
    def _on_stream_heavy(store_data: Optional[Dict[str, Any]]):
        empty = _empty_fig()
        if not store_data or not store_data.get("run_id"):
            return empty, empty, empty

        if not store_data.get("has_rows", False):
            return no_update, no_update, no_update

        run_root_key = store_data["run_root_key"]
        data_version = int(store_data.get("data_version", 0))
        is_terminal = store_data.get("is_terminal", False)

        if not _should_render_tier(
            run_root_key, "heavy", data_version,
            min_interval_s=HEAVY_FIG_UPDATE_EVERY * SSE_LOOP_SLEEP_SECONDS,
            force=is_terminal,
        ):
            return no_update, no_update, no_update

        cached = _get_cached_metrics(run_root_key)
        rows = list(cached.rows) if cached else []
        if not rows:
            return empty, empty, empty

        fig_price_dist = _build_price_distribution_figure(rows)
        fig_step_dist = _build_pnl_per_block_distribution_figure(rows)
        fig_lvr = _build_normalized_lvr_figure(rows)
        # No transition for histogram / distribution figures — bin positions
        # shift between updates causing distracting morph artifacts.
        return fig_price_dist, fig_step_dist, fig_lvr

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
    # Disable reloader to avoid duplicate process trees with multiprocessing.
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
