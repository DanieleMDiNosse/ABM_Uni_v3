#!/usr/bin/env python3
"""Generate stylized-facts diagnostics for a price or returns series."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.artifacts import build_run_manifest, make_unique_dir, write_json


@dataclass(frozen=True)
class FigureSaveResult:
    """Container for persisted figure artifact metadata."""

    name: str
    html_path: Path
    png_path: Path
    png_ok: bool
    png_error: str


def _to_builtin(value: Any) -> Any:
    """Convert nested NumPy values/arrays into JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        vf = float(value)
        return vf if np.isfinite(vf) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _load_series(path: Path) -> np.ndarray:
    """Load a numeric 1D series from a NumPy file."""
    arr = np.asarray(np.load(path), dtype=float).reshape(-1)
    if arr.size < 10:
        raise ValueError(f"Series is too short ({arr.size} points); need at least 10.")
    return arr


def _clean_input_series(raw: np.ndarray, *, input_kind: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """Clean raw input values and return cleaned series plus preprocessing counters."""
    arr = np.asarray(raw, dtype=float).reshape(-1)
    finite_mask = np.isfinite(arr)
    dropped_nonfinite = int(np.sum(~finite_mask))

    dropped_nonpositive = 0
    keep_mask = finite_mask.copy()
    if input_kind == "prices":
        positive_mask = arr > 0.0
        dropped_nonpositive = int(np.sum(finite_mask & ~positive_mask))
        keep_mask &= positive_mask

    cleaned = arr[keep_mask]
    info = {
        "raw_size": int(arr.size),
        "dropped_nonfinite_count": dropped_nonfinite,
        "dropped_nonpositive_count": dropped_nonpositive,
        "final_size": int(cleaned.size),
    }
    return cleaned, info


def _returns_from_sampled_prices(prices: np.ndarray, horizon: int, *, return_type: str = "log") -> np.ndarray:
    """Sample prices every `horizon` steps and compute returns on sampled prices."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    sampled = np.asarray(prices, dtype=float)[::horizon]
    sampled = sampled[np.isfinite(sampled)]
    if sampled.size < 2:
        return np.array([], dtype=float)

    if return_type == "log":
        if np.any(sampled <= 0.0):
            sampled = sampled[sampled > 0.0]
            if sampled.size < 2:
                return np.array([], dtype=float)
        out = np.diff(np.log(sampled))
    elif return_type == "simple":
        out = np.diff(sampled) / sampled[:-1]
    else:
        raise ValueError(f"Unsupported return_type: {return_type}")

    out = np.asarray(out, dtype=float)
    out = out[np.isfinite(out)]
    return out


def _aggregate_input_returns(
    returns: np.ndarray,
    horizon: int,
    *,
    return_type: str = "log",
) -> np.ndarray:
    """Aggregate precomputed returns into non-overlapping horizon blocks."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if r.size < 2:
        return np.array([], dtype=float)
    if horizon == 1:
        return r

    n_full = (r.size // horizon) * horizon
    if n_full < horizon:
        return np.array([], dtype=float)

    blocks = r[:n_full].reshape(-1, horizon)
    if return_type == "log":
        out = np.sum(blocks, axis=1)
    elif return_type == "simple":
        out = np.prod(1.0 + blocks, axis=1) - 1.0
    else:
        raise ValueError(f"Unsupported return_type: {return_type}")
    out = np.asarray(out, dtype=float)
    out = out[np.isfinite(out)]
    return out


def _compute_max_lag_acf(n_values: int) -> int:
    """Default ACF lag cap from sample size: min(250, floor(n/10)), at least 1."""
    return int(max(1, min(250, n_values // 10)))


def _acf_series(values: np.ndarray, *, nlags: int) -> np.ndarray:
    """Compute sample ACF up to `nlags` with lag 0 included."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.array([], dtype=float)

    lag_cap = int(max(1, min(nlags, x.size - 1)))
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.array([], dtype=float)

    acf_vals = np.empty(lag_cap + 1, dtype=float)
    acf_vals[0] = 1.0
    for lag in range(1, lag_cap + 1):
        acf_vals[lag] = float(np.dot(x[:-lag], x[lag:]) / denom)
    return acf_vals


def _lagged_corr_curve(returns: np.ndarray, *, max_lag: int, future_proxy: str) -> np.ndarray:
    """Compute corr(r_t, proxy(r_{t+lag})) for lag >= 1."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 5:
        return np.array([], dtype=float)

    lag_cap = int(max(1, min(max_lag, r.size - 2)))
    out = np.full(lag_cap, np.nan, dtype=float)
    for lag in range(1, lag_cap + 1):
        x = r[:-lag]
        if future_proxy == "abs":
            y = np.abs(r[lag:])
        elif future_proxy == "sq":
            y = r[lag:] ** 2
        else:
            raise ValueError(f"Unsupported future_proxy: {future_proxy}")
        if x.size < 5 or y.size < 5:
            continue
        sx = float(np.std(x, ddof=1))
        sy = float(np.std(y, ddof=1))
        if sx <= 0.0 or sy <= 0.0 or not np.isfinite(sx) or not np.isfinite(sy):
            continue
        out[lag - 1] = float(np.corrcoef(x, y)[0, 1])
    return out


def _zscore(values: np.ndarray) -> np.ndarray:
    """Standardize finite values and guard degenerate variance."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.array([], dtype=float)
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    if sigma <= 0.0 or not np.isfinite(sigma):
        return np.array([], dtype=float)
    return (x - mu) / sigma


def _tail_values(returns: np.ndarray, *, side: str) -> np.ndarray:
    """Extract positive tail magnitudes for one side of returns."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if side == "upper":
        x = r[r > 0.0]
    elif side == "lower":
        x = -r[r < 0.0]
    elif side == "abs":
        x = np.abs(r[r != 0.0])
    else:
        raise ValueError(f"Unsupported tail side: {side}")
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0.0]
    return x


def _build_k_grid(n_tail: int, *, k_min: int = 10, k_max_cap: int = 1000, n_grid: int = 100) -> np.ndarray:
    """Build an integer k-grid for Hill diagnostics."""
    if n_tail < 3:
        return np.array([], dtype=int)
    k_max = int(min(k_max_cap, n_tail // 2))
    if k_max < 2:
        return np.array([], dtype=int)
    k_lo = int(max(2, min(k_min, k_max)))
    if k_lo == k_max:
        return np.array([k_lo], dtype=int)
    n_points = int(min(n_grid, k_max - k_lo + 1))
    k_vals = np.linspace(k_lo, k_max, num=max(2, n_points), dtype=int)
    k_vals = np.unique(k_vals)
    return k_vals[k_vals >= 2]


def _hill_curve(tail_values: np.ndarray, *, k_values: np.ndarray, ci_level: float = 0.95) -> Dict[str, np.ndarray]:
    """Compute Hill gamma/alpha curves with asymptotic CI bands."""
    x = np.asarray(tail_values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0.0]
    if x.size < 3 or k_values.size == 0:
        empty = np.array([], dtype=float)
        return {
            "k": np.array([], dtype=int),
            "gamma": empty,
            "alpha": empty,
            "gamma_ci_low": empty,
            "gamma_ci_high": empty,
            "alpha_ci_low": empty,
            "alpha_ci_high": empty,
        }

    x_sorted = np.sort(x)
    zcrit = float(stats.norm.ppf(0.5 + 0.5 * ci_level))

    out_k: List[int] = []
    gamma_vals: List[float] = []
    alpha_vals: List[float] = []
    gamma_lo_vals: List[float] = []
    gamma_hi_vals: List[float] = []
    alpha_lo_vals: List[float] = []
    alpha_hi_vals: List[float] = []

    n = x_sorted.size
    for k_raw in np.asarray(k_values, dtype=int):
        k = int(k_raw)
        if k < 2 or (k + 1) > n:
            continue
        threshold = float(x_sorted[n - k - 1])
        if threshold <= 0.0 or not np.isfinite(threshold):
            continue

        top_k = x_sorted[n - k :]
        log_excess = np.log(top_k) - math.log(threshold)
        gamma_hat = float(np.mean(log_excess))
        if gamma_hat <= 0.0 or not np.isfinite(gamma_hat):
            continue
        alpha_hat = float(1.0 / gamma_hat)

        se_gamma = float(gamma_hat / math.sqrt(k))
        g_lo = max(gamma_hat - zcrit * se_gamma, 1e-12)
        g_hi = gamma_hat + zcrit * se_gamma

        # Delta method: alpha = 1/gamma => se(alpha) ~= se(gamma) / gamma^2
        se_alpha = float(se_gamma / (gamma_hat**2))
        a_lo = max(alpha_hat - zcrit * se_alpha, 0.0)
        a_hi = alpha_hat + zcrit * se_alpha

        out_k.append(k)
        gamma_vals.append(gamma_hat)
        alpha_vals.append(alpha_hat)
        gamma_lo_vals.append(g_lo)
        gamma_hi_vals.append(g_hi)
        alpha_lo_vals.append(a_lo)
        alpha_hi_vals.append(a_hi)

    return {
        "k": np.asarray(out_k, dtype=int),
        "gamma": np.asarray(gamma_vals, dtype=float),
        "alpha": np.asarray(alpha_vals, dtype=float),
        "gamma_ci_low": np.asarray(gamma_lo_vals, dtype=float),
        "gamma_ci_high": np.asarray(gamma_hi_vals, dtype=float),
        "alpha_ci_low": np.asarray(alpha_lo_vals, dtype=float),
        "alpha_ci_high": np.asarray(alpha_hi_vals, dtype=float),
    }


def _choose_stable_k(curve: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """Choose a working k* where alpha(k) is locally stable after mild smoothing."""
    k = np.asarray(curve["k"], dtype=int)
    alpha = np.asarray(curve["alpha"], dtype=float)
    gamma = np.asarray(curve["gamma"], dtype=float)
    alpha_lo = np.asarray(curve["alpha_ci_low"], dtype=float)
    alpha_hi = np.asarray(curve["alpha_ci_high"], dtype=float)
    gamma_lo = np.asarray(curve["gamma_ci_low"], dtype=float)
    gamma_hi = np.asarray(curve["gamma_ci_high"], dtype=float)

    valid = np.isfinite(alpha) & np.isfinite(gamma) & np.isfinite(k.astype(float))
    if np.sum(valid) < 5:
        return {
            "k_star": None,
            "alpha_star": float("nan"),
            "gamma_star": float("nan"),
            "alpha_star_ci_low": float("nan"),
            "alpha_star_ci_high": float("nan"),
            "gamma_star_ci_low": float("nan"),
            "gamma_star_ci_high": float("nan"),
            "alpha_mid50_min": float("nan"),
            "alpha_mid50_max": float("nan"),
            "gamma_mid50_min": float("nan"),
            "gamma_mid50_max": float("nan"),
        }

    kv = k[valid]
    av = alpha[valid]
    gv = gamma[valid]
    av_lo = alpha_lo[valid]
    av_hi = alpha_hi[valid]
    gv_lo = gamma_lo[valid]
    gv_hi = gamma_hi[valid]

    logk = np.log(kv.astype(float))
    n = av.size
    window = min(9, n)
    if window % 2 == 0:
        window -= 1
    if window >= 3:
        kernel = np.ones(window, dtype=float) / float(window)
        av_smooth = np.convolve(av, kernel, mode="same")
    else:
        av_smooth = av

    slope = np.abs(np.gradient(av_smooth, logk))
    lo = int(math.floor(0.2 * n))
    hi = int(math.ceil(0.8 * n))
    cand_idx = np.arange(max(1, lo), min(n - 1, max(lo + 1, hi)))
    if cand_idx.size == 0:
        cand_idx = np.arange(n)
    best_local = int(cand_idx[np.argmin(slope[cand_idx])])

    mid_lo = int(math.floor(0.25 * n))
    mid_hi = int(math.ceil(0.75 * n))
    mid_idx = np.arange(mid_lo, max(mid_lo + 1, mid_hi))
    if mid_idx.size == 0:
        mid_idx = np.arange(n)

    return {
        "k_star": int(kv[best_local]),
        "alpha_star": float(av[best_local]),
        "gamma_star": float(gv[best_local]),
        "alpha_star_ci_low": float(av_lo[best_local]),
        "alpha_star_ci_high": float(av_hi[best_local]),
        "gamma_star_ci_low": float(gv_lo[best_local]),
        "gamma_star_ci_high": float(gv_hi[best_local]),
        "alpha_mid50_min": float(np.nanmin(av[mid_idx])),
        "alpha_mid50_max": float(np.nanmax(av[mid_idx])),
        "gamma_mid50_min": float(np.nanmin(gv[mid_idx])),
        "gamma_mid50_max": float(np.nanmax(gv[mid_idx])),
    }


def _compute_hill_diagnostic(
    returns: np.ndarray,
    *,
    side: str,
    k_min: int = 10,
    k_max_cap: int = 1000,
    n_grid: int = 100,
    ci_level: float = 0.95,
) -> Dict[str, Any]:
    """Compute Hill diagnostic details for one tail side."""
    tail = _tail_values(returns, side=side)
    n_tail = int(tail.size)
    k_grid = _build_k_grid(n_tail, k_min=k_min, k_max_cap=k_max_cap, n_grid=n_grid)
    curve = _hill_curve(tail, k_values=k_grid, ci_level=ci_level)
    stable = _choose_stable_k(curve)

    k_max = int(min(k_max_cap, n_tail // 2))
    warning = ""
    if n_tail < 50:
        warning = "Too few tail points for stable inference (n_tail < 50)."
    if curve["k"].size == 0 and n_tail > 0:
        warning = (warning + " " if warning else "") + "Hill curve unavailable for requested k-grid."

    return {
        "side": side,
        "n_tail": n_tail,
        "k_min": int(k_min),
        "k_max": int(max(0, k_max)),
        "k_grid_size": int(curve["k"].size),
        "ci_level": float(ci_level),
        "warning": warning,
        "k": curve["k"],
        "gamma": curve["gamma"],
        "alpha": curve["alpha"],
        "gamma_ci_low": curve["gamma_ci_low"],
        "gamma_ci_high": curve["gamma_ci_high"],
        "alpha_ci_low": curve["alpha_ci_low"],
        "alpha_ci_high": curve["alpha_ci_high"],
        **stable,
    }


def _compute_tail_metrics(returns: np.ndarray) -> Dict[str, Any]:
    """Compute fat-tail and robust tail diagnostics from returns."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 10:
        return {
            "n_returns": int(r.size),
            "n_pos": int(np.sum(r > 0.0)),
            "n_neg": int(np.sum(r < 0.0)),
            "n_abs": int(np.sum(r != 0.0)),
            "kurtosis_excess": float("nan"),
            "tail_exceed_3sigma_share": float("nan"),
            "iqr_outlier_share": float("nan"),
            "mad_scale": float("nan"),
            "q99_over_q95_abs": float("nan"),
        }

    std = float(np.std(r, ddof=1))
    exceed_share = float(np.mean(np.abs(r) > 3.0 * std)) if (std > 0 and np.isfinite(std)) else float("nan")

    q1 = float(np.quantile(r, 0.25))
    q3 = float(np.quantile(r, 0.75))
    iqr = q3 - q1
    if iqr > 0:
        lo = q1 - 3.0 * iqr
        hi = q3 + 3.0 * iqr
        iqr_outlier_share = float(np.mean((r < lo) | (r > hi)))
    else:
        iqr_outlier_share = float("nan")

    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    mad_scale = 1.4826 * mad

    abs_r = np.abs(r)
    q95 = float(np.quantile(abs_r, 0.95))
    q99 = float(np.quantile(abs_r, 0.99))
    q99_over_q95 = (q99 / q95) if q95 > 0 else float("nan")

    return {
        "n_returns": int(r.size),
        "n_pos": int(np.sum(r > 0.0)),
        "n_neg": int(np.sum(r < 0.0)),
        "n_abs": int(np.sum(r != 0.0)),
        "kurtosis_excess": float(stats.kurtosis(r, fisher=True, bias=False)),
        "tail_exceed_3sigma_share": exceed_share,
        "iqr_outlier_share": iqr_outlier_share,
        "mad_scale": float(mad_scale),
        "q99_over_q95_abs": float(q99_over_q95),
    }


def _compute_horizon_summary(returns: np.ndarray, *, acf_max_lag: int, leverage_max_lag: int) -> Dict[str, Any]:
    """Compute horizon summary metrics (distribution + clustering + leverage)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {
            "n_returns": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "skewness": float("nan"),
            "kurtosis_excess": float("nan"),
            "acf_abs_lag1": float("nan"),
            "acf_sq_lag1": float("nan"),
            "acf_ret_lag1": float("nan"),
            "leverage_lag1": float("nan"),
            "leverage_sq_lag1": float("nan"),
            "leverage_abs_curve": np.array([], dtype=float),
            "leverage_sq_curve": np.array([], dtype=float),
        }

    acf_abs = _acf_series(np.abs(r), nlags=acf_max_lag)
    acf_sq = _acf_series(r**2, nlags=acf_max_lag)
    acf_ret = _acf_series(r, nlags=acf_max_lag)
    lev_abs = _lagged_corr_curve(r, max_lag=leverage_max_lag, future_proxy="abs")
    lev_sq = _lagged_corr_curve(r, max_lag=leverage_max_lag, future_proxy="sq")

    return {
        "n_returns": int(r.size),
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=1)) if r.size > 1 else float("nan"),
        "skewness": float(stats.skew(r, bias=False)) if r.size > 2 else float("nan"),
        "kurtosis_excess": float(stats.kurtosis(r, fisher=True, bias=False)) if r.size > 3 else float("nan"),
        "acf_abs_lag1": float(acf_abs[1]) if acf_abs.size > 1 else float("nan"),
        "acf_sq_lag1": float(acf_sq[1]) if acf_sq.size > 1 else float("nan"),
        "acf_ret_lag1": float(acf_ret[1]) if acf_ret.size > 1 else float("nan"),
        "leverage_lag1": float(lev_abs[0]) if lev_abs.size > 0 else float("nan"),
        "leverage_sq_lag1": float(lev_sq[0]) if lev_sq.size > 0 else float("nan"),
        "leverage_abs_curve": lev_abs,
        "leverage_sq_curve": lev_sq,
    }


def _save_plotly_figure(
    fig: go.Figure,
    *,
    name: str,
    figures_dir: Path,
    allow_png: bool = True,
    png_disabled_reason: str = "",
) -> FigureSaveResult:
    """Persist a Plotly figure to HTML and best-effort PNG."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    html_path = figures_dir / f"{name}.html"
    png_path = figures_dir / f"{name}.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    if not allow_png:
        reason = png_disabled_reason or "PNG export disabled for this run."
        return FigureSaveResult(
            name=name,
            html_path=html_path,
            png_path=png_path,
            png_ok=False,
            png_error=reason,
        )

    has_kaleido = importlib.util.find_spec("kaleido") is not None
    if not has_kaleido:
        return FigureSaveResult(
            name=name,
            html_path=html_path,
            png_path=png_path,
            png_ok=False,
            png_error="kaleido not installed (install kaleido for PNG export).",
        )

    chrome_candidates = [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "chrome",
    ]
    if not any(shutil.which(cmd) for cmd in chrome_candidates):
        return FigureSaveResult(
            name=name,
            html_path=html_path,
            png_path=png_path,
            png_ok=False,
            png_error="No local Chrome/Chromium executable found; PNG export skipped.",
        )

    def _timeout_handler(signum: int, frame: Any) -> None:
        raise TimeoutError("Timed out during Plotly PNG export")

    try:
        prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(10)
        try:
            fig.write_image(str(png_path), width=1400, height=900, scale=1.0)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)
        return FigureSaveResult(name=name, html_path=html_path, png_path=png_path, png_ok=True, png_error="")
    except Exception as exc:  # pragma: no cover
        return FigureSaveResult(
            name=name,
            html_path=html_path,
            png_path=png_path,
            png_ok=False,
            png_error=f"{exc} (install kaleido and a local Chrome/Chromium).",
        )


def _fmt(x: Any, digits: int = 4) -> str:
    """Format helper for report text."""
    try:
        xf = float(x)
    except Exception:
        return "n/a"
    if not np.isfinite(xf):
        return "n/a"
    return f"{xf:.{digits}g}"


def _subplot_shape(n_panels: int) -> Tuple[int, int]:
    """Compute a compact subplot grid."""
    cols = 2 if n_panels > 1 else 1
    rows = int(math.ceil(n_panels / cols))
    return rows, cols


def _color_for_horizon(index: int) -> str:
    """Color palette helper."""
    palette = ["#60A5FA", "#F59E0B", "#22C55E", "#E879F9", "#F43F5E", "#2DD4BF", "#A78BFA"]
    return palette[index % len(palette)]


def _build_qq_figure(returns_by_h: Mapping[int, np.ndarray]) -> go.Figure:
    """Build QQ plots versus Normal for all horizons."""
    horizons = sorted(returns_by_h.keys())
    rows, cols = _subplot_shape(len(horizons))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"h={h}" for h in horizons])
    for idx, h in enumerate(horizons):
        row = idx // cols + 1
        col = idx % cols + 1
        z = _zscore(returns_by_h[h])
        if z.size < 10:
            continue
        theo_q, samp_q = stats.probplot(z, dist="norm", fit=False)
        theo = np.asarray(theo_q, dtype=float)
        samp = np.asarray(samp_q, dtype=float)
        r_qq = float(np.corrcoef(theo, samp)[0, 1]) if theo.size > 1 else float("nan")
        fig.add_trace(
            go.Scatter(
                x=theo,
                y=samp,
                mode="markers",
                marker=dict(size=4, color="#60A5FA", opacity=0.75),
                name=f"h={h} QQ",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        lo = float(np.nanmin(theo))
        hi = float(np.nanmax(theo))
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color="#E2E8F0", width=1.8, dash="dash"),
                name="45-degree",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_annotation(
            text=f"corr={r_qq:.3f}",
            x=0.98,
            y=0.02,
            xref=f"x{idx+1} domain" if idx > 0 else "x domain",
            yref=f"y{idx+1} domain" if idx > 0 else "y domain",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=10, color="#A1A1AA"),
        )
    fig.update_layout(template="plotly_dark", title="Fat tails: QQ plots vs Normal", height=max(450, 360 * rows))
    fig.update_xaxes(title_text="Normal quantiles")
    fig.update_yaxes(title_text="Empirical quantiles")
    return fig


def _build_vol_clustering_figure(returns_by_h: Mapping[int, np.ndarray], *, max_lag: int) -> go.Figure:
    """Build ACF diagnostics for |r| and r^2, starting at lag 1."""
    horizons = sorted(returns_by_h.keys())
    rows, cols = _subplot_shape(len(horizons))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"h={h}" for h in horizons])
    for idx, h in enumerate(horizons):
        row = idx // cols + 1
        col = idx % cols + 1
        r = np.asarray(returns_by_h[h], dtype=float)
        acf_abs = _acf_series(np.abs(r), nlags=max_lag)
        acf_sq = _acf_series(r**2, nlags=max_lag)
        if acf_abs.size > 1:
            lags = np.arange(1, acf_abs.size)
            fig.add_trace(
                go.Scatter(
                    x=lags,
                    y=acf_abs[1:],
                    mode="lines",
                    line=dict(color="#F59E0B", width=1.8),
                    name="ACF |r|",
                    showlegend=(idx == 0),
                ),
                row=row,
                col=col,
            )
        if acf_sq.size > 1:
            lags = np.arange(1, acf_sq.size)
            fig.add_trace(
                go.Scatter(
                    x=lags,
                    y=acf_sq[1:],
                    mode="lines",
                    line=dict(color="#2DD4BF", width=1.8, dash="dash"),
                    name="ACF r^2",
                    showlegend=(idx == 0),
                ),
                row=row,
                col=col,
            )
    fig.update_layout(
        template="plotly_dark",
        title=f"Volatility clustering by horizon (ACF lags 1..{max_lag})",
        height=max(450, 360 * rows),
    )
    fig.update_xaxes(title_text="Lag")
    fig.update_yaxes(title_text="ACF")
    return fig


def _build_leverage_figure(
    leverage_abs_by_h: Mapping[int, np.ndarray],
    leverage_sq_by_h: Mapping[int, np.ndarray],
) -> go.Figure:
    """Build lagged asymmetry curves for |r| and r^2 proxies."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("corr(r_t, |r_{t+lag}|)", "corr(r_t, r_{t+lag}^2)"),
    )

    horizons = sorted(leverage_abs_by_h.keys())
    for idx, h in enumerate(horizons):
        color = _color_for_horizon(idx)
        abs_vals = np.asarray(leverage_abs_by_h[h], dtype=float)
        sq_vals = np.asarray(leverage_sq_by_h[h], dtype=float)
        if abs_vals.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=np.arange(1, abs_vals.size + 1),
                    y=abs_vals,
                    mode="lines+markers",
                    line=dict(width=1.8, color=color),
                    marker=dict(size=4),
                    name=f"h={h}",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        if sq_vals.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=np.arange(1, sq_vals.size + 1),
                    y=sq_vals,
                    mode="lines+markers",
                    line=dict(width=1.8, color=color),
                    marker=dict(size=4),
                    name=f"h={h}",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    fig.add_hline(y=0.0, line=dict(color="#94A3B8", width=1, dash="dot"), row=1, col=1)
    fig.add_hline(y=0.0, line=dict(color="#94A3B8", width=1, dash="dot"), row=1, col=2)
    fig.update_layout(template="plotly_dark", title="Leverage/asymmetry diagnostics by horizon", height=520)
    fig.update_xaxes(title_text="Lag")
    fig.update_yaxes(title_text="Correlation")
    return fig


def _build_scaling_kde_figure(returns_by_h: Mapping[int, np.ndarray]) -> go.Figure:
    """Build standardized return density comparison across horizons with Gaussian reference."""
    fig = go.Figure()
    x_grid = np.linspace(-8.0, 8.0, 1200)
    horizons = sorted(returns_by_h.keys())

    for idx, h in enumerate(horizons):
        color = _color_for_horizon(idx)
        z = _zscore(returns_by_h[h])
        if z.size < 30:
            continue
        try:
            kde = stats.gaussian_kde(z)
            density = np.asarray(kde.evaluate(x_grid), dtype=float)
        except Exception:
            hist, edges = np.histogram(z, bins=80, density=True)
            centers = 0.5 * (edges[1:] + edges[:-1])
            density = np.interp(x_grid, centers, hist, left=np.nan, right=np.nan)

        density[~np.isfinite(density)] = np.nan
        density[density <= 0] = np.nan
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=density,
                mode="lines",
                line=dict(width=2.0, color=color),
                name=f"h={h}",
            )
        )

    gauss = stats.norm.pdf(x_grid)
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=gauss,
            mode="lines",
            line=dict(width=2, color="#E2E8F0", dash="dash"),
            name="N(0,1)",
        )
    )

    fig.update_layout(template="plotly_dark", title="Scaling: standardized return densities (log-y)", height=540)
    fig.update_xaxes(title_text="Standardized return")
    fig.update_yaxes(title_text="Density", type="log")
    return fig


def _build_return_acf_figure(returns_by_h: Mapping[int, np.ndarray], *, max_lag: int) -> go.Figure:
    """Build return ACF comparison by horizon (lag 1 onward)."""
    fig = go.Figure()
    horizons = sorted(returns_by_h.keys())
    for idx, h in enumerate(horizons):
        color = _color_for_horizon(idx)
        acf_r = _acf_series(np.asarray(returns_by_h[h], dtype=float), nlags=max_lag)
        if acf_r.size <= 1:
            continue
        lags = np.arange(1, acf_r.size)
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=acf_r[1:],
                mode="lines",
                line=dict(width=1.8, color=color),
                name=f"h={h}",
            )
        )
    fig.add_hline(y=0.0, line=dict(color="#94A3B8", width=1, dash="dot"))
    fig.update_layout(template="plotly_dark", title=f"Return ACF by horizon (lags 1..{max_lag})", height=520)
    fig.update_xaxes(title_text="Lag")
    fig.update_yaxes(title_text="ACF")
    return fig


def _build_hill_plot(hill_diag: Mapping[str, Any], *, horizon: int, side: str) -> go.Figure:
    """Build one Hill plot (alpha with CI, gamma as optional trace)."""
    k = np.asarray(hill_diag["k"], dtype=float)
    alpha = np.asarray(hill_diag["alpha"], dtype=float)
    alpha_lo = np.asarray(hill_diag["alpha_ci_low"], dtype=float)
    alpha_hi = np.asarray(hill_diag["alpha_ci_high"], dtype=float)
    gamma = np.asarray(hill_diag["gamma"], dtype=float)

    fig = go.Figure()
    if k.size > 0:
        fill_x = np.concatenate([k, k[::-1]])
        fill_y = np.concatenate([alpha_hi, alpha_lo[::-1]])
        fig.add_trace(
            go.Scatter(
                x=fill_x,
                y=fill_y,
                fill="toself",
                fillcolor="rgba(96,165,250,0.18)",
                line=dict(color="rgba(96,165,250,0)"),
                hoverinfo="skip",
                name="95% CI alpha",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=k,
                y=alpha,
                mode="lines",
                line=dict(color="#60A5FA", width=2),
                name="alpha(k)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=k,
                y=gamma,
                mode="lines",
                line=dict(color="#F59E0B", width=1.6, dash="dash"),
                name="gamma(k)",
                yaxis="y2",
                visible="legendonly",
            )
        )

    k_star = hill_diag.get("k_star")
    alpha_star = hill_diag.get("alpha_star")
    if isinstance(k_star, int) and np.isfinite(float(alpha_star)):
        fig.add_vline(x=float(k_star), line=dict(color="#E2E8F0", width=1.2, dash="dot"))
        fig.add_annotation(
            x=float(k_star),
            y=float(alpha_star),
            text=f"k*={k_star}, alpha={float(alpha_star):.3g}",
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-25,
            font=dict(size=10, color="#E2E8F0"),
        )

    side_title = {"upper": "Upper tail", "lower": "Lower tail", "abs": "Absolute tail"}[side]
    fig.update_layout(
        template="plotly_dark",
        title=f"Hill plot ({side_title}) - horizon h={horizon}",
        height=520,
        yaxis=dict(title="alpha(k)"),
        yaxis2=dict(title="gamma(k)", overlaying="y", side="right", showgrid=False),
    )
    fig.update_xaxes(title_text="k (tail points used)")
    warning = str(hill_diag.get("warning", "")).strip()
    if warning:
        fig.add_annotation(
            text=warning,
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=10, color="#FCA5A5"),
            bgcolor="rgba(127,29,29,0.2)",
        )
    return fig


def _write_tables(
    *,
    tables_dir: Path,
    horizons: Sequence[int],
    horizon_summary: Mapping[int, Mapping[str, Any]],
    tail_metrics: Mapping[int, Mapping[str, Any]],
    hill_diagnostics: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Path]:
    """Write CSV tables used by the report."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    horizon_path = tables_dir / "horizon_summary.csv"
    with horizon_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "horizon",
                "n_returns",
                "mean",
                "std",
                "skewness",
                "kurtosis_excess",
                "acf_abs_lag1",
                "acf_sq_lag1",
                "acf_ret_lag1",
                "leverage_lag1",
                "leverage_sq_lag1",
            ],
        )
        writer.writeheader()
        for h in horizons:
            hs = horizon_summary[h]
            writer.writerow(
                {
                    "horizon": h,
                    "n_returns": hs["n_returns"],
                    "mean": hs["mean"],
                    "std": hs["std"],
                    "skewness": hs["skewness"],
                    "kurtosis_excess": hs["kurtosis_excess"],
                    "acf_abs_lag1": hs["acf_abs_lag1"],
                    "acf_sq_lag1": hs["acf_sq_lag1"],
                    "acf_ret_lag1": hs["acf_ret_lag1"],
                    "leverage_lag1": hs["leverage_lag1"],
                    "leverage_sq_lag1": hs["leverage_sq_lag1"],
                }
            )

    tail_path = tables_dir / "tail_metrics.csv"
    with tail_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "horizon",
                "n_pos",
                "n_neg",
                "n_abs",
                "kurtosis_excess",
                "tail_exceed_3sigma_share",
                "iqr_outlier_share",
                "mad_scale",
                "q99_over_q95_abs",
            ],
        )
        writer.writeheader()
        for h in horizons:
            tm = tail_metrics[h]
            writer.writerow(
                {
                    "horizon": h,
                    "n_pos": tm["n_pos"],
                    "n_neg": tm["n_neg"],
                    "n_abs": tm["n_abs"],
                    "kurtosis_excess": tm["kurtosis_excess"],
                    "tail_exceed_3sigma_share": tm["tail_exceed_3sigma_share"],
                    "iqr_outlier_share": tm["iqr_outlier_share"],
                    "mad_scale": tm["mad_scale"],
                    "q99_over_q95_abs": tm["q99_over_q95_abs"],
                }
            )

    hill_path = tables_dir / "hill_working_estimates.csv"
    with hill_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "horizon",
                "tail_side",
                "n_tail",
                "k_min",
                "k_max",
                "k_star",
                "alpha_star",
                "alpha_star_ci_low",
                "alpha_star_ci_high",
                "gamma_star",
                "gamma_star_ci_low",
                "gamma_star_ci_high",
                "alpha_mid50_min",
                "alpha_mid50_max",
                "warning",
            ],
        )
        writer.writeheader()
        for h in horizons:
            for side in ("upper", "lower", "abs"):
                hd = hill_diagnostics[h][side]
                writer.writerow(
                    {
                        "horizon": h,
                        "tail_side": side,
                        "n_tail": hd["n_tail"],
                        "k_min": hd["k_min"],
                        "k_max": hd["k_max"],
                        "k_star": hd["k_star"],
                        "alpha_star": hd["alpha_star"],
                        "alpha_star_ci_low": hd["alpha_star_ci_low"],
                        "alpha_star_ci_high": hd["alpha_star_ci_high"],
                        "gamma_star": hd["gamma_star"],
                        "gamma_star_ci_low": hd["gamma_star_ci_low"],
                        "gamma_star_ci_high": hd["gamma_star_ci_high"],
                        "alpha_mid50_min": hd["alpha_mid50_min"],
                        "alpha_mid50_max": hd["alpha_mid50_max"],
                        "warning": hd["warning"],
                    }
                )

    return {
        "horizon_summary": horizon_path,
        "tail_metrics": tail_path,
        "hill_working_estimates": hill_path,
    }


def _build_report_text(
    *,
    input_path: Path,
    input_kind: str,
    return_type: str,
    horizons: Sequence[int],
    preprocess_info: Mapping[str, int],
    horizon_summary: Mapping[int, Mapping[str, Any]],
    tail_metrics: Mapping[int, Mapping[str, Any]],
    hill_diagnostics: Mapping[int, Mapping[str, Mapping[str, Any]]],
    figure_results: Sequence[FigureSaveResult],
    table_paths: Mapping[str, Path],
    acf_max_lag: int,
    leverage_max_lag: int,
) -> str:
    """Build the markdown report body."""

    def f(x: Any, digits: int = 4) -> str:
        return _fmt(x, digits=digits)

    lines: List[str] = []
    lines.append("# Stylized Facts Report")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("- Goal: first-pass econophysics diagnostics for market-like behavior in ABM-generated series.")
    lines.append("- Scope: fat tails, volatility clustering, leverage/asymmetry, scaling, and return autocorrelation.")
    lines.append("- Inference level: diagnostics only (not proof of a generative model).")
    lines.append("")

    lines.append("## Data provenance and preprocessing")
    lines.append(f"- Input path: `{input_path}`")
    lines.append(f"- Input kind: **{input_kind}**")
    lines.append(f"- Return type analyzed: **{return_type}**")
    lines.append(f"- Raw sample size: **{preprocess_info['raw_size']}**")
    lines.append(f"- Dropped non-finite values: **{preprocess_info['dropped_nonfinite_count']}**")
    if input_kind == "prices":
        lines.append(f"- Dropped non-positive prices (<=0): **{preprocess_info['dropped_nonpositive_count']}**")
    lines.append(f"- Final clean sample size: **{preprocess_info['final_size']}**")
    lines.append("- Horizons analyzed: **1, 5, 10, 20** (non-overlapping construction).")
    if input_kind == "prices":
        lines.append("- Construction rule: downsample prices at horizon h, then compute returns on sampled prices.")
    else:
        lines.append("- Construction rule: aggregate precomputed returns in non-overlapping blocks per horizon.")
    lines.append("")

    lines.append("## Sample size table")
    lines.append("| Horizon | n_returns | n_pos | n_neg | n_abs |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for h in horizons:
        hs = horizon_summary[h]
        tm = tail_metrics[h]
        lines.append(f"| {h} | {hs['n_returns']} | {tm['n_pos']} | {tm['n_neg']} | {tm['n_abs']} |")
    lines.append("")

    lines.append("## 1) Fat tails")
    lines.append("- QQ plots vs Normal are qualitative diagnostics; tail uncertainty dominates finite samples.")
    lines.append("| Horizon | excess kurtosis | |r|>3σ share | IQR outlier share | MAD-scale | q(0.99)/q(0.95) for |r| |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for h in horizons:
        tm = tail_metrics[h]
        lines.append(
            f"| {h} | {f(tm['kurtosis_excess'])} | {f(tm['tail_exceed_3sigma_share'])} | "
            f"{f(tm['iqr_outlier_share'])} | {f(tm['mad_scale'])} | {f(tm['q99_over_q95_abs'])} |"
        )
    lines.append("")
    lines.append("### Hill tail index diagnostics (alpha = 1/gamma)")
    lines.append("| Horizon | Tail side | n_tail | k-range | k* | alpha(k*) [95% CI] | gamma(k*) [95% CI] | alpha range (middle 50% k-grid) |")
    lines.append("| --- | --- | ---: | --- | ---: | --- | --- | --- |")
    for h in horizons:
        for side in ("upper", "lower", "abs"):
            hd = hill_diagnostics[h][side]
            k_star = hd["k_star"]
            k_range = f"{hd['k_min']}..{hd['k_max']}"
            alpha_star = f"{f(hd['alpha_star'])} [{f(hd['alpha_star_ci_low'])}, {f(hd['alpha_star_ci_high'])}]"
            gamma_star = f"{f(hd['gamma_star'])} [{f(hd['gamma_star_ci_low'])}, {f(hd['gamma_star_ci_high'])}]"
            alpha_range = f"[{f(hd['alpha_mid50_min'])}, {f(hd['alpha_mid50_max'])}]"
            lines.append(
                f"| {h} | {side} | {hd['n_tail']} | {k_range} | {k_star if k_star is not None else 'n/a'} | "
                f"{alpha_star} | {gamma_star} | {alpha_range} |"
            )
    lines.append("- Hill warning: for light-tailed data Hill can imply gamma≈0 and alpha→infinity, which is misleading.")
    lines.append("- Hill warning: returns are heteroskedastic; i.i.d. tail assumptions are violated by volatility clustering.")
    lines.append("- Use Hill plots as stability diagnostics, not as single-point proof.")
    low_tail_rows: List[str] = []
    for h in horizons:
        for side in ("upper", "lower", "abs"):
            w = str(hill_diagnostics[h][side].get("warning", "")).strip()
            if w:
                low_tail_rows.append(f"- h={h}, side={side}: {w}")
    if low_tail_rows:
        lines.append("- Tail-sample warnings:")
        lines.extend(low_tail_rows)
    lines.append("")

    lines.append("## 2) Volatility clustering")
    lines.append(f"- ACF lag cap uses default rule `min(250, floor(n/10))` with chosen `max_lag_acf={acf_max_lag}`.")
    lines.append("| Horizon | ACF(|r|, lag1) | ACF(r^2, lag1) |")
    lines.append("| --- | ---: | ---: |")
    for h in horizons:
        hs = horizon_summary[h]
        lines.append(f"| {h} | {f(hs['acf_abs_lag1'])} | {f(hs['acf_sq_lag1'])} |")
    lines.append("- Slow decay in ACF(|r|) and ACF(r^2) is consistent with clustering / long-memory proxies.")
    lines.append("- Optional GARCH(1,1) fit skipped in this first-pass report.")
    lines.append("")

    lines.append("## 3) Leverage and asymmetry")
    lines.append(f"- Lagged correlations computed for lags 1..{leverage_max_lag}.")
    lines.append("| Horizon | corr(r_t, |r_{t+1}|) | corr(r_t, r_{t+1}^2) |")
    lines.append("| --- | ---: | ---: |")
    for h in horizons:
        hs = horizon_summary[h]
        lines.append(f"| {h} | {f(hs['leverage_lag1'])} | {f(hs['leverage_sq_lag1'])} |")
    lines.append("- Negative values are consistent with leverage effect (common in equities, not universal).")
    lines.append("")

    lines.append("## 4) Scaling / aggregation")
    lines.append("- Distributions are compared across horizons using non-overlapping construction only.")
    lines.append("- KDEs are shown on standardized returns with log-y scale and Gaussian reference.")
    lines.append("- Caveat: KDE bandwidth and tail sparsity affect large-horizon interpretation.")
    lines.append("")

    lines.append("## 5) Return autocorrelation")
    lines.append("- Return ACF is reported by horizon, starting at lag 1 (lag 0 removed).")
    lines.append("- Near-zero short-lag ACF is typical for liquid daily data; microstructure can alter this at high frequency.")
    lines.append("")

    stylized_total = 0
    stylized_hits = 0
    for h in horizons:
        hs = horizon_summary[h]
        tm = tail_metrics[h]
        stylized_total += 3
        if np.isfinite(tm["kurtosis_excess"]) and tm["kurtosis_excess"] > 0:
            stylized_hits += 1
        if np.isfinite(hs["acf_abs_lag1"]) and hs["acf_abs_lag1"] > 0:
            stylized_hits += 1
        if np.isfinite(hs["leverage_lag1"]) and hs["leverage_lag1"] < 0:
            stylized_hits += 1
    hit_ratio = (stylized_hits / stylized_total) if stylized_total > 0 else 0.0

    lines.append("## 6) Conclusion")
    if hit_ratio >= 2.0 / 3.0:
        verdict = "Consistent with stylized facts"
    elif hit_ratio >= 1.0 / 3.0:
        verdict = "Partially consistent with stylized facts"
    else:
        verdict = "Weak alignment with stylized facts"
    lines.append(f"- Verdict: **{verdict}** (diagnostic score = {stylized_hits}/{stylized_total}).")
    lines.append("- Plausible deviation sources: microstructure noise, illiquidity, regime shifts, sampling artifacts, or missing-data handling.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("- Diagnostics are descriptive, not formal hypothesis tests.")
    lines.append("- Hill estimates are sensitive to threshold choice and dependence in returns.")
    lines.append("- Finite-sample uncertainty is substantial at large horizons and in one-sided tails.")
    lines.append("")

    lines.append("## What I would do next")
    lines.append("1. Add threshold-stability checks with alternative EVT estimators (Pickands, moment).")
    lines.append("2. Repeat tail analysis after volatility normalization / declustering.")
    lines.append("3. Check sub-sample stability across regimes and simulation seeds.")
    lines.append("4. Compare non-overlapping vs overlapping horizons as a sensitivity analysis.")
    lines.append("")

    lines.append("## Generated tables")
    lines.append(f"- Horizon summary: `{table_paths['horizon_summary']}`")
    lines.append(f"- Tail metrics: `{table_paths['tail_metrics']}`")
    lines.append(f"- Hill working estimates: `{table_paths['hill_working_estimates']}`")
    lines.append("")

    lines.append("## Figures")
    lines.append("| Figure | HTML | PNG | PNG status |")
    lines.append("| --- | --- | --- | --- |")
    for fr in figure_results:
        png_status = "ok" if fr.png_ok else f"failed ({fr.png_error})"
        lines.append(f"| `{fr.name}` | `{fr.html_path.name}` | `{fr.png_path.name}` | {png_status} |")

    return "\n".join(lines)


def run_stylized_facts(
    input_path: Path,
    *,
    horizons: Sequence[int] = (1, 5, 10, 20),
    input_kind: str = "prices",
    return_type: str = "log",
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run stylized-facts diagnostics and persist report + figures.

    Parameters
    ----------
    input_path
        Path to a `.npy` file containing a 1D price or returns series.
    horizons
        Sampling horizons in observation steps.
    input_kind
        Either `"prices"` (downsample prices then compute returns) or `"returns"`
        (aggregate precomputed returns in non-overlapping blocks).
    return_type
        Return construction: `"log"` or `"simple"`.
    out_dir
        Output directory. If `None`, defaults to
        `<input_path.parent>/stylized_facts_<input_path.stem>/`.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable summary with output paths and diagnostics.

    Notes
    -----
    - This is a diagnostic workflow; it does not prove model validity.
    - Hill confidence intervals are asymptotic and should be treated as rough guidance.
    - PNG export is best-effort and depends on `kaleido` plus a local browser runtime.

    Examples
    --------
    >>> from pathlib import Path
    >>> import json
    >>> latest = json.loads(Path("abm_results/scenarios/test/latest_run.json").read_text())
    >>> series_path = Path(latest["run_root"]) / "output_data" / "dex_price_end_of_block.npy"
    >>> summary = run_stylized_facts(series_path, input_kind="prices", return_type="log")  # doctest: +SKIP
    >>> print(summary["report_path"])
    """
    input_kind = str(input_kind).strip().lower()
    return_type = str(return_type).strip().lower()
    if input_kind not in {"prices", "returns"}:
        raise ValueError(f"input_kind must be 'prices' or 'returns', got {input_kind!r}")
    if return_type not in {"log", "simple"}:
        raise ValueError(f"return_type must be 'log' or 'simple', got {return_type!r}")

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_series = _load_series(input_path)
    clean_series, preprocess_info = _clean_input_series(raw_series, input_kind=input_kind)
    if clean_series.size < 10:
        raise ValueError("Not enough observations after preprocessing (need at least 10).")

    if out_dir is None:
        out_dir = input_path.parent / f"stylized_facts_{input_path.stem}"
    out_dir = Path(out_dir)
    if out_dir.exists():
        # Avoid silently overwriting an existing report folder; keep the existing
        # folder as an experimental record and create a suffixed sibling.
        try:
            has_any = any(out_dir.iterdir())
        except OSError:
            has_any = True
        if has_any:
            out_dir = make_unique_dir(out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(script="stylized_facts_report", run_id=out_dir.name, config_path=None)
    write_json(
        out_dir / "metadata.json",
        {
            **manifest.to_dict(),
            "input_path": str(input_path),
            "input_kind": str(input_kind),
            "return_type": str(return_type),
            "horizons": [int(h) for h in horizons],
        },
    )

    horizons_sorted = sorted({int(h) for h in horizons if int(h) > 0})
    if not horizons_sorted:
        raise ValueError("No valid positive horizons were provided.")

    returns_by_h: Dict[int, np.ndarray] = {}
    for h in horizons_sorted:
        if input_kind == "prices":
            returns_h = _returns_from_sampled_prices(clean_series, h, return_type=return_type)
        else:
            returns_h = _aggregate_input_returns(clean_series, h, return_type=return_type)
        returns_by_h[h] = np.asarray(returns_h, dtype=float)

    nonempty_lengths = [int(r.size) for r in returns_by_h.values() if int(r.size) > 0]
    if not nonempty_lengths:
        raise ValueError("No non-empty return series were produced across horizons.")
    n_min = min(nonempty_lengths)
    acf_max_lag = _compute_max_lag_acf(n_min)
    leverage_max_lag = int(max(1, min(30, n_min // 10)))

    horizon_summary: Dict[int, Dict[str, Any]] = {}
    tail_metrics: Dict[int, Dict[str, Any]] = {}
    hill_diagnostics: Dict[int, Dict[str, Dict[str, Any]]] = {}
    leverage_abs_by_h: Dict[int, np.ndarray] = {}
    leverage_sq_by_h: Dict[int, np.ndarray] = {}

    for h in horizons_sorted:
        r_h = returns_by_h[h]
        hs = _compute_horizon_summary(r_h, acf_max_lag=acf_max_lag, leverage_max_lag=leverage_max_lag)
        leverage_abs_by_h[h] = np.asarray(hs.pop("leverage_abs_curve"), dtype=float)
        leverage_sq_by_h[h] = np.asarray(hs.pop("leverage_sq_curve"), dtype=float)
        horizon_summary[h] = hs
        tail_metrics[h] = _compute_tail_metrics(r_h)
        hill_diagnostics[h] = {
            "upper": _compute_hill_diagnostic(r_h, side="upper"),
            "lower": _compute_hill_diagnostic(r_h, side="lower"),
            "abs": _compute_hill_diagnostic(r_h, side="abs"),
        }

    figure_results: List[FigureSaveResult] = []
    png_enabled = True
    png_disabled_reason = ""

    def save_figure(fig: go.Figure, *, name: str) -> FigureSaveResult:
        nonlocal png_enabled, png_disabled_reason
        res = _save_plotly_figure(
            fig,
            name=name,
            figures_dir=figures_dir,
            allow_png=png_enabled,
            png_disabled_reason=png_disabled_reason,
        )
        if png_enabled and not res.png_ok:
            png_enabled = False
            png_disabled_reason = f"PNG export disabled after first failure: {res.png_error}"
        return res

    figure_results.append(save_figure(_build_qq_figure(returns_by_h), name="fat_tails_qq_by_horizon"))
    figure_results.append(
        save_figure(
            _build_vol_clustering_figure(returns_by_h, max_lag=acf_max_lag),
            name="volatility_clustering_acf_by_horizon",
        )
    )
    figure_results.append(
        save_figure(
            _build_leverage_figure(leverage_abs_by_h, leverage_sq_by_h),
            name="leverage_lagged_correlation_by_horizon",
        )
    )
    figure_results.append(
        save_figure(
            _build_scaling_kde_figure(returns_by_h),
            name="scaling_kde_logy_by_horizon",
        )
    )
    figure_results.append(
        save_figure(
            _build_return_acf_figure(returns_by_h, max_lag=acf_max_lag),
            name="return_acf_by_horizon",
        )
    )

    for h in horizons_sorted:
        for side in ("upper", "lower", "abs"):
            fig = _build_hill_plot(hill_diagnostics[h][side], horizon=h, side=side)
            figure_results.append(save_figure(fig, name=f"hill_plot_{side}_h{h}"))

    table_paths = _write_tables(
        tables_dir=tables_dir,
        horizons=horizons_sorted,
        horizon_summary=horizon_summary,
        tail_metrics=tail_metrics,
        hill_diagnostics=hill_diagnostics,
    )

    report_text = _build_report_text(
        input_path=input_path,
        input_kind=input_kind,
        return_type=return_type,
        horizons=horizons_sorted,
        preprocess_info=preprocess_info,
        horizon_summary=horizon_summary,
        tail_metrics=tail_metrics,
        hill_diagnostics=hill_diagnostics,
        figure_results=figure_results,
        table_paths=table_paths,
        acf_max_lag=acf_max_lag,
        leverage_max_lag=leverage_max_lag,
    )
    report_path = out_dir / "stylized_facts_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    summary: MutableMapping[str, Any] = {
        "input_path": str(input_path),
        "input_kind": input_kind,
        "return_type": return_type,
        "output_dir": str(out_dir),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "report_path": str(report_path),
        "raw_size": preprocess_info["raw_size"],
        "sample_size": preprocess_info["final_size"],
        "dropped_nonfinite_count": preprocess_info["dropped_nonfinite_count"],
        "dropped_nonpositive_count": preprocess_info["dropped_nonpositive_count"],
        "horizons": horizons_sorted,
        "acf_max_lag": acf_max_lag,
        "leverage_max_lag": leverage_max_lag,
        "horizon_summary": horizon_summary,
        "tail_metrics": tail_metrics,
        "hill_diagnostics": hill_diagnostics,
        "figures": [
            {
                "name": fr.name,
                "html_path": str(fr.html_path),
                "png_path": str(fr.png_path),
                "png_ok": fr.png_ok,
                "png_error": fr.png_error,
            }
            for fr in figure_results
        ],
        "tables": {k: str(v) for k, v in table_paths.items()},
    }
    summary_json = _to_builtin(summary)
    summary_path = out_dir / "stylized_facts_summary.json"
    summary_json["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    return dict(summary_json)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point for stylized-facts diagnostics.

    Parameters
    ----------
    argv
        Optional argument list. If `None`, arguments are read from `sys.argv`.

    Returns
    -------
    int
        Process exit status (`0` on success).

    Notes
    -----
    - Figures are always saved as HTML.
    - PNG export is attempted when `kaleido` is installed and usable.

    Examples
    --------
    >>> # Use `latest_run.json` written by `python -m scripts.run ...` to locate the newest run:
    >>> main(["/path/to/abm_results/scenarios/<scenario>/runs/<run_id>/output_data/dex_price_end_of_block.npy"])  # doctest: +SKIP
    0
    """
    parser = argparse.ArgumentParser(description="Generate stylized-facts diagnostics for a price or returns series.")
    parser.add_argument("input_path", type=Path, help="Path to .npy input series file")
    parser.add_argument(
        "--input-kind",
        choices=["prices", "returns"],
        default="prices",
        help="Interpret input as prices or precomputed returns (default: prices)",
    )
    parser.add_argument(
        "--return-type",
        choices=["log", "simple"],
        default="log",
        help="Return type for analysis and aggregation (default: log)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: sibling stylized_facts_<input_stem>)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="Sampling horizons (default: 1 5 10 20)",
    )
    args = parser.parse_args(argv)

    summary = run_stylized_facts(
        args.input_path,
        horizons=args.horizons,
        input_kind=args.input_kind,
        return_type=args.return_type,
        out_dir=args.out_dir,
    )
    print(f"[stylized-facts] report: {summary['report_path']}")
    print(f"[stylized-facts] summary: {summary['summary_path']}")
    print(f"[stylized-facts] figures: {summary['figures_dir']}")
    print(f"[stylized-facts] tables: {summary['tables_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
