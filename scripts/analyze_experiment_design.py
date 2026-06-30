#!/usr/bin/env python3
"""Analyze an experiment cache: screening, Sobol indices, and summaries.

This script consumes the cached point summaries produced by
`scripts/run_experiment_design.py` and produces lightweight scientific artifacts:
- parameter screening via permutation importance (RandomForest surrogate),
- Sobol (Saltelli) first/total-order indices when the design is saltelli,
- top-point tables and convergence traces.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.artifacts import write_json


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


def _infer_param_order(dataframe: pd.DataFrame, meta: Optional[Mapping[str, Any]]) -> List[str]:
    if meta is not None:
        space = meta.get("space")
        if isinstance(space, list):
            names = []
            for entry in space:
                if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                    names.append(str(entry["name"]))
            if names:
                return names
    cols = []
    for c in dataframe.columns:
        if str(c).startswith("p__"):
            cols.append(str(c)[len("p__") :])
    if not cols:
        raise SystemExit("No parameter columns found (expected p__<param> columns).")
    return cols


def _direction_from_meta(meta: Optional[Mapping[str, Any]], default: str) -> str:
    if meta is not None:
        design = meta.get("design")
        if isinstance(design, dict):
            direction = design.get("direction")
            if isinstance(direction, str) and direction.lower() in ("maximize", "minimize"):
                return direction.lower()
    return default


def _dedup_last_by_point_id(df: pd.DataFrame) -> pd.DataFrame:
    if "point_id" not in df.columns:
        return df
    try:
        df = df.sort_values(by=["point_id"], kind="mergesort")
        df = df.drop_duplicates(subset=["point_id"], keep="last")
    except Exception:
        pass
    return df


def _sobol_saltelli_indices(
    y: np.ndarray,
    *,
    n_base: int,
    d: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Saltelli first-order and total-order indices (S1, ST).

    Parameters
    ----------
    y
        Output vector in Saltelli order: A (N), B (N), then AB_i blocks (each N).
    n_base
        Base sample size N.
    d
        Number of parameters.

    Returns
    -------
    (S1, ST)
        Arrays of shape (d,) for first-order and total-order indices.

    Notes
    -----
    Uses common Saltelli/Jansen-style estimators:
    - Var(Y) computed on concatenated [Y_A, Y_B] with `ddof=1`.
    - S1_i = mean( Y_B * (Y_ABi - Y_A) ) / Var(Y)
    - ST_i = 0.5 * mean( (Y_A - Y_ABi)^2 ) / Var(Y)
    """
    y = np.asarray(y, dtype=float)
    n = int(n_base)
    d = int(d)
    expected = n * (2 + d)
    if y.size < expected:
        raise ValueError(f"Expected at least {expected} outputs but got {y.size}.")

    yA = y[0:n]
    yB = y[n : 2 * n]
    var_y = float(np.var(np.concatenate([yA, yB]), ddof=1))
    if not np.isfinite(var_y) or var_y <= 0.0:
        raise ValueError(f"Invalid variance for Sobol indices: Var(Y)={var_y}.")

    s1 = np.zeros(d, dtype=float)
    st = np.zeros(d, dtype=float)
    base = 2 * n
    for i in range(d):
        yAB = y[base + i * n : base + (i + 1) * n]
        s1[i] = float(np.mean(yB * (yAB - yA)) / var_y)
        st[i] = float(0.5 * np.mean((yA - yAB) ** 2) / var_y)
    return s1, st


def _plot_importance(importances: pd.DataFrame, *, title: str, output_html: Path) -> None:
    fig = go.Figure(
        data=[
            go.Bar(
                x=importances["param"].tolist(),
                y=importances["importance_mean"].tolist(),
                error_y=dict(type="data", array=importances["importance_std"].tolist(), visible=True),
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Parameter",
        yaxis_title="Permutation importance (mean ± std)",
        template="plotly_white",
        margin=dict(l=50, r=30, t=60, b=80),
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")


def _plot_convergence(
    df: pd.DataFrame,
    *,
    metric: str,
    direction: str,
    title: str,
    output_html: Path,
) -> None:
    x = df["point_id"].astype(int).to_numpy()
    y = df[metric].astype(float).to_numpy()
    if direction == "minimize":
        best = np.minimum.accumulate(y)
    else:
        best = np.maximum.accumulate(y)
    fig = go.Figure(
        data=[
            go.Scatter(x=x, y=y, mode="markers", name=metric, marker=dict(size=6, opacity=0.7)),
            go.Scatter(x=x, y=best, mode="lines", name="best-so-far", line=dict(width=3)),
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="point_id",
        yaxis_title=metric,
        template="plotly_white",
        margin=dict(l=50, r=30, t=60, b=50),
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn")


def main() -> None:
    """Entry point for experiment cache analysis.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - Designed to be a fast, research-friendly post-processing step.
    - Does not re-run simulations; it only reads cached CSV/meta artifacts.

    Examples
    --------
    `python -m scripts.analyze_experiment_design --cache .../points_<tag>.csv --meta .../meta_<tag>.json --metric fee_mean`
    """
    parser = argparse.ArgumentParser(description="Analyze an experiment cache CSV.")
    parser.add_argument("--cache", type=Path, required=True, help="Cached points CSV (points_<tag>.csv).")
    parser.add_argument("--meta", type=Path, default=None, help="Optional meta JSON (meta_<tag>.json).")
    parser.add_argument("--metric", type=str, required=True, help="Metric column to analyze (e.g., fee_mean).")
    parser.add_argument("--direction", type=str, default="maximize", help="maximize|minimize (default: maximize).")
    parser.add_argument("--top-k", type=int, default=20, help="How many best points to export.")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory (default: <run_root>/analysis).")
    parser.add_argument("--random-state", type=int, default=1, help="Random seed for surrogate models.")
    parser.add_argument("--regime-threshold", type=float, default=None, help="Optional threshold for boundary candidates.")
    args = parser.parse_args()

    cache_path = args.cache.expanduser().resolve()
    if not cache_path.exists():
        raise SystemExit(f"Cache CSV not found: {cache_path}")
    df = pd.read_csv(cache_path)
    if df.empty:
        raise SystemExit(f"Cache CSV is empty: {cache_path}")

    meta = _load_meta(args.meta)
    param_order = _infer_param_order(df, meta)

    metric = str(args.metric)
    if metric not in df.columns:
        raise SystemExit(f"Metric column not found in cache: {metric}")

    direction = str(args.direction).strip().lower()
    direction = _direction_from_meta(meta, direction)
    if direction not in ("maximize", "minimize"):
        raise SystemExit("--direction must be maximize|minimize.")

    outdir = args.outdir
    if outdir is None:
        # Try to infer run root from the cache path structure: <run_root>/data/points_*.csv
        outdir = cache_path.parents[1] / "analysis" if cache_path.parent.name == "data" else cache_path.parent / "analysis"
    outdir = Path(outdir).expanduser().resolve()
    plots_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    df = _dedup_last_by_point_id(df)
    if "ok" in df.columns:
        df_ok = df[df["ok"] == True].copy()  # noqa: E712
    else:
        df_ok = df.copy()

    df_ok = df_ok[np.isfinite(df_ok[metric].astype(float))].copy()
    if df_ok.empty:
        raise SystemExit(f"No finite rows for metric {metric!r} (after filtering ok points).")

    # Feature matrix
    X_cols = [f"p__{p}" for p in param_order]
    missing = [c for c in X_cols if c not in df_ok.columns]
    if missing:
        raise SystemExit(f"Missing parameter columns in cache: {missing}")

    X = df_ok[X_cols].astype(float).to_numpy()
    y = df_ok[metric].astype(float).to_numpy()

    # Screening via permutation importance (RandomForest surrogate).
    # Use MAE scoring to avoid small-sample issues with R^2, and guard on very
    # small caches (importance is not meaningful with a handful of points).
    imp_path = outdir / f"importance_{metric}.csv"
    importance_note: Optional[str] = None
    if int(len(df_ok)) < 5:
        importance_note = "Not enough points for permutation importance (need >= 5 ok+finite points)."
        imp = pd.DataFrame(
            {
                "param": list(param_order),
                "importance_mean": [np.nan for _ in param_order],
                "importance_std": [np.nan for _ in param_order],
            }
        )
        imp.to_csv(imp_path, index=False)
    else:
        rf = RandomForestRegressor(
            n_estimators=300,
            random_state=int(args.random_state),
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2,
        )
        rf.fit(X, y)

        perm = permutation_importance(
            rf,
            X,
            y,
            n_repeats=10,
            random_state=int(args.random_state),
            n_jobs=-1,
            scoring="neg_mean_absolute_error",
        )
        imp = pd.DataFrame(
            {
                "param": list(param_order),
                "importance_mean": perm.importances_mean.tolist(),
                "importance_std": perm.importances_std.tolist(),
            }
        ).sort_values(by="importance_mean", ascending=False, kind="mergesort")
        imp.to_csv(imp_path, index=False)

    # Best points (table)
    df_rank = df_ok[["point_id", metric] + X_cols].copy()
    df_rank = df_rank.sort_values(by=metric, ascending=(direction == "minimize"), kind="mergesort")
    top_k = max(1, int(args.top_k))
    top = df_rank.head(top_k)
    top_path = outdir / f"top_points_{metric}.csv"
    top.to_csv(top_path, index=False)

    # Plots
    if importance_note is None:
        _plot_importance(
            imp, title=f"Permutation importance ({metric})", output_html=plots_dir / f"importance_{metric}.html"
        )
    if "point_id" in df_ok.columns:
        df_conv = df_ok.sort_values(by="point_id", kind="mergesort")[["point_id", metric]].copy()
        _plot_convergence(
            df_conv,
            metric=metric,
            direction=direction,
            title=f"Convergence trace ({metric}, {direction})",
            output_html=plots_dir / f"convergence_{metric}.html",
        )

    # Sobol indices (Saltelli) if available.
    sobol_path = None
    sobol_summary = None
    if meta is not None:
        design = meta.get("design")
        if isinstance(design, dict) and str(design.get("type", "")) == "sobol_saltelli":
            meta_design = design.get("meta")
            saltelli = meta_design.get("saltelli") if isinstance(meta_design, dict) else None
            if isinstance(saltelli, dict) and saltelli.get("n_base") and saltelli.get("d"):
                n_base = int(saltelli["n_base"])
                d = int(saltelli["d"])
                expected = n_base * (2 + d)

                # Build Y in point_id order (0..expected-1).
                df_s = df_ok.copy()
                if "point_id" not in df_s.columns:
                    df_s = df.copy()
                df_s = _dedup_last_by_point_id(df_s)
                df_s = df_s[np.isfinite(df_s[metric].astype(float))].copy()
                if "point_id" in df_s.columns:
                    df_s = df_s.sort_values(by="point_id", kind="mergesort")
                    y_map = {int(r["point_id"]): float(r[metric]) for _, r in df_s.iterrows()}
                    ys = np.array([y_map.get(i, np.nan) for i in range(expected)], dtype=float)
                    if np.isfinite(ys).all():
                        s1, st = _sobol_saltelli_indices(ys, n_base=n_base, d=d)
                        sob = pd.DataFrame({"param": list(param_order)[:d], "S1": s1.tolist(), "ST": st.tolist()})
                        sobol_path = outdir / f"sobol_{metric}.csv"
                        sob.to_csv(sobol_path, index=False)
                        sobol_summary = {"n_base": n_base, "d": d, "path": str(sobol_path)}

    # Regime boundary candidates (optional).
    regime_threshold = args.regime_threshold
    if regime_threshold is None and meta is not None:
        design = meta.get("design")
        if isinstance(design, dict) and design.get("regime_threshold") is not None:
            try:
                regime_threshold = float(design["regime_threshold"])
            except Exception:
                regime_threshold = None
    regime_path = None
    if regime_threshold is not None:
        df_reg = df_ok.copy()
        df_reg["abs_gap"] = np.abs(df_reg[metric].astype(float) - float(regime_threshold))
        df_reg = df_reg.sort_values(by="abs_gap", kind="mergesort").head(min(200, len(df_reg)))
        regime_path = outdir / f"regime_boundary_candidates_{metric}.csv"
        df_reg[["point_id", metric, "abs_gap"] + X_cols].to_csv(regime_path, index=False)

    summary = {
        "cache": str(cache_path),
        "meta": None if args.meta is None else str(Path(args.meta).expanduser().resolve()),
        "metric": metric,
        "direction": direction,
        "n_rows_total": int(len(df)),
        "n_rows_ok_finite": int(len(df_ok)),
        "importance_csv": str(imp_path),
        "importance_note": importance_note,
        "top_points_csv": str(top_path),
        "plots_dir": str(plots_dir),
        "sobol": sobol_summary,
        "regime_boundary_candidates_csv": None if regime_path is None else str(regime_path),
    }
    write_json(outdir / "summary.json", summary)

    print(f"[experiment_analyze] cache: {cache_path}")
    print(f"[experiment_analyze] outdir: {outdir}")
    print(f"[experiment_analyze] summary: {outdir / 'summary.json'}")
    if sobol_path is not None:
        print(f"[experiment_analyze] sobol: {sobol_path}")


if __name__ == "__main__":
    main()
