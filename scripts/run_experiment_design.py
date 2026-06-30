#!/usr/bin/env python3
"""Run a reproducible experiment design (grid/LHS/Sobol/Saltelli) and cache results.

This script generalizes the ND grid runner to support non-grid designs that are
more compute-efficient in high dimensions (e.g., Latin Hypercube, Sobol).

Outputs (under an experiment-scoped run folder):
- `data/points_<tag>.csv`: one row per evaluated point (restartable, append-only cache)
- `data/meta_<tag>.json`: provenance + design spec + cache schema info
- `data/errors_<tag>.csv`: failures (subset of cached rows with error details)
- `data/base_config_snapshot_<tag>.yml`: snapshot of the base scenario YAML
- `data/experiment_snapshot_<tag>.yml`: snapshot of the experiment YAML

Notes
-----
- The point list is deterministic given the experiment YAML and seeds.
- Worker simulations write to isolated temp folders under `tmp/_tmp_runs/<tag>/`
  and are cleaned automatically (unless `outputs.keep_worker_tmp=true`).
"""

from __future__ import annotations

import atexit
import argparse
import logging
import math
import shutil
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow `python scripts/run_experiment_design.py ...` to work from any CWD by ensuring
# the repo root (parent of `scripts/`) is on `sys.path` so `import core` succeeds.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import run as run_module
from core.utils import load_simulation_parameters
from core.artifacts import build_run_manifest, safe_tag, snapshot_file, write_json
from core.experiment_design import (
    DesignPoint,
    ExperimentSpec,
    ParameterSpec,
    experiment_yaml_content_hash,
    load_experiment_spec,
    map_unit_to_point,
    generate_design_points,
    stable_content_hash,
)


def _silent_tqdm(iterable=None, **kwargs):
    """Silence tqdm inside scripts.run.simulate to avoid nested progress bars."""
    if iterable is None:
        total = int(kwargs.get("total", 0))
        return range(total)
    return iterable


run_module.tqdm = _silent_tqdm  # type: ignore[attr-defined]
simulate = run_module.simulate


CACHE_SCHEMA_VERSION = 1
SCRIPT_VERSION = "experiment_design_runner_v1"


def _slice_series(values: Sequence[float], skip: int) -> np.ndarray:
    """Slice a numeric series after an integer burn-in."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    s0 = max(0, min(int(skip), int(arr.size)))
    return arr[s0:]


def _append_rows_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    """Append rows to a CSV cache file."""
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def _load_cache(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _required_metric_columns(*, pnl_metrics: Sequence[str], include_fee: bool, include_sr: bool) -> List[str]:
    cols: List[str] = []
    for key in pnl_metrics:
        cols.append(f"pnl_rate_p50_{key}")
    if include_fee:
        cols.extend(["fee_mean", "fee_median"])
    if include_sr:
        cols.extend(["smart_router_dex_share_mean", "smart_router_dex_share_median"])
    return cols


def _existing_status(
    dataframe: pd.DataFrame,
    *,
    required_cols: Sequence[str],
) -> Tuple[set[int], set[int]]:
    """Return sets (ok_point_ids, failed_point_ids) for an existing cache."""
    if dataframe.empty:
        return set(), set()
    if "point_id" not in dataframe.columns:
        return set(), set()
    if not set(required_cols).issubset(set(dataframe.columns)):
        return set(), set()

    df = dataframe
    try:
        df = df.drop_duplicates(subset=["point_id"], keep="last")
    except Exception:
        pass

    ok_ids: set[int] = set()
    failed_ids: set[int] = set()
    for _, row in df.iterrows():
        try:
            pid = int(row["point_id"])
        except Exception:
            continue
        is_ok = True
        if "ok" in df.columns:
            try:
                is_ok = is_ok and bool(row["ok"])
            except Exception:
                is_ok = False
        for col in required_cols:
            try:
                is_ok = is_ok and bool(np.isfinite(float(row[col])))
            except Exception:
                is_ok = False
        if is_ok:
            ok_ids.add(pid)
        else:
            failed_ids.add(pid)
    return ok_ids, failed_ids


def _make_worker_run_root(
    *,
    worker_temp_root: Path,
    point_id: int,
    run_index: int,
    seed_value: int,
) -> Path:
    """Create a unique temporary run directory for one worker simulation."""
    point_root = worker_temp_root / f"pt_{int(point_id):08d}"
    point_root.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f"run_{int(run_index):04d}_seed_{int(seed_value)}_",
            dir=str(point_root),
        )
    )


def _evaluate_point(
    point: DesignPoint,
    *,
    base_params: Dict[str, Any],
    pnl_metrics: Sequence[str],
    pnl_quantiles: Sequence[float],
    include_fee_hist: bool,
    fee_bins: np.ndarray,
    include_sr_dex_share_hist: bool,
    sr_bins: np.ndarray,
    runs_per_point: int,
    common_seeds: bool,
    global_seed_base: int,
    point_seed_base: int,
    worker_temp_root: str,
    compute_overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    """Worker evaluation for one design point (runs multiple seeds and aggregates)."""
    logging.getLogger("uniswapv3_pool").setLevel(logging.ERROR)
    worker_temp_root_path = Path(worker_temp_root)
    point_temp_root = worker_temp_root_path / f"pt_{int(point.point_id):08d}"

    params = dict(base_params)
    params.update(dict(point.values))
    params.update(dict(compute_overrides))

    skip_step = max(0, int(params.get("skip_step", 0)))

    ok = True
    error_type = ""
    error_message = ""
    error_run_index: Optional[int] = None
    error_seed: Optional[int] = None

    pnl_run_rates: Dict[str, List[float]] = {k: [] for k in pnl_metrics}

    fee_values: List[float] = []
    fee_hist = np.zeros(int(fee_bins.size - 1), dtype=np.int64)

    sr_values: List[float] = []
    sr_hist = np.zeros(int(sr_bins.size - 1), dtype=np.int64)

    try:
        for run_index in range(int(runs_per_point)):
            if common_seeds:
                seed_value = int(global_seed_base + run_index)
            else:
                seed_value = int(point_seed_base + run_index)
            params["seed"] = seed_value

            run_temp_root: Optional[Path] = None
            try:
                run_temp_root = _make_worker_run_root(
                    worker_temp_root=worker_temp_root_path,
                    point_id=point.point_id,
                    run_index=run_index,
                    seed_value=seed_value,
                )
                params["results_root"] = run_temp_root
                output = simulate(**params)
            except Exception as exc:
                ok = False
                error_type = type(exc).__name__
                error_message = str(exc)
                error_run_index = int(run_index)
                error_seed = int(seed_value)
                break
            finally:
                if run_temp_root is not None:
                    shutil.rmtree(run_temp_root, ignore_errors=True)

            # PnL rates
            for key in pnl_metrics:
                series = _slice_series(output.get(key, []), skip_step)
                if series.size < 2:
                    raise ValueError(f"Series {key!r} too short after skip_step={skip_step} (len={series.size}).")
                rate = float(np.mean(np.diff(series)))
                if not np.isfinite(rate):
                    raise ValueError(f"Non-finite per-step rate for {key!r} (seed={seed_value}).")
                pnl_run_rates[key].append(rate)

            # Fees
            if include_fee_hist:
                fee_series = _slice_series(output.get("fee_series", []), skip_step)
                if fee_series.size == 0:
                    raise ValueError("fee_series empty after applying skip_step.")
                fee_values.extend([float(v) for v in fee_series.tolist()])
                fee_hist += np.histogram(fee_series, bins=fee_bins)[0].astype(np.int64)

            # Smart-router DEX share
            if include_sr_dex_share_hist:
                sr_steps_raw = output.get("smart_router_dex_share_steps", [])
                sr_series_raw = output.get("smart_router_dex_share_series", [])
                try:
                    sr_steps = [int(v) for v in sr_steps_raw]
                except Exception:
                    sr_steps = []
                try:
                    sr_series = [float(v) for v in sr_series_raw]
                except Exception:
                    sr_series = []

                sr_vals_run: List[float] = []
                if sr_steps and len(sr_steps) == len(sr_series):
                    for step, ratio in zip(sr_steps, sr_series):
                        if int(step) < int(skip_step):
                            continue
                        if not np.isfinite(ratio):
                            continue
                        sr_vals_run.append(float(ratio))
                else:
                    for ratio in sr_series:
                        if not np.isfinite(ratio):
                            continue
                        sr_vals_run.append(float(ratio))

                if sr_vals_run:
                    arr = np.asarray(sr_vals_run, dtype=float)
                    sr_values.extend([float(v) for v in arr.tolist()])
                    sr_hist += np.histogram(arr, bins=sr_bins)[0].astype(np.int64)
    except Exception as exc:
        ok = False
        error_type = type(exc).__name__
        error_message = str(exc)
    finally:
        shutil.rmtree(point_temp_root, ignore_errors=True)

    row: Dict[str, Any] = {
        "point_id": int(point.point_id),
        "ok": bool(ok),
        "error_type": str(error_type),
        "error_message": str(error_message)[:500],
        "error_run_index": "" if error_run_index is None else int(error_run_index),
        "error_seed": "" if error_seed is None else int(error_seed),
        "runs_per_point": int(runs_per_point),
        "seed_base": int(point_seed_base),
    }

    # Parameter columns (always include, even on failures).
    for name, value in point.values.items():
        row[f"p__{name}"] = value
    for name, idx in point.indices.items():
        row[f"i__{name}"] = int(idx)
    role = point.meta.get("role")
    if isinstance(role, str) and role:
        row["design_role"] = role
    i_param = point.meta.get("i_param")
    if i_param is not None:
        try:
            row["design_i_param"] = int(i_param)
        except Exception:
            pass

    # Metric columns
    if ok:
        for key in pnl_metrics:
            rates = np.asarray(pnl_run_rates[key], dtype=float)
            for q in pnl_quantiles:
                q_name = f"p{int(round(100.0 * float(q)))}"
                row[f"pnl_rate_{q_name}_{key}"] = float(np.quantile(rates, float(q))) if rates.size else np.nan
            row[f"pnl_rate_p_loss_{key}"] = float(np.mean(rates < 0.0)) if rates.size else np.nan

        if include_fee_hist:
            fee_arr = np.asarray(fee_values, dtype=float)
            row["fee_mean"] = float(np.mean(fee_arr)) if fee_arr.size else np.nan
            row["fee_median"] = float(np.median(fee_arr)) if fee_arr.size else np.nan
            for b, count in enumerate(fee_hist.tolist()):
                row[f"fee_hist_{int(b)}"] = int(count)

        if include_sr_dex_share_hist:
            sr_arr = np.asarray(sr_values, dtype=float)
            row["smart_router_dex_share_mean"] = float(np.mean(sr_arr)) if sr_arr.size else np.nan
            row["smart_router_dex_share_median"] = float(np.median(sr_arr)) if sr_arr.size else np.nan
            for b, count in enumerate(sr_hist.tolist()):
                row[f"smart_router_dex_share_hist_{int(b)}"] = int(count)
    else:
        for key in pnl_metrics:
            for q in pnl_quantiles:
                q_name = f"p{int(round(100.0 * float(q)))}"
                row[f"pnl_rate_{q_name}_{key}"] = np.nan
            row[f"pnl_rate_p_loss_{key}"] = np.nan
        if include_fee_hist:
            row["fee_mean"] = np.nan
            row["fee_median"] = np.nan
            for b in range(int(fee_bins.size - 1)):
                row[f"fee_hist_{int(b)}"] = 0
        if include_sr_dex_share_hist:
            row["smart_router_dex_share_mean"] = np.nan
            row["smart_router_dex_share_median"] = np.nan
            for b in range(int(sr_bins.size - 1)):
                row[f"smart_router_dex_share_hist_{int(b)}"] = 0

    return row


def _canon_fingerprint_payload(
    *,
    experiment_content_hash: str,
    config_content_hash: str,
    script_version: str,
    cache_schema_version: int,
) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "experiment_content_hash": str(experiment_content_hash),
        "config_content_hash": str(config_content_hash),
        "script_version": str(script_version),
        "cache_schema_version": int(cache_schema_version),
    }
    fingerprint = stable_content_hash(payload, n_hex=12)
    return fingerprint, payload


def _expected_metric_columns(
    *,
    pnl_metrics: Sequence[str],
    pnl_quantiles: Sequence[float],
    include_fee_hist: bool,
    fee_hist_bins: int,
    include_sr_dex_share_hist: bool,
    sr_hist_bins: int,
) -> set[str]:
    cols: set[str] = set()
    for key in pnl_metrics:
        for q in pnl_quantiles:
            q_name = f"p{int(round(100.0 * float(q)))}"
            cols.add(f"pnl_rate_{q_name}_{key}")
        cols.add(f"pnl_rate_p_loss_{key}")
    if include_fee_hist:
        cols.update(["fee_mean", "fee_median"])
        for b in range(int(fee_hist_bins)):
            cols.add(f"fee_hist_{b}")
    if include_sr_dex_share_hist:
        cols.update(["smart_router_dex_share_mean", "smart_router_dex_share_median"])
        for b in range(int(sr_hist_bins)):
            cols.add(f"smart_router_dex_share_hist_{b}")
    return cols


def _point_key(*, space: Sequence[ParameterSpec], values: Mapping[str, Any], indices: Mapping[str, Any]) -> Tuple[Any, ...]:
    parts: List[Any] = []
    for spec in space:
        name = spec.name
        if spec.kind == "continuous":
            v = values.get(name)
            parts.append(("c", name, round(float(v), 12)))
        else:
            idx = indices.get(name)
            if idx is None:
                raw = values.get(name)
                if spec.values is None:
                    raise ValueError(f"Missing values list for discrete parameter {name!r}.")
                # Best-effort index recovery (only used for uniqueness checks).
                vals = np.asarray([float(x) for x in spec.values], dtype=float)
                idx = int(np.argmin(np.abs(vals - float(raw))))
            parts.append(("d", name, int(idx)))
    return tuple(parts)


def _unit_features_from_row(row: Mapping[str, Any], *, space: Sequence[ParameterSpec]) -> np.ndarray:
    feats: List[float] = []
    for spec in space:
        name = spec.name
        if spec.kind == "continuous":
            if spec.bounds is None:
                raise ValueError(f"Continuous parameter {name!r} missing bounds.")
            lo, hi = float(spec.bounds[0]), float(spec.bounds[1])
            v = float(row[f"p__{name}"])
            if spec.transform == "log":
                u = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
            else:
                u = (v - lo) / (hi - lo)
            feats.append(float(min(1.0, max(0.0, u))))
        else:
            k = 1 if spec.values is None else int(len(spec.values))
            idx_col = f"i__{name}"
            if idx_col in row:
                try:
                    idx = int(row[idx_col])
                except Exception:
                    idx = 0
            else:
                idx = 0
            denom = max(1, k - 1)
            feats.append(float(min(1.0, max(0.0, idx / denom))))
    return np.asarray(feats, dtype=float)


def _load_sequential_state(
    *,
    csv_cache: Path,
    space: Sequence[ParameterSpec],
    target_metric: str,
) -> Tuple[int, set[Tuple[Any, ...]], List[np.ndarray], List[float]]:
    """Load cached state for sequential designs (next_id, existing_keys, X_train, y_train)."""
    cached = _load_cache(csv_cache)
    if cached.empty or "point_id" not in cached.columns:
        return 0, set(), [], []

    try:
        cached = cached.drop_duplicates(subset=["point_id"], keep="last")
    except Exception:
        pass
    try:
        cached = cached.sort_values(by=["point_id"], kind="mergesort")
    except Exception:
        pass

    point_ids = []
    for v in cached["point_id"].tolist():
        try:
            point_ids.append(int(v))
        except Exception:
            continue
    next_id = 0 if not point_ids else int(max(point_ids) + 1)

    # Detect gaps (resume is ambiguous if point_id is not dense from 0..max).
    if point_ids:
        max_id = int(max(point_ids))
        present = set(point_ids)
        missing = [i for i in range(max_id + 1) if i not in present]
        if missing:
            raise SystemExit(
                "Sequential design cache has missing point_ids (gaps). "
                "This usually indicates manual slicing or partial file corruption. "
                "Fix the cache or re-run with --recompute."
            )

    existing_keys: set[Tuple[Any, ...]] = set()
    X_train: List[np.ndarray] = []
    y_train: List[float] = []

    for _, row in cached.iterrows():
        row_map = row.to_dict()
        values = {spec.name: row_map.get(f"p__{spec.name}") for spec in space}
        idxs = {spec.name: row_map.get(f"i__{spec.name}") for spec in space if spec.kind == "discrete"}
        try:
            existing_keys.add(_point_key(space=space, values=values, indices=idxs))
        except Exception:
            pass

        ok = True
        if "ok" in cached.columns:
            try:
                ok = bool(row_map.get("ok", True))
            except Exception:
                ok = False
        if not ok:
            continue
        try:
            yv = float(row_map.get(target_metric))
        except Exception:
            continue
        if not np.isfinite(yv):
            continue
        try:
            X_train.append(_unit_features_from_row(row_map, space=space))
            y_train.append(float(yv))
        except Exception:
            continue

    return next_id, existing_keys, X_train, y_train


def _run_sequential_design(
    *,
    spec: ExperimentSpec,
    base_params: Dict[str, Any],
    csv_cache: Path,
    errors_path: Path,
    worker_tmp_root: Path,
    fee_bin_edges: np.ndarray,
    sr_bin_edges: np.ndarray,
    max_workers: int,
    compute_overrides: Mapping[str, Any],
    target_metric: str,
    direction: str,
    total_points: int,
    failed_ids: set[int],
    retry_failed: bool,
    recompute: bool,
) -> Tuple[int, int, int]:
    """Run an adaptive_refine or bayesopt design sequentially (batch-evaluated in parallel)."""
    # Lazy imports: these are only needed for sequential designs.
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from scipy.stats import norm, qmc

    space = list(spec.space)
    d = int(len(space))
    n_total = int(total_points)
    if n_total <= 0:
        return 0, 0, 0

    design_seed = int(spec.seed_base if spec.design.seed is None else spec.design.seed)
    n_init_default = min(n_total, max(20, 4 * d))
    n_init = int(n_init_default if spec.design.n_init is None else spec.design.n_init)
    n_init = max(1, min(n_total, n_init))

    if spec.design.type == "bayesopt":
        batch_default = 1
    else:
        batch_default = min(64, max(8, 2 * d))
    batch_size = int(batch_default if spec.design.batch_size is None else spec.design.batch_size)
    batch_size = max(1, min(n_total, batch_size))

    n_candidates = 6000
    if d >= 12:
        n_candidates = 12000

    if recompute:
        next_id, existing_keys, X_train, y_train = 0, set(), [], []
    else:
        # Load cache-derived state for resume.
        next_id, existing_keys, X_train, y_train = _load_sequential_state(
            csv_cache=csv_cache,
            space=space,
            target_metric=target_metric,
        )

    ran = 0
    failed = 0

    # Optional: retry failed point_ids by reusing the last cached parameters.
    if (not recompute) and retry_failed and failed_ids:
        cached = _load_cache(csv_cache)
        if not cached.empty and "point_id" in cached.columns:
            try:
                cached = cached.drop_duplicates(subset=["point_id"], keep="last")
            except Exception:
                pass
            retry_rows = cached[cached["point_id"].isin(list(failed_ids))].copy()
            if not retry_rows.empty:
                points_retry: List[DesignPoint] = []
                for _, r in retry_rows.iterrows():
                    row_map = r.to_dict()
                    pid = int(row_map.get("point_id", -1))
                    vals = {spec_p.name: row_map.get(f"p__{spec_p.name}") for spec_p in space}
                    idxs = {spec_p.name: row_map.get(f"i__{spec_p.name}") for spec_p in space if spec_p.kind == "discrete"}
                    points_retry.append(
                        DesignPoint(point_id=pid, values=vals, indices=idxs, meta={"phase": "retry_failed"})
                    )

                if points_retry:
                    with ProcessPoolExecutor(max_workers=int(max_workers)) as executor:
                        futures = []
                        for pt in points_retry:
                            pid = int(pt.point_id)
                            point_seed_base = int(spec.seed_base + pid * int(spec.runs_per_point))
                            futures.append(
                                executor.submit(
                                    _evaluate_point,
                                    pt,
                                    base_params=base_params,
                                    pnl_metrics=list(spec.metrics.pnl_metrics),
                                    pnl_quantiles=list(spec.metrics.pnl_quantiles),
                                    include_fee_hist=bool(spec.metrics.include_fee_hist),
                                    fee_bins=fee_bin_edges,
                                    include_sr_dex_share_hist=bool(spec.metrics.include_sr_dex_share_hist),
                                    sr_bins=sr_bin_edges,
                                    runs_per_point=int(spec.runs_per_point),
                                    common_seeds=bool(spec.common_seeds),
                                    global_seed_base=int(spec.seed_base),
                                    point_seed_base=point_seed_base,
                                    worker_temp_root=str(worker_tmp_root),
                                    compute_overrides=compute_overrides,
                                )
                            )
                        rows = [f.result() for f in futures]
                    _append_rows_csv(csv_cache, rows)
                    error_rows = []
                    for row in rows:
                        ran += 1
                        if not bool(row.get("ok", True)):
                            failed += 1
                            error_rows.append(
                                {
                                    "point_id": int(row.get("point_id", -1)),
                                    "seed_base": int(row.get("seed_base", 0)),
                                    "error_type": str(row.get("error_type", "")),
                                    "error_message": str(row.get("error_message", ""))[:500],
                                    "error_run_index": row.get("error_run_index", ""),
                                    "error_seed": row.get("error_seed", ""),
                                }
                            )
                    if error_rows:
                        _append_rows_csv(errors_path, error_rows)

                    # Refresh training data after retries.
                    next_id, existing_keys, X_train, y_train = _load_sequential_state(
                        csv_cache=csv_cache,
                        space=space,
                        target_metric=target_metric,
                    )

    # Initial LHS points (deterministic).
    u_init = qmc.LatinHypercube(d=d, seed=int(design_seed)).random(n=int(n_init))

    start_id = int(next_id)
    progress = (
        tqdm(total=max(0, n_total - next_id), desc="Points (sequential)", unit="pt") if next_id < n_total else None
    )

    def _update_training_from_rows(rows: Sequence[Mapping[str, Any]]) -> None:
        nonlocal existing_keys, X_train, y_train, failed
        for row in rows:
            if not bool(row.get("ok", True)):
                continue
            try:
                yv = float(row.get(target_metric))
            except Exception:
                continue
            if not np.isfinite(yv):
                continue
            try:
                X_train.append(_unit_features_from_row(row, space=space))
                y_train.append(float(yv))
            except Exception:
                continue
            # Track keys to avoid duplicate proposals.
            vals = {spec_p.name: row.get(f"p__{spec_p.name}") for spec_p in space}
            idxs = {spec_p.name: row.get(f"i__{spec_p.name}") for spec_p in space if spec_p.kind == "discrete"}
            try:
                existing_keys.add(_point_key(space=space, values=vals, indices=idxs))
            except Exception:
                pass

    def _eval_points(points_batch: Sequence[DesignPoint]) -> None:
        nonlocal ran, failed
        if not points_batch:
            return
        pending_rows: List[Dict[str, Any]] = []
        pending_error_rows: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=int(max_workers)) as executor:
            futures = []
            for pt in points_batch:
                pid = int(pt.point_id)
                point_seed_base = int(spec.seed_base + pid * int(spec.runs_per_point))
                futures.append(
                    executor.submit(
                        _evaluate_point,
                        pt,
                        base_params=base_params,
                        pnl_metrics=list(spec.metrics.pnl_metrics),
                        pnl_quantiles=list(spec.metrics.pnl_quantiles),
                        include_fee_hist=bool(spec.metrics.include_fee_hist),
                        fee_bins=fee_bin_edges,
                        include_sr_dex_share_hist=bool(spec.metrics.include_sr_dex_share_hist),
                        sr_bins=sr_bin_edges,
                        runs_per_point=int(spec.runs_per_point),
                        common_seeds=bool(spec.common_seeds),
                        global_seed_base=int(spec.seed_base),
                        point_seed_base=point_seed_base,
                        worker_temp_root=str(worker_tmp_root),
                        compute_overrides=compute_overrides,
                    )
                )
            for fut in futures:
                row = fut.result()
                pending_rows.append(row)
                ran += 1
                if not bool(row.get("ok", True)):
                    failed += 1
                    pending_error_rows.append(
                        {
                            "point_id": int(row.get("point_id", -1)),
                            "seed_base": int(row.get("seed_base", 0)),
                            "error_type": str(row.get("error_type", "")),
                            "error_message": str(row.get("error_message", ""))[:500],
                            "error_run_index": row.get("error_run_index", ""),
                            "error_seed": row.get("error_seed", ""),
                        }
                    )
        _append_rows_csv(csv_cache, pending_rows)
        if pending_error_rows:
            _append_rows_csv(errors_path, pending_error_rows)
        _update_training_from_rows(pending_rows)
        if progress is not None:
            progress.update(len(points_batch))

    # Evaluate remaining points.
    current_id = int(next_id)
    while current_id < n_total:
        if current_id < n_init:
            # Evaluate remaining initial LHS points (in batches).
            batch_end = min(n_init, n_total, current_id + batch_size)
            pts = []
            for pid in range(int(current_id), int(batch_end)):
                vals, idxs = map_unit_to_point(space, u_init[int(pid), :].tolist())
                pts.append(DesignPoint(point_id=int(pid), values=vals, indices=idxs, meta={"phase": "init_lhs"}))
            _eval_points(pts)
            current_id = int(batch_end)
            continue

        # Fit surrogate and propose candidates.
        pts_next: List[DesignPoint] = []
        rng = np.random.default_rng(int(design_seed + 100000 + current_id))
        u_cand = rng.random((int(n_candidates), int(d)))

        have_training = len(X_train) >= max(10, d + 1)
        scores = None
        if have_training:
            X_tr = np.vstack(X_train).astype(float)
            y_tr = np.asarray(y_train, dtype=float)

            if spec.design.type == "adaptive_refine" and spec.design.regime_threshold is not None:
                thr = float(spec.design.regime_threshold)
                y_lab = (y_tr >= thr).astype(int)
                clf = RandomForestClassifier(
                    n_estimators=400,
                    random_state=int(design_seed),
                    n_jobs=-1,
                    min_samples_leaf=2,
                )
                clf.fit(X_tr, y_lab)
                p = clf.predict_proba(u_cand)[:, 1]
                scores = -np.abs(p - 0.5)
            elif spec.design.type == "adaptive_refine":
                rf = RandomForestRegressor(
                    n_estimators=500,
                    random_state=int(design_seed),
                    n_jobs=-1,
                    min_samples_leaf=2,
                )
                rf.fit(X_tr, y_tr)
                # Uncertainty proxy: std across trees.
                preds = np.stack([t.predict(u_cand) for t in rf.estimators_], axis=0)
                scores = np.std(preds, axis=0)
            else:
                # Bayesian optimization via GP + Expected Improvement.
                kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2))
                kernel += WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-1))
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-8,
                    normalize_y=True,
                    random_state=int(design_seed),
                )
                gp.fit(X_tr, y_tr)
                mu, sigma = gp.predict(u_cand, return_std=True)
                sigma = np.maximum(sigma, 1e-12)
                if direction == "minimize":
                    best = float(np.min(y_tr))
                    imp = best - mu
                else:
                    best = float(np.max(y_tr))
                    imp = mu - best
                z = imp / sigma
                scores = imp * norm.cdf(z) + sigma * norm.pdf(z)

        if scores is None:
            # Cold start fallback: random proposal.
            scores = rng.random(int(u_cand.shape[0]))

        order = np.argsort(-scores, kind="mergesort")
        for j in order.tolist():
            if len(pts_next) >= min(batch_size, n_total - current_id):
                break
            u_row = u_cand[int(j), :].tolist()
            vals, idxs = map_unit_to_point(space, u_row)
            key = _point_key(space=space, values=vals, indices=idxs)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            pts_next.append(DesignPoint(point_id=int(current_id + len(pts_next)), values=vals, indices=idxs, meta={"phase": "proposed"}))

        if not pts_next:
            # Extremely unlikely (all candidates duplicates); expand candidate pool once.
            rng2 = np.random.default_rng(int(design_seed + 200000 + current_id))
            u2 = rng2.random((int(n_candidates * 2), int(d)))
            for u_row in u2.tolist():
                if len(pts_next) >= min(batch_size, n_total - current_id):
                    break
                vals, idxs = map_unit_to_point(space, u_row)
                key = _point_key(space=space, values=vals, indices=idxs)
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                pts_next.append(
                    DesignPoint(
                        point_id=int(current_id + len(pts_next)),
                        values=vals,
                        indices=idxs,
                        meta={"phase": "proposed"},
                    )
                )

        _eval_points(pts_next)
        current_id += int(len(pts_next))

    if progress is not None:
        progress.close()

    return ran, failed, start_id


def main() -> None:
    """Entry point for running an experiment design cache.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - Reads an experiment YAML, generates a deterministic point list, runs the simulator
      for each point (multiple seeds), and appends a row to a cache CSV.
    - Use `--index-start/--index-stop` to run a slice (useful for distributed runs).

    Examples
    --------
    - Preview the design:
      `python -m scripts.run_experiment_design --experiment abm_results/experiments/example.yml --dry-run`
    """
    parser = argparse.ArgumentParser(description="Run an experiment design and cache point summaries to CSV.")
    parser.add_argument("--experiment", type=Path, required=True, help="Experiment YAML file.")
    parser.add_argument("--max-workers", type=int, default=None, help="Override compute.max_workers from YAML.")
    parser.add_argument("--index-start", type=int, default=0, help="Point index start (inclusive).")
    parser.add_argument("--index-stop", type=int, default=None, help="Point index stop (exclusive).")
    parser.add_argument("--retry-failed", action="store_true", help="Re-run failed points in the cache.")
    parser.add_argument("--recompute", action="store_true", help="Ignore cache and recompute all points.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved design and exit.")
    args = parser.parse_args()

    experiment_path = args.experiment.expanduser().resolve()
    spec = load_experiment_spec(experiment_path)

    # Allow a CLI override for max_workers without mutating the spec object.
    max_workers = int(spec.max_workers if args.max_workers is None else args.max_workers)

    scenario_label, loaded_params = load_simulation_parameters(spec.base_config, simulate_func=simulate)
    base_params = dict(loaded_params)

    config_content_hash = stable_content_hash(
        {"scenario_label": str(scenario_label), "simulate_params": base_params},
        n_hex=16,
    )
    experiment_content_hash = experiment_yaml_content_hash(experiment_path)
    fingerprint, fingerprint_payload = _canon_fingerprint_payload(
        experiment_content_hash=experiment_content_hash,
        config_content_hash=config_content_hash,
        script_version=SCRIPT_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION,
    )

    tag = f"{safe_tag(spec.outputs_tag)}_{fingerprint}"
    run_root = spec.outputs_root / tag
    data_root = run_root / "data"
    csv_cache = data_root / f"points_{tag}.csv"
    meta_path = data_root / f"meta_{tag}.json"
    errors_path = data_root / f"errors_{tag}.csv"
    worker_tmp_root = run_root / "tmp" / "_tmp_runs" / tag

    points: List[DesignPoint]
    design_meta: Mapping[str, Any]
    if spec.design.type in ("grid", "lhs", "sobol", "sobol_saltelli"):
        points, design_meta = generate_design_points(space=list(spec.space), design=spec.design, seed_base=spec.seed_base)
        total_points = int(len(points))
    else:
        if spec.design.n_points is None or int(spec.design.n_points) <= 0:
            raise SystemExit(f"design.type={spec.design.type!r} requires design.n_points > 0.")
        points = []
        design_meta = {
            "type": str(spec.design.type),
            "seed": int(spec.seed_base if spec.design.seed is None else spec.design.seed),
            "sequential": True,
        }
        total_points = int(spec.design.n_points)

    index_start = max(0, int(args.index_start))
    index_stop = int(total_points) if args.index_stop is None else int(args.index_stop)
    index_stop = max(index_start, min(int(total_points), int(index_stop)))
    slice_total = int(index_stop - index_start)
    if spec.design.type in ("adaptive_refine", "bayesopt") and (index_start != 0 or index_stop != total_points):
        raise SystemExit("Sequential designs (adaptive_refine/bayesopt) do not support --index-start/--index-stop slicing.")

    # Histogram bin edges (from base config)
    fee_hist_bins = int(spec.metrics.fee_hist_bins)
    sr_hist_bins = int(spec.metrics.sr_dex_share_hist_bins)

    f_min = float(base_params.get("f_min", np.nan))
    f_max = float(base_params.get("f_max", np.nan))
    if spec.metrics.include_fee_hist:
        if not (np.isfinite(f_min) and np.isfinite(f_max) and f_max > f_min):
            raise SystemExit(f"Invalid fee bounds from base_config: f_min={f_min}, f_max={f_max}")
        fee_bin_edges = np.linspace(f_min, f_max, fee_hist_bins + 1, dtype=float)
    else:
        fee_bin_edges = np.array([0.0, 1.0], dtype=float)
    sr_bin_edges = np.linspace(0.0, 1.0, sr_hist_bins + 1, dtype=float)

    if args.dry_run:
        print("Resolved experiment design:")
        print(f"  experiment: {experiment_path}")
        print(f"  base_config: {spec.base_config}")
        print(f"  scenario_label: {scenario_label}")
        print(f"  design: {spec.design.type} (total points: {total_points})")
        if spec.design.type in ("adaptive_refine", "bayesopt"):
            print(f"  target_metric: {spec.design.target_metric}")
            print(f"  direction: {spec.design.direction}")
        if slice_total != total_points:
            print(f"  point slice: [{index_start}, {index_stop}) => {slice_total} points")
        print(f"  cache: {csv_cache}")
        print(f"  meta:  {meta_path}")
        print(f"  worker tmp root: {worker_tmp_root} (created + cleaned at runtime)")
        print(f"  experiment_content_hash: {experiment_content_hash}")
        print(f"  config_content_hash: {config_content_hash}")
        return

    # --- metadata snapshots --------------------------------------------------
    data_root.mkdir(parents=True, exist_ok=True)
    config_snapshot = data_root / f"base_config_snapshot_{tag}.yml"
    experiment_snapshot = data_root / f"experiment_snapshot_{tag}.yml"
    snapshot_file(spec.base_config, config_snapshot)
    snapshot_file(experiment_path, experiment_snapshot)

    manifest = build_run_manifest(script="run_experiment_design", run_id=tag, config_path=spec.base_config)

    # --- cache / resume ------------------------------------------------------
    required_cols = _required_metric_columns(
        pnl_metrics=spec.metrics.pnl_metrics,
        include_fee=bool(spec.metrics.include_fee_hist),
        include_sr=bool(spec.metrics.include_sr_dex_share_hist),
    )
    cached = _load_cache(csv_cache)
    ok_cached, failed_cached = _existing_status(cached, required_cols=required_cols)
    if args.recompute:
        existing: set[int] = set()
    else:
        existing = set(ok_cached)
        if not args.retry_failed:
            existing |= set(failed_cached)

    print(
        f"[experiment] name={spec.name} | design={spec.design.type} | points={total_points} | "
        f"slice=[{index_start},{index_stop}) ({slice_total}) | workers={max_workers}"
    )
    print(f"[experiment] cache: {csv_cache}")
    print(f"[experiment] run_root: {run_root}")

    meta_payload = {
        "tag": tag,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "cache_schema_version": int(CACHE_SCHEMA_VERSION),
        "script_version": str(SCRIPT_VERSION),
        "experiment_path": str(experiment_path),
        "experiment_snapshot": str(experiment_snapshot),
        "experiment_content_hash": str(experiment_content_hash),
        "base_config_path": str(spec.base_config),
        "base_config_snapshot": str(config_snapshot),
        "config_content_hash": str(config_content_hash),
        "scenario_label": str(scenario_label),
        "design": _to_jsonable_design(spec, design_meta),
        "space": [
            {
                "name": s.name,
                "kind": s.kind,
                "bounds": None if s.bounds is None else [float(s.bounds[0]), float(s.bounds[1])],
                "values": None if s.values is None else list(s.values),
                "transform": s.transform,
                "cast": s.cast,
            }
            for s in spec.space
        ],
        "seed": {
            "seed_base": int(spec.seed_base),
            "common_seeds": bool(spec.common_seeds),
            "runs_per_point": int(spec.runs_per_point),
        },
        "compute": {
            "max_workers": int(max_workers),
            "light_mode": bool(spec.light_mode),
            "visualize": bool(spec.visualize),
            "verbose": bool(spec.verbose),
        },
        "metrics": {
            "pnl_metrics": list(spec.metrics.pnl_metrics),
            "pnl_quantiles": [float(q) for q in spec.metrics.pnl_quantiles],
            "include_fee_hist": bool(spec.metrics.include_fee_hist),
            "fee_hist_bins": int(fee_hist_bins),
            "fee_bin_edges": [float(v) for v in fee_bin_edges.tolist()],
            "include_sr_dex_share_hist": bool(spec.metrics.include_sr_dex_share_hist),
            "sr_dex_share_hist_bins": int(sr_hist_bins),
            "sr_dex_share_bin_edges": [float(v) for v in sr_bin_edges.tolist()],
        },
        "created_at_utc": manifest.created_at_utc,
        "git_commit": manifest.git_commit,
        "python": manifest.python,
        "platform": manifest.platform,
        "worker_temp_root": str(worker_tmp_root),
    }
    write_json(meta_path, meta_payload)

    # --- run simulations -----------------------------------------------------
    worker_tmp_root.mkdir(parents=True, exist_ok=True)
    atexit.register(shutil.rmtree, worker_tmp_root, True)

    compute_overrides = {
        "visualize": bool(spec.visualize),
        "verbose": bool(spec.verbose),
        "light_mode": bool(spec.light_mode),
        # Avoid heavy outputs for sweeps by default (portable across configs).
        "liquidity_for_gif": False,
    }

    pending_rows: List[Dict[str, Any]] = []
    pending_error_rows: List[Dict[str, Any]] = []

    progress: Optional[tqdm] = None
    if spec.design.type not in ("adaptive_refine", "bayesopt") and slice_total > 0:
        progress = tqdm(total=int(slice_total), desc="Points", unit="pt")
    processed = 0
    cached_in_slice = 0
    run_in_slice = 0
    failed_in_slice = 0

    if spec.design.type in ("adaptive_refine", "bayesopt"):
        target_metric = str(spec.design.target_metric or "").strip()
        if not target_metric:
            raise SystemExit(f"design.type={spec.design.type!r} requires design.target_metric.")
        expected_cols = _expected_metric_columns(
            pnl_metrics=list(spec.metrics.pnl_metrics),
            pnl_quantiles=list(spec.metrics.pnl_quantiles),
            include_fee_hist=bool(spec.metrics.include_fee_hist),
            fee_hist_bins=int(fee_hist_bins),
            include_sr_dex_share_hist=bool(spec.metrics.include_sr_dex_share_hist),
            sr_hist_bins=int(sr_hist_bins),
        )
        if target_metric not in expected_cols:
            preview = sorted(expected_cols)
            preview_txt = ", ".join(preview[:20]) + (" ..." if len(preview) > 20 else "")
            raise SystemExit(
                f"design.target_metric={target_metric!r} is not produced by this experiment's metrics configuration. "
                f"Expected one of: {preview_txt}"
            )
        direction = str(spec.design.direction).strip().lower()
        ran, failed, start_id = _run_sequential_design(
            spec=spec,
            base_params=base_params,
            csv_cache=csv_cache,
            errors_path=errors_path,
            worker_tmp_root=worker_tmp_root,
            fee_bin_edges=fee_bin_edges,
            sr_bin_edges=sr_bin_edges,
            max_workers=int(max_workers),
            compute_overrides=compute_overrides,
            target_metric=target_metric,
            direction=direction,
            total_points=int(total_points),
            failed_ids=set(failed_cached),
            retry_failed=bool(args.retry_failed),
            recompute=bool(args.recompute),
        )
        run_in_slice = int(ran)
        failed_in_slice = int(failed)
        cached_in_slice = 0 if args.recompute else int(start_id)
        processed = int(slice_total)
    elif slice_total > 0:
        max_pending = max(1, int(max_workers) * 4)
        with ProcessPoolExecutor(max_workers=int(max_workers)) as executor:
            pending: set = set()
            for point in points[index_start:index_stop]:
                pid = int(point.point_id)
                if not args.recompute and pid in existing:
                    cached_in_slice += 1
                    if progress is not None:
                        progress.update(1)
                    processed += 1
                    continue

                point_seed_base = int(spec.seed_base + pid * int(spec.runs_per_point))
                future = executor.submit(
                    _evaluate_point,
                    point,
                    base_params=base_params,
                    pnl_metrics=list(spec.metrics.pnl_metrics),
                    pnl_quantiles=list(spec.metrics.pnl_quantiles),
                    include_fee_hist=bool(spec.metrics.include_fee_hist),
                    fee_bins=fee_bin_edges,
                    include_sr_dex_share_hist=bool(spec.metrics.include_sr_dex_share_hist),
                    sr_bins=sr_bin_edges,
                    runs_per_point=int(spec.runs_per_point),
                    common_seeds=bool(spec.common_seeds),
                    global_seed_base=int(spec.seed_base),
                    point_seed_base=point_seed_base,
                    worker_temp_root=str(worker_tmp_root),
                    compute_overrides=compute_overrides,
                )
                pending.add(future)
                run_in_slice += 1

                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        row = fut.result()
                        pending_rows.append(row)
                        if not bool(row.get("ok", True)):
                            failed_in_slice += 1
                            pending_error_rows.append(
                                {
                                    "point_id": int(row.get("point_id", -1)),
                                    "seed_base": int(row.get("seed_base", 0)),
                                    "error_type": str(row.get("error_type", "")),
                                    "error_message": str(row.get("error_message", ""))[:500],
                                    "error_run_index": row.get("error_run_index", ""),
                                    "error_seed": row.get("error_seed", ""),
                                }
                            )
                        if progress is not None:
                            progress.update(1)
                        processed += 1

                        if len(pending_rows) >= 25:
                            _append_rows_csv(csv_cache, pending_rows)
                            pending_rows.clear()
                        if len(pending_error_rows) >= 10:
                            _append_rows_csv(errors_path, pending_error_rows)
                            pending_error_rows.clear()

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    row = fut.result()
                    pending_rows.append(row)
                    if not bool(row.get("ok", True)):
                        failed_in_slice += 1
                        pending_error_rows.append(
                            {
                                "point_id": int(row.get("point_id", -1)),
                                "seed_base": int(row.get("seed_base", 0)),
                                "error_type": str(row.get("error_type", "")),
                                "error_message": str(row.get("error_message", ""))[:500],
                                "error_run_index": row.get("error_run_index", ""),
                                "error_seed": row.get("error_seed", ""),
                            }
                        )
                    if progress is not None:
                        progress.update(1)
                    processed += 1

                    if len(pending_rows) >= 25:
                        _append_rows_csv(csv_cache, pending_rows)
                        pending_rows.clear()
                    if len(pending_error_rows) >= 10:
                        _append_rows_csv(errors_path, pending_error_rows)
                        pending_error_rows.clear()

        if pending_rows:
            _append_rows_csv(csv_cache, pending_rows)
        if pending_error_rows:
            _append_rows_csv(errors_path, pending_error_rows)

    if progress is not None:
        progress.close()

    if not spec.keep_worker_tmp:
        shutil.rmtree(worker_tmp_root, ignore_errors=True)

    print(
        f"[experiment] slice summary: ran={run_in_slice}, cached={cached_in_slice}, failed={failed_in_slice}, total={slice_total}"
    )
    print(f"[experiment] cache complete: {csv_cache}")
    print(
        "[experiment] build dashboard with:\n"
        f"  python -m scripts.build_experiment_design_dashboard --cache {csv_cache} --meta {meta_path}"
    )
    print(
        "[experiment] analyze with:\n"
        f"  python -m scripts.analyze_experiment_design --cache {csv_cache} --meta {meta_path} --metric fee_mean"
    )


def _to_jsonable_design(spec: ExperimentSpec, design_meta: Mapping[str, Any]) -> Mapping[str, Any]:
    """Build a JSON-serializable design section for meta payload."""
    out: Dict[str, Any] = {
        "type": str(spec.design.type),
        "n_points": None if spec.design.n_points is None else int(spec.design.n_points),
        "n_base": None if spec.design.n_base is None else int(spec.design.n_base),
        "seed": None if spec.design.seed is None else int(spec.design.seed),
        "n_init": None if spec.design.n_init is None else int(spec.design.n_init),
        "batch_size": None if spec.design.batch_size is None else int(spec.design.batch_size),
        "target_metric": None if spec.design.target_metric is None else str(spec.design.target_metric),
        "direction": str(spec.design.direction),
        "regime_threshold": None if spec.design.regime_threshold is None else float(spec.design.regime_threshold),
        "meta": dict(design_meta),
    }
    return out


if __name__ == "__main__":
    main()
