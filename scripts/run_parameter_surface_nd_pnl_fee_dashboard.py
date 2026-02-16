#!/usr/bin/env python3
"""
Precompute an N-parameter grid and cache results to CSV.

This script is intentionally "cache-only": it runs the expensive simulations and writes a
scenario-scoped CSV that can be resumed/incrementally extended.

Stored results (written under `abm_results/grid_search/dashboard_nd/`):
  - `data/grid_<config_stem>_<fingerprint>.csv`: one row per grid point with:
      - Sweep coordinates:
          - `i__<param>`: index of the swept value for `<param>` (0..len(values)-1)
          - `v__<param>`: actual swept value used for `<param>` (cast to int for `INT_PARAMS`)
      - PnL summaries (horizon-normalized, aggregated across runs/seeds):
          - First, for each run/seed and each PnL time series `x_t`, we compute the per-step rate:
              `rate = mean(diff(x_{t0:}))`, where `t0 = skip_step`.
            (Equivalently: `(x_T - x_{t0}) / (T - t0)` when the series length is constant.)
          - Then we aggregate these per-run rates across `--runs-per-point` seeds:
              - `pnl_rate_p10_<metric>`, `pnl_rate_p50_<metric>`, `pnl_rate_p90_<metric>`: 10/50/90%
                quantiles of the per-run rates for that grid point and metric.
              - `pnl_rate_p_loss_<metric>`: fraction of runs with negative per-step rate.
      - Fee summaries (aggregated across time and runs):
          - `fee_mean`, `fee_median`: mean/median of the concatenated `fee_series` across all runs,
            after dropping the first `skip_step` observations in each run.
          - `fee_hist_0..fee_hist_<fee_hist_bins-1>`: histogram counts of `fee_series` values using the
            bin edges stored in the meta JSON (also aggregated across time and runs).
      - Smart-router routing share summaries (aggregated across time windows and runs):
          - We use the per-window series reported by the simulation:
              `smart_router_dex_share_series[w] = dex / (cex + dex)` computed every `n_block_SR_ratio` steps.
          - `smart_router_dex_share_mean`, `smart_router_dex_share_median`: mean/median of the concatenated
            `smart_router_dex_share_series` across all runs, after dropping observations whose associated
            window end-step is `< skip_step`.
          - `smart_router_dex_share_hist_0..smart_router_dex_share_hist_<fee_hist_bins-1>`: histogram counts
            of the concatenated `smart_router_dex_share_series` values using fixed edges in [0, 1] (stored in meta).
      - Provenance:
          - `runs_per_point`: number of runs used for this row
          - `seed_base`: base seed for the grid point (each run uses `seed_base + run_index`)
    Failed points are still appended, but with NaNs/zeros in the summary columns; error details
    are recorded separately in `errors_*.csv` (see below).
  - `data/meta_<config_stem>_<fingerprint>.json`: reproducibility metadata (sweeps, param order,
    metrics, seed mode, fee histogram bin edges).
  - `data/errors_<config_stem>_<fingerprint>.csv`: per-failure diagnostics (error type/message and
    the corresponding `i__*` / `v__*` sweep coordinates).

To build the interactive HTML dashboard (PnL surface + fee histogram) from the cached CSV, use:
  `scripts/build_parameter_surface_nd_pnl_fee_dashboard.py`

Notes:
  - This script runs the full simulation for every parameter combination (potentially expensive).
  - Results are cached to a scenario-scoped CSV to support resume / incremental runs.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts import run as run_module
from core.utils import load_simulation_parameters


def _silent_tqdm(iterable=None, **kwargs):
    """Silence tqdm inside scripts.run.simulate to avoid nested progress bars."""
    if iterable is None:
        total = int(kwargs.get("total", 0))
        return range(total)
    return iterable


run_module.tqdm = _silent_tqdm  # type: ignore[attr-defined]
simulate = run_module.simulate


BASE_CONFIG_PATH = Path("abm_results/scenarios/test.yml")

RUNS_PER_POINT_DEFAULT = 15
SEED_BASE_DEFAULT = 1
FEE_HIST_BINS_DEFAULT = 60

# PnL aggregation strategy for the cache (kept in the fingerprint/meta for reproducibility).
PNL_SUMMARY_KIND = "step_rate_mean_diff"
PNL_RATE_QUANTILES: Tuple[Tuple[str, float], ...] = (("p10", 0.10), ("p50", 0.50), ("p90", 0.90))


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
    # "passive_lp_share": np.linspace(0.0, 1.0, 3).tolist(),
    "narrow_mints_per_block": linspace_int(0, 2, 5),
    "smart_trades_per_block": linspace_int(0, 2, 5),
    "passive_mints_per_block": linspace_int(0, 2, 5),
    "noise_trades_per_block": linspace_int(0, 2, 5),
    "passive_burns_per_block": linspace_int(0, 2, 5),
    "k_sigma": np.linspace(0.0, 2, 5).tolist(),
    "mint_mu": np.linspace(-1, -0.1, 5).tolist(),
    "mint_sigma": np.linspace(1, 2, 5).tolist(),
    # "theta_T": [0.95, 0.98, 0.99999],
    # "p_jit": np.linspace(0.0, 1.0, 3).tolist(),
}

# DEFAULT_SWEEPS: Dict[str, Sequence[float | int]] = {
#     "passive_lp_share": [0.7, 0.5],
#     "narrow_mints_per_block": [0, 5],
#     "passive_mints_per_block": [0, 5],
#     "noise_trades_per_block": [0, 5],
#     "passive_burns_per_block": [0, 5],
#     "k_sigma": [0.0, 5.0],
#     "theta_T": [0.95, 0.9999],
#     "p_jit": [0.0, 1.0],

# }


# Parameters treated as integers for UI display + casting.
# INT_PARAMS: set[str] = {
#     "narrow_mints_per_block",
#     "passive_mints_per_block",
#     "noise_trades_per_block",
#     "passive_burns_per_block",
# }
INT_PARAMS: set[str] = {}
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
    smart_router_dex_share_hist_bins: int,
    pnl_summary: str,
) -> str:
    payload = {
        "sweeps": {k: list(v) for k, v in sorted(sweeps.items())},
        "runs_per_point": int(runs_per_point),
        "seed_base": int(seed_base),
        "common_seeds": bool(common_seeds),
        "fee_hist_bins": int(fee_hist_bins),
        "smart_router_dex_share_hist_bins": int(smart_router_dex_share_hist_bins),
        "pnl_summary": str(pnl_summary),
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
    required |= {f"pnl_rate_p50_{metric_key}" for metric_key, _ in PNL_METRICS}
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
            is_ok = bool(np.isfinite(row["fee_mean"]) and np.isfinite(row["fee_median"]))
            for metric_key, _ in PNL_METRICS:
                is_ok = is_ok and bool(np.isfinite(row[f"pnl_rate_p50_{metric_key}"]))
            if is_ok:
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
    smart_router_dex_share_bins: np.ndarray,
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
    params["liquidity_for_gif"] = False

    skip_step = max(0, int(params.get("skip_step", 0)))

    pnl_run_rates_by_key: Dict[str, List[float]] = {key: [] for key, _ in PNL_METRICS}
    fee_values: List[float] = []
    fee_hist = np.zeros(int(fee_bins.size - 1), dtype=np.int64)
    sr_dex_share_values: List[float] = []
    sr_dex_share_hist = np.zeros(int(smart_router_dex_share_bins.size - 1), dtype=np.int64)

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
                if series.size < 2:
                    raise ValueError(
                        f"Series '{metric_key}' too short after applying skip_step={skip_step} (len={series.size})."
                    )
                # Horizon-normalized per-run summary:
                #   rate = mean(ΔPnL_t) over the post-burn-in window.
                rate = float(np.mean(np.diff(series)))
                if not np.isfinite(rate):
                    raise ValueError(
                        f"Non-finite per-step rate for '{metric_key}' (run_index={run_index}, seed={seed_value})."
                    )
                pnl_run_rates_by_key[metric_key].append(rate)

            fee_series = _slice_series(output.get("fee_series", []), skip_step)
            if fee_series.size == 0:
                raise ValueError("fee_series empty after applying skip_step.")
            fee_values.extend([float(v) for v in fee_series.tolist()])
            fee_hist += np.histogram(fee_series, bins=fee_bins)[0].astype(np.int64)

            # Smart-router DEX share time series is defined over coarse windows (n_block_SR_ratio).
            # We aggregate its empirical distribution over time windows and runs, matching the fee histogram approach.
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
                # Fallback: if step alignment is unavailable, keep all finite ratios.
                for ratio in sr_series:
                    if not np.isfinite(ratio):
                        continue
                    sr_vals_run.append(float(ratio))

            if sr_vals_run:
                sr_arr_run = np.asarray(sr_vals_run, dtype=float)
                sr_dex_share_values.extend([float(v) for v in sr_arr_run.tolist()])
                sr_dex_share_hist += np.histogram(sr_arr_run, bins=smart_router_dex_share_bins)[0].astype(np.int64)
    except Exception as exc:
        ok = False
        error_type = type(exc).__name__
        error_message = str(exc)

    if ok:
        fee_arr = np.asarray(fee_values, dtype=float)
        fee_mean = float(np.mean(fee_arr)) if fee_arr.size else np.nan
        fee_median = float(np.median(fee_arr)) if fee_arr.size else np.nan

        sr_arr = np.asarray(sr_dex_share_values, dtype=float)
        sr_mean = float(np.mean(sr_arr)) if sr_arr.size else np.nan
        sr_median = float(np.median(sr_arr)) if sr_arr.size else np.nan

        pnl_summaries: Dict[str, float] = {}
        for metric_key, _ in PNL_METRICS:
            rates = np.asarray(pnl_run_rates_by_key[metric_key], dtype=float)
            if rates.size:
                for q_name, q in PNL_RATE_QUANTILES:
                    pnl_summaries[f"pnl_rate_{q_name}_{metric_key}"] = float(np.quantile(rates, q))
                pnl_summaries[f"pnl_rate_p_loss_{metric_key}"] = float(np.mean(rates < 0.0))
            else:
                for q_name, _ in PNL_RATE_QUANTILES:
                    pnl_summaries[f"pnl_rate_{q_name}_{metric_key}"] = np.nan
                pnl_summaries[f"pnl_rate_p_loss_{metric_key}"] = np.nan
        fee_hist_out = fee_hist.tolist()
        sr_dex_share_hist_out = sr_dex_share_hist.tolist()
    else:
        fee_mean = np.nan
        fee_median = np.nan
        sr_mean = np.nan
        sr_median = np.nan
        pnl_summaries = {
            f"pnl_rate_{q_name}_{metric_key}": np.nan
            for metric_key, _ in PNL_METRICS
            for q_name, _ in PNL_RATE_QUANTILES
        }
        pnl_summaries |= {f"pnl_rate_p_loss_{metric_key}": np.nan for metric_key, _ in PNL_METRICS}
        fee_hist_out = [0 for _ in range(int(fee_bins.size - 1))]
        sr_dex_share_hist_out = [0 for _ in range(int(smart_router_dex_share_bins.size - 1))]

    return {
        "grid_index": int(point.index),
        "indices": point.indices,
        "seed_base": int(point.seed_base),
        "ok": bool(ok),
        "error_type": error_type,
        "error_message": error_message,
        "error_run_index": error_run_index,
        "error_seed": error_seed,
        **pnl_summaries,
        "fee_mean": fee_mean,
        "fee_median": fee_median,
        "fee_hist": fee_hist_out,
        "smart_router_dex_share_mean": sr_mean,
        "smart_router_dex_share_median": sr_median,
        "smart_router_dex_share_hist": sr_dex_share_hist_out,
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
        "smart_router_dex_share_mean": float(result["smart_router_dex_share_mean"]),
        "smart_router_dex_share_median": float(result["smart_router_dex_share_median"]),
    }
    for metric_key, _ in PNL_METRICS:
        for q_name, _ in PNL_RATE_QUANTILES:
            row[f"pnl_rate_{q_name}_{metric_key}"] = float(result[f"pnl_rate_{q_name}_{metric_key}"])
        row[f"pnl_rate_p_loss_{metric_key}"] = float(result[f"pnl_rate_p_loss_{metric_key}"])

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

    sr_hist = result["smart_router_dex_share_hist"]
    if not isinstance(sr_hist, list) or len(sr_hist) != int(fee_hist_bins):
        raise ValueError("Internal error: smart_router_dex_share_hist has unexpected shape.")
    for b in range(int(fee_hist_bins)):
        row[f"smart_router_dex_share_hist_{b}"] = int(sr_hist[b])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="N-parameter grid runner (cached CSV outputs). Use scripts/build_parameter_surface_nd_pnl_fee_dashboard.py to render HTML."
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
    smart_router_dex_share_bin_edges = np.linspace(0.0, 1.0, fee_hist_bins + 1, dtype=float)

    fingerprint = _canon_fingerprint_payload(
        sweeps=sweeps,
        runs_per_point=runs_per_point,
        seed_base=int(args.seed_base),
        common_seeds=bool(args.common_seeds),
        fee_hist_bins=fee_hist_bins,
        smart_router_dex_share_hist_bins=fee_hist_bins,
        pnl_summary=PNL_SUMMARY_KIND,
    )

    # --- outputs -------------------------------------------------------------
    global_root = Path("abm_results") / "grid_search" / "dashboard_nd"

    stem = args.config.stem
    tag = f"{stem}_{fingerprint}"

    csv_global = global_root / "data" / f"grid_{tag}.csv"
    meta_global = global_root / "data" / f"meta_{tag}.json"
    errors_global = global_root / "data" / f"errors_{tag}.csv"
    verbose_progress = global_root / "logs" / f"progress_{tag}.txt"

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
        print(f"  smart_router_dex_share histogram bins: {fee_hist_bins} (fixed edges in [0, 1])")
        if slice_total != total_points:
            print(f"  index slice: [{index_start}, {index_stop}) => {slice_total} points")
        print("  swept parameters:")
        for name in param_order:
            values = sweeps[name]
            preview = values if len(values) <= 10 else (list(values[:5]) + ["..."] + list(values[-2:]))
            print(f"    - {name}: {len(values)} values: {preview}")
        print(f"  total grid points: {total_points}")
        print(f"  cache (global): {csv_global}")
        print(f"  meta (global):  {meta_global}")
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
    print(f"[dashboard_nd] cache (global): {csv_global}")
    meta_global.parent.mkdir(parents=True, exist_ok=True)
    meta_payload = {
        "tag": tag,
        "fingerprint": fingerprint,
        "scenario_label": scenario_label,
        "config_path": str(args.config),
        "param_order": list(param_order),
        "sweeps": {k: list(v) for k, v in sweeps.items()},
        "int_params": sorted(INT_PARAMS),
        "metrics": [{"key": k, "label": label} for k, label in PNL_METRICS],
        "pnl_summary": {
            "kind": PNL_SUMMARY_KIND,
            "within_run": "mean(diff(series[skip_step:]))",
            "across_runs": {
                "quantiles": [{"name": name, "q": q} for name, q in PNL_RATE_QUANTILES],
                "p_loss": "mean(rate < 0)",
            },
        },
        "runs_per_point": int(runs_per_point),
        "seed_base": int(args.seed_base),
        "common_seeds": bool(args.common_seeds),
        "fee_hist_bins": int(fee_hist_bins),
        "fee_bin_edges": fee_bin_edges.tolist(),
        "smart_router_dex_share_hist_bins": int(fee_hist_bins),
        "smart_router_dex_share_bin_edges": smart_router_dex_share_bin_edges.tolist(),
    }
    meta_global.write_text(json.dumps(meta_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[dashboard_nd] meta (global):  {meta_global}")

    # --- run simulations -----------------------------------------------------
    pending_rows: List[Dict[str, Any]] = []
    progress_overall: Optional[tqdm] = (
        tqdm(total=int(slice_total), desc="Grid points (slice)", unit="pt") if slice_total > 0 else None
    )

    cached_in_slice = 0
    run_in_slice = 0
    failed_in_slice = 0
    pending_error_rows: List[Dict[str, Any]] = []
    processed_in_slice = 0

    def _maybe_append_progress() -> None:
        if slice_total <= 0 or processed_in_slice <= 0:
            return
        if processed_in_slice % 5000 != 0:
            return
        verbose_progress.parent.mkdir(parents=True, exist_ok=True)
        with verbose_progress.open("a", encoding="utf-8") as handle:
            handle.write(f"{processed_in_slice}/{slice_total}\n")

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
                    processed_in_slice += 1
                    _maybe_append_progress()
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
                    smart_router_dex_share_bins=smart_router_dex_share_bin_edges,
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
                        processed_in_slice += 1
                        _maybe_append_progress()
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
                    processed_in_slice += 1
                    _maybe_append_progress()
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
    html_default = global_root / "html" / f"dashboard_{tag}.html"
    print(f"[dashboard_nd] cache complete: {csv_global}")
    print(
        "[dashboard_nd] build HTML with:\n"
        f"  python -m scripts.build_parameter_surface_nd_pnl_fee_dashboard --cache {csv_global} --meta {meta_global} --output {html_default}"
    )


if __name__ == "__main__":
    main()
