#!/usr/bin/env python3
"""
Precompute and plot a 3D PnL surface with a k_sigma slider.

Axes:
  x = Passive LP width (percentage, interpreted as +/- width/2 around price)
  y = Noise trades per block
  z = Median(final hedged passive LP PnL) across N runs

We run the full grid for fee_mode="volatility" and k_sigma in a discrete list
(default: np.linspace(1e-2, 10.0, 20)). Plotly frames + slider switch between
surfaces. Seeds are held fixed across k_sigma for each (x, y) cell (common
random numbers).
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

import run as run_module
from utils import build_empty_pool, load_simulation_parameters, scenario_output_root


def _silent_tqdm(iterable=None, **kwargs):
    """Silence tqdm inside run.simulate to avoid nested progress bars."""
    if iterable is None:
        total = int(kwargs.get("total", 0))
        return range(total)
    return iterable


run_module.tqdm = _silent_tqdm  # type: ignore[attr-defined]
simulate = run_module.simulate
save_plotly_figure = run_module.save_plotly_figure


BASE_CONFIG_PATH = Path("abm_results/scenarios/test.yml")

DEFAULT_WIDTH_PCTS: Tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
DEFAULT_NOISE_VALUES: Tuple[float, ...] = (1.0, 3.0, 5.0, 10.0, 15.0, 20.0)
# DEFAULT_WIDTH_PCTS: Tuple[float, ...] = (1.0, 2.0, 5.0)
# DEFAULT_NOISE_VALUES: Tuple[float, ...] = (1.0, 3.0, 5.0)
DEFAULT_K_SIGMA_VALUES = np.linspace(1e-2, 10.0, 20)

RUNS_PER_POINT_DEFAULT = 20
SEED_BASE_DEFAULT = 1


@dataclass(frozen=True)
class GridCell:
    width_pct: float
    noise_trades_per_block: float
    passive_width_ticks: int
    run_seed_base: int

    def label(self) -> str:
        return (
            f"width={self.width_pct:g}%, noise={self.noise_trades_per_block:g}, "
            f"ticks={self.passive_width_ticks}, seed_base={self.run_seed_base}"
        )


@dataclass(frozen=True)
class GridPoint:
    k_sigma_index: int
    k_sigma: float
    cell: GridCell

    def key(self) -> Tuple[int, float, float]:
        return (int(self.k_sigma_index), _canon_float(self.cell.width_pct), _canon_float(self.cell.noise_trades_per_block))


def _canon_float(value: float, *, ndigits: int = 12) -> float:
    return round(float(value), ndigits)


def _parse_float_list(value: Optional[str], *, name: str) -> List[float]:
    if value is None:
        raise ValueError("Internal error: expected a string.")
    parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not parts:
        raise SystemExit(f"No values provided for --{name}.")
    parsed: List[float] = []
    for part in parts:
        try:
            parsed.append(float(part))
        except ValueError as exc:
            raise SystemExit(f"Invalid float '{part}' in --{name}.") from exc
    return parsed


def _tick_from_price(pool, price: float) -> int:
    S = math.sqrt(max(1e-18, float(price)))
    tick_real = math.log(S / pool.base_s, pool.g)
    return int(round(tick_real))


def passive_width_ticks_from_percent(width_pct: float) -> int:
    """
    Convert a total width percentage into a tick-width, by:
      P_low  = P * (1 - width_pct/200)
      P_high = P * (1 + width_pct/200)
    then mapping both prices to ticks and snapping to tick_spacing.

    The absolute price P cancels in continuous math, but snapping to tick_spacing
    introduces small dependence; we use the same pool init as simulate().
    """
    pool, m0 = build_empty_pool()
    half = float(width_pct) / 200.0
    if half <= 0.0 or half >= 1.0:
        raise ValueError(f"width_pct must be in (0, 200): got {width_pct}")
    p_low = m0 * (1.0 - half)
    p_high = m0 * (1.0 + half)
    tick_low = pool._snap(_tick_from_price(pool, p_low))
    tick_high = pool._snap(_tick_from_price(pool, p_high))
    width_ticks = int(tick_high - tick_low)
    return max(int(pool.tick_spacing), width_ticks)


def _slice_series(values: Sequence[float], skip: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    skip_clamped = max(0, min(int(skip), array.size))
    return array[skip_clamped:]


def _point_to_row(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "k_sigma_index": int(result["k_sigma_index"]),
        "k_sigma": float(result["k_sigma"]),
        "passive_width_pct": float(result["width_pct"]),
        "passive_width_ticks": int(result["width_ticks"]),
        "noise_trades_per_block": float(result["noise_trades_per_block"]),
        "runs_per_point": int(result["runs_per_point"]),
        "seed_base": int(result["seed_base"]),
        "median_final_lp_pnl_passive": float(result["median_pnl"]),
        "median_fee": float(result["median_fee"]),
    }


def _evaluate_grid_point(
    point: GridPoint,
    base_params: Dict[str, Any],
    *,
    runs_per_point: int,
) -> Dict[str, Any]:
    params = dict(base_params)
    params["fee_mode"] = "volatility"
    params["k_sigma"] = float(point.k_sigma)
    params["noise_trades_per_block"] = float(point.cell.noise_trades_per_block)
    params["passive_width_ticks"] = int(point.cell.passive_width_ticks)

    params["visualize"] = False
    params["verbose"] = False
    params["light_mode"] = True

    skip_step = max(0, int(params.get("skip_step", 0)))
    pnl_samples: List[float] = []
    fee_medians: List[float] = []
    for run_index in range(int(runs_per_point)):
        params["seed"] = int(point.cell.run_seed_base + run_index)
        output = simulate(**params)
        series = _slice_series(output["lp_pnl_passive"], skip_step)
        if series.size == 0:
            raise ValueError("lp_pnl_passive is empty after applying skip_step.")
        pnl_samples.append(float(series[-1]))
        fee_series = _slice_series(output["fee_series"], skip_step)
        if fee_series.size == 0:
            raise ValueError("fee_series is empty after applying skip_step.")
        fee_medians.append(float(np.median(fee_series)))

    median_pnl = float(np.median(np.asarray(pnl_samples, dtype=float)))
    median_fee = float(np.median(np.asarray(fee_medians, dtype=float)))
    return {
        "k_sigma_index": point.k_sigma_index,
        "k_sigma": point.k_sigma,
        "width_pct": point.cell.width_pct,
        "width_ticks": point.cell.passive_width_ticks,
        "noise_trades_per_block": point.cell.noise_trades_per_block,
        "runs_per_point": runs_per_point,
        "seed_base": point.cell.run_seed_base,
        "median_pnl": median_pnl,
        "median_fee": median_fee,
    }


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


def _existing_keys(dataframe: pd.DataFrame) -> set[Tuple[int, float, float]]:
    required = {
        "k_sigma_index",
        "passive_width_pct",
        "noise_trades_per_block",
        "median_final_lp_pnl_passive",
        "median_fee",
    }
    if dataframe.empty or not required.issubset(set(dataframe.columns)):
        return set()
    keys: set[Tuple[int, float, float]] = set()
    for _, row in dataframe.iterrows():
        try:
            if not (np.isfinite(row["median_final_lp_pnl_passive"]) and np.isfinite(row["median_fee"])):
                continue
            keys.add(
                (
                    int(row["k_sigma_index"]),
                    _canon_float(row["passive_width_pct"]),
                    _canon_float(row["noise_trades_per_block"]),
                )
            )
        except Exception:
            continue
    return keys


def _build_frames(
    *,
    k_sigma_values: Sequence[float],
    width_pcts: Sequence[float],
    noise_values: Sequence[float],
    width_ticks_by_pct: Dict[float, int],
    pnl_by_k: np.ndarray,
    pnl_min: float,
    pnl_max: float,
    fee_by_k: np.ndarray,
    fee_min: float,
    fee_max: float,
) -> Tuple[List[go.Frame], List[Dict[str, Any]]]:
    frames: List[go.Frame] = []
    steps: List[Dict[str, Any]] = []
    width_ticks_ordered = [float(width_ticks_by_pct[float(pct)]) for pct in width_pcts]
    width_ticks_matrix = np.tile(np.asarray(width_ticks_ordered, dtype=float), (len(noise_values), 1))

    for k_idx, k_val in enumerate(k_sigma_values):
        z_pnl = pnl_by_k[k_idx]
        z_fee = fee_by_k[k_idx]

        customdata_pnl = np.dstack(
            [
                np.full_like(z_pnl, float(k_val), dtype=float),
                width_ticks_matrix,
            ]
        )
        customdata_fee = np.dstack(
            [
                np.full_like(z_fee, float(k_val), dtype=float),
                width_ticks_matrix,
            ]
        )
        frame = go.Frame(
            name=f"k_sigma_{k_idx}",
            data=[
                go.Surface(
                    scene="scene",
                    x=np.asarray(width_pcts, dtype=float),
                    y=np.asarray(noise_values, dtype=float),
                    z=z_pnl,
                    customdata=customdata_pnl,
                    cmin=pnl_min,
                    cmax=pnl_max,
                ),
                go.Surface(
                    scene="scene2",
                    x=np.asarray(width_pcts, dtype=float),
                    y=np.asarray(noise_values, dtype=float),
                    z=z_fee,
                    customdata=customdata_fee,
                    cmin=fee_min,
                    cmax=fee_max,
                ),
            ],
            layout=go.Layout(title=f"Median passive LP PnL and fee (k_sigma={k_val:.4g})"),
        )
        frames.append(frame)
        steps.append(
            {
                "method": "animate",
                "label": f"{k_val:.4g}",
                "args": [
                    [frame.name],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    return frames, steps


def build_figure(
    *,
    k_sigma_values: Sequence[float],
    width_pcts: Sequence[float],
    noise_values: Sequence[float],
    width_ticks_by_pct: Dict[float, int],
    pnl_by_k: np.ndarray,
    fee_by_k: np.ndarray,
) -> go.Figure:
    pnl_min = float(np.nanmin(pnl_by_k))
    pnl_max = float(np.nanmax(pnl_by_k))
    fee_min = float(np.nanmin(fee_by_k))
    fee_max = float(np.nanmax(fee_by_k))

    width_ticks_ordered = [float(width_ticks_by_pct[float(pct)]) for pct in width_pcts]
    width_ticks_matrix = np.tile(np.asarray(width_ticks_ordered, dtype=float), (len(noise_values), 1))

    z0 = pnl_by_k[0]
    f0 = fee_by_k[0]
    customdata0 = np.dstack(
        [
            np.full_like(z0, float(k_sigma_values[0]), dtype=float),
            width_ticks_matrix,
        ]
    )
    customdataf0 = np.dstack(
        [
            np.full_like(f0, float(k_sigma_values[0]), dtype=float),
            width_ticks_matrix,
        ]
    )

    surface_pnl = go.Surface(
        scene="scene",
        x=np.asarray(width_pcts, dtype=float),
        y=np.asarray(noise_values, dtype=float),
        z=z0,
        customdata=customdata0,
        colorscale="Viridis",
        cmin=pnl_min,
        cmax=pnl_max,
        colorbar={"title": "Median PnL", "x": 0.46},
        hovertemplate=(
            "Width=%{x:.3g}% (ticks=%{customdata[1]:.0f})<br>"
            "Noise trades/block=%{y:.3g}<br>"
            "k_sigma=%{customdata[0]:.4g}<br>"
            "Median final PnL=%{z:.6g}"
            "<extra></extra>"
        ),
    )
    surface_fee = go.Surface(
        scene="scene2",
        x=np.asarray(width_pcts, dtype=float),
        y=np.asarray(noise_values, dtype=float),
        z=f0,
        customdata=customdataf0,
        colorscale="Plasma",
        cmin=fee_min,
        cmax=fee_max,
        colorbar={"title": "Median fee", "x": 1.02},
        hovertemplate=(
            "Width=%{x:.3g}% (ticks=%{customdata[1]:.0f})<br>"
            "Noise trades/block=%{y:.3g}<br>"
            "k_sigma=%{customdata[0]:.4g}<br>"
            "Median fee=%{z:.6g}"
            "<extra></extra>"
        ),
    )

    frames, steps = _build_frames(
        k_sigma_values=k_sigma_values,
        width_pcts=width_pcts,
        noise_values=noise_values,
        width_ticks_by_pct=width_ticks_by_pct,
        pnl_by_k=pnl_by_k,
        pnl_min=pnl_min,
        pnl_max=pnl_max,
        fee_by_k=fee_by_k,
        fee_min=fee_min,
        fee_max=fee_max,
    )

    fig = go.Figure(data=[surface_pnl, surface_fee], frames=frames)
    fig.update_layout(
        template="plotly_white",
        title=f"Median passive LP PnL and fee (k_sigma={k_sigma_values[0]:.4g})",
        scene=dict(
            xaxis_title="Passive LP width (%)",
            yaxis_title="Noise trades per block",
            zaxis_title="Median final hedged passive LP PnL",
            zaxis=dict(range=[pnl_min, pnl_max]),
        ),
        scene2=dict(
            xaxis_title="Passive LP width (%)",
            yaxis_title="Noise trades per block",
            zaxis_title="Median fee",
            zaxis=dict(range=[fee_min, fee_max]),
        ),
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "k_sigma: "},
                "pad": {"t": 50},
                "steps": steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.12,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 250, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D passive LP PnL and fee surfaces with k_sigma slider (fee_mode=volatility)."
    )
    parser.add_argument("--config", type=Path, default=BASE_CONFIG_PATH, help="Base YAML scenario config path.")
    parser.add_argument(
        "--runs-per-point",
        type=int,
        default=RUNS_PER_POINT_DEFAULT,
        help="Number of simulations per (k_sigma, width, noise) grid point.",
    )
    parser.add_argument(
        "--k-sigma",
        type=str,
        default=None,
        help="Comma-separated k_sigma values (overrides default linspace).",
    )
    parser.add_argument(
        "--width-pcts",
        type=str,
        default=",".join(str(v) for v in DEFAULT_WIDTH_PCTS),
        help="Comma-separated total width percentages (e.g. '1,2,5,10,20,40').",
    )
    parser.add_argument(
        "--noise",
        type=str,
        default=",".join(str(v) for v in DEFAULT_NOISE_VALUES),
        help="Comma-separated noise_trades_per_block values.",
    )
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--seed-base", type=int, default=SEED_BASE_DEFAULT, help="Base seed for deterministic grids.")
    parser.add_argument("--recompute", action="store_true", help="Ignore cache and recompute all grid points.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved grid and exit.")
    args = parser.parse_args()

    width_pcts = [float(v) for v in _parse_float_list(args.width_pcts, name="width-pcts")]
    noise_values = [float(v) for v in _parse_float_list(args.noise, name="noise")]

    if args.k_sigma is None:
        k_sigma_values = np.asarray(DEFAULT_K_SIGMA_VALUES, dtype=float)
    else:
        k_sigma_values = np.asarray(_parse_float_list(args.k_sigma, name="k-sigma"), dtype=float)

    runs_per_point = int(args.runs_per_point)
    if runs_per_point <= 0:
        raise SystemExit("--runs-per-point must be positive.")

    scenario_label, base_params = load_simulation_parameters(args.config, simulate_func=simulate)
    if float(base_params.get("passive_lp_share", 1.0)) <= 0.0:
        raise SystemExit("Scenario has passive_lp_share=0; passive LP PnL is not defined for this surface.")

    # Keep outputs scenario-scoped (no per-run logs in light_mode, but keep paths consistent).
    scenario_root = scenario_output_root(args.config)
    base_params["results_root"] = scenario_root / "grid_search"

    width_ticks_by_pct: Dict[float, int] = {}
    for pct in width_pcts:
        width_ticks_by_pct[float(pct)] = passive_width_ticks_from_percent(float(pct))

    if args.dry_run:
        print("Resolved grid:")
        print(f"  config: {args.config}")
        print(f"  scenario_label (from YAML): {scenario_label}")
        print(f"  fee_mode override: volatility")
        print(f"  runs_per_point: {runs_per_point}")
        print(f"  k_sigma values: {k_sigma_values.tolist()}")
        print(f"  noise_trades_per_block: {noise_values}")
        print("  passive width mapping (% -> ticks):")
        for pct in width_pcts:
            print(f"    {pct:g}% -> {width_ticks_by_pct[float(pct)]} ticks")
        print(f"  total grid points: {len(k_sigma_values) * len(width_pcts) * len(noise_values)}")
        return

    # --- outputs -------------------------------------------------------------
    global_root = Path("abm_results") / "grid_search"
    scenario_grid_dir = scenario_root / "grid_search"

    data_dir_global = global_root / "surface_3d" / "data"
    data_dir_scenario = scenario_grid_dir / "surface_3d" / "data"

    stem = args.config.stem
    grid_tag = f"w{len(width_pcts)}_n{len(noise_values)}_k{len(k_sigma_values)}_r{runs_per_point}"
    csv_global = data_dir_global / f"surface_passive_lp_pnl_and_fee_medians_{stem}_{grid_tag}.csv"
    csv_scenario = data_dir_scenario / f"surface_passive_lp_pnl_and_fee_medians_{grid_tag}.csv"

    png_dir_global = global_root / "surface_3d" / "png"
    html_dir_global = global_root / "surface_3d" / "html"
    png_dir_scenario = scenario_grid_dir / "surface_3d" / "png"
    html_dir_scenario = scenario_grid_dir / "surface_3d" / "html"

    fig_base = f"surface_passive_lp_pnl_and_fee_medians_{stem}_{grid_tag}_k_sigma_slider"
    png_path_global = png_dir_global / f"{fig_base}.png"
    html_path_global = html_dir_global / f"{fig_base}.html"
    png_path_scenario = png_dir_scenario / f"{fig_base}.png"
    html_path_scenario = html_dir_scenario / f"{fig_base}.html"

    cached = _load_cache(csv_scenario)
    existing_keys = _existing_keys(cached)

    # --- grid construction ---------------------------------------------------
    cells: List[GridCell] = []
    seed_cursor = int(args.seed_base)
    for width_pct in width_pcts:
        for noise in noise_values:
            cells.append(
                GridCell(
                    width_pct=float(width_pct),
                    noise_trades_per_block=float(noise),
                    passive_width_ticks=int(width_ticks_by_pct[float(width_pct)]),
                    run_seed_base=seed_cursor,
                )
            )
            seed_cursor += runs_per_point

    points: List[GridPoint] = []
    for k_idx, k_val in enumerate(k_sigma_values.tolist()):
        for cell in cells:
            points.append(GridPoint(k_sigma_index=int(k_idx), k_sigma=float(k_val), cell=cell))

    if args.recompute:
        points_to_run = points
    else:
        points_to_run = [pt for pt in points if pt.key() not in existing_keys]

    print(
        f"[surface_3d] config={args.config} | fee_mode=volatility | "
        f"grid={len(points)} points ({len(points_to_run)} to run, {len(points) - len(points_to_run)} cached) | "
        f"workers={args.max_workers}"
    )
    if points_to_run:
        print(f"[surface_3d] cache (scenario): {csv_scenario}")
        print(f"[surface_3d] cache (global):   {csv_global}")

    # --- run simulations -----------------------------------------------------
    pending_rows: List[Dict[str, Any]] = []
    progress_overall: Optional[tqdm] = None
    if points:
        progress_overall = tqdm(total=len(points), desc="Grid points (overall)", unit="pt")
        # Count cached points as already completed
        cached_count = len(points) - len(points_to_run)
        if cached_count > 0:
            progress_overall.update(cached_count)

    if points_to_run:
        with ProcessPoolExecutor(max_workers=int(args.max_workers)) as executor:
            futures = {
                executor.submit(
                    _evaluate_grid_point,
                    point,
                    base_params,
                    runs_per_point=runs_per_point,
                ): point
                for point in points_to_run
            }
            for future in as_completed(futures):
                result = future.result()
                pending_rows.append(_point_to_row(result))
                if progress_overall is not None:
                    progress_overall.update(1)
                if len(pending_rows) >= 25:
                    _append_rows_csv(csv_scenario, pending_rows)
                    _append_rows_csv(csv_global, pending_rows)
                    pending_rows.clear()

        if pending_rows:
            _append_rows_csv(csv_scenario, pending_rows)
            _append_rows_csv(csv_global, pending_rows)
            pending_rows.clear()
    if progress_overall is not None:
        progress_overall.close()

    # --- build figure from cache --------------------------------------------
    dataframe = _load_cache(csv_scenario)
    if dataframe.empty:
        raise SystemExit("No data found; nothing to plot.")

    dataframe = dataframe[
        dataframe["passive_width_pct"].isin(width_pcts)
        & dataframe["noise_trades_per_block"].isin(noise_values)
        & dataframe["k_sigma_index"].isin(list(range(len(k_sigma_values))))
    ].copy()

    z_by_k: List[np.ndarray] = []
    fee_by_k: List[np.ndarray] = []
    for k_idx in range(len(k_sigma_values)):
        sub = dataframe[dataframe["k_sigma_index"] == k_idx]
        pivot_pnl = sub.pivot_table(
            index="noise_trades_per_block",
            columns="passive_width_pct",
            values="median_final_lp_pnl_passive",
            aggfunc="median",
        )
        pivot_fee = sub.pivot_table(
            index="noise_trades_per_block",
            columns="passive_width_pct",
            values="median_fee",
            aggfunc="median",
        )
        pivot_pnl = pivot_pnl.reindex(index=noise_values, columns=width_pcts)
        pivot_fee = pivot_fee.reindex(index=noise_values, columns=width_pcts)
        if pivot_pnl.isna().any().any() or pivot_fee.isna().any().any():
            missing = pivot_pnl.isna() | pivot_fee.isna()
            missing_pairs = [
                (float(noise_values[i]), float(width_pcts[j]))
                for i, j in zip(*np.where(missing.to_numpy()))
            ]
            raise SystemExit(
                f"Missing grid values for k_sigma_index={k_idx}. "
                f"Rerun without cache or with --recompute. Missing (noise,width%): {missing_pairs[:10]}"
                + (" ..." if len(missing_pairs) > 10 else "")
            )
        z_by_k.append(pivot_pnl.to_numpy(dtype=float))
        fee_by_k.append(pivot_fee.to_numpy(dtype=float))

    z_stack = np.stack(z_by_k, axis=0)
    fee_stack = np.stack(fee_by_k, axis=0)
    figure = build_figure(
        k_sigma_values=k_sigma_values.tolist(),
        width_pcts=width_pcts,
        noise_values=noise_values,
        width_ticks_by_pct=width_ticks_by_pct,
        pnl_by_k=z_stack,
        fee_by_k=fee_stack,
    )

    save_plotly_figure(figure, png_path=png_path_global, html_path=html_path_global, source="surface_3d")
    save_plotly_figure(figure, png_path=png_path_scenario, html_path=html_path_scenario, source="surface_3d")

    print(f"[surface_3d] HTML written to {html_path_global} and {html_path_scenario}")
    print(f"[surface_3d] PNG written to {png_path_global} and {png_path_scenario}")


if __name__ == "__main__":
    main()
