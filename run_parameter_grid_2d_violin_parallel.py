#!/usr/bin/env python3
"""
Parallel 2D grid runner that sweeps:

- Dynamic fee sensitivity (e.g. k_sigma or k_basis), and
- A second user-selected simulation parameter (default: smart/noise trade ratio),

using a base YAML config (default: abm_results/scenarios/test.yml).

For each 2D grid point, the script:
- Runs the ABM several times.
- Collects final hedged LP PnL samples (passive/active, when present).
- Collects the fee samples over time.
- Builds 3D "violin columns" over the 2D grid, following the geometry from
  3d_violin_plot.py.

It produces up to three 3D figures (each in its own window):
- Passive LP hedged PnL (if passive_lp_share > 0).
- Active LP hedged PnL (if passive_lp_share < 1).
- Fee level.

Each figure is saved as both PNG and a simple HTML wrapper next to it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tqdm import tqdm
import plotly.graph_objects as go

import run as run_module
from utils import load_simulation_parameters


# Silence the tqdm progress bar inside run.simulate to avoid nested bars
def _silent_tqdm(iterable=None, **kwargs):
    if iterable is None:
        total = int(kwargs.get("total", 0))
        return range(total)
    return iterable


run_module.tqdm = _silent_tqdm  # type: ignore[attr-defined]
simulate = run_module.simulate
save_plotly_figure = run_module.save_plotly_figure


PNL_ARBITRAGEUR_KEY: Tuple[str, str] = ("arb_pnl_cum", "Arbitrageur PnL")
PNL_PASSIVE_KEY: Tuple[str, str] = ("lp_pnl_passive", "Passive LP hedged PnL")
PNL_ACTIVE_KEY: Tuple[str, str] = ("lp_pnl_active", "Active LP hedged PnL")


def resolve_pnl_keys(params: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Include only the PnL distributions for LP cohorts that exist in the scenario.

    - passive_lp_share = 1.0 => only passive LP PnL
    - passive_lp_share = 0.0 => only active LP PnL
    - otherwise include both LP cohorts
    """
    try:
        passive_share = float(params.get("passive_lp_share", 1.0))
    except (TypeError, ValueError):
        passive_share = 1.0
    passive_share = max(0.0, min(1.0, passive_share))
    keys: List[Tuple[str, str]] = []
    if passive_share > 0.0:
        keys.append(PNL_PASSIVE_KEY)
    if passive_share < 1.0:
        keys.append(PNL_ACTIVE_KEY)
    return keys


# --- configuration -----------------------------------------------------------
BASE_CONFIG_PATH = Path("abm_results/scenarios/test.yml")
FEE_MODE_CONFIG = {
    "static": {
        "param_name": None,
        "values": [None],  # constant fee
        "xlabel": "Static fee",
    },
    "volatility_cex": {
        "param_name": "k_sigma",
        "values": np.linspace(1e-2, 10.0, 5),
        "xlabel": r"$k_{\sigma}$ (volatility CEX)",
    },
    "volatility_dex": {
        "param_name": "k_sigma",
        "values": np.linspace(1e-2, 10.0, 5),
        "xlabel": r"$k_{\sigma}$ (volatility DEX)",
    },
    # "toxicity": {
    #     "param_name": "k_basis",
    #     "values": np.linspace(1e-5, 1e-1, 10),
    #     "xlabel": r"$k_{b}$ (toxicity)",
    # },
}

RUNS_PER_POINT = 20
SEED_BASE = 1

# -----------------------------------------------------------------------------
# 2D grid configuration (edit these at the top of the file)
# -----------------------------------------------------------------------------
# Choose which 2 parameters define the 2D grid, or run all 2-combinations.
# Supported axes:
# - "sensitivity"            (mapped automatically to k_sigma/k_basis depending on fee_mode)
# - "ratio"                  (smart_trades_per_block / noise_trades_per_block)
# - "noise_trades_per_block"
# - "smart_trades_per_block"
# - "passive_width_pct"
# - "passive_width_ticks"  # legacy tick widths
#
# Grid modes:
# - "single": use GRID_AXIS_X / GRID_AXIS_Y
# - "all_pairs": run all 2-combinations from GRID_ALL_AXES (4 choose 2 = 6 by default)
GRID_MODE = "all_pairs"  # "single" | "all_pairs"
GRID_ALL_AXES: Tuple[str, ...] = (
    "sensitivity",
    "noise_trades_per_block",
    # "smart_trades_per_block",
    "passive_width_pct",
)
GRID_AXIS_X = "sensitivity"
GRID_AXIS_Y = "ratio"

# Sweep values for axes other than "sensitivity".
# - For "ratio", you can either set values here OR pass --ratios on the CLI.
# - For "sensitivity", values are taken from FEE_MODE_CONFIG[fee_mode]["values"].
GRID_SWEEP_VALUES: Dict[str, Sequence[float]] = {
    "ratio": np.linspace(0.0, 1.0, 5),
    "noise_trades_per_block": [1.0, 3.0, 10.0, 20.0],
    "smart_trades_per_block": [0.0, 5.0, 10.0, 20.0],
    "passive_width_pct": [1.0, 2.0, 5.0, 10.0, 20.0],
    # passive_width_ticks are ticks (ints), but we store them as floats for plotting.
    "passive_width_ticks": [50.0, 100.0, 500.0, 1_000.0],
}

# Default sweep for smart/noise ratio (only used when "ratio" is selected and no override is provided)
DEFAULT_RATIO_VALUES = np.linspace(0.0, 1.0, 5)  # smart_trades_per_block / noise_trades_per_block

OUTPUT_ROOT = Path("abm_results") / "grid_search"
PLOTS_3D_ROOT = OUTPUT_ROOT / "plots_3d"

KEEP_VISUALS = False       # pass through to simulate()
VERBOSE_RUNS = False       # print per-run progress inside workers if True
RECOMPUTE = True          # If True, run simulations and regenerate cached CSVs
# ---------------------------------------------------------------------------


def _slice_series(values: Sequence[float], skip: int) -> np.ndarray:
    """Utility matching run_scenarios_mean_std._slice_series."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    skip_clamped = max(0, min(int(skip), array.size))
    return array[skip_clamped:]


def _sigma_label_from_params(params: Dict[str, Any]) -> str:
    mode = str(params.get("cex_sigma_mode", "static")).lower()
    if mode.startswith("regime"):
        low_value = params.get("cex_sigma_low")
        high_value = params.get("cex_sigma_high")
        return f"regime_{low_value}-{high_value}"
    if mode == "noisy_sine":
        center = params.get("cex_sigma")
        amplitude = params.get("cex_sigma_sine_amp")
        if amplitude is None and params.get("cex_sigma_low") is not None and params.get("cex_sigma_high") is not None:
            amplitude = 0.5 * abs(float(params["cex_sigma_high"]) - float(params["cex_sigma_low"]))
            center = 0.5 * (float(params["cex_sigma_low"]) + float(params["cex_sigma_high"]))
        noise = params.get("cex_sigma_sine_noise", 0.0)
        period = params.get("cex_sigma_sine_period", "p")
        amplitude_label = "auto" if amplitude is None else amplitude
        return f"noisy_sine_{center}_amp{amplitude_label}_per{period}_noise{noise}"
    return str(params.get("cex_sigma"))


def _parse_ratio_values(arg_value: Optional[str], base_ratio: float) -> List[float]:
    """
    Parse a comma-separated list of ratios or fall back to a default sweep
    that always includes the base_ratio from the YAML.
    """
    if arg_value:
        raw_parts = [chunk.strip() for chunk in arg_value.split(",") if chunk.strip()]
        ratio_values: List[float] = []
        for part in raw_parts:
            try:
                ratio_values.append(float(part))
            except ValueError:
                raise SystemExit(f"Invalid ratio value '{part}' in --ratios") from None
        if not ratio_values:
            raise SystemExit("No valid ratio values parsed from --ratios.")
        return ratio_values

    default_values = list(DEFAULT_RATIO_VALUES.astype(float))
    if base_ratio not in default_values:
        default_values.append(base_ratio)
    return sorted(set(default_values))


GRID_AXIS_LABELS: Dict[str, str] = {
    "ratio": "Smart/noise ratio",
    "noise_trades_per_block": "Noise trades per block",
    "smart_trades_per_block": "Smart trades per block",
    "passive_width_pct": "Passive LP width (%)",
    "passive_width_ticks": "Passive LP width (ticks)",
}

SUPPORTED_GRID_AXES = {"sensitivity", *GRID_AXIS_LABELS}


def _validate_grid_axes(axis_x: str, axis_y: str) -> None:
    if axis_x == axis_y:
        raise SystemExit(f"GRID_AXIS_X and GRID_AXIS_Y must be different (both were '{axis_x}').")
    unknown = sorted({axis_x, axis_y} - SUPPORTED_GRID_AXES)
    if unknown:
        raise SystemExit(
            "Unsupported grid axis/axes: "
            f"{', '.join(unknown)}. Supported: {', '.join(sorted(SUPPORTED_GRID_AXES))}"
        )
    axes_set = {axis_x, axis_y}
    if "ratio" in axes_set and "smart_trades_per_block" in axes_set:
        raise SystemExit(
            "Invalid grid: 'ratio' sets smart_trades_per_block implicitly, so it cannot be combined with "
            "'smart_trades_per_block' as the other axis. Use 'noise_trades_per_block' + 'ratio' instead."
        )


def _resolve_axis_pairs() -> List[Tuple[str, str]]:
    mode = str(GRID_MODE).lower().strip()
    if mode == "single":
        return [(GRID_AXIS_X, GRID_AXIS_Y)]
    if mode == "all_pairs":
        axes = list(GRID_ALL_AXES)
        if len(axes) < 2:
            raise SystemExit("GRID_ALL_AXES must contain at least 2 axes when GRID_MODE='all_pairs'.")
        return [(str(a), str(b)) for a, b in combinations(axes, 2)]
    raise SystemExit(f"Invalid GRID_MODE='{GRID_MODE}'. Expected 'single' or 'all_pairs'.")


def _grid_axis_label(axis: str, fee_mode: str) -> str:
    if axis == "sensitivity":
        return str(FEE_MODE_CONFIG[fee_mode]["xlabel"])
    return GRID_AXIS_LABELS.get(axis, axis)


def _format_axis_values(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(value):g}" for value in values) + "]"


def _resolve_axis_values(
    axis: str,
    *,
    base_params: Dict[str, Any],
    fee_mode: str,
    args: argparse.Namespace,
    base_ratio: float,
) -> List[float]:
    if axis == "sensitivity":
        param_name = FEE_MODE_CONFIG[fee_mode]["param_name"]
        values = FEE_MODE_CONFIG[fee_mode]["values"]
        if param_name is None:
            return [0.0]
        return [float(v) for v in values if v is not None]

    if axis == "ratio":
        override = GRID_SWEEP_VALUES.get("ratio")
        if override is not None:
            return [float(v) for v in override]
        return [float(v) for v in _parse_ratio_values(args.ratios, base_ratio)]

    if axis not in GRID_SWEEP_VALUES:
        raise SystemExit(
            f"Missing GRID_SWEEP_VALUES['{axis}']. Define its sweep values near the top of the script."
        )
    return [float(v) for v in GRID_SWEEP_VALUES[axis]]


def _apply_axis_value(
    params: Dict[str, Any],
    *,
    axis: str,
    value: float,
    fee_mode: str,
) -> None:
    if axis == "sensitivity":
        param_name = FEE_MODE_CONFIG[fee_mode]["param_name"]
        if param_name is not None:
            params[param_name] = float(value)
        return

    if axis == "noise_trades_per_block":
        params["noise_trades_per_block"] = float(value)
        return

    if axis == "smart_trades_per_block":
        params["smart_trades_per_block"] = float(value)
        return

    if axis == "passive_width_ticks":
        params["passive_width_ticks"] = int(round(float(value)))
        params["passive_width_pct"] = None
        return

    if axis == "passive_width_pct":
        params["passive_width_pct"] = float(value)
        params["passive_width_ticks"] = None
        return

    if axis == "ratio":
        # Applied later, after all direct parameter overrides, so it can use the final noise_trades_per_block.
        return

    raise KeyError(f"Unsupported axis '{axis}'")


def _grid_output_basename(
    *,
    scenario_label: str,
    fee_mode: str,
    sigma_label: str,
    theta_T_label: str,
    axis_x: str,
    axis_y: str,
) -> str:
    """
    Base name used for cached CSVs and plot outputs.

    Keeps legacy filenames when using the original (sensitivity, ratio) grid.
    """
    base = f"3d_violin_{scenario_label}_{fee_mode}_sigma_{sigma_label}_thetaT_{theta_T_label}"
    if axis_x == "sensitivity" and axis_y == "ratio":
        return base
    return f"{base}_x-{axis_x}_y-{axis_y}"


@dataclass(frozen=True)
class GridPoint2D:
    fee_mode: str
    cex_sigma_label: str
    axis_x: str
    axis_y: str
    x_value: float
    y_value: float
    run_seed_base: int

    def label(self) -> str:
        return (
            f"fee={self.fee_mode}, sigma={self.cex_sigma_label}, "
            f"{self.axis_x}={self.x_value}, {self.axis_y}={self.y_value}"
        )


def build_grid_2d(
    base_params: Dict[str, Any],
    fee_mode: str,
    axis_x: str,
    axis_y: str,
    axis_x_values: Sequence[float],
    axis_y_values: Sequence[float],
) -> List[GridPoint2D]:
    """
    Build the 2D grid over (axis_x, axis_y) for a single fee_mode.

    Notes:
    - "sensitivity" is mapped automatically to the appropriate fee controller
      parameter (k_sigma or k_basis) based on fee_mode. For fee modes without a
      tunable parameter (e.g. static), it degenerates to a constant coordinate.
    - "ratio" sets smart_trades_per_block as ratio * noise_trades_per_block.
    """
    grid: List[GridPoint2D] = []
    seed_cursor = SEED_BASE
    sigma_label = _sigma_label_from_params(base_params)

    for x_val in axis_x_values:
        for y_val in axis_y_values:
            grid.append(
                GridPoint2D(
                    fee_mode=fee_mode,
                    cex_sigma_label=sigma_label,
                    axis_x=axis_x,
                    axis_y=axis_y,
                    x_value=float(x_val),
                    y_value=float(y_val),
                    run_seed_base=seed_cursor,
                )
            )
            seed_cursor += RUNS_PER_POINT

    return grid


def run_grid_point_2d(
    base_params: Dict[str, Any],
    point: GridPoint2D,
    pnl_keys: Sequence[Tuple[str, str]],
) -> Tuple[Dict[str, List[float]], List[float]]:
    """
    Execute RUNS_PER_POINT simulations for a single 2D grid point.

    Returns:
        pnl_samples: dict[key -> list of final PnL samples]
        fee_samples: list of fee values over time across all runs
    """
    params = dict(base_params)
    params["fee_mode"] = point.fee_mode

    # Apply direct axis overrides first.
    _apply_axis_value(params, axis=point.axis_x, value=point.x_value, fee_mode=point.fee_mode)
    _apply_axis_value(params, axis=point.axis_y, value=point.y_value, fee_mode=point.fee_mode)

    # Apply ratio override last, since it depends on the final noise_trades_per_block.
    if point.axis_x == "ratio" or point.axis_y == "ratio":
        ratio_value = point.x_value if point.axis_x == "ratio" else point.y_value
        noise_value = float(params.get("noise_trades_per_block", 0.0))
        params["smart_trades_per_block"] = float(ratio_value) * noise_value

    if not KEEP_VISUALS:
        params["visualize"] = False
    params["light_mode"] = True
    if not VERBOSE_RUNS:
        params["verbose"] = False

    skip_step = max(0, int(params.get("skip_step", 0)))

    pnl_samples: Dict[str, List[float]] = {key: [] for key, _ in pnl_keys}
    fee_samples: List[float] = []

    for run_index in range(RUNS_PER_POINT):
        seed_value = point.run_seed_base + run_index
        params["seed"] = seed_value
        if VERBOSE_RUNS:
            print(f"[grid_2d:{point.label()}] run {run_index + 1}/{RUNS_PER_POINT} (seed={seed_value})")
        output = simulate(**params)

        for key, _ in pnl_keys:
            series = _slice_series(output[key], skip_step)
            if series.size == 0:
                raise ValueError(f"Series '{key}' empty after skip_step.")
            pnl_samples[key].append(float(series[-1]))

        fee_series = _slice_series(output["fee_series"], skip_step)
        fee_samples.extend([float(value) for value in fee_series])

    return pnl_samples, fee_samples


def _evaluate_point_2d(
    point: GridPoint2D,
    base_params: Dict[str, Any],
    pnl_keys: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    pnl_samples, fee_samples = run_grid_point_2d(base_params, point, pnl_keys)
    return {
        "point": point,
        "pnl_samples": pnl_samples,
        "fee_samples": fee_samples,
    }


def plot_3d_violin_grid(
    dataframe: pd.DataFrame,
    param1: str,
    param2: str,
    value_col: str,
    *,
    param1_label: Optional[str] = None,
    param2_label: Optional[str] = None,
    z_label: Optional[str] = None,
    n_z: int = 40,
    n_theta: int = 24,
    max_radius_factor: float = 0.35,
    bandwidth: str | float = "scott",
    title: Optional[str] = None,
    add_mean_surface: bool = True,
) -> go.Figure:
    """
    Plot a 3D grid of "violin columns" for value_col distributions over a 2D
    parameter grid (param1, param2), following the geometry in 3d_violin_plot.py,
    using Plotly Surface traces (interactive HTML-ready). Optionally overlays a
    smooth surface connecting the mean of each distribution.
    """
    if dataframe.empty:
        raise ValueError("Input DataFrame is empty; nothing to plot.")

    value_array = dataframe[value_col].values
    value_min, value_max = float(value_array.min()), float(value_array.max())
    z_grid = np.linspace(value_min, value_max, n_z)

    grouped = dataframe.groupby([param1, param2])[value_col]

    densities: Dict[Tuple[float, float], np.ndarray] = {}
    global_max_density = 0.0
    mean_values: Dict[Tuple[float, float], float] = {}

    for (param1_value, param2_value), series in grouped:
        values = series.to_numpy(dtype=float)
        if values.size < 2:
            continue
        key = (float(param1_value), float(param2_value))

        # Handle singular/near-constant data: if variance is ~0, fall back to a narrow spike.
        std_val = float(values.std())
        if std_val <= 1e-12:
            mean_val = float(values.mean())
            eps = 1e-6
            density = np.exp(-0.5 * ((z_grid - mean_val) / eps) ** 2)
        else:
            try:
                kernel = gaussian_kde(values, bw_method=bandwidth)
                density = kernel(z_grid)
            except np.linalg.LinAlgError:
                # Fallback: add tiny jitter to break covariance singularity
                jittered = values + np.random.normal(scale=1e-9, size=values.shape)
                kernel = gaussian_kde(jittered, bw_method=bandwidth)
                density = kernel(z_grid)
        densities[key] = density
        mean_values[key] = float(values.mean())
        local_max = float(density.max())
        if local_max > global_max_density:
            global_max_density = local_max

    if global_max_density == 0.0:
        raise ValueError("All densities are zero; check your input data.")

    param1_unique = np.sort(dataframe[param1].unique().astype(float))
    param2_unique = np.sort(dataframe[param2].unique().astype(float))

    if len(param1_unique) > 1:
        spacing_x = float(np.min(np.diff(param1_unique)))
    else:
        spacing_x = 1.0
    if len(param2_unique) > 1:
        spacing_y = float(np.min(np.diff(param2_unique)))
    else:
        spacing_y = 1.0

    base_spacing = min(spacing_x, spacing_y)
    max_radius = max_radius_factor * base_spacing

    angles = np.linspace(0.0, 2.0 * np.pi, n_theta)

    fig = go.Figure()

    # Optional mean surface connecting the average value per (param1, param2).
    if add_mean_surface and mean_values:
        x_grid = param1_unique
        y_grid = param2_unique
        z_mean = np.full((y_grid.size, x_grid.size), np.nan, dtype=float)
        for (p1_val, p2_val), mean_val in mean_values.items():
            x_idx = np.where(x_grid == p1_val)[0]
            y_idx = np.where(y_grid == p2_val)[0]
            if x_idx.size == 0 or y_idx.size == 0:
                continue
            z_mean[y_idx[0], x_idx[0]] = mean_val
        fig.add_surface(
            x=x_grid,
            y=y_grid,
            z=z_mean,
            colorscale="Viridis",
            showscale=True,
            opacity=0.85,
            name="Mean surface",
        )

    for (param1_value, param2_value), density in densities.items():
        radii = max_radius * (density / global_max_density)

        z_matrix = np.repeat(z_grid[:, None], n_theta, axis=1)
        radius_matrix = np.repeat(radii[:, None], n_theta, axis=1)
        cosine = np.cos(angles)[None, :]
        sine = np.sin(angles)[None, :]

        x_matrix = param1_value + radius_matrix * cosine
        y_matrix = param2_value + radius_matrix * sine

        fig.add_surface(
            x=x_matrix,
            y=y_matrix,
            z=z_matrix,
            showscale=False,
            opacity=0.7,
        )

    # Match seaborn's default "darkgrid" feel (as in sns.set_theme()).
    light_bg = "rgb(234,234,242)"  # ~ #EAEAF2
    tick_font_size = 12
    label_font_size = 14
    title_font_size = 15
    axis_style = dict(
        showbackground=True,
        backgroundcolor=light_bg,
        gridcolor="white",
        zerolinecolor="white",
        showgrid=True,
        tickfont=dict(color="#2a3f5f", size=tick_font_size),
    )

    fig.update_layout(
        title_text=title or "",
        title_font=dict(size=title_font_size, color="#2a3f5f"),
        template="seaborn",
        paper_bgcolor="white",
        font=dict(color="#2a3f5f", size=tick_font_size),
        scene=dict(
            xaxis=dict(
                axis_style,
                title=dict(text=param1_label or param1, font=dict(color="#2a3f5f", size=label_font_size)),
            ),
            yaxis=dict(
                axis_style,
                title=dict(text=param2_label or param2, font=dict(color="#2a3f5f", size=label_font_size)),
            ),
            zaxis=dict(
                axis_style,
                title=dict(text=z_label or value_col, font=dict(color="#2a3f5f", size=label_font_size)),
            ),
            bgcolor=light_bg,
        ),
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel 2D grid runner producing 3D violin plots for LP PnLs and fee. "
            "The swept axes are configured via GRID_MODE / GRID_AXIS_X / GRID_AXIS_Y / GRID_ALL_AXES at the top of this file."
        )
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: cpu_count).",
    )
    parser.add_argument(
        "--fee-mode",
        type=str,
        default=None,
        choices=sorted(FEE_MODE_CONFIG.keys()),
        help="Fee mode to sweep (default: sweep all fee modes in FEE_MODE_CONFIG).",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default=None,
        help=(
            "Comma-separated list of smart/noise ratios to sweep, "
            "e.g. '0.1,0.2,0.5'. Only used when the selected grid includes axis 'ratio' "
            "and GRID_SWEEP_VALUES does not override it."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved 2D grid (axes, values, filenames) and exit without running simulations.",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--use-cache",
        action="store_true",
        help="Load cached CSV data and regenerate plots without running simulations.",
    )
    cache_group.add_argument(
        "--recompute",
        action="store_true",
        help="Force re-running simulations and overwriting cached CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recompute = RECOMPUTE
    if args.use_cache:
        recompute = False
    elif args.recompute:
        recompute = True

    scenario_label, base_params = load_simulation_parameters(BASE_CONFIG_PATH, simulate_func=simulate)
    base_params = dict(base_params)
    base_params["results_root"] = OUTPUT_ROOT
    theta_T_value = base_params.get("theta_T", None)
    theta_T_label = "na" if theta_T_value is None else f"{float(theta_T_value):g}"

    pnl_keys = resolve_pnl_keys(base_params)
    include_passive_lp_pnl = PNL_PASSIVE_KEY in pnl_keys
    include_active_lp_pnl = PNL_ACTIVE_KEY in pnl_keys

    base_smart = float(base_params.get("smart_trades_per_block", 1.0))
    base_noise = float(base_params.get("noise_trades_per_block", 10.0))
    base_ratio = base_smart / base_noise if base_noise > 0.0 else 0.0
    axis_pairs = _resolve_axis_pairs()
    for axis_x, axis_y in axis_pairs:
        _validate_grid_axes(axis_x, axis_y)

    fee_modes = [args.fee_mode] if args.fee_mode else list(FEE_MODE_CONFIG.keys())
    sigma_label = _sigma_label_from_params(base_params)

    if args.dry_run:
        for fee_mode in fee_modes:
            if fee_mode not in FEE_MODE_CONFIG:
                raise SystemExit(
                    f"Fee mode '{fee_mode}' not in FEE_MODE_CONFIG; "
                    f"valid options: {', '.join(sorted(FEE_MODE_CONFIG.keys()))}"
                )
            fee_cfg = FEE_MODE_CONFIG[fee_mode]
            for axis_x, axis_y in axis_pairs:
                axis_x_values = sorted(
                    set(
                        _resolve_axis_values(
                            axis_x,
                            base_params=base_params,
                            fee_mode=fee_mode,
                            args=args,
                            base_ratio=base_ratio,
                        )
                    )
                )
                axis_y_values = sorted(
                    set(
                        _resolve_axis_values(
                            axis_y,
                            base_params=base_params,
                            fee_mode=fee_mode,
                            args=args,
                            base_ratio=base_ratio,
                        )
                    )
                )
                axis_x_label = _grid_axis_label(axis_x, fee_mode)
                axis_y_label = _grid_axis_label(axis_y, fee_mode)
                base_filename = _grid_output_basename(
                    scenario_label=scenario_label,
                    fee_mode=fee_mode,
                    sigma_label=sigma_label,
                    theta_T_label=theta_T_label,
                    axis_x=axis_x,
                    axis_y=axis_y,
                )
                total_points = len(axis_x_values) * len(axis_y_values)
                print(
                    f"[DRY RUN] 2D grid (x={axis_x}, y={axis_y}):\n"
                    f"  base scenario: {scenario_label} (output -> {OUTPUT_ROOT})\n"
                    f"  sigma profile: {sigma_label}\n"
                    f"  fee_mode: {fee_mode} (sensitivity_param={fee_cfg['param_name']})\n"
                    f"  axis_x: {axis_x} ({axis_x_label}) values: {_format_axis_values(axis_x_values)}\n"
                    f"  axis_y: {axis_y} ({axis_y_label}) values: {_format_axis_values(axis_y_values)}\n"
                    f"  runs per point: {RUNS_PER_POINT}\n"
                    f"  total grid points: {total_points}\n"
                    f"  output basename: {base_filename}\n"
                )
        return

    # Prepare output directories (global only).
    png_dir_global = PLOTS_3D_ROOT / "png"
    html_dir_global = PLOTS_3D_ROOT / "html"
    png_dir_global.mkdir(parents=True, exist_ok=True)
    html_dir_global.mkdir(parents=True, exist_ok=True)

    # Data cache directories
    data_dir_global = PLOTS_3D_ROOT / "data"
    data_dir_global.mkdir(parents=True, exist_ok=True)

    for fee_mode in fee_modes:
        if fee_mode not in FEE_MODE_CONFIG:
            raise SystemExit(
                f"Fee mode '{fee_mode}' not in FEE_MODE_CONFIG; "
                f"valid options: {', '.join(sorted(FEE_MODE_CONFIG.keys()))}"
            )
        fee_cfg = FEE_MODE_CONFIG[fee_mode]

        for axis_x, axis_y in axis_pairs:
            base_filename = _grid_output_basename(
                scenario_label=scenario_label,
                fee_mode=fee_mode,
                sigma_label=sigma_label,
                theta_T_label=theta_T_label,
                axis_x=axis_x,
                axis_y=axis_y,
            )
            data_passive_path = data_dir_global / f"{base_filename}_passive.csv"
            data_active_path = data_dir_global / f"{base_filename}_active.csv"
            data_fee_path = data_dir_global / f"{base_filename}_fee.csv"

            axis_x_label = _grid_axis_label(axis_x, fee_mode)
            axis_y_label = _grid_axis_label(axis_y, fee_mode)

            if not recompute:
                # Load cached data
                if not data_fee_path.exists():
                    raise FileNotFoundError(
                        f"Cached data not found for fee_mode '{fee_mode}'. "
                        f"Expected at least {data_fee_path} (run with --recompute to regenerate)."
                    )
                dataframe_passive = pd.read_csv(data_passive_path) if data_passive_path.exists() else pd.DataFrame()
                dataframe_active = pd.read_csv(data_active_path) if data_active_path.exists() else pd.DataFrame()
                dataframe_fee = pd.read_csv(data_fee_path)
            else:
                axis_x_values = sorted(
                    set(
                        _resolve_axis_values(
                            axis_x,
                            base_params=base_params,
                            fee_mode=fee_mode,
                            args=args,
                            base_ratio=base_ratio,
                        )
                    )
                )
                axis_y_values = sorted(
                    set(
                        _resolve_axis_values(
                            axis_y,
                            base_params=base_params,
                            fee_mode=fee_mode,
                            args=args,
                            base_ratio=base_ratio,
                        )
                    )
                )

                grid = build_grid_2d(
                    base_params=base_params,
                    fee_mode=fee_mode,
                    axis_x=axis_x,
                    axis_y=axis_y,
                    axis_x_values=axis_x_values,
                    axis_y_values=axis_y_values,
                )

                total_points = len(grid)
                if total_points == 0:
                    print(f"[{fee_mode} | {axis_x}×{axis_y}] No grid points constructed; skipping.")
                    continue

                print(
                    "2D grid parameters:\n"
                    f"  base scenario: {scenario_label} (output -> {OUTPUT_ROOT})\n"
                    f"  sigma profile: {sigma_label}\n"
                    f"  fee_mode: {fee_mode} (sensitivity_param={fee_cfg['param_name']})\n"
                    f"  axis_x: {axis_x} ({axis_x_label}) values: {_format_axis_values(axis_x_values)}\n"
                    f"  axis_y: {axis_y} ({axis_y_label}) values: {_format_axis_values(axis_y_values)}\n"
                    f"  runs per point: {RUNS_PER_POINT}\n"
                    f"  passive LP PnL included: {'yes' if include_passive_lp_pnl else 'no'}\n"
                    f"  active LP PnL included: {'yes' if include_active_lp_pnl else 'no'}\n"
                    f"  total grid points: {total_points}\n"
                )

                point_results: List[Dict[str, Any]] = []
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(_evaluate_point_2d, point, base_params, pnl_keys): point
                        for point in grid
                    }
                    for future in tqdm(
                        as_completed(futures),
                        total=total_points,
                        desc=f"Grid points ({fee_mode} | {axis_x}×{axis_y})",
                        unit="point",
                    ):
                        result = future.result()
                        point_results.append(result)

                passive_rows: List[Dict[str, float]] = []
                active_rows: List[Dict[str, float]] = []
                fee_rows: List[Dict[str, float]] = []

                for result in point_results:
                    point = result["point"]
                    pnl_samples = result["pnl_samples"]
                    fee_samples = result["fee_samples"]

                    x_value = float(point.x_value)
                    y_value = float(point.y_value)

                    if include_passive_lp_pnl and "lp_pnl_passive" in pnl_samples:
                        for pnl_value in pnl_samples["lp_pnl_passive"]:
                            passive_rows.append(
                                {
                                    axis_x: x_value,
                                    axis_y: y_value,
                                    "value": float(pnl_value),
                                }
                            )
                    if include_active_lp_pnl and "lp_pnl_active" in pnl_samples:
                        for pnl_value in pnl_samples["lp_pnl_active"]:
                            active_rows.append(
                                {
                                    axis_x: x_value,
                                    axis_y: y_value,
                                    "value": float(pnl_value),
                                }
                            )
                    for fee_value in fee_samples:
                        fee_rows.append(
                            {
                                axis_x: x_value,
                                axis_y: y_value,
                                "fee": float(fee_value),
                            }
                        )

                dataframe_passive = pd.DataFrame(passive_rows)
                dataframe_active = pd.DataFrame(active_rows)
                dataframe_fee = pd.DataFrame(fee_rows)

                # Cache data to CSV (global)
                if not dataframe_passive.empty:
                    dataframe_passive.to_csv(data_passive_path, index=False)
                if not dataframe_active.empty:
                    dataframe_active.to_csv(data_active_path, index=False)
                if not dataframe_fee.empty:
                    dataframe_fee.to_csv(data_fee_path, index=False)

            figures_to_show: List[go.Figure] = []

            if include_passive_lp_pnl and not dataframe_passive.empty:
                title_passive = f"[{fee_mode}] Passive LP hedged PnL — 2D {axis_x_label} × {axis_y_label}"
                figure_passive = plot_3d_violin_grid(
                    dataframe=dataframe_passive,
                    param1=axis_x,
                    param2=axis_y,
                    value_col="value",
                    param1_label=axis_x_label,
                    param2_label=axis_y_label,
                    z_label="Final hedged PnL",
                    title=title_passive,
                )
                figures_to_show.append(figure_passive)

                filename_passive = f"{base_filename}_lp_passive_pnl"
                png_path_global_passive = png_dir_global / f"{filename_passive}.png"
                html_path_global_passive = html_dir_global / f"{filename_passive}.html"
                save_plotly_figure(
                    figure_passive,
                    png_path_global_passive,
                    html_path_global_passive,
                    source="grid_2d_violin",
                )

            if include_active_lp_pnl and not dataframe_active.empty:
                title_active = f"[{fee_mode}] Active LP hedged PnL — 2D {axis_x_label} × {axis_y_label}"
                figure_active = plot_3d_violin_grid(
                    dataframe=dataframe_active,
                    param1=axis_x,
                    param2=axis_y,
                    value_col="value",
                    param1_label=axis_x_label,
                    param2_label=axis_y_label,
                    z_label="Final hedged PnL",
                    title=title_active,
                )
                figures_to_show.append(figure_active)

                filename_active = f"{base_filename}_lp_active_pnl"
                png_path_global_active = png_dir_global / f"{filename_active}.png"
                html_path_global_active = html_dir_global / f"{filename_active}.html"
                save_plotly_figure(
                    figure_active,
                    png_path_global_active,
                    html_path_global_active,
                    source="grid_2d_violin",
                )

            if not dataframe_fee.empty:
                title_fee = f"[{fee_mode}] Fee level — 2D {axis_x_label} × {axis_y_label}"
                figure_fee = plot_3d_violin_grid(
                    dataframe=dataframe_fee,
                    param1=axis_x,
                    param2=axis_y,
                    value_col="fee",
                    param1_label=axis_x_label,
                    param2_label=axis_y_label,
                    z_label="Fee",
                    title=title_fee,
                )
                figures_to_show.append(figure_fee)

                filename_fee = f"{base_filename}_fee"
                png_path_global_fee = png_dir_global / f"{filename_fee}.png"
                html_path_global_fee = html_dir_global / f"{filename_fee}.html"
                save_plotly_figure(
                    figure_fee,
                    png_path_global_fee,
                    html_path_global_fee,
                    source="grid_2d_violin",
                )

            # Optionally display figures interactively (separate windows/tabs).
            # for fig in figures_to_show:
            #     fig.show()

            print(
                f"[{fee_mode} | {axis_x}×{axis_y}] 3D violin PNGs written to {png_dir_global}\n"
                f"[{fee_mode} | {axis_x}×{axis_y}] 3D violin HTML files written to {html_dir_global}"
            )


if __name__ == "__main__":
    main()
