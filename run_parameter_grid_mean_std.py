#!/usr/bin/env python3
"""
Sweep cex_sigma × controller-sensitivity for each fee_mode and plot final PnL.

For every combination:
  • run `RUNS_PER_POINT` simulations with distinct seeds,
  • aggregate the mean/std of each agent’s final PnL,
  • write the summary rows to CSV,
  • for each cex_sigma produce one figure with three subplots
    (static, volatility, toxicity) plotting mean ± std versus sensitivity.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import run as run_module
from run_scenarios_mean_std import SERIES_DEFS, _slice_series, aggregate_runs
from utils import load_simulation_parameters

# Silence the tqdm progress bar inside run.simulate to avoid nested bars.


def _silent_tqdm(iterable=None, **kwargs):
    if iterable is None:
        total = int(kwargs.get("total", 0))
        return range(total)
    return iterable


run_module.tqdm = _silent_tqdm  # type: ignore[attr-defined]
simulate = run_module.simulate


# --- configuration -----------------------------------------------------------
BASE_CONFIG_PATH = Path("tests/test.yml")

CEX_SIGMAS = [1e-5, 1e-4, 1e-3, 1e-2]
FEE_MODE_CONFIG = {
    "static": {
        "param_name": None,
        "values": [None],           # constant fee
        "xlabel": "Static fee",
    },
    "volatility": {
        "param_name": "k_sigma",
        "values": [1, 10, 25, 50, 100],
        "xlabel": "k_sigma (volatility sensitivity)",
    },
    # "toxicity": {
    #     "param_name": "k_basis",
    #     "values": [1e-5, 1e-4, 1e-3, 1e-2],
    #     "xlabel": "k_basis (toxicity sensitivity)",
    # },
}

RUNS_PER_POINT = 5
SEED_BASE = 1
OUTPUT_DIR = Path("abm_results") / "grid_search"
SUMMARY_CSV = OUTPUT_DIR / "grid_summary.csv"
PLOTS_DIR = OUTPUT_DIR / "plots"

SAVE_SERIES = False        # set True if you still want per-point NPZ files
KEEP_VISUALS = False       # pass through to simulate()
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GridPoint:
    fee_mode: str
    cex_sigma: float
    param_name: Optional[str]
    sensitivity: Optional[float]
    run_seed_base: int

    def label(self) -> str:
        if self.param_name is None:
            return f"fee={self.fee_mode}, cex_sigma={self.cex_sigma}"
        return f"fee={self.fee_mode}, cex_sigma={self.cex_sigma}, {self.param_name}={self.sensitivity}"

    def slug(self) -> str:
        def _fmt(value: Any) -> str:
            if value is None:
                return "const"
            if isinstance(value, str):
                return value
            text = f"{value:.6g}"
            return text.replace("-", "m").replace(".", "p")
        return (
            f"fee-{_fmt(self.fee_mode)}_"
            f"cexsigma-{_fmt(self.cex_sigma)}_"
            f"{self.param_name or 'const'}-{_fmt(self.sensitivity)}"
        )


def build_grid() -> List[GridPoint]:
    grid: List[GridPoint] = []
    seed_cursor = SEED_BASE
    for fee_mode, cfg in FEE_MODE_CONFIG.items():
        for sigma in CEX_SIGMAS:
            for value in cfg["values"]:
                grid.append(
                    GridPoint(
                        fee_mode=fee_mode,
                        cex_sigma=float(sigma),
                        param_name=cfg["param_name"],
                        sensitivity=None if value is None else float(value),
                        run_seed_base=seed_cursor,
                    )
                )
                seed_cursor += RUNS_PER_POINT
    return grid


def run_grid_point(
    base_params: Dict[str, Any],
    point: GridPoint,
) -> tuple[np.ndarray, Dict[str, List[np.ndarray]], Dict[str, Any]]:
    params = dict(base_params)
    params.update(
        fee_mode=point.fee_mode,
        cex_sigma=point.cex_sigma,
    )
    if point.param_name and point.sensitivity is not None:
        params[point.param_name] = point.sensitivity
    if not KEEP_VISUALS:
        params["visualize"] = False

    skip_step = max(0, int(params.get("skip_step", 0)))
    series_data = {key: [] for key, *_ in SERIES_DEFS}
    steps_ref: Optional[np.ndarray] = None

    for idx in range(RUNS_PER_POINT):
        seed = point.run_seed_base + idx
        params["seed"] = seed
        print(f"[grid:{point.label()}] run {idx + 1}/{RUNS_PER_POINT} (seed={seed})")
        out = simulate(**params)

        reference = _slice_series(out[SERIES_DEFS[0][0]], skip_step)
        if reference.size == 0:
            raise ValueError("Empty PnL series after skip_step.")
        if steps_ref is None:
            total_len = len(out[SERIES_DEFS[0][0]])
            steps_ref = np.arange(total_len, dtype=int)[skip_step:]
        for key, _, _ in SERIES_DEFS:
            series = _slice_series(out[key], skip_step)
            if series.size != reference.size:
                raise ValueError(f"Series '{key}' length mismatch.")
            series_data[key].append(series)

    assert steps_ref is not None
    metadata = {
        "fee_mode": point.fee_mode,
        "cex_sigma": point.cex_sigma,
        "param_name": point.param_name,
        "sensitivity": point.sensitivity,
        "runs": RUNS_PER_POINT,
        "skip_step": skip_step,
        "steps_len": int(steps_ref.size),
    }
    return steps_ref, series_data, metadata


def summarise(point: GridPoint, stats: Dict[str, tuple[np.ndarray, np.ndarray]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, label, _ in SERIES_DEFS:
        mean_series, std_series = stats[key]
        rows.append(
            {
                "fee_mode": point.fee_mode,
                "cex_sigma": point.cex_sigma,
                "param_name": point.param_name or "constant",
                "sensitivity": point.sensitivity if point.param_name else 0.0,
                "series_key": key,
                "series_label": label,
                "mean_final_pnl": float(mean_series[-1]),
                "std_final_pnl": float(std_series[-1]),
                "runs": metadata["runs"],
                "skip_step": metadata["skip_step"],
                "steps": metadata["steps_len"],
            }
        )
    return rows


def plot_per_sigma(summary_rows: Sequence[Dict[str, Any]]) -> None:
    df = pd.DataFrame(summary_rows)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for sigma_value, sigma_df in df.groupby("cex_sigma"):
        fig, axes = plt.subplots(1, len(FEE_MODE_CONFIG), figsize=(6 * len(FEE_MODE_CONFIG), 4), sharey=True)
        for ax, (fee_mode, cfg) in zip(axes, FEE_MODE_CONFIG.items()):
            sub = sigma_df[sigma_df["fee_mode"] == fee_mode]
            if sub.empty:
                ax.set_title(f"{fee_mode} (no data)")
                ax.axis("off")
                continue
            for series_label in sorted(sub["series_label"].unique()):
                series_block = sub[sub["series_label"] == series_label].sort_values("sensitivity")
                ax.errorbar(
                    series_block["sensitivity"],
                    series_block["mean_final_pnl"],
                    yerr=series_block["std_final_pnl"],
                    label=series_label,
                    marker="o",
                    capsize=4,
                    alpha=0.85,
                )
            ax.set_title(f"{fee_mode} — cex_sigma={sigma_value:g}")
            ax.set_xlabel(cfg["xlabel"])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        axes[0].set_ylabel("Final PnL (mean ± std)")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"cex_sigma_{sigma_value:.6g}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    scenario_label, base_params = load_simulation_parameters(BASE_CONFIG_PATH, simulate_func=simulate)
    print(f"Loaded base scenario '{scenario_label}'")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    series_dir = OUTPUT_DIR / "npz"

    grid = build_grid()
    for point in tqdm(grid, desc="Grid combinations", unit="combo"):
        steps, series_data, metadata = run_grid_point(base_params, point)
        stats = aggregate_runs(series_data)
        summary_rows.extend(summarise(point, stats, metadata))
        if SAVE_SERIES:
            npz_path = series_dir / f"{point.slug()}.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"steps": steps}
            for key, (mean, std) in stats.items():
                payload[f"{key}_mean"] = mean
                payload[f"{key}_std"] = std
            np.savez(npz_path, **payload)

    # write CSV
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fee_mode",
                "cex_sigma",
                "param_name",
                "sensitivity",
                "series_key",
                "series_label",
                "mean_final_pnl",
                "std_final_pnl",
                "runs",
                "skip_step",
                "steps",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_per_sigma(summary_rows)
    print(f"Summary CSV written to {SUMMARY_CSV}")
    print(f"Plots written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
