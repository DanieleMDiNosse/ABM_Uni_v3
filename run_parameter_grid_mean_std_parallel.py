#!/usr/bin/env python3
"""
Parallel grid runner: sweep fee modes/sensitivities using the base YAML config
for sigma (static or regime), aggregate PnL/fee distributions, and plot violins.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import run as run_module
from run_scenarios_mean_std import SERIES_DEFS, _slice_series, aggregate_runs
from utils import load_simulation_parameters

# Silence the tqdm progress bar inside run.simulate to avoid nested bars
def _silent_tqdm(iterable=None, **kwargs):
    if iterable is None:
        total = int(kwargs.get("total", 0))
        return range(total)
    return iterable
run_module.tqdm = _silent_tqdm  # type: ignore[attr-defined]
simulate = run_module.simulate

PNL_KEYS = [
    ("arb_pnl_cum", "Arbitrageur PnL"),
    ("lp_pnl_passive", "Passive LP hedged PnL"),
]
PNL_COLORS = {
    "arb_pnl_cum": "#2ca02c",
    "lp_pnl_passive": "#8c564b",
}

# --- configuration -----------------------------------------------------------
BASE_CONFIG_PATH = Path("tests/test.yml")
FEE_MODE_CONFIG = {
    "static": {
        "param_name": None,
        "values": [None],           # constant fee
        "xlabel": "Static fee",
    },
    "volatility": {
        "param_name": "k_sigma",
        "values": np.linspace(1e-2, 10, 10),
        "xlabel": "k_sigma (volatility sensitivity)",
    },
    "toxicity": {
        "param_name": "k_basis",
        "values": np.linspace(1e-5, 1e-1, 10),
        "xlabel": "k_basis (toxicity sensitivity)",
    },
    # "gas": {
    #     "param_name": "k_gas_sigma",
    #     "values": np.linspace(1e-2, 10, 10),
    #     "xlabel": "k_gas_sigma (GAS volatility sensitivity)",
    # },
}

RUNS_PER_POINT = 30
SEED_BASE = 1
OUTPUT_DIR = Path("abm_results") / "grid_search"
SUMMARY_CSV = OUTPUT_DIR / "grid_summary.csv"
PLOTS_DIR = OUTPUT_DIR / "plots"

SAVE_SERIES = False        # set True if you still want per-point NPZ files
KEEP_VISUALS = False       # pass through to simulate()
VERBOSE_RUNS = False       # set True to print per-run progress inside workers
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GridPoint:
    fee_mode: str
    cex_sigma_label: str
    param_name: Optional[str]
    sensitivity: Optional[float]
    run_seed_base: int

    def label(self) -> str:
        if self.param_name is None:
            return f"fee={self.fee_mode}, sigma={self.cex_sigma_label}"
        return f"fee={self.fee_mode}, sigma={self.cex_sigma_label}, {self.param_name}={self.sensitivity}"

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
            f"sigma-{_fmt(self.cex_sigma_label)}_"
            f"{self.param_name or 'const'}-{_fmt(self.sensitivity)}"
        )


def _sigma_label_from_params(params: Dict[str, Any]) -> str:
    mode = str(params.get("cex_sigma_mode", "static")).lower()
    if mode.startswith("regime"):
        lo = params.get("cex_sigma_low")
        hi = params.get("cex_sigma_high")
        return f"regime_{lo}-{hi}"
    if mode == "noisy_sine":
        center = params.get("cex_sigma")
        amp = params.get("cex_sigma_sine_amp")
        if amp is None and params.get("cex_sigma_low") is not None and params.get("cex_sigma_high") is not None:
            amp = 0.5 * abs(float(params["cex_sigma_high"]) - float(params["cex_sigma_low"]))
            center = 0.5 * (float(params["cex_sigma_low"]) + float(params["cex_sigma_high"]))
        noise = params.get("cex_sigma_sine_noise", 0.0)
        period = params.get("cex_sigma_sine_period", "p")
        amp_label = "auto" if amp is None else amp
        return f"noisy_sine_{center}_amp{amp_label}_per{period}_noise{noise}"
    return str(params.get("cex_sigma"))


def _format_values(values: Sequence[Any]) -> str:
    formatted: List[str] = []
    for val in values:
        if val is None:
            formatted.append("const")
        elif isinstance(val, (float, int, np.floating)):
            formatted.append(f"{float(val):.4g}")
        else:
            formatted.append(str(val))
    return ", ".join(formatted)


def build_grid(base_params: Dict[str, Any]) -> List[GridPoint]:
    grid: List[GridPoint] = []
    seed_cursor = SEED_BASE
    sigma_label = _sigma_label_from_params(base_params)
    for fee_mode, cfg in FEE_MODE_CONFIG.items():
        for value in cfg["values"]:
            grid.append(
                GridPoint(
                    fee_mode=fee_mode,
                    cex_sigma_label=sigma_label,
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
) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]], Dict[str, Any], Dict[str, List[float]], List[float]]:
    params = dict(base_params)
    params.update(fee_mode=point.fee_mode)
    if point.param_name and point.sensitivity is not None:
        params[point.param_name] = point.sensitivity
    if not KEEP_VISUALS:
        params["visualize"] = False

    skip_step = max(0, int(params.get("skip_step", 0)))
    series_data = {key: [] for key, *_ in SERIES_DEFS}
    series_data["fee_series"] = []
    pnl_samples: Dict[str, List[float]] = {key: [] for key, _ in PNL_KEYS}
    fee_samples: List[float] = []
    steps_ref: Optional[np.ndarray] = None

    for idx in range(RUNS_PER_POINT):
        seed = point.run_seed_base + idx
        params["seed"] = seed
        if VERBOSE_RUNS:
            print(f"[grid:{point.label()}] run {idx + 1}/{RUNS_PER_POINT} (seed={seed})")
        out = simulate(**params)

        reference = _slice_series(out[SERIES_DEFS[0][0]], skip_step)
        if reference.size == 0:
            raise ValueError("Empty PnL series after skip_step.")
        if steps_ref is None:
            total_len = len(out[SERIES_DEFS[0][0]])
            steps_ref = np.arange(total_len, dtype=int)[skip_step:]
        for key in series_data:
            series = _slice_series(out[key], skip_step)
            if series.size != reference.size:
                raise ValueError(f"Series '{key}' length mismatch.")
            series_data[key].append(series)
        for key, _ in PNL_KEYS:
            series = _slice_series(out[key], skip_step)
            if series.size == 0:
                raise ValueError(f"Series '{key}' empty after skip_step.")
            pnl_samples[key].append(float(series[-1]))
        fee_series = _slice_series(out["fee_series"], skip_step)
        fee_samples.extend(fee_series.tolist())

    assert steps_ref is not None
    metadata = {
        "fee_mode": point.fee_mode,
        "cex_sigma_label": point.cex_sigma_label,
        "param_name": point.param_name,
        "sensitivity": point.sensitivity,
        "runs": RUNS_PER_POINT,
        "skip_step": skip_step,
        "steps_len": int(steps_ref.size),
    }
    return steps_ref, series_data, metadata, pnl_samples, fee_samples


def summarise(point: GridPoint, stats: Dict[str, tuple[np.ndarray, np.ndarray]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, label, _ in SERIES_DEFS:
        mean_series, std_series = stats[key]
        rows.append(
            {
                "fee_mode": point.fee_mode,
                "cex_sigma_label": point.cex_sigma_label,
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


def plot_per_sigma(plot_data: Sequence[Dict[str, Any]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    sigma_values = sorted({entry["sigma_label"] for entry in plot_data})
    for sigma_value in sigma_values:
        fig, axes = plt.subplots(
            2,
            len(FEE_MODE_CONFIG),
            figsize=(6 * len(FEE_MODE_CONFIG), 6),
            sharex="col",
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
        if len(FEE_MODE_CONFIG) == 1:
            axes = np.array([[axes[0]], [axes[1]]])  # ensure 2D indexing when len=1

        for col_idx, (fee_mode, cfg) in enumerate(FEE_MODE_CONFIG.items()):
            ax_pnl = axes[0, col_idx]
            ax_fee = axes[1, col_idx]

            entries = [
                entry for entry in plot_data
                if entry["sigma_label"] == sigma_value and entry["fee_mode"] == fee_mode
            ]
            if not entries:
                ax_pnl.set_title(f"{fee_mode} (no data)")
                ax_pnl.axis("off")
                ax_fee.axis("off")
                continue

            sens_values = list(cfg["values"])
            labels = ["const" if v is None else str(v)[:6] for v in sens_values]
            pos_base = np.arange(len(sens_values)) + 1
            width = 0.18

            for offset, (key, label) in zip([-0.12, 0.12], PNL_KEYS):
                positions = pos_base + offset
                samples_per_sens = []
                for sens in sens_values:
                    target = next((e for e in entries if e["sensitivity"] == sens and e["param_name"] == cfg["param_name"]), None)
                    samples_per_sens.append(target["pnl_samples"][key] if target else [])
                for pos, samples in zip(positions, samples_per_sens):
                    if not samples:
                        continue
                    vp = ax_pnl.violinplot(samples, positions=[pos], widths=width, showmeans=True, showextrema=False)
                    for pc in vp["bodies"]:
                        pc.set_facecolor(PNL_COLORS.get(key, "#444"))
                        pc.set_alpha(0.4)
                    vp["cmeans"].set_color(PNL_COLORS.get(key, "#444"))
                    vp["cmeans"].set_linewidth(1.4)
                ax_pnl.plot([], [], color=PNL_COLORS.get(key, "#444"), label=label)

            # ax_pnl.set_title(f"{fee_mode} — sigma={sigma_value}")
            ax_pnl.set_xticks(pos_base, labels)
            ax_pnl.grid(True, alpha=0.3)
            ax_pnl.legend(fontsize=8)

            for pos, sens in zip(pos_base, sens_values):
                entry = next((e for e in entries if e["sensitivity"] == sens and e["param_name"] == cfg["param_name"]), None)
                if entry is None or not entry["fee_samples"]:
                    continue
                vp = ax_fee.violinplot(entry["fee_samples"], positions=[pos], widths=0.35, showmeans=True, showextrema=False)
                for pc in vp["bodies"]:
                    pc.set_facecolor("#1f77b4")
                    pc.set_alpha(0.4)
                vp["cmeans"].set_color("#1f77b4")
                vp["cmeans"].set_linewidth(1.4)
            ax_fee.set_xticks(pos_base, labels, rotation=90)
            ax_fee.set_xlabel(cfg["xlabel"])
            ax_fee.set_ylabel("Fee")
            ax_fee.grid(True, alpha=0.3)

        axes[0, 0].set_ylabel("Final PnL")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"sigma_{sigma_value}.png", dpi=200)
        plt.close(fig)


def _evaluate_point(point: GridPoint, base_params: Dict[str, Any]) -> Dict[str, Any]:
    steps, series_data, metadata, pnl_samples, fee_samples = run_grid_point(base_params, point)
    stats = aggregate_runs(series_data)
    summary_rows = summarise(point, stats, metadata)
    plot_entry = {
        "fee_mode": point.fee_mode,
        "sigma_label": point.cex_sigma_label,
        "param_name": point.param_name,
        "sensitivity": point.sensitivity,
        "pnl_samples": pnl_samples,
        "fee_samples": fee_samples,
    }
    return {
        "point": point,
        "steps": steps,
        "stats": stats,
        "summary_rows": summary_rows,
        "plot_entry": plot_entry,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel grid search over fee modes/sensitivities.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: cpu_count).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_label, base_params = load_simulation_parameters(BASE_CONFIG_PATH, simulate_func=simulate)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    series_dir = OUTPUT_DIR / "npz"
    plot_data: List[Dict[str, Any]] = []

    grid = build_grid(base_params)
    total = len(grid)
    fee_lines = []
    for fee_mode, cfg in FEE_MODE_CONFIG.items():
        param_label = cfg["param_name"] or "constant"
        values_label = _format_values(cfg["values"])
        fee_lines.append(f"    - {fee_mode}: {param_label} = [{values_label}]")
    fee_block = "\n".join(fee_lines)
    print(
        "Grid search parameters:\n"
        f"  base scenario: {scenario_label}\n"
        f"  sigma profile: {_sigma_label_from_params(base_params)}\n"
        "  fee_mode from YAML will be overridden\n"
        f"  combinations: {total} ({len(FEE_MODE_CONFIG)} fee modes)\n"
        f"  runs per point: {RUNS_PER_POINT}\n"
        "  sweeps:\n"
        f"{fee_block}"
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_evaluate_point, point, base_params): point
            for point in grid
        }
        for fut in tqdm(as_completed(futures), total=total, desc="Grid combinations", unit="combo"):
            result = fut.result()
            summary_rows.extend(result["summary_rows"])
            plot_data.append(result["plot_entry"])

            if SAVE_SERIES:
                point = result["point"]
                steps = result["steps"]
                stats = result["stats"]
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
                "cex_sigma_label",
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

    plot_per_sigma(plot_data)
    print(f"Summary CSV written to {SUMMARY_CSV}")
    print(f"Plots written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
