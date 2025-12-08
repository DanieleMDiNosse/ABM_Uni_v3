#!/usr/bin/env python3
"""
Batch runner that executes every scenario configuration multiple times and
builds aggregated agent PnL plots (mean ± std) per scenario.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

from run import save_plotly_figure, simulate
from utils import load_simulation_parameters, scenario_output_root

SeriesDef = Tuple[str, str, str]

# (output key, legend label, hex color)
SERIES_DEFS: Tuple[SeriesDef, ...] = (
    ("smart_router_pnl_cum", "Smart router PnL", "#1f77b4"),
    ("noise_trader_pnl_cum", "Noise trader PnL", "#ff7f0e"),
    ("arb_pnl_cum", "Arbitrageur PnL", "#2ca02c"),
    ("lp_pnl_active", "Active LP hedged (fees - LVR)", "#9467bd"),
    ("lp_pnl_passive", "Passive LP hedged (fees - LVR)", "#8c564b"),
    ("lp_unhedged_active", "Active LP unhedged", "#c5b0d5"),
    ("lp_unhedged_passive", "Passive LP unhedged", "#c49c94"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run every scenario config multiple times with distinct seeds and "
            "create aggregated PnL charts (mean ± std)."
        )
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=Path("scenarios"),
        help="Directory that contains YAML scenario configs.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="How many times to run each scenario (must be >= 1).",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1,
        help="Base seed used when deriving deterministic seeds per run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("abm_results") / "mean_std",
        help="Root directory used for the aggregated plot outputs.",
    )
    parser.add_argument(
        "--keep-visuals",
        action="store_true",
        help="Keep per-run visualizations instead of forcing visualize=False.",
    )
    return parser.parse_args()


def list_scenarios(dir_path: Path) -> List[Path]:
    patterns = ("*.yml", "*.yaml")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(dir_path.glob(pattern))
    return sorted(p.resolve() for p in files if p.is_file())


def _slice_series(values: Sequence[float], skip: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    skip_clamped = max(0, min(int(skip), arr.size))
    return arr[skip_clamped:]


def _hex_to_rgba(color: str, alpha: float) -> str:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got '{color}'")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _build_fill_trace(
    steps: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    label: str,
) -> go.Scatter:
    upper = mean + std
    lower = mean - std
    x_values = np.concatenate([steps, steps[::-1]])
    y_values = np.concatenate([upper, lower[::-1]])
    return go.Scatter(
        x=x_values.tolist(),
        y=y_values.tolist(),
        fill="toself",
        fillcolor=_hex_to_rgba(color, 0.18),
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
        legendgroup=label,
    )


def _build_mean_trace(
    steps: np.ndarray,
    mean: np.ndarray,
    color: str,
    label: str,
) -> go.Scatter:
    return go.Scatter(
        x=steps.tolist(),
        y=mean.tolist(),
        mode="lines",
        name=label,
        line=dict(color=color),
        legendgroup=label,
    )


def aggregate_runs(data: Dict[str, List[np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key, series_list in data.items():
        if not series_list:
            raise ValueError(f"No data collected for series '{key}'")
        stack = np.vstack(series_list)
        stats[key] = (stack.mean(axis=0), stack.std(axis=0))
    return stats


def render_pnl_figure(
    steps: np.ndarray,
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title_suffix: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"))
    for key, label, color in SERIES_DEFS:
        mean, std = stats[key]
        fig.add_trace(_build_fill_trace(steps, mean, std, color, label))
        fig.add_trace(_build_mean_trace(steps, mean, color, label))
    fig.update_layout(
        template="plotly_white",
        title=f"Agent PnL (mean ± std) — {title_suffix}",
        xaxis_title="Step",
        yaxis_title="Token1 value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def run_scenario(
    config_path: Path,
    runs: int,
    seed_base: int,
    keep_visuals: bool,
) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]], Dict[str, Any]]:
    scenario_label, params = load_simulation_parameters(config_path, simulate_func=simulate)
    run_params = dict(params)
    # Ensure all runs for this scenario share the same per-scenario output root.
    scenario_root = scenario_output_root(config_path)
    run_params["results_root"] = scenario_root
    skip_step = max(0, int(run_params.get("skip_step", 0)))
    if not keep_visuals:
        run_params["visualize"] = False

    series_data: Dict[str, List[np.ndarray]] = {key: [] for key, *_ in SERIES_DEFS}
    steps_reference: np.ndarray | None = None

    for run_idx in range(runs):
        seed = seed_base + run_idx
        run_params["seed"] = seed
        print(
            f"[scenario:{config_path.name}] run {run_idx + 1}/{runs} "
            f"(seed={seed}) — fee_mode={scenario_label} -> {scenario_root}"
        )
        out = simulate(**run_params)

        reference_series = _slice_series(out[SERIES_DEFS[0][0]], skip_step)
        if reference_series.size == 0:
            raise ValueError(f"Scenario '{config_path}' produced empty PnL series after skip_step.")
        if steps_reference is None:
            full_length = len(out[SERIES_DEFS[0][0]])
            steps_reference = np.arange(full_length, dtype=int)[skip_step:]
        for key, _, _ in SERIES_DEFS:
            series = _slice_series(out[key], skip_step)
            if series.size != reference_series.size:
                raise ValueError(
                    f"Series '{key}' length {series.size} does not match reference length {reference_series.size}"
                )
            series_data[key].append(series)

    assert steps_reference is not None
    metadata = {
        "scenario_label": scenario_label,
        "params": params,
        "skip_step": skip_step,
        "steps_len": steps_reference.size,
    }
    return steps_reference, series_data, metadata


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    scenario_files = list_scenarios(args.scenarios_dir)
    if not scenario_files:
        raise SystemExit(f"No scenario configs found in {args.scenarios_dir}")

    for scenario_index, config_path in enumerate(scenario_files):
        per_scenario_seed_base = args.seed_base + scenario_index * args.runs
        steps, series_data, metadata = run_scenario(
            config_path=config_path,
            runs=args.runs,
            seed_base=per_scenario_seed_base,
            keep_visuals=args.keep_visuals,
        )
        stats = aggregate_runs(series_data)
        scenario_suffix = config_path.stem
        fig = render_pnl_figure(steps, stats, scenario_suffix)

        fee_mode = metadata["scenario_label"]
        cex_sigma = metadata["params"]["cex_sigma"]
        total_steps = metadata["steps_len"]
        prefix = f"abm_fee_{fee_mode}_{cex_sigma}_{scenario_suffix}"
        filename = f"{prefix}_6_pnl_meanstd_steps{total_steps}"

        # Global aggregated outputs (kept for backwards compatibility)
        global_png = args.output_dir / "png"
        global_html = args.output_dir / "html"
        global_png.mkdir(parents=True, exist_ok=True)
        global_html.mkdir(parents=True, exist_ok=True)
        save_plotly_figure(
            fig,
            global_png / f"{filename}.png",
            global_html / f"{filename}.html",
            source="mean_std",
        )

        # Scenario-local aggregated outputs under abm_results/scenarios/<scenario>/mean_std
        scenario_root = scenario_output_root(config_path)
        scenario_png = scenario_root / "mean_std" / "png"
        scenario_html = scenario_root / "mean_std" / "html"
        scenario_png.mkdir(parents=True, exist_ok=True)
        scenario_html.mkdir(parents=True, exist_ok=True)
        save_plotly_figure(
            fig,
            scenario_png / f"{filename}.png",
            scenario_html / f"{filename}.html",
            source="mean_std",
        )

        print(
            f"[scenario:{config_path.name}] wrote {filename}.png/.html "
            f"to {args.output_dir} and {scenario_root / 'mean_std'}"
        )


if __name__ == "__main__":
    main()
