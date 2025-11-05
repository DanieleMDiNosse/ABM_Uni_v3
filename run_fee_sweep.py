#!/usr/bin/env python3

"""
Sweep the ABM across fee modes and fee-control parameters, aggregating LP PnL.

For every scenario (static, volatility, toxicity) this script runs several
simulations at different fee-parameter values, collects the end-of-run PnL for
active/passive LP cohorts, and plots the mean ± SEM for each configuration.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from abm_dynamical_fee import SCENARIOS, simulate  # noqa: E402


DEFAULT_STATIC_VALUES: Sequence[float] = (0.0,)
DEFAULT_VOLATILITY_VALUES: Sequence[float] = np.linspace(1,200,40)
# DEFAULT_TOXICITY_VALUES: Sequence[float] = (5e-5, 1e-4, 2e-4, 3e-4, 5e-4)
DEFAULT_TOXICITY_VALUES: Sequence[float] = np.linspace(1e-6,1e-2,40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fee parameter sweeps for the ABM and plot LP PnL (mean ± SEM)."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of random seeds per fee parameter value (default: 5).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of simulation steps T for each run (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for the RNG that draws per-run simulation seeds.",
    )
    parser.add_argument(
        "--static-values",
        type=float,
        nargs="+",
        default=DEFAULT_STATIC_VALUES,
        help="Dummy fee parameter values for the static baseline (default: 0.0).",
    )
    parser.add_argument(
        "--volatility-values",
        type=float,
        nargs="+",
        default=DEFAULT_VOLATILITY_VALUES,
        help="Fee parameter grid for k_sigma in the volatility baseline.",
    )
    parser.add_argument(
        "--toxicity-values",
        type=float,
        nargs="+",
        default=DEFAULT_TOXICITY_VALUES,
        help="Fee parameter grid for k_basis in the toxicity baseline.",
    )
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=None,
        help="Override noise trader execution probability (noise_floor). Must be in [0, 1].",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("abm_results/fee_sweep.png"),
        help="Path to save the summary plot (default: abm_results/fee_sweep.png).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("abm_results/fee_sweep_results.csv"),
        help="Optional CSV dump of the aggregated statistics.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel worker processes to use (default: 10).",
    )
    return parser.parse_args()


def scenario_title(name: str) -> str:
    return {
        "static_baseline": "Static Baseline",
        "volatility_baseline": "Volatility Baseline",
        "toxicity_baseline": "Toxicity Baseline",
    }.get(name, name)


def _simulate_once(
    scenario: str,
    fee_param: Optional[float],
    steps: int,
    seed: int,
    noise_floor_override: Optional[float],
) -> Tuple[int, float, float, float]:
    """Run a single simulation and return final LP wealth components."""
    params = SCENARIOS[scenario].copy()
    params.update(
        {
            "T": steps,
            "seed": int(seed),
            "visualize": False,
            "skip_step": 0,
        }
    )
    if noise_floor_override is not None:
        params["noise_floor"] = noise_floor_override

    if scenario == "volatility_baseline" and fee_param is not None:
        params["k_sigma"] = fee_param
    elif scenario == "toxicity_baseline" and fee_param is not None:
        params["k_basis"] = fee_param

    result = simulate(**params)
    active_final = float(result["lp_wealth_active_series"][-1])
    passive_final = float(result["lp_wealth_passive_series"][-1])
    total_final = float(result["lp_wealth_series"][-1])
    return seed, active_final, passive_final, total_final


def run_single_configuration(
    scenario: str,
    fee_param: Optional[float],
    steps: int,
    rng: np.random.Generator,
    n_runs: int,
    noise_floor_override: Optional[float],
    executor: ProcessPoolExecutor,
    progress_bar: Optional[tqdm],
) -> Tuple[np.ndarray, List[int]]:
    """Run the simulation `n_runs` times (in parallel) and return final LP PnL samples."""
    seeds = [int(s) for s in rng.integers(0, 1_000_000, size=n_runs, dtype=np.int64)]
    futures = {
        executor.submit(
            _simulate_once,
            scenario,
            fee_param,
            steps,
            seed,
            noise_floor_override,
        ): seed
        for seed in seeds
    }

    results = []
    for future in as_completed(futures):
        seed, active_final, passive_final, total_final = future.result()
        results.append((seed, active_final, passive_final, total_final))
        if progress_bar is not None:
            progress_bar.update(1)

    results.sort(key=lambda x: x[0])
    samples = np.asarray([r[1:] for r in results], dtype=float)
    seeds_sorted = [r[0] for r in results]
    return samples, seeds_sorted


def summarise(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, SEM) along axis 0."""
    mean = samples.mean(axis=0)
    if samples.shape[0] > 1:
        sem = samples.std(axis=0, ddof=1) / np.sqrt(samples.shape[0])
    else:
        sem = np.zeros_like(mean)
    return mean, sem


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ensure_parent(args.output)
    ensure_parent(args.csv)

    sweep_plan: Dict[str, Dict[str, Sequence[float]]] = {
        "static_baseline": {"param_name": None, "values": args.static_values},
        "volatility_baseline": {"param_name": "k_sigma", "values": args.volatility_values},
        "toxicity_baseline": {"param_name": "k_basis", "values": args.toxicity_values},
    }

    aggregated_rows: List[str] = [
        "scenario,parameter_name,parameter_value,component,mean,sem,num_runs,seeds"
    ]

    plot_payload = []

    total_iterations = sum(len(conf["values"]) for conf in sweep_plan.values()) * args.runs
    progress = tqdm(total=total_iterations, desc="Running simulations", unit="run")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for scenario, conf in sweep_plan.items():
            param_name = conf["param_name"]
            values = list(conf["values"])
            means_active, sems_active = [], []
            means_passive, sems_passive = [], []
            means_total, sems_total = [], []
            for fee_value in values:
                fee_value = float(fee_value)
                fee_param_for_sim = fee_value if param_name is not None else None
                samples, seeds_used = run_single_configuration(
                    scenario,
                    fee_param_for_sim,
                    args.steps,
                    rng,
                    args.runs,
                    args.noise_floor,
                    executor,
                    progress,
                )
                mean, sem = summarise(samples)
                means_active.append(mean[0])
                sems_active.append(sem[0])
                means_passive.append(mean[1])
                sems_passive.append(sem[1])
                means_total.append(mean[2])
                sems_total.append(sem[2])

                param_str = "baseline" if param_name is None else f"{fee_value:.6g}"
                seeds_str = "|".join(str(s) for s in seeds_used)
                aggregated_rows.append(
                    f"{scenario},{param_name or 'N/A'},{param_str},active,{mean[0]:.6f},{sem[0]:.6f},{args.runs},{seeds_str}"
                )
                aggregated_rows.append(
                    f"{scenario},{param_name or 'N/A'},{param_str},passive,{mean[1]:.6f},{sem[1]:.6f},{args.runs},{seeds_str}"
                )
                aggregated_rows.append(
                    f"{scenario},{param_name or 'N/A'},{param_str},total,{mean[2]:.6f},{sem[2]:.6f},{args.runs},{seeds_str}"
                )

            plot_payload.append(
                {
                    "scenario": scenario,
                    "param_name": param_name,
                    "param_values": np.array(values, dtype=float),
                    "means": {
                        "active": np.array(means_active),
                        "passive": np.array(means_passive),
                        "total": np.array(means_total),
                    },
                    "sems": {
                        "active": np.array(sems_active),
                        "passive": np.array(sems_passive),
                        "total": np.array(sems_total),
                    },
                }
            )

    # Save CSV summary
    args.csv.write_text("\n".join(aggregated_rows) + "\n")

    # Plot results
    n_plots = len(plot_payload)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(5.5 * n_plots, 4.5), sharey=True, squeeze=False
    )

    for idx, (ax, payload) in enumerate(zip(axes[0], plot_payload)):
        param_vals = payload["param_values"]
        means = payload["means"]
        sems = payload["sems"]
        scenario = payload["scenario"]
        param_label = payload["param_name"] or "N/A"

        if payload["param_name"] is None:
            ax.set_xticks(param_vals)
            ax.set_xticklabels(["static fee" for _ in param_vals])

        ax.errorbar(
            param_vals,
            means["active"],
            yerr=sems["active"],
            fmt="o-",
            capsize=4,
            label="Active LPs",
            alpha=0.7
        )
        ax.errorbar(
            param_vals,
            means["passive"],
            yerr=sems["passive"],
            fmt="s-",
            capsize=4,
            label="Passive LPs",
            alpha=0.7
        )
        # ax.errorbar(
        #     param_vals,
        #     means["total"],
        #     yerr=sems["total"],
        #     fmt="^-",
        #     capsize=4,
        #     label="All LPs",
        # )

        ax.set_title(scenario_title(scenario))
        if payload["param_name"] == "k_sigma":
            ax.set_xlabel(r"$k_\sigma$")
        elif payload["param_name"] == "k_basis":
            ax.set_xlabel(r"$k_{\text{basis}}$")
        elif payload["param_name"] is None:
            ax.set_xlabel("")
        else:
            ax.set_xlabel(param_label if payload["param_name"] else "")
        ax.grid(True, alpha=0.3)

    axes[0][0].set_ylabel("LP PnL (token1)")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    progress.close()
    print(f"[RESULT] Plot saved to {args.output}")
    print(f"[RESULT] CSV saved to {args.csv}")


if __name__ == "__main__":
    main()
