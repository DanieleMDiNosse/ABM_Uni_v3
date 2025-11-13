#!/usr/bin/env python3

"""
Sweep the ABM across fee modes and fee-control parameters, aggregating LP PnL.

Parameters are supplied via a YAML configuration. For every scenario (static,
volatility, toxicity) this script runs several simulations at different
fee-parameter values, collects the end-of-run PnL for active/passive LP
cohorts, and plots the mean ± SEM for each configuration.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

sys.path.append(str(Path(__file__).resolve().parent))

from run import simulate  # noqa: E402
from utils import load_simulation_parameters  # noqa: E402


SCENARIO_FEE_MODES: Dict[str, str] = {
    "static_baseline": "static",
    "volatility_baseline": "volatility",
    "toxicity_baseline": "toxicity",
}


@dataclass
class ScenarioConfig:
    param_name: Optional[str]
    values: Sequence[float]


@dataclass
class SweepConfig:
    runs: int
    steps: int
    seed: int
    workers: int
    noise_floor: Optional[float]
    output: Path
    csv: Path
    base_simulation_config: Path
    scenarios: Dict[str, ScenarioConfig]


def load_base_parameters(config_path: Path) -> Dict[str, Any]:
    """Load base simulate() parameters from a YAML config, requiring all keys to be specified."""
    scenario_label, params = load_simulation_parameters(config_path, simulate_func=simulate)
    # We don't need the label here; enforce explicit fee_mode later.
    return params


def build_scenario_base_params(common_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Create per-scenario parameter dictionaries starting from common defaults."""
    scenario_params: Dict[str, Dict[str, Any]] = {}
    for scenario, fee_mode in SCENARIO_FEE_MODES.items():
        params = dict(common_params)
        params["fee_mode"] = fee_mode
        params["visualize"] = False  # disable plotting during sweeps
        scenario_params[scenario] = params
    return scenario_params


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for run_fee_sweep.py configuration parsing. Install 'pyyaml'.")


def load_sweep_config(config_path: Path) -> SweepConfig:
    """Load and validate the fee sweep configuration from YAML."""
    _require_yaml()

    resolved = config_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Fee sweep configuration file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Root of {resolved} must be a mapping.")

    fee_sweep_cfg = data.get("fee_sweep")
    if not isinstance(fee_sweep_cfg, dict):
        raise ValueError(f"Configuration must contain a 'fee_sweep' mapping in {resolved}.")

    required_keys = [
        "runs",
        "steps",
        "seed",
        "workers",
        "noise_floor",
        "output",
        "csv",
        "base_simulation_config",
        "scenarios",
    ]
    missing = [key for key in required_keys if key not in fee_sweep_cfg]
    if missing:
        raise ValueError(f"Missing required fee sweep keys in {resolved}: {missing}")

    try:
        runs = int(fee_sweep_cfg["runs"])
        steps = int(fee_sweep_cfg["steps"])
        seed = int(fee_sweep_cfg["seed"])
        workers = int(fee_sweep_cfg["workers"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Runs, steps, seed, and workers must be integers.") from exc

    noise_floor_raw = fee_sweep_cfg["noise_floor"]
    if noise_floor_raw is None:
        noise_floor = None
    else:
        try:
            noise_floor = float(noise_floor_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("noise_floor must be a float or null.") from exc
        if not (0.0 <= noise_floor <= 1.0):
            raise ValueError("noise_floor must be within [0, 1].")

    cfg_dir = resolved.parent
    output_path = (cfg_dir / Path(fee_sweep_cfg["output"])).resolve()
    csv_path = (cfg_dir / Path(fee_sweep_cfg["csv"])).resolve()
    base_sim_path = (cfg_dir / Path(fee_sweep_cfg["base_simulation_config"])).resolve()
    if not base_sim_path.exists():
        raise FileNotFoundError(f"Base simulation configuration not found: {base_sim_path}")

    scenarios_raw = fee_sweep_cfg["scenarios"]
    if not isinstance(scenarios_raw, dict):
        raise ValueError("'scenarios' must be a mapping of scenario names to configuration blocks.")

    missing_scenarios = [name for name in SCENARIO_FEE_MODES if name not in scenarios_raw]
    if missing_scenarios:
        raise ValueError(f"Missing scenario entries: {missing_scenarios}")

    parsed_scenarios: Dict[str, ScenarioConfig] = {}
    for scenario, fee_mode in SCENARIO_FEE_MODES.items():
        block = scenarios_raw.get(scenario)
        if not isinstance(block, dict):
            raise ValueError(f"Scenario '{scenario}' configuration must be a mapping.")
        if "param_name" not in block or "values" not in block:
            raise ValueError(f"Scenario '{scenario}' must specify 'param_name' and 'values'.")

        param_name = block["param_name"]
        if scenario == "static_baseline":
            if param_name is not None:
                raise ValueError("Scenario 'static_baseline' must have param_name set to null.")
        else:
            if not isinstance(param_name, str) or not param_name:
                raise ValueError(f"Scenario '{scenario}' requires a non-empty string 'param_name'.")

        raw_values = block["values"]
        if not isinstance(raw_values, (list, tuple)):
            raise ValueError(f"Scenario '{scenario}' 'values' must be a list of numbers.")
        if len(raw_values) == 0:
            raise ValueError(f"Scenario '{scenario}' must provide at least one value.")
        try:
            values = [float(v) for v in raw_values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Scenario '{scenario}' values must be numeric.") from exc

        parsed_scenarios[scenario] = ScenarioConfig(
            param_name=None if param_name is None else str(param_name),
            values=tuple(values),
        )

    return SweepConfig(
        runs=runs,
        steps=steps,
        seed=seed,
        workers=workers,
        noise_floor=noise_floor,
        output=output_path,
        csv=csv_path,
        base_simulation_config=base_sim_path,
        scenarios=parsed_scenarios,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fee parameter sweeps for the ABM using a YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file describing the sweep.",
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
    base_params: Dict[str, Any],
) -> Tuple[int, float, float, float]:
    """Run a single simulation and return final LP wealth components."""
    params = dict(base_params)
    params.update(
        {
            "T": steps,
            "seed": int(seed),
            "visualize": False,
        }
    )
    params["fee_mode"] = SCENARIO_FEE_MODES[scenario]
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
    scenario_params: Dict[str, Any],
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
            scenario_params,
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
    sweep_cfg = load_sweep_config(args.config)

    rng = np.random.default_rng(sweep_cfg.seed)
    ensure_parent(sweep_cfg.output)
    ensure_parent(sweep_cfg.csv)

    common_base_params = load_base_parameters(sweep_cfg.base_simulation_config)
    scenario_base_params = build_scenario_base_params(common_base_params)

    sweep_plan: Dict[str, Dict[str, Sequence[float]]] = {
        scenario: {
            "param_name": sweep_cfg.scenarios[scenario].param_name,
            "values": sweep_cfg.scenarios[scenario].values,
        }
        for scenario in SCENARIO_FEE_MODES
    }

    aggregated_rows: List[str] = [
        "scenario,parameter_name,parameter_value,component,mean,sem,num_runs,seeds"
    ]

    plot_payload = []

    total_iterations = sum(len(conf["values"]) for conf in sweep_plan.values()) * sweep_cfg.runs
    if total_iterations <= 0:
        raise ValueError("Sweep configuration must provide at least one parameter value.")

    progress = tqdm(total=total_iterations, desc="Running simulations", unit="run")

    try:
        with ProcessPoolExecutor(max_workers=sweep_cfg.workers) as executor:
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
                        sweep_cfg.steps,
                        rng,
                        sweep_cfg.runs,
                        sweep_cfg.noise_floor,
                        executor,
                        progress,
                        scenario_base_params[scenario],
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
                        f"{scenario},{param_name or 'N/A'},{param_str},active,{mean[0]:.6f},{sem[0]:.6f},{sweep_cfg.runs},{seeds_str}"
                    )
                    aggregated_rows.append(
                        f"{scenario},{param_name or 'N/A'},{param_str},passive,{mean[1]:.6f},{sem[1]:.6f},{sweep_cfg.runs},{seeds_str}"
                    )
                    aggregated_rows.append(
                        f"{scenario},{param_name or 'N/A'},{param_str},total,{mean[2]:.6f},{sem[2]:.6f},{sweep_cfg.runs},{seeds_str}"
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
    finally:
        progress.close()

    # Save CSV summary
    sweep_cfg.csv.write_text("\n".join(aggregated_rows) + "\n")

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
    fig.savefig(sweep_cfg.output, dpi=150, bbox_inches="tight")
    print(f"[RESULT] Plot saved to {sweep_cfg.output}")
    print(f"[RESULT] CSV saved to {sweep_cfg.csv}")


if __name__ == "__main__":
    main()
