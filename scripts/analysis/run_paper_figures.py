"""scripts/analysis/run_paper_figures.py — from-scratch paper figure runner.

Regenerates the simulation-backed data and figures needed by all active figures
in ``paper/ABM_paper.tex``.  This is the from-scratch runner: it executes fresh
simulations for the microstructure diagnostics, cross-scenario outcome summary,
and Model 2 block-size diagnostics.  The companion
``scripts.analysis.regenerate_paper_figures`` script remains cache-first: it
rebuilds the paper image snapshots from already saved tables/arrays.

Usage
-----
::

    python -m scripts.analysis.run_paper_figures \
        --config configs/scenarios/section4_microstructure_model0_static.yml \
        --runs 100 --max-workers 4 \
        --image-dir paper/images --table-dir paper/tables

Models
------
- Model 0 : passive_lp_share=1.0, p_jit=0  (passive LPs only)
- Model 1 : passive_lp_share=0.5, p_jit=0  (passive + active LPs)
- Model 2 : passive_lp_share=0.5, p_jit=1  (passive + active + JIT; each JIT attempt targets the single largest pending swap)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.utils import load_simulation_parameters, scenario_output_root
from scripts.analysis import plot_model2_delta_lvr_combined as delta_lvr
from scripts.analysis import plot_model2_lvr_ratio_combined as lvr_ratio
from scripts.analysis.common import iter_multi_seed, save_figure
from scripts.analysis.pnl_heatmap import pnl_summary_table
from scripts.analysis.scalar_heatmaps import dex_share_barplot, fee_value_barplot, mean_fee_barplot
from scripts.analysis.volatility_conditioned import volatility_binned_analysis_from_artifacts

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODELS: Dict[str, Dict[str, Any]] = {
    "Model0": {"passive_lp_share": 1.0, "p_jit": 0},
    "Model1": {"passive_lp_share": 0.5, "p_jit": 0},
    "Model2": {"passive_lp_share": 0.5, "p_jit": 1},
}

FEE_MODES = ("static", "toxicity", "volatility_dex", "volatility_cex", "linear_asymmetric")
LINEAR_ASYMMETRIC_DEFAULT_SLOPE = 0.5

BLOCKSIZE_CONFIG_NAMES = {
    "static": "model2_static",
    "toxicity": "model2_tox",
    "volatility_cex": "model2_vol_cex",
    "volatility_dex": "model2_vol_dex",
    "linear_asymmetric": "model2_linear_asym",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label(model: str, fee_mode: str) -> str:
    return f"{model} — {fee_mode}"


def _apply_linear_asymmetric_defaults(params: Dict[str, Any], fee_mode: str, slope: float | None) -> None:
    """Ensure paper linear-asymmetric runs are not silently equivalent to static fees."""
    if fee_mode != "linear_asymmetric":
        return
    if slope is not None:
        params["asymmetric_fee_slope"] = float(slope)
        return
    if "asymmetric_fee_slope" not in params:
        params["asymmetric_fee_slope"] = LINEAR_ASYMMETRIC_DEFAULT_SLOPE


def _scenario_params(
    base_params: Dict[str, Any],
    model_name: str,
    fee_mode: str,
    *,
    fee_use_ewma: bool | None = None,
    linear_asymmetric_slope: float | None = None,
) -> Dict[str, Any]:
    """Build the simulation parameter dict for one (model, fee_mode) scenario."""
    params = dict(base_params)
    model_overrides = MODELS[model_name]
    params.update(model_overrides)
    params["fee_mode"] = fee_mode
    _apply_linear_asymmetric_defaults(params, fee_mode, linear_asymmetric_slope)
    if fee_use_ewma is not None:
        params["fee_use_ewma"] = bool(fee_use_ewma)
    return params


def _run_command(cmd: List[str]) -> None:
    """Run a subprocess command, echoing it for provenance."""
    print("[run_paper_figures] $ " + " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


def _write_fee_override_config(base_config_path: Path, output_path: Path, *, fee_use_ewma: bool) -> Path:
    """Write a scenario YAML copy with an explicit EWMA fee-controller override."""
    data = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("simulate"), dict):
        raise ValueError(f"Expected {base_config_path} to contain a mapping with a simulate block")

    data = dict(data)
    simulate_block = dict(data["simulate"])
    simulate_block["fee_use_ewma"] = bool(fee_use_ewma)
    data["simulate"] = simulate_block

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output_path


def _write_model2_blocksize_config(
    base_config_path: Path,
    fee_mode: str,
    output_dir: Path,
    *,
    fee_use_ewma: bool | None = None,
    linear_asymmetric_slope: float | None = None,
) -> Path:
    """Write a generated Model 2 scenario YAML for one block-size sweep."""
    data = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("simulate"), dict):
        raise ValueError(f"Expected {base_config_path} to contain a mapping with a simulate block")

    config_name = BLOCKSIZE_CONFIG_NAMES[fee_mode]
    data = dict(data)
    simulate_block = dict(data["simulate"])
    simulate_block.update(
        {
            "config_name": config_name,
            "fee_mode": fee_mode,
            "passive_lp_share": 0.5,
            "p_jit": 1,
        }
    )
    _apply_linear_asymmetric_defaults(simulate_block, fee_mode, linear_asymmetric_slope)
    if fee_use_ewma is not None:
        simulate_block["fee_use_ewma"] = bool(fee_use_ewma)
    data["fee_mode"] = fee_mode
    data["simulate"] = simulate_block

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{config_name}.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _latest_blocksize_output_dir(config_path: Path) -> Path:
    """Return the latest LVR-vs-blocksize output directory for ``config_path``."""
    root = scenario_output_root(config_path) / "lvr_vs_blocksize"
    candidates = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No block-size output directories found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _single_csv(out_dir: Path, pattern: str) -> Path:
    matches = sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} under {out_dir}")
    return matches[-1]


def _stage_blocksize_sources(blocksize_dirs: Dict[str, Path], staging_root: Path) -> None:
    """Copy fresh block-size summaries into the source layout expected by plotters."""
    if staging_root.exists():
        shutil.rmtree(staging_root)
    for schedule, out_dir in blocksize_dirs.items():
        ratio_dest = staging_root / lvr_ratio.SCHEDULES[schedule]["source_csv"]
        delta_dest = staging_root / delta_lvr.SCHEDULES[schedule]["source_csv"]
        ratio_dest.parent.mkdir(parents=True, exist_ok=True)
        delta_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_single_csv(out_dir, "dLVR_over_dFees_summary*.csv"), ratio_dest)
        shutil.copy2(_single_csv(out_dir, "dLVR_summary*.csv"), delta_dest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate all active paper figures from fresh simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", required=True, type=Path, help="Base YAML scenario config.")
    p.add_argument("--runs", type=int, default=10, help="Number of seeds per (model, fee_mode) point.")
    p.add_argument("--seed-base", type=int, default=1)
    p.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/images/analysis"),
        help="Root directory for cross-scenario output figures.",
    )
    p.add_argument(
        "--image-dir",
        type=Path,
        default=Path("paper/images"),
        help="Paper image snapshot directory for active manuscript figures.",
    )
    p.add_argument(
        "--table-dir",
        type=Path,
        default=Path("paper/tables"),
        help="Paper table/provenance directory.",
    )
    p.add_argument("--style-config", type=Path, default=Path("paper/figure_style.yml"))
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    p.add_argument("--fee-modes", nargs="+", default=list(FEE_MODES), choices=list(FEE_MODES))
    p.add_argument("--skip", type=int, default=None, help="Override skip_step (burn-in blocks).")
    p.add_argument("--skip-microstructure", action="store_true", help="Skip fresh microstructure diagnostic runs.")
    p.add_argument("--skip-cross-scenario", action="store_true", help="Skip fresh cross-scenario outcome runs.")
    p.add_argument("--skip-blocksize", action="store_true", help="Skip fresh Model 2 block-size runs.")
    p.add_argument(
        "--no-ewma",
        action="store_true",
        help="Disable EWMA smoothing for all generated fee-schedule simulations and configs.",
    )
    p.add_argument(
        "--linear-asymmetric-slope",
        type=float,
        default=None,
        help=(
            "Slope used when generating linear_asymmetric paper scenarios. If omitted, "
            f"uses an explicit nonzero default ({LINEAR_ASYMMETRIC_DEFAULT_SLOPE}) "
            "when the base config does not already define asymmetric_fee_slope."
        ),
    )
    p.add_argument("--blocksize-runs", type=int, default=50)
    p.add_argument("--blocksize-seed-base", type=int, default=10)
    p.add_argument("--blocksize-seed-step", type=int, default=1)
    p.add_argument("--blocksize-min", type=int, default=2)
    p.add_argument("--blocksize-max", type=int, default=16)
    args = p.parse_args()

    from scripts.run import simulate

    config_path = args.config.expanduser().resolve()
    _, base_params = load_simulation_parameters(config_path, simulate_func=simulate)
    base_params = dict(base_params)
    fee_use_ewma_override = False if args.no_ewma else None
    if args.no_ewma:
        base_params["fee_use_ewma"] = False
    skip = args.skip if args.skip is not None else int(base_params.get("skip_step", 0))

    out_dir = args.output_dir.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()
    table_dir = args.table_dir.expanduser().resolve()
    style_config = args.style_config.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 0: active paper microstructure figures from fresh simulations
    # -----------------------------------------------------------------------
    if not args.skip_microstructure:
        print("[run_paper_figures] Phase 0: running microstructure diagnostics ...")
        microstructure_static_config = _REPO_ROOT / "configs/scenarios/section4_microstructure_model0_static.yml"
        microstructure_toxicity_config = _REPO_ROOT / "configs/scenarios/section4_microstructure_model0_toxicity.yml"
        if args.no_ewma:
            generated_config_dir = _REPO_ROOT / "configs/paper/run_paper_figures_no_ewma_configs"
            microstructure_static_config = _write_fee_override_config(
                microstructure_static_config,
                generated_config_dir / "section4_microstructure_model0_static_no_ewma.yml",
                fee_use_ewma=False,
            )
            microstructure_toxicity_config = _write_fee_override_config(
                microstructure_toxicity_config,
                generated_config_dir / "section4_microstructure_model0_toxicity_no_ewma.yml",
                fee_use_ewma=False,
            )
        _run_command(
            [
                sys.executable,
                "-m",
                "scripts.analysis.microstructure_fee_diagnostics",
                "--static-config",
                str(microstructure_static_config),
                "--toxicity-config",
                str(microstructure_toxicity_config),
                "--image-dir",
                str(image_dir),
                "--table-dir",
                str(table_dir),
                "--force",
            ]
        )
        _run_command(
            [
                sys.executable,
                "-m",
                "scripts.analysis.regenerate_paper_figures",
                "--figures",
                "representative-price",
                "--image-dir",
                str(image_dir),
                "--table-dir",
                str(table_dir),
                "--style-config",
                str(style_config),
                "--source-root",
                str(_REPO_ROOT),
                "--manifest",
                str(table_dir / "regenerate_paper_figures_manifest.json"),
            ]
        )

    # -----------------------------------------------------------------------
    # Phase 1: run one scenario at a time and generate scenario-local figures
    # -----------------------------------------------------------------------
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    if not args.skip_cross_scenario:
        print("[run_paper_figures] Phase 1: running simulations and streaming scenario artifacts ...")
        for model in args.models:
            for fee_mode in args.fee_modes:
                label = _label(model, fee_mode)
                params = _scenario_params(
                    base_params,
                    model,
                    fee_mode,
                    fee_use_ewma=fee_use_ewma_override,
                    linear_asymmetric_slope=args.linear_asymmetric_slope,
                )
                print(f"  [{model}, {fee_mode}] running {args.runs} seeds ...")
                with tempfile.TemporaryDirectory(prefix=f"abm_paper_{model}_{fee_mode}_") as tmp_dir:
                    artifact_dir = Path(tmp_dir)
                    scenario_scalars: List[Dict[str, Any]] = []
                    artifact_records: List[tuple[int, Path]] = []
                    for payload in iter_multi_seed(
                        params,
                        args.runs,
                        seed_base=args.seed_base,
                        max_workers=args.max_workers,
                        mode="paper",
                        artifact_dir=artifact_dir,
                        skip=skip,
                    ):
                        scenario_scalars.append(dict(payload["scalars"]))
                        artifact_records.append((int(payload["seed"]), Path(payload["artifact_path"])))
                    scenario_scalars.sort(key=lambda row: int(row.get("seed", 0)))
                    artifact_paths = [path for _, path in sorted(artifact_records, key=lambda item: item[0])]
                    all_results[label] = scenario_scalars

                    for cohort in ("passive", "active"):
                        if model == "Model0" and cohort == "active":
                            continue
                        tag = f"vol_cond_{model}_{fee_mode}_{cohort}"
                        print(f"  → {tag}")
                        fig = volatility_binned_analysis_from_artifacts(
                            artifact_paths,
                            n_bins=5,
                            cohort=cohort,
                        )
                        save_figure(fig, out_dir, tag)

        # -------------------------------------------------------------------
        # Phase 2: generate cross-scenario figures from scalar summaries only
        # -------------------------------------------------------------------
        print("[run_paper_figures] Phase 2: generating cross-scenario figures ...")
        print("  → PnL heatmap")
        save_figure(pnl_summary_table(all_results), out_dir, "pnl_heatmap")

        print("  → DEX share bar plot")
        save_figure(dex_share_barplot(all_results), out_dir, "dex_share_barplot")

        print("  → Fee value bar plot")
        save_figure(fee_value_barplot(all_results), out_dir, "fee_value_barplot")

        print("  → Mean fee level bar plot")
        save_figure(mean_fee_barplot(all_results), out_dir, "mean_fee_barplot")

    # -----------------------------------------------------------------------
    # Phase 3: Model 2 block-size figures from fresh sweeps
    # -----------------------------------------------------------------------
    if not args.skip_blocksize:
        print("[run_paper_figures] Phase 3: running Model 2 block-size sweeps ...")
        generated_config_dir = _REPO_ROOT / "configs/paper/run_paper_figures_blocksize_configs"
        blocksize_dirs: Dict[str, Path] = {}
        for fee_mode in FEE_MODES:
            cfg = _write_model2_blocksize_config(
                config_path,
                fee_mode,
                generated_config_dir,
                fee_use_ewma=fee_use_ewma_override,
                linear_asymmetric_slope=args.linear_asymmetric_slope,
            )
            _run_command(
                [
                    sys.executable,
                    "-m",
                    "scripts.LVR_vs_blocksize",
                    "--config",
                    str(cfg),
                    "--runs",
                    str(args.blocksize_runs),
                    "--block-min",
                    str(args.blocksize_min),
                    "--block-max",
                    str(args.blocksize_max),
                    "--seed-base",
                    str(args.blocksize_seed_base),
                    "--seed-step",
                    str(args.blocksize_seed_step),
                    "--max-workers",
                    str(args.max_workers),
                    "--summary-only",
                    "--skip-png",
                    "--fee-definition",
                    "flow",
                    "--fee-modes",
                    fee_mode,
                ]
            )
            blocksize_dirs[fee_mode] = _latest_blocksize_output_dir(cfg)

        staging_root = table_dir / "run_paper_figures_blocksize_sources"
        _stage_blocksize_sources(blocksize_dirs, staging_root)

        if int(args.blocksize_min) != 2 or int(args.blocksize_max) != 16:
            print(
                "[run_paper_figures] Skipping combined block-size paper plots because "
                "the plotting validators require the paper grid B=2..16. "
                "Fresh summary CSVs were still generated and staged."
            )
            print(f"[run_paper_figures] Done. Cross-scenario outputs in {out_dir}/; active paper images in {image_dir}/")
            return

        print("  → Model 2 fee-coverage ratio")
        _run_command(
            [
                sys.executable,
                "-m",
                "scripts.analysis.plot_model2_lvr_ratio_combined",
                "--rebuild-data-from-source-csv",
                "--source-root",
                str(staging_root),
                "--data",
                str(table_dir / "model2_blocksize_ratio_values.csv"),
                "--output",
                str(image_dir / "model2_blocksize_ratio_combined.png"),
                "--style-config",
                str(style_config),
                "--manifest",
                str(table_dir / "model2_blocksize_ratio_combined_manifest.json"),
            ]
        )

        print("  → Model 2 ΔLVR")
        _run_command(
            [
                sys.executable,
                "-m",
                "scripts.analysis.plot_model2_delta_lvr_combined",
                "--rebuild-data-from-source-csv",
                "--source-root",
                str(staging_root),
                "--data",
                str(table_dir / "model2_delta_lvr_blocksize_values.csv"),
                "--output",
                str(image_dir / "model2_delta_lvr_blocksize_combined.png"),
                "--style-config",
                str(style_config),
                "--manifest",
                str(table_dir / "model2_delta_lvr_blocksize_combined_manifest.json"),
            ]
        )

    print(f"[run_paper_figures] Done. Cross-scenario outputs in {out_dir}/; active paper images in {image_dir}/")


if __name__ == "__main__":
    main()
