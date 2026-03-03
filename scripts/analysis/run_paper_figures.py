"""scripts/analysis/run_paper_figures.py — CLI orchestrator for paper analyses.

Runs all (model, fee_mode) combinations and generates the full set of
analysis figures.  Each analysis is written under the chosen output directory.

Usage
-----
::

    python -m scripts.analysis.run_paper_figures \\
        --config abm_results/scenarios/test.yml \\
        --runs 10 --max-workers 4 \\
        --output-dir paper/images/analysis

Models
------
- Model 0 : passive_lp_share=1.0, p_jit=0  (passive LPs only)
- Model 1 : passive_lp_share=0.5, p_jit=0  (passive + active LPs)
- Model 2 : passive_lp_share=0.5, p_jit=1  (passive + active + JIT)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.utils import load_simulation_parameters
from scripts.analysis.common import (
    ANALYSIS_KEYS, run_multi_seed, save_figure,
)
from scripts.analysis.pnl_heatmap import pnl_summary_table
from scripts.analysis.scalar_heatmaps import dex_share_barplot, fee_value_barplot, mean_fee_barplot
from scripts.analysis.volatility_conditioned import volatility_binned_analysis

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODELS: Dict[str, Dict[str, Any]] = {
    "Model0": {"passive_lp_share": 1.0, "p_jit": 0},
    "Model1": {"passive_lp_share": 0.5, "p_jit": 0},
    "Model2": {"passive_lp_share": 0.5, "p_jit": 1},
}

FEE_MODES = ("static", "toxicity", "volatility_dex", "volatility_cex")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label(model: str, fee_mode: str) -> str:
    return f"{model} — {fee_mode}"


def _run_scenario(
    base_params: Dict[str, Any],
    model_name: str,
    fee_mode: str,
    *,
    n_seeds: int,
    seed_base: int,
    max_workers: int,
) -> List[Dict[str, Any]]:
    """Run a single (model, fee_mode) scenario and return per-seed outputs."""
    params = dict(base_params)
    model_overrides = MODELS[model_name]
    params.update(model_overrides)
    params["fee_mode"] = fee_mode
    print(f"  [{model_name}, {fee_mode}] running {n_seeds} seeds ...")
    return run_multi_seed(
        params, n_seeds,
        seed_base=seed_base, max_workers=max_workers,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate all paper analysis figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", required=True, type=Path,
                   help="Base YAML scenario config.")
    p.add_argument("--runs", type=int, default=10,
                   help="Number of seeds per (model, fee_mode) point.")
    p.add_argument("--seed-base", type=int, default=1)
    p.add_argument("--max-workers", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--output-dir", type=Path, default=Path("paper/images/analysis"),
                   help="Root directory for output figures.")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()),
                   help="Which model variants to run.")
    p.add_argument("--fee-modes", nargs="+", default=list(FEE_MODES),
                   choices=list(FEE_MODES),
                   help="Which fee modes to run.")
    p.add_argument("--skip", type=int, default=None,
                   help="Override skip_step (burn-in blocks).")
    args = p.parse_args()

    from scripts.run import simulate
    config_path = args.config.expanduser().resolve()
    _, base_params = load_simulation_parameters(config_path, simulate_func=simulate)
    base_params = dict(base_params)
    skip = args.skip if args.skip is not None else int(base_params.get("skip_step", 0))

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1: run all (model, fee_mode) combinations
    # -----------------------------------------------------------------------
    print("[run_paper_figures] Phase 1: running simulations ...")
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for model in args.models:
        for fee_mode in args.fee_modes:
            label = _label(model, fee_mode)
            all_results[label] = _run_scenario(
                base_params, model, fee_mode,
                n_seeds=args.runs, seed_base=args.seed_base,
                max_workers=args.max_workers,
            )

    # -----------------------------------------------------------------------
    # Phase 2: generate analysis figures
    # -----------------------------------------------------------------------
    print("[run_paper_figures] Phase 2: generating figures ...")

    # 1. PnL heatmap
    print("  → PnL heatmap")
    fig = pnl_summary_table(all_results)
    save_figure(fig, out_dir, "pnl_heatmap")

    # 2. DEX share bar plot
    print("  → DEX share bar plot")
    fig = dex_share_barplot(all_results)
    save_figure(fig, out_dir, "dex_share_barplot")

    # 3. Fee value bar plot
    print("  → Fee value bar plot")
    fig = fee_value_barplot(all_results)
    save_figure(fig, out_dir, "fee_value_barplot")

    # 4. Mean fee level bar plot (dynamic fee modes only)
    print("  → Mean fee level bar plot")
    fig = mean_fee_barplot(all_results)
    save_figure(fig, out_dir, "mean_fee_barplot")

    # 5. Volatility-conditioned analysis
    for model in args.models:
        for fee_mode in args.fee_modes:
            label = _label(model, fee_mode)
            results = all_results[label]
            for cohort in ("passive", "active"):
                if model == "Model0" and cohort == "active":
                    continue
                tag = f"vol_cond_{model}_{fee_mode}_{cohort}"
                print(f"  → {tag}")
                fig = volatility_binned_analysis(
                    results, n_bins=3, skip=skip, cohort=cohort,
                )
                save_figure(fig, out_dir, tag)

    print(f"[run_paper_figures] Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
